"""
Real-time Voice Conversational Chat with Ollama — Sarvam AI TTS Edition

Architecture: LLM streams tokens → clause buffer → Sarvam AI cloud TTS
synthesizes each clause → sounddevice plays it. Indian-English voice via
Bulbul v3 model.  STT uses faster-whisper (base model, beam-search).

Ctrl+C → interrupt / text fallback / quit

Setup:
  1. pip install faster-whisper sounddevice numpy ollama requests
  2. Set env var:  SARVAM_API_KEY=<your key from dashboard.sarvam.ai>
  3. Have Ollama running with gemma2:2b (or change OLLAMA_MODEL)
"""

import re
import os
import io
from dotenv import load_dotenv

load_dotenv()  # Load .env file automatically
import time
import base64
import threading
import queue
import numpy as np
import sounddevice as sd
import ollama
import requests
from faster_whisper import WhisperModel

# ── Config ──────────────────────────────────────────────────────────────────
WHISPER_MODEL = "base"          # upgraded from tiny → base for better accuracy
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
OLLAMA_MODEL = "gemma2:2b"
OLLAMA_NUM_GPU = 99
SAMPLE_RATE = 16000
CHANNELS = 1

# Sarvam AI TTS
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_API_KEY_HERE")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_MODEL = "bulbul:v3"
SARVAM_SPEAKER = "shubh"       # Indian-English male voice
SARVAM_LANGUAGE = "en-IN"
SARVAM_SAMPLE_RATE = 22050     # supported: 8000, 16000, 22050, 24000
SARVAM_PACE = 0.9              # slower, more natural conversational pace (0.3-3)
SARVAM_LOUDNESS = 1.0          # (0.3-3)
SARVAM_PITCH = 0.0             # (-0.75 to 0.75, v2 only)

# VAD — tight settings for snappy turn-taking
SILENCE_THRESHOLD = 0.010
SPEECH_THRESHOLD = 0.015
SILENCE_DURATION = 0.4
PRE_SPEECH_BUFFER = 0.5        # increased from 0.3 → 0.5 to capture word onsets
MIN_SPEECH_DURATION = 0.25
MAX_SPEECH_DURATION = 30
CHUNK_MS = 50

# Idle timeout — AI follows up if user stays silent
IDLE_TIMEOUT = 8.0             # seconds of silence before AI prompts user
MAX_IDLE_FOLLOWUPS = 3         # max consecutive follow-ups before gentle exit hint

# Flush triggers — send whole response as one TTS chunk (cloud API, fewer calls = fewer pauses)
SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[;:])\s+')
CLAUSE_BREAK = re.compile(r'(?<=[,\-\—])\s+|\s+(?=and\s|but\s|or\s|so\s|because\s|then\s)')
MIN_FLUSH_LEN = 9999         # disable mid-stream sentence flush (whole response as one chunk)
MIN_CLAUSE_LEN = 9999        # disable mid-stream clause flush
SAFETY_FLUSH_LEN = 500       # only split extremely long responses

SYSTEM_PROMPT = (
    "You are a helpful conversational assistant in a voice call. "
    "Keep answers concise — 1 to 3 sentences maximum. "
    "Never use markdown, lists, formatting, or emojis. Speak naturally "
    "as plain text only."
)

# Regex to strip emojis and special characters before TTS
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\U00002600-\U000026FF"   # misc symbols
    "\U0000200D"              # zero width joiner
    "\U00002B50-\U00002B55"   # stars
    "]+")
_SPECIAL_RE = re.compile(r'[*#_~`|<>{}\\[\]\\]+')

# Whitelist: only keep characters that are actually speakable
# Letters, digits, spaces, and basic punctuation
_SPEAKABLE_RE = re.compile(r"[^a-zA-Z0-9\s.,;:!?'\"\-\—\(\)]")


def clean_for_tts(text: str) -> str:
    """Strip everything non-speakable from text, keeping only plain English + punctuation."""
    # First pass: remove emojis (broad ranges)
    text = _EMOJI_RE.sub(' ', text)
    
    # Second pass: remove markdown/special chars
    text = _SPECIAL_RE.sub(' ', text)
    
    # Third pass (safety net): strip ANY remaining non-speakable character
    text = _SPEAKABLE_RE.sub(' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,;:.])', r'\1', text)
    text = re.sub(r'([,;:.!?])(\S)', r'\1 \2', text)
    return text.strip()


# ── Globals ─────────────────────────────────────────────────────────────────
messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
_interrupted = False
_tts_queue: queue.Queue[str | None] = queue.Queue()
_TIMEOUT_SENTINEL = np.array([], dtype=np.float32)  # unique sentinel for idle timeout
_tts_thread: threading.Thread | None = None
_sarvam_ready = threading.Event()
_tts_done = threading.Event()   # set when TTS worker is truly idle (done playing)
_speech_thresh_global = 0.015  # will be set after calibration

# Persistent mic monitor (callback-based, coexists with sd.play)
_mic_level = 0.0               # RMS level updated by audio callback
_mic_stream: sd.InputStream | None = None

# Whisper hallucination filter
# AI follow-up prompts when user is idle (cycled through)
_IDLE_PROMPTS = [
    "The user has been silent for a while. Gently ask if they're still there or if they need a moment to think. Keep it brief and natural.",
    "The user hasn't responded. Ask a brief follow-up question related to what you were just talking about, to re-engage them.",
    "The user is still silent. Politely let them know you're here whenever they're ready, and suggest they can say 'quit' to end the conversation.",
]

_HALLUCINATIONS = {
    "thank you", "thanks for watching", "bye", "you", "",
    "thank you for watching", "thanks", "the end",
    "please subscribe", "like and subscribe",
    "subtitles by", "subtitle", "silence",
}


# ── TTS: Sarvam AI worker thread ─────────────────────────────────────────────

def _sarvam_synthesize(text: str) -> np.ndarray | None:
    """Call Sarvam AI TTS API and return audio as float32 numpy array."""
    text = text.strip()
    if not text:
        return None

    # Ensure proper sentence ending
    if text[-1] not in '.!?':
        text += '.'

    payload = {
        "inputs": [text],
        "target_language_code": SARVAM_LANGUAGE,
        "speaker": SARVAM_SPEAKER,
        "model": SARVAM_MODEL,
        "pace": SARVAM_PACE,
        "speech_sample_rate": SARVAM_SAMPLE_RATE,
        "enable_preprocessing": True,
    }

    # Pitch and loudness are only supported on bulbul:v2, not v3
    if "v2" in SARVAM_MODEL:
        payload["pitch"] = SARVAM_PITCH
        payload["loudness"] = SARVAM_LOUDNESS

    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
    }

    try:
        r = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Response: {"request_id": "...", "audios": ["base64..."]}
        audios = data.get("audios", [])
        if not audios or not audios[0]:
            print("\n  [Sarvam: empty audio response]", end="", flush=True)
            return None

        raw_bytes = base64.b64decode(audios[0])

        # Sarvam returns WAV — parse it to get raw PCM
        audio_array = _wav_bytes_to_float32(raw_bytes)
        return audio_array

    except requests.exceptions.HTTPError as e:
        print(f"\n  [Sarvam API error {e.response.status_code}: {e.response.text[:200]}]", end="", flush=True)
        return None
    except Exception as e:
        print(f"\n  [Sarvam TTS error: {e}]", end="", flush=True)
        return None


def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    """Parse WAV bytes into a float32 numpy array."""
    import wave
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    return audio


def _tts_worker():
    """Dedicated thread: consume text queue → Sarvam AI TTS → play via sounddevice."""
    global _interrupted
    # Warmup — verify API key works
    print("  Verifying Sarvam AI TTS...", end="", flush=True)
    test_audio = _sarvam_synthesize("Hello.")
    if test_audio is not None:
        print(f" ok ({SARVAM_SAMPLE_RATE}Hz, speaker={SARVAM_SPEAKER})")
    else:
        print(" WARNING: warmup failed — check your SARVAM_API_KEY!")

    _sarvam_ready.set()
    _tts_done.set()  # initially idle

    prefetched_audio: np.ndarray | None = None

    while True:
        # ── Get audio for current clause ───────────────────────────────
        if prefetched_audio is not None:
            audio = prefetched_audio
            prefetched_audio = None
        else:
            # Wait for work — only signal _tts_done when queue is truly empty
            try:
                text = _tts_queue.get(timeout=0.2)
            except queue.Empty:
                _tts_done.set()  # genuinely idle: queue empty AND nothing playing
                text = _tts_queue.get()  # block until new work arrives
            _tts_done.clear()  # got work, mark as busy
            if text is None:
                _tts_done.set()
                break
            if _interrupted:
                while not _tts_queue.empty():
                    try:
                        _tts_queue.get_nowait()
                    except queue.Empty:
                        break
                continue
            text = clean_for_tts(text)
            if not text:
                continue
            try:
                audio = _sarvam_synthesize(text)
            except Exception as e:
                print(f"\n  [TTS error: {e}]", end="", flush=True)
                continue
            if audio is None:
                continue

        # Check interruption AFTER synthesis (user may have spoken during API call)
        if _interrupted:
            prefetched_audio = None
            continue

        # ── Play current audio (non-blocking) ─────────────────────────
        sd.play(audio, samplerate=SARVAM_SAMPLE_RATE)
        play_duration = len(audio) / SARVAM_SAMPLE_RATE
        play_start = time.time()
        consecutive_speech = 0

        # ── Wait for playback with barge-in detection ─────────────────
        # Poll every 50ms: check _interrupted flag AND mic level for
        # user speech (barge-in). This replaces sd.wait() which can't
        # be interrupted on Windows.
        while time.time() - play_start < play_duration + 0.1:
            if _interrupted:
                sd.stop()
                prefetched_audio = None
                break
            # Check mic level for barge-in (2 consecutive = ~100ms, faster detection)
            if _mic_level > _speech_thresh_global:
                consecutive_speech += 1
                if consecutive_speech >= 2:  # ~100ms of speech → immediate stop
                    elapsed_play = time.time() - play_start
                    print(f"\n  ✋ [interrupted at {elapsed_play:.1f}s]", end="", flush=True)
                    sd.stop()
                    _interrupted = True
                    # Drain TTS queue
                    while not _tts_queue.empty():
                        try:
                            _tts_queue.get_nowait()
                        except queue.Empty:
                            break
                    prefetched_audio = None
                    break
            else:
                consecutive_speech = 0
            time.sleep(0.05)


def start_tts():
    """Start the TTS worker thread."""
    global _tts_thread
    _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    _tts_thread.start()


def stop_speaking():
    """Interrupt current speech and drain the queue."""
    global _interrupted
    _interrupted = True
    try:
        sd.stop()  # immediately stop audio playback
    except Exception:
        pass
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
        except queue.Empty:
            break
    _tts_done.set()  # unblock any wait loops immediately


# ── Persistent mic monitor (callback-based) ─────────────────────────────────

def _mic_callback(indata, frames, time_info, status):
    """Audio callback: runs in native audio thread, updates mic level."""
    global _mic_level
    _mic_level = float(np.sqrt(np.mean(indata ** 2))) if indata.size else 0.0


def start_mic_monitor():
    """Start persistent callback-based mic stream for barge-in detection."""
    global _mic_stream
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    _mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
        callback=_mic_callback, blocksize=chunk_samples,
    )
    _mic_stream.start()


def stop_mic_monitor():
    """Stop the persistent mic stream."""
    global _mic_stream
    if _mic_stream is not None:
        try:
            _mic_stream.stop()
            _mic_stream.close()
        except Exception:
            pass
        _mic_stream = None


def pause_mic_monitor():
    """Pause mic monitor (before listen() opens its own stream)."""
    if _mic_stream is not None:
        try:
            _mic_stream.stop()
        except Exception:
            pass


def resume_mic_monitor():
    """Resume mic monitor (after listen() closes its stream)."""
    if _mic_stream is not None:
        try:
            _mic_stream.start()
        except Exception:
            pass


# ── Helpers ─────────────────────────────────────────────────────────────────
def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0


def calibrate_noise() -> float:
    print("  Calibrating mic...", end="", flush=True)
    samples = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as s:
        for _ in range(5):
            s.read(int(SAMPLE_RATE * 0.05))
        for _ in range(30):
            d, _ = s.read(int(SAMPLE_RATE * 0.05))
            samples.append(rms(d))
    ambient = float(np.mean(samples))
    print(f" {ambient:.5f}")
    if ambient < 0.0001:
        print("  ⚠ Mic near-zero — using defaults.")
    return ambient


# ── STT (improved) ──────────────────────────────────────────────────────────
def load_whisper() -> WhisperModel:
    print(f"  Loading Whisper ({WHISPER_MODEL})...", end="", flush=True)
    m = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print(" ok")
    return m


def listen(speech_thresh: float, silence_thresh: float, timeout: float = 0) -> np.ndarray | None:
    """Listen for speech. Returns audio, None (interrupted), or _TIMEOUT_SENTINEL (idle timeout)."""
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    pre_chunks = int(PRE_SPEECH_BUFFER / (CHUNK_MS / 1000))
    ring: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    started = False
    sil_start: float | None = None
    rec_start: float | None = None
    wait_start = time.time()  # track how long we've been waiting for speech to begin

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as s:
            while True:
                d, _ = s.read(chunk_samples)
                c = d.copy()
                lv = rms(c)
                if not started:
                    # Check idle timeout (only before speech starts)
                    if timeout > 0 and (time.time() - wait_start) >= timeout:
                        return _TIMEOUT_SENTINEL
                    ring.append(c)
                    if len(ring) > pre_chunks:
                        ring.pop(0)
                    if lv > speech_thresh:
                        started = True
                        rec_start = time.time()
                        sil_start = None
                        frames.extend(ring)
                        ring.clear()
                        frames.append(c)
                        print("\r  🎙 ...", end="", flush=True)
                else:
                    frames.append(c)
                    elapsed = time.time() - rec_start  # type: ignore
                    if lv < silence_thresh:
                        if sil_start is None:
                            sil_start = time.time()
                        elif time.time() - sil_start >= SILENCE_DURATION:
                            print(f"\r  🎙 {elapsed - SILENCE_DURATION:.1f}s", flush=True)
                            break
                    else:
                        sil_start = None
                    if elapsed >= MAX_SPEECH_DURATION:
                        break
    except KeyboardInterrupt:
        return None

    if not frames:
        return None
    audio = np.concatenate(frames).flatten()
    trim = int(SILENCE_DURATION * SAMPLE_RATE * 0.8)
    if len(audio) > trim:
        audio = audio[:-trim]
    return audio if len(audio) / SAMPLE_RATE >= MIN_SPEECH_DURATION else None


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    """Transcribe audio with improved settings and hallucination filtering."""
    # Reject audio that's too quiet to be real speech
    if rms(audio) < 0.01:
        return ""

    segs, _ = model.transcribe(
        audio,
        beam_size=5,                # upgraded from 1 → 5 for better accuracy
        language="en",
        initial_prompt="This is a casual voice conversation.",  # primes vocabulary
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = " ".join(s.text for s in segs).strip()

    # Filter common Whisper hallucinations
    if text.lower().strip(".!? ") in _HALLUCINATIONS:
        return ""

    # Filter very short single-character artifacts
    if len(text) <= 1:
        return ""

    return text


# ── LLM → clause queue → Sarvam TTS (overlapping) ──────────────────────────

def chat_and_speak(user_text: str):
    """
    Stream LLM tokens → split into clauses → Sarvam AI speaks each.

    Sarvam synthesizes in the cloud with natural Indian-English voice.
    Each clause plays as soon as synthesis completes.
    Barge-in: mic is monitored — if user speaks, AI stops immediately.
    """
    global _interrupted
    _interrupted = False

    messages.append({"role": "user", "content": user_text})

    full_tokens: list[str] = []
    sentence_buf: list[str] = []
    t0 = time.time()
    was_interrupted = False

    print("  AI: ", end="", flush=True)

    try:
        for chunk in ollama.chat(
            model=OLLAMA_MODEL, messages=messages, stream=True,
            options={"num_gpu": OLLAMA_NUM_GPU, "num_predict": 150},
        ):
            if _interrupted:
                was_interrupted = True
                break
            try:
                token = chunk.message.content
            except AttributeError:
                token = chunk["message"]["content"]
            if not token:
                continue

            print(token, end="", flush=True)
            full_tokens.append(token)
            sentence_buf.append(token)
            buf_text = "".join(sentence_buf)

            # 1) Sentence boundary
            if len(buf_text) >= MIN_FLUSH_LEN:
                parts = SENTENCE_END.split(buf_text)
                if len(parts) > 1:
                    to_flush = " ".join(
                        p.strip() for p in parts[:-1] if p.strip()
                    )
                    if to_flush:
                        _tts_queue.put(to_flush)
                    tail = parts[-1]
                    sentence_buf = [tail] if tail.strip() else []
                    continue

            # 2) Soft clause boundary
            if len(buf_text) >= MIN_CLAUSE_LEN:
                cm = CLAUSE_BREAK.search(buf_text)
                if cm:
                    head = buf_text[:cm.end()].strip()
                    tail = buf_text[cm.end():]
                    if head:
                        _tts_queue.put(head)
                    sentence_buf = [tail] if tail.strip() else []
                    continue

            # 3) Safety flush
            if len(buf_text) > SAFETY_FLUSH_LEN:
                _tts_queue.put(buf_text.strip())
                sentence_buf = []

    except KeyboardInterrupt:
        _interrupted = True
        was_interrupted = True
    except Exception as e:
        print(f"\n  [LLM error: {e}]")
        messages.pop()
        stop_speaking()
        return

    print(f"  ({time.time()-t0:.1f}s)", flush=True)

    # Flush remaining text (only if not interrupted)
    remainder = "".join(sentence_buf).strip()
    if remainder and not _interrupted:
        _tts_done.clear()  # mark busy BEFORE queuing so wait loop can't exit early
        _tts_queue.put(remainder)

    reply = "".join(full_tokens)

    if _interrupted or was_interrupted:
        # On interruption: store only a note that AI was cut off
        # so the AI doesn't reference things the user never heard
        stop_speaking()
        if reply:
            # Truncate to what was likely heard (first sentence or partial)
            first_sentence = re.split(r'[.!?]', reply, maxsplit=1)[0].strip()
            if first_sentence:
                messages.append({"role": "assistant", "content": first_sentence + "... [user interrupted]"})
            else:
                messages.pop()  # remove user message too since AI said nothing audible
        else:
            messages.pop()
    elif reply:
        messages.append({"role": "assistant", "content": reply})
        # Wait for TTS worker to finish ALL playback
        while not _tts_done.wait(timeout=0.1):
            if _interrupted:
                stop_speaking()
                break
    else:
        messages.pop()


def chat_and_speak_idle():
    """
    AI proactive follow-up when user is idle.
    Messages are already set up by the caller — just stream LLM and speak.
    Shorter response limit to keep follow-ups brief.
    """
    global _interrupted
    _interrupted = False

    full_tokens: list[str] = []
    sentence_buf: list[str] = []
    t0 = time.time()

    print("  AI: ", end="", flush=True)

    try:
        for chunk in ollama.chat(
            model=OLLAMA_MODEL, messages=messages, stream=True,
            options={"num_gpu": OLLAMA_NUM_GPU, "num_predict": 80},
        ):
            if _interrupted:
                break
            try:
                token = chunk.message.content
            except AttributeError:
                token = chunk["message"]["content"]
            if not token:
                continue

            print(token, end="", flush=True)
            full_tokens.append(token)
            sentence_buf.append(token)

    except KeyboardInterrupt:
        _interrupted = True
    except Exception as e:
        print(f"\n  [LLM error: {e}]")
        # Remove the injected system + user messages
        if len(messages) >= 2:
            messages.pop()  # system nudge
            messages.pop()  # [silence]
        stop_speaking()
        return

    print(f"  ({time.time()-t0:.1f}s)", flush=True)

    # Flush all text as one TTS chunk
    remainder = "".join(sentence_buf).strip()
    if remainder and not _interrupted:
        _tts_done.clear()  # mark busy BEFORE queuing so wait loop can't exit early
        _tts_queue.put(remainder)

    reply = "".join(full_tokens)
    if reply:
        # Remove the injected [silence] + system nudge, replace with clean assistant reply
        if len(messages) >= 2 and messages[-1].get("role") == "system":
            messages.pop()  # system nudge
            messages.pop()  # [silence]
        messages.append({"role": "assistant", "content": reply})
    else:
        # Remove injected messages if LLM gave nothing
        if len(messages) >= 2:
            messages.pop()
            messages.pop()

    if _interrupted:
        stop_speaking()
    else:
        # Wait for TTS worker to finish ALL playback before returning
        while not _tts_done.wait(timeout=0.1):
            if _interrupted:
                stop_speaking()
                break


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    global OLLAMA_NUM_GPU

    print("Starting voice chat (Sarvam AI TTS)...")
    print(f"  Model: {SARVAM_MODEL}, Speaker: {SARVAM_SPEAKER}, Lang: {SARVAM_LANGUAGE}")

    if SARVAM_API_KEY == "YOUR_API_KEY_HERE":
        print("\n  ⚠ SARVAM_API_KEY not set!")
        print("  Set it via: set SARVAM_API_KEY=<your key>")
        print("  Get a key at: https://dashboard.sarvam.ai/\n")
        return

    # Start TTS worker
    start_tts()
    # Wait for Sarvam warmup
    _sarvam_ready.wait(timeout=20)

    try:
        models = ollama.list()
        try:
            names = [m.model for m in models.models]
        except AttributeError:
            names = [m["name"] for m in models.get("models", [])]
        print(f"  Ollama: {', '.join(names[:3])}")
        if not any(OLLAMA_MODEL.split(':')[0] in n for n in names):
            print(f"  ⚠ '{OLLAMA_MODEL}' not found!")
            return
        print("  Warming up LLM...", end="", flush=True)
        try:
            ollama.chat(model=OLLAMA_MODEL,
                        messages=[{"role": "user", "content": "hi"}],
                        options={"num_gpu": OLLAMA_NUM_GPU})
            print(" ok")
        except Exception as e:
            if "CUDA" in str(e) or "gpu" in str(e).lower():
                OLLAMA_NUM_GPU = 0
                ollama.chat(model=OLLAMA_MODEL,
                            messages=[{"role": "user", "content": "hi"}],
                            options={"num_gpu": 0})
                print(" ok (CPU)")
            else:
                raise
    except Exception as e:
        print(f"  Ollama FAILED: {e}")
        return

    whisper = load_whisper()
    ambient = calibrate_noise()
    global _speech_thresh_global
    sp_t = max(SPEECH_THRESHOLD, ambient * 2.5)
    si_t = max(SILENCE_THRESHOLD, ambient * 1.8)
    _speech_thresh_global = sp_t  # share with barge-in monitor

    # Start persistent mic monitor for barge-in detection
    start_mic_monitor()
    print("  Mic monitor started for barge-in detection.")

    print("\n  ✓ Voice chat ready (Sarvam AI) — just start talking!")
    print("  Ctrl+C → text input / quit\n")

    idle_count = 0  # track consecutive idle timeouts

    while True:
        try:
            print("  🎧 ...", end="", flush=True)
            pause_mic_monitor()   # pause before listen() opens its own mic
            audio = listen(sp_t, si_t, timeout=IDLE_TIMEOUT)
            resume_mic_monitor()  # resume after listen() closes its mic

            # ── Idle timeout: AI proactively follows up ──────────────
            if audio is not None and audio.size == 0:  # _TIMEOUT_SENTINEL
                idle_count += 1
                if idle_count > MAX_IDLE_FOLLOWUPS:
                    # Too many consecutive silences — just wait quietly
                    print("\r  ⏳ (waiting...)", end="", flush=True)
                    continue
                prompt_idx = min(idle_count - 1, len(_IDLE_PROMPTS) - 1)
                follow_up = _IDLE_PROMPTS[prompt_idx]
                print(f"\r  ⏳ (no response — AI following up...)")
                # Inject as a system nudge so LLM responds naturally
                messages.append({"role": "user", "content": "[silence]"})
                messages.append({"role": "system", "content": follow_up})
                print("  🔊", flush=True)
                chat_and_speak_idle()
                continue

            # ── Normal flow ──────────────────────────────────────────
            idle_count = 0  # reset on any user activity

            if audio is None:
                try:
                    txt = input("\n  > ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if not txt or txt.lower() in ("quit", "exit", "q"):
                    break
            else:
                txt = transcribe(whisper, audio)
                if not txt:
                    print(" (nothing)")
                    continue
                print(f"  You: {txt}")

            print("  🔊", flush=True)
            chat_and_speak(txt)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  Error: {e}")
            continue

    stop_speaking()
    stop_mic_monitor()
    _tts_queue.put(None)
    if _tts_thread:
        _tts_thread.join(timeout=3)
    print("\nBye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
