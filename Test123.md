# 🎙️ Voice Chat — Sarvam AI TTS Edition

Real-time voice conversational chat powered by **Ollama** (local LLM) and **Sarvam AI** (cloud TTS). Speak naturally, get spoken responses in an Indian-English voice — with barge-in support, idle follow-ups, and streaming playback.

## Architecture

```
Mic → faster-whisper (STT) → Ollama LLM (streaming) → Sarvam AI TTS (cloud) → Speaker
```

- **STT**: `faster-whisper` with the `base` model and beam-search decoding for accurate transcription.
- **LLM**: Ollama running `gemma2:2b` locally — streams tokens in real-time.
- **TTS**: Sarvam AI's **Bulbul v3** model via cloud API — natural Indian-English voice.
- **Playback**: `sounddevice` plays synthesized audio with real-time barge-in detection (mic monitoring stops AI speech when you start talking).

## Features

- 🗣️ **Natural conversation** — speak freely, AI responds with voice
- ⚡ **Streaming LLM** — tokens stream to TTS as they arrive
- ✋ **Barge-in** — start talking mid-response and the AI stops immediately
- ⏳ **Idle follow-ups** — AI proactively re-engages if you go silent
- 🔇 **Hallucination filter** — filters common Whisper artifacts ("thank you", "thanks for watching", etc.)
- ⌨️ **Text fallback** — press `Ctrl+C` to type instead of speak
- 🎛️ **Auto mic calibration** — adapts to ambient noise on startup

---

## Requirements

### Python

- **Python 3.10+** (uses `X | Y` union type syntax)

### System Dependencies

| Dependency | Purpose | Install |
|---|---|---|
| **Ollama** | Local LLM inference | [ollama.com](https://ollama.com/) |
| **gemma2:2b** model | Default chat model | `ollama pull gemma2:2b` |
| **PortAudio** | Audio I/O (required by sounddevice) | Bundled on Windows; `brew install portaudio` on macOS; `sudo apt install portaudio19-dev` on Linux |
| **Sarvam AI API Key** | Cloud TTS | Sign up at [dashboard.sarvam.ai](https://dashboard.sarvam.ai/) |

### Python Packages

```
faster-whisper
sounddevice
numpy
ollama
requests
python-dotenv
```

Install all at once:

```bash
pip install faster-whisper sounddevice numpy ollama requests python-dotenv
```

---

## Setup

### 1. Clone & install dependencies

```bash
cd voice-chat
pip install faster-whisper sounddevice numpy ollama requests python-dotenv
```

### 2. Pull the Ollama model

Make sure Ollama is running, then:

```bash
ollama pull gemma2:2b
```

### 3. Set your Sarvam AI API key

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_api_key_here
```

Or set it as an environment variable:

```bash
# Windows (cmd)
set SARVAM_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:SARVAM_API_KEY="your_api_key_here"

# Linux / macOS
export SARVAM_API_KEY=your_api_key_here
```

### 4. Run

```bash
python voice_chat_sarvam.py
```

---

## Usage

| Action | How |
|---|---|
| **Talk** | Just speak — the mic auto-detects speech |
| **Interrupt AI** | Start talking while AI is speaking |
| **Type instead** | Press `Ctrl+C` then type your message |
| **Quit** | Press `Ctrl+C` twice, or type `quit` / `exit` / `q` |

---

## Configuration

All configuration constants are at the top of `voice_chat_sarvam.py`:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `OLLAMA_MODEL` | `"gemma2:2b"` | Ollama model to use for chat |
| `SARVAM_MODEL` | `"bulbul:v3"` | Sarvam TTS model |
| `SARVAM_SPEAKER` | `"shubh"` | Voice speaker name |
| `SARVAM_LANGUAGE` | `"en-IN"` | TTS language code |
| `SARVAM_PACE` | `0.9` | Speech speed (0.3–3.0) |
| `SARVAM_SAMPLE_RATE` | `22050` | Audio sample rate (8000, 16000, 22050, 24000) |
| `SILENCE_DURATION` | `0.4` | Seconds of silence to end a turn |
| `IDLE_TIMEOUT` | `8.0` | Seconds before AI follows up on silence |
| `MAX_IDLE_FOLLOWUPS` | `3` | Max consecutive idle follow-ups |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `SARVAM_API_KEY not set!` | Set the key via `.env` or environment variable (see Setup step 3) |
| Sarvam warmup failed | Verify your API key is valid at [dashboard.sarvam.ai](https://dashboard.sarvam.ai/) |
| Ollama model not found | Run `ollama pull gemma2:2b` |
| Mic near-zero warning | Check your microphone is connected and set as default input device |
| `PortAudio` errors | Install PortAudio — see System Dependencies table above |
| CUDA / GPU errors | The script auto-falls back to CPU if GPU fails |

---

## License

Private project — not licensed for redistribution.
