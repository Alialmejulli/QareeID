# QuranID V1 — Project Summary
*Session date: May 13, 2026*

---

## What This Program Does

A local Python CLI tool that identifies Quran reciters (Qaris) from audio clips using
voice fingerprinting (embeddings). Given any audio file or microphone recording, it
returns the most likely reciter with a confidence score.

No internet required for identification. Claude API is only called as a fallback for
uncertain matches. Everything else runs fully offline on your Windows machine.

---

## How It Works — The Full Pipeline

```
Audio file / Microphone
        ↓
audio.py — load & convert to 16kHz mono, slice into 5-second chunks
        ↓
embedder.py — run each chunk through ECAPA-TDNN AI model → 192 numbers (embedding)
        ↓
embedder.py — average all chunk embeddings → one final fingerprint
        ↓
database.py — load all stored embeddings for all enrolled reciters
        ↓
matcher.py — compare fingerprints using cosine similarity → ranked list
        ↓
score ≥ 0.55  → HIGH confidence → done
score 0.20–0.55 → LOW confidence → ask Claude API for help
score < 0.20  → UNKNOWN → not enrolled or too noisy
        ↓
database.py — save result to history
        ↓
Print result to screen
```

---

## The Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Comfortable + all ML libraries available |
| Audio loading | pydub + soundfile + librosa | Handles mp3, wav, flac, m4a |
| AI model | ECAPA-TDNN via SpeechBrain | Converts voice → 192-number fingerprint |
| Similarity | Cosine similarity (numpy) | Measures how alike two fingerprints are |
| Database | SQLite (data/quranid.db) | Single file, no server needed |
| AI fallback | Claude API (claude-sonnet-4-6) | Only for low-confidence matches |
| Transcription | Whisper (optional) | Arabic transcription to help Claude |
| Mic recording | sounddevice | Real-time microphone identification |
| Config | .env file | Stores ANTHROPIC_API_KEY |

---

## Project File Structure

```
quranID/
├── core/
│   ├── __init__.py       — makes core a Python package
│   ├── audio.py          — load_audio() + slice_audio()
│   ├── embedder.py       — get_embedding() + average_embedding() — ECAPA-TDNN wrapper
│   ├── database.py       — all SQLite operations (reciters, embeddings, history)
│   └── matcher.py        — cosine similarity + Claude API fallback
├── data/
│   ├── quranid.db        — SQLite database (auto-created)
│   └── samples/          — downloaded audio samples per reciter
│       ├── Al-Minshawy/
│       ├── Muhammad-Ayyoub/
│       └── Yasser-Al-Dossary/
├── pretrained_models/
│   └── spkrec-ecapa-voxceleb/  — AI model weights (~150MB, auto-downloaded)
├── venv/                 — Python virtual environment (never touch)
├── main.py               — CLI entry point, all commands live here
├── download_samples.py   — automated downloader from everyayah.com
├── requirements.txt      — all pip dependencies
├── app.html              — web UI (not yet wired up)
└── .env                  — ANTHROPIC_API_KEY (never share or commit)
```

---

## What Each Core File Does

### `core/audio.py` — The Ears
Converts any audio format into what the AI expects: 16kHz mono float32 numpy array.
Slices long clips into 5-second chunks so the model gets consistent input.

### `core/embedder.py` — The Fingerprinter
Wraps the ECAPA-TDNN model. Takes a 5-second audio chunk and returns 192 numbers
that uniquely represent that voice. Model loads once and stays in memory (lazy singleton).

### `core/database.py` — The Memory
Manages the SQLite database. Three tables:
- `reciters` — one row per enrolled reciter (name, style, nationality)
- `embeddings` — many rows per reciter (one per audio segment, stored as binary blob)
- `history` — log of every identification ever run

### `core/matcher.py` — The Judge
Compares the query embedding against all stored embeddings using cosine similarity.
Returns a ranked list. Applies thresholds to decide confidence level.
Calls Claude API when score is in the uncertain range.

### `main.py` — The Front Door
Receives CLI commands and routes them to the right core functions. Never do computation
here — it just dispatches.

---

## CLI Commands

```bash
# Enroll a reciter from a folder of audio samples
python main.py enroll --name "Al-Husary" --folder data/samples/Al-Husary --style Murattal --nationality Egyptian

# Identify the reciter in an audio file
python main.py identify clip.mp3

# Identify + run Whisper transcription (helps Claude fallback)
python main.py identify clip.mp3 --transcribe

# Record from microphone and identify
python main.py listen --duration 15

# Process a whole folder of clips, export to CSV
python main.py batch ./clips --output results.csv

# List all enrolled reciters
python main.py list

# Remove a reciter and all their embeddings
python main.py remove "Al-Husary"

# Show identification history
python main.py history --limit 50
```

---

## Downloading Samples (everyayah.com)

URL pattern: `https://everyayah.com/data/{reciter_folder}/{surah}{ayah}.mp3`

```bash
# Download 30 ayahs for all configured reciters
python download_samples.py --ayahs 30

# Download for one specific reciter
python download_samples.py --reciter "Alafasy"

# See all configured reciters
python download_samples.py --list
```

15 reciters are pre-configured in `download_samples.py` with their correct folder names.

---

## Current State (End of Session 1)

### Enrolled reciters
| Reciter | Style | Nationality | Samples |
|---|---|---|---|
| Al-Minshawy | Murattal | Egyptian | 78 embeddings (20 files) |
| Muhammad-Ayyoub | Murattal | Saudi | 85 embeddings (20 files) |
| Yasser-Al-Dossary | Murattal | Saudi | 74 embeddings (20 files) |

### Identification scores so far
| Test clip | Correct? | Score | Confidence |
|---|---|---|---|
| Al-Minshawy 002255.mp3 | ✅ Yes | 0.531 | LOW |
| Muhammad-Ayyoub 002255.mp3 | ✅ Yes | 0.613 | LOW |
| Mic recording (Muhammad-Ayyoub) | ✅ Ranked #1 but | 0.205 | UNKNOWN |

### Current thresholds (core/matcher.py)
```python
CONFIDENT_THRESHOLD = 0.55   # above this → HIGH confidence
UNCERTAIN_THRESHOLD = 0.20   # below this → UNKNOWN (lowered from 0.35 for mic use)
```

---

## Known Issues & Observations

1. **Scores are in the LOW range (0.5–0.65)** — normal for ECAPA-TDNN on Quran audio.
   The model was trained on regular speech, not Quran recitation. Scores will improve
   with more samples per reciter.

2. **Mic identification scores are lower (~0.2)** — because the model was enrolled on
   clean mp3 downloads but mic recording adds room noise. The model hears them as
   slightly different voices. Fix: enroll some mic-recorded samples too.

3. **SpeechBrain symlink warning on Windows** — harmless, just a Windows limitation.
   The model works fine despite the warning.

---

## Next Session — TODO List

### Priority 1: More samples per reciter
```bash
# Download 10 more per reciter (skips existing 20)
python download_samples.py --ayahs 30

# Re-enroll all three with the extra samples
python main.py remove "Al-Minshawy"
python main.py remove "Muhammad-Ayyoub"
python main.py remove "Yasser-Al-Dossary"

python main.py enroll --name "Al-Minshawy" --folder data/samples/Al-Minshawy --style Murattal --nationality Egyptian
python main.py enroll --name "Muhammad-Ayyoub" --folder data/samples/Muhammad-Ayyoub --style Murattal --nationality Saudi
python main.py enroll --name "Yasser-Al-Dossary" --folder data/samples/Yasser-Al-Dossary --style Murattal --nationality Saudi
```
Then test again and compare scores. Goal: push confident matches above 0.55.

### Priority 2: Add more reciters
Once scores improve with 30 samples, enroll more from the 15 pre-configured:
Al-Husary, Abdul-Basit, As-Sudais, Alafasy, Maher-Al-Muaiqly, etc.

### Priority 3: Mic improvement
Add `--save` flag to the `listen` command so mic recordings can be saved and
re-enrolled as additional samples, making mic identification more accurate.

### Priority 4: Wire up app.html
Connect the existing web UI to the Python backend so identification can be done
by dragging and dropping a file in the browser instead of typing commands.

---

## How to Start Each Session

```bash
# 1. Open terminal in quranID/ folder
# 2. Activate virtual environment
venv\Scripts\activate

# 3. Verify everything still works
python main.py list

# 4. Continue from TODO list above
```
