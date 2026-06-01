# QareeID

**Identify Quran reciters from audio clips using voice fingerprinting.**

[![Download](https://img.shields.io/badge/Download-QareeID.exe-blue?style=for-the-badge)](https://github.com/Alialmejulli/QareeID/releases/latest)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?style=for-the-badge)
![Offline](https://img.shields.io/badge/Works-Offline-green?style=for-the-badge)

---

## What it does
QareeID listens to a Quran recitation and tells you which reciter is speaking, with support for 51+ reciters and both Arabic and English interfaces.

## How to use
1. Download the latest release using the button above
2. Extract the zip
3. Run `QareeID.exe`

## Features
- 51+ enrolled reciters with Arabic and English names
- Real-time microphone identification
- Clean desktop UI
- Works fully offline — no internet required
- Claude API fallback for uncertain matches (optional)

## For developers
See [CONTEXT.md](CONTEXT.md) for full technical documentation.

## Built with
- Python 3.11
- SpeechBrain ECAPA-TDNN
- PyWebView
- SQLite
