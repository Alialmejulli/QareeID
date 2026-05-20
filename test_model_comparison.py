#!/usr/bin/env python3
"""
QuranID — ECAPA-TDNN vs WavLM model comparison.

ECAPA same/cross scores come from the enrolled-database pipeline
(core.embedder + core.matcher), so they reflect real-world identify() quality.

WavLM scores come from direct file-to-file verification via
speechbrain/spkrec-wavlm-voxceleb (downloads ~360 MB on first run).
"""

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
RECITERS = [
    "Ahmad Talib bin Humaid",
    "Al-Minshawy",
    "Ash-Shuraym",
    "Alafasy",
    "Bader Alturki",
]

CROSS_PAIRS = [
    ("Ahmad Talib bin Humaid", "Al-Minshawy"),
    ("Ahmad Talib bin Humaid", "Alafasy"),
    ("Alafasy", "Ash-Shuraym"),
]

SAMPLES_DIR = Path(__file__).parent / "data" / "samples"


# ─────────────────────────────────────────────────────────────────
#  File selection
# ─────────────────────────────────────────────────────────────────
def pick_files(reciter: str) -> tuple[Path, Path]:
    """Return (largest, median) mp3 from a reciter's sample folder."""
    folder = SAMPLES_DIR / reciter
    files = sorted(folder.glob("*.mp3"), key=lambda f: f.stat().st_size)
    if len(files) < 2:
        raise ValueError(f"need ≥2 files, found {len(files)}")
    largest = files[-1]
    median  = files[len(files) // 2]
    return largest, median


# ─────────────────────────────────────────────────────────────────
#  ECAPA helpers  (uses the live enrolled database)
# ─────────────────────────────────────────────────────────────────
_enrolled_cache = None

def _get_enrolled():
    global _enrolled_cache
    if _enrolled_cache is None:
        from core import database as db
        _enrolled_cache = db.get_all_embeddings()
    return _enrolled_cache


def _embed_file(path: Path):
    from core.audio import load_audio, slice_audio
    from core.embedder import get_embedding, average_embedding
    audio, sr = load_audio(str(path))
    segs = slice_audio(audio, sr, segment_sec=5.0)
    return average_embedding([get_embedding(s) for s in segs])


def _ranked_score_for(query_emb, target_name: str) -> float:
    """Return the matcher score assigned to target_name for this query."""
    from core import matcher
    ranked = matcher.match(query_emb, _get_enrolled())
    # exact match first
    for r in ranked:
        if r["name"].lower() == target_name.lower():
            return r["score"]
    # case-insensitive substring
    for r in ranked:
        n = r["name"].lower()
        t = target_name.lower()
        if t in n or n in t:
            return r["score"]
    return float("nan")


def ecapa_same(reciter: str, f1: Path, f2: Path) -> float:
    """Average score the model gives reciter when queried with their own files."""
    s1 = _ranked_score_for(_embed_file(f1), reciter)
    s2 = _ranked_score_for(_embed_file(f2), reciter)
    valid = [s for s in (s1, s2) if not math.isnan(s)]
    return sum(valid) / len(valid) if valid else float("nan")


def ecapa_cross(f_a: Path, name_b: str) -> float:
    """Score that file A gets when matched against reciter B (should be low)."""
    return _ranked_score_for(_embed_file(f_a), name_b)


# ─────────────────────────────────────────────────────────────────
#  WavLM helpers
# ─────────────────────────────────────────────────────────────────
_wavlm = None

def _load_wavlm():
    global _wavlm
    if _wavlm is not None:
        return _wavlm
    print("\n  [WavLM] Loading model — first run downloads ~360 MB …", flush=True)
    from speechbrain.inference.speaker import SpeakerRecognition
    _wavlm = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-wavlm-voxceleb",
        savedir=str(Path(__file__).parent / "pretrained_models" / "spkrec-wavlm-voxceleb"),
        run_opts={"device": "cpu"},
    )
    print("  [WavLM] Model ready.", flush=True)
    return _wavlm


def _to_wav(mp3: Path) -> str:
    """Convert mp3 → temp wav so torchaudio can read it reliably."""
    import soundfile as sf
    from core.audio import load_audio
    audio, sr = load_audio(str(mp3))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    return tmp.name


def wavlm_verify(p1: Path, p2: Path) -> float:
    """Return WavLM cosine similarity between two audio files."""
    model = _load_wavlm()
    w1 = _to_wav(p1)
    w2 = _to_wav(p2)
    try:
        score, _ = model.verify_files(w1, w2)
        return float(score)
    finally:
        for p in (w1, w2):
            try:
                os.unlink(p)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────────────────────────
NC = 28   # name / pair column width

def _fmt(v: float) -> str:
    return "  N/A " if math.isnan(v) else f"{v:.3f}"

def _avg(vals: list[float]) -> float:
    ok = [v for v in vals if not math.isnan(v)]
    return sum(ok) / len(ok) if ok else float("nan")

def _sep(): print("─" * 51)
def _hdr(col1_label: str):
    _sep()
    print(f"  {col1_label:<{NC}}  {'ECAPA':>6}   {'WavLM':>6}")
    _sep()


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main():
    from dotenv import load_dotenv
    load_dotenv()
    from core import database as db
    db.init_db()

    print()
    print("═" * 51)
    print("  QuranID — Model Comparison Test")
    print("  ECAPA-TDNN vs WavLM")
    print("═" * 51)

    # ── Select test files ──────────────────────────────
    print("\n[Setup] Selecting test files…")
    files: dict[str, tuple[Path, Path]] = {}
    for name in RECITERS:
        try:
            f1, f2 = pick_files(name)
            files[name] = (f1, f2)
            print(f"  {name:<28}  {f1.name}  /  {f2.name}")
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")

    if not files:
        print("\nNo test files found. Check data/samples/ directory.")
        return

    # ── ECAPA same-person ──────────────────────────────
    print("\n[ECAPA] Same-person similarity…")
    ecapa_same_scores: dict[str, float] = {}
    for name, (f1, f2) in files.items():
        print(f"  {name}…", end="  ", flush=True)
        try:
            s = ecapa_same(name, f1, f2)
            ecapa_same_scores[name] = s
            print(_fmt(s))
        except Exception as exc:
            print(f"ERROR: {exc}")
            ecapa_same_scores[name] = float("nan")

    # ── WavLM same-person ──────────────────────────────
    wavlm_same_scores: dict[str, float] = {}
    wavlm_ok = True
    try:
        print("\n[WavLM] Same-person similarity…")
        for name, (f1, f2) in files.items():
            print(f"  {name}…", end="  ", flush=True)
            try:
                s = wavlm_verify(f1, f2)
                wavlm_same_scores[name] = s
                print(_fmt(s))
            except Exception as exc:
                print(f"ERROR: {exc}")
                wavlm_same_scores[name] = float("nan")
    except Exception as exc:
        print(f"\n  [WavLM] Failed to load: {exc}")
        wavlm_ok = False
        for name in files:
            wavlm_same_scores[name] = float("nan")

    # ── ECAPA cross-person ─────────────────────────────
    print("\n[ECAPA] Cross-person similarity…")
    ecapa_cross_scores: dict[tuple, float] = {}
    for na, nb in CROSS_PAIRS:
        if na not in files or nb not in files:
            continue
        label = f"{na} vs {nb}"
        print(f"  {label}…", end="  ", flush=True)
        try:
            s = ecapa_cross(files[na][0], nb)
            ecapa_cross_scores[(na, nb)] = s
            print(_fmt(s))
        except Exception as exc:
            print(f"ERROR: {exc}")
            ecapa_cross_scores[(na, nb)] = float("nan")

    # ── WavLM cross-person ─────────────────────────────
    wavlm_cross_scores: dict[tuple, float] = {}
    if wavlm_ok:
        print("\n[WavLM] Cross-person similarity…")
        for na, nb in CROSS_PAIRS:
            if na not in files or nb not in files:
                continue
            label = f"{na} vs {nb}"
            print(f"  {label}…", end="  ", flush=True)
            try:
                s = wavlm_verify(files[na][0], files[nb][0])
                wavlm_cross_scores[(na, nb)] = s
                print(_fmt(s))
            except Exception as exc:
                print(f"ERROR: {exc}")
                wavlm_cross_scores[(na, nb)] = float("nan")
    else:
        for pair in CROSS_PAIRS:
            wavlm_cross_scores[pair] = float("nan")

    # ── Results table ──────────────────────────────────
    print()
    print("═" * 51)
    print()
    print("SAME-PERSON SIMILARITY  (↑ higher is better, target > 0.5)")
    _hdr("Reciter")
    for name in RECITERS:
        if name not in files:
            continue
        e = _fmt(ecapa_same_scores.get(name, float("nan")))
        w = _fmt(wavlm_same_scores.get(name, float("nan")))
        print(f"  {name:<{NC}}  {e:>6}   {w:>6}")
    _sep()
    avg_es = _avg(list(ecapa_same_scores.values()))
    avg_ws = _avg(list(wavlm_same_scores.values()))
    print(f"  {'Average':<{NC}}  {_fmt(avg_es):>6}   {_fmt(avg_ws):>6}")

    print()
    print("DIFFERENT-PERSON SIMILARITY  (↓ lower is better, target < 0.3)")
    _hdr("Pair")
    for na, nb in CROSS_PAIRS:
        if na not in files or nb not in files:
            continue
        label = f"{na} vs {nb}"
        if len(label) > NC:
            label = label[:NC - 1] + "…"
        e = _fmt(ecapa_cross_scores.get((na, nb), float("nan")))
        w = _fmt(wavlm_cross_scores.get((na, nb), float("nan")))
        print(f"  {label:<{NC}}  {e:>6}   {w:>6}")
    _sep()
    avg_ec = _avg(list(ecapa_cross_scores.values()))
    avg_wc = _avg(list(wavlm_cross_scores.values()))
    print(f"  {'Average':<{NC}}  {_fmt(avg_ec):>6}   {_fmt(avg_wc):>6}")

    # ── Verdict ────────────────────────────────────────
    print()
    print("VERDICT:")
    if not wavlm_ok or math.isnan(avg_ws):
        print("  WavLM results unavailable — cannot compare.")
    else:
        same_winner = "WavLM" if avg_ws > avg_es else "ECAPA"
        diff_winner = "WavLM" if avg_wc < avg_ec else "ECAPA"
        same_delta  = abs(avg_ws - avg_es)
        diff_delta  = abs(avg_wc - avg_ec)

        if same_winner == diff_winner:
            print(f"  {same_winner} wins on both metrics.")
            print(f"  Same-person gap: {same_delta:+.3f}  |  "
                  f"Different-person gap: {diff_delta:+.3f}")
        else:
            print(f"  {same_winner} has better same-person scores  "
                  f"(Δ {same_delta:+.3f})")
            print(f"  {diff_winner} has better cross-person separation  "
                  f"(Δ {diff_delta:+.3f})")
            print("  Mixed results — no clear winner.")
    print()


if __name__ == "__main__":
    main()
