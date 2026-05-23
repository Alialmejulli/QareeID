#!/usr/bin/env python3
"""
QuranID - Qari voice identification tool
Usage: python main.py <command> [options]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

DEBUG = os.getenv("QAREE_DEBUG", "0") == "1"

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def cmd_enroll(args):
    from core.audio import load_audio, slice_audio
    from core.embedder import get_embedding, average_embedding
    from core import database as db

    db.init_db()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Error: folder not found: {folder}")

    files = [f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS]
    if not files:
        sys.exit(f"No audio files found in {folder}")

    reciter_id = db.add_reciter(args.name, style=args.style or "", nationality=args.nationality or "", arabic_name=args.arabic or "", display_name=args.display or "")
    print(f"Enrolling '{args.name}' from {len(files)} file(s)...")

    enrolled = 0
    for fpath in files:
        try:
            audio, sr = load_audio(str(fpath))
            segments = slice_audio(audio, sr, segment_sec=5.0)
            for seg in segments:
                emb = get_embedding(seg)
                db.save_embedding(reciter_id, emb, source_file=str(fpath))
                enrolled += 1
            print(f"  [ok] {fpath.name} - {len(segments)} segment(s)")
        except Exception as e:
            print(f"  [skip] {fpath.name}: {e}")

    print(f"\nDone. {enrolled} embedding(s) saved for '{args.name}'.")


def cmd_list(args):
    from core import database as db
    db.init_db()
    reciters = db.list_reciters()
    if not reciters:
        print("No reciters enrolled yet. Use: python main.py enroll --name ... --folder ...")
        return

    print(f"\n{'Name':<30} {'Style':<15} {'Nationality':<15} {'Samples':>7}  {'Enrolled'}")
    print("-" * 85)
    for r in reciters:
        print(f"{r['name']:<30} {r['style'] or '-':<15} {r['nationality'] or '-':<15} "
              f"{r['sample_count']:>7}  {r['enrolled_at'][:10]}")
    print(f"\nTotal: {len(reciters)} reciter(s)")


def cmd_identify(args):
    from core.audio import load_audio, slice_audio
    from core.embedder import get_embedding, average_embedding
    from core import database as db
    from core import matcher

    db.init_db()
    audio_path = args.file
    if not os.path.isfile(audio_path):
        sys.exit(f"Error: file not found: {audio_path}")

    enrolled = db.get_all_embeddings()
    if not enrolled:
        sys.exit("No reciters enrolled. Run: python main.py enroll --name ... --folder ...")

    print(f"Loading '{audio_path}'...")
    audio, sr = load_audio(audio_path)
    segments = slice_audio(audio, sr, segment_sec=5.0)
    print(f"  {len(segments)} segment(s) found. Extracting embeddings...")

    embeddings = [get_embedding(seg) for seg in segments]
    query_emb = average_embedding(embeddings)

    ranked = matcher.match(query_emb, enrolled)
    result = matcher.interpret(ranked)

    transcript = ""
    if args.transcribe:
        transcript = _transcribe(audio_path)
        if transcript:
            print(f"  Transcript: {transcript[:120]}...")

    method = "cosine"
    if result["needs_claude"]:
        print("  Confidence is low - asking Claude API...")
        claude_result = matcher.ask_claude(audio_path, ranked, transcript=transcript)
        result.update(claude_result)
        method = claude_result.get("method", "claude")
    else:
        method = "cosine"

    db.save_history(audio_path, result["name"] or "unknown", result["score"], method)

    _print_result(result, ranked)
    return result


def cmd_batch(args):
    from core.audio import load_audio, slice_audio
    from core.embedder import get_embedding, average_embedding
    from core import database as db
    from core import matcher

    db.init_db()
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Error: folder not found: {folder}")

    files = [f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS]
    if not files:
        sys.exit(f"No audio files found in {folder}")

    enrolled = db.get_all_embeddings()
    if not enrolled:
        sys.exit("No reciters enrolled.")

    results = []
    for fpath in sorted(files):
        print(f"Processing {fpath.name}...", end=" ", flush=True)
        try:
            audio, sr = load_audio(str(fpath))
            segments = slice_audio(audio, sr, segment_sec=5.0)
            embeddings = [get_embedding(seg) for seg in segments]
            query_emb = average_embedding(embeddings)
            ranked = matcher.match(query_emb, enrolled)
            result = matcher.interpret(ranked)

            method = "cosine"
            if result["needs_claude"]:
                claude_result = matcher.ask_claude(str(fpath), ranked)
                result.update(claude_result)
                method = claude_result.get("method", "claude")

            db.save_history(str(fpath), result["name"] or "unknown", result["score"], method)
            results.append({"file": fpath.name, "match": result["name"] or "unknown",
                             "score": f"{result['score']:.3f}", "confidence": result["confidence"]})
            display = result.get("display_name") or result["name"] or "unknown"
            print(f"-> {display} ({result['score']:.3f}, {result['confidence']})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"file": fpath.name, "match": "error", "score": "0", "confidence": "error"})

    if args.output:
        out_path = args.output
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "match", "score", "confidence"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {out_path}")

    print(f"\nProcessed {len(results)} file(s).")


def cmd_update(args):
    from core import database as db
    db.init_db()
    ok = db.update_reciter(args.name, args.arabic)
    if ok:
        print(f"Updated '{args.name}': arabic_name set to '{args.arabic}'.")
    else:
        print(f"Reciter '{args.name}' not found.")


def cmd_remove(args):
    from core import database as db
    db.init_db()
    name = args.name
    ok = db.remove_reciter(name)
    if ok:
        print(f"Removed '{name}' and all their embeddings.")
    else:
        print(f"Reciter '{name}' not found.")


def cmd_listen(args):
    import tempfile
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    from core import database as db 
    from core.audio import slice_audio
    from core.embedder import get_embedding, average_embedding
    from core import matcher

    db.init_db()
    enrolled = db.get_all_embeddings()
    if not enrolled:
        sys.exit("No reciters enrolled. Run: python main.py enroll --name ... --folder ...")

    duration = args.duration
    sr = 16000

    print(f"Listening for {duration} seconds... (play your audio now)")
    print("-" * 40)

    # Try WASAPI HyperX first (device 32), fallback to default
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr,
                       channels=1, dtype='float32', device=32,
                       extra_settings=sd.WasapiSettings(exclusive=False))
        # show a simple countdown
        import time
        for remaining in range(duration, 0, -1):
            print(f"  {remaining}s remaining...", end="\r")
            time.sleep(1)
        sd.wait()
    except Exception:
        try:
            # Try with 48000 native rate then resample
            import librosa
            audio_48k = sd.rec(int(duration * 48000), samplerate=48000,
                               channels=1, dtype='float32', device=32)
            sd.wait()
            audio = librosa.resample(audio_48k.flatten(),
                                     orig_sr=48000, target_sr=sr)
        except Exception:
            # Fallback to default device
            audio = sd.rec(int(duration * sr), samplerate=sr,
                           channels=1, dtype='float32')
            sd.wait()

    if audio.ndim > 1:
        audio = audio.flatten()
    print("  Recording done.          ")
    print("-" * 40)

    audio = audio.flatten()

    import numpy as np
    audio_flat = audio.flatten() if audio.ndim > 1 else audio
    rms = float(np.sqrt(np.mean(audio_flat ** 2)))
    peak = float(np.max(np.abs(audio_flat)))
    if DEBUG: print(f"[DEBUG] RMS: {rms:.5f}  Peak: {peak:.5f}  Device used: default")

    # Amplify quiet mic input
    rms = float(np.sqrt(np.mean(audio_flat ** 2)))
    if rms < 0.02 and rms > 0.001:
        gain = 0.02 / rms
        gain = min(gain, 10.0)
        audio_flat = audio_flat * gain
        if DEBUG: print(f"[DEBUG] Applied gain: {gain:.1f}x  New RMS: {float(np.sqrt(np.mean(audio_flat**2))):.5f}")

    audio = audio_flat

    # save to temp file so identify logic can reuse it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()

    segments = slice_audio(audio, sr, segment_sec=5.0)
    if not segments:
        print("Recording too short - try a longer duration with --duration")
        return

    print(f"  {len(segments)} segment(s) captured. Identifying...")
    embeddings = [get_embedding(seg) for seg in segments]
    query_emb = average_embedding(embeddings)

    ranked = matcher.match(query_emb, enrolled)
    result = matcher.interpret(ranked)

    method = "cosine"
    if result["needs_claude"]:
        print("  Confidence is low - asking Claude API...")
        claude_result = matcher.ask_claude(tmp.name, ranked)
        result.update(claude_result)
        method = claude_result.get("method", "claude")

    db.save_history("[microphone]", result["name"] or "unknown", result["score"], method)
    os.unlink(tmp.name)

    _print_result(result, ranked)


def cmd_history(args):
    from core import database as db
    db.init_db()
    rows = db.get_history(limit=args.limit)
    if not rows:
        print("No identification history yet.")
        return

    print(f"\n{'File':<35} {'Match':<25} {'Score':>6}  {'Method':<8}  {'Time'}")
    print("-" * 95)
    for r in rows:
        fname = Path(r["query_file"]).name if r["query_file"] else "-"
        print(f"{fname:<35} {r['matched_name']:<25} {r['score']:>6.3f}  {r['method']:<8}  {r['identified_at'][:16]}")


# ---------- helpers ----------

def _transcribe(audio_path: str) -> str:
    try:
        import whisper
        print("  Loading Whisper model for transcription...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language="ar")
        return result.get("text", "").strip()
    except Exception as e:
        print(f"  [transcribe] {e}")
        return ""


def _print_result(result: dict, ranked: list[dict]):
    print()
    if result["name"]:
        conf_label = {"high": "HIGH", "low": "LOW", "claude": "CLAUDE-ASSISTED"}.get(
            result["confidence"], result["confidence"].upper()
        )
        display = result.get("display_name") or result["name"]
        print(f"  Match    : {display}")
        print(f"  Score    : {result['score']:.3f}")
        print(f"  Confidence: {conf_label}")
    else:
        print("  Result   : Unknown / not enrolled")
        print(f"  Best score: {result['score']:.3f} (below threshold)")

    if len(ranked) > 1:
        print("\n  Top candidates:")
        for i, r in enumerate(ranked[:5], 1):
            marker = " <-- match" if i == 1 and result["name"] else ""
            display = r.get("display_name") or r["name"]
            print(f"    {i}. {display:<28} {r['score']:.3f}{marker}")


# ---------- CLI ----------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="QuranID — identify Qari reciters from audio",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a reciter from a folder of audio samples")
    p_enroll.add_argument("--name", required=True, help="Reciter name (e.g. 'Abdul-Basit Murattal')")
    p_enroll.add_argument("--folder", required=True, help="Path to folder of audio samples")
    p_enroll.add_argument("--arabic", default="", help="Reciter name in Arabic script")
    p_enroll.add_argument("--style", default="", help="Recitation style (e.g. Murattal)")
    p_enroll.add_argument("--nationality", default="", help="Reciter nationality")
    p_enroll.add_argument("--display", default="", help="Display name shown to users (default: same as --name)")

    # list
    sub.add_parser("list", help="List all enrolled reciters")

    # identify
    p_id = sub.add_parser("identify", help="Identify the reciter in an audio clip")
    p_id.add_argument("file", help="Path to audio file (mp3, wav, flac, etc.)")
    p_id.add_argument("--transcribe", action="store_true",
                      help="Also transcribe the audio via Whisper (helps Claude fallback)")

    # batch
    p_batch = sub.add_parser("batch", help="Identify all clips in a folder")
    p_batch.add_argument("folder", help="Path to folder of audio clips")
    p_batch.add_argument("--output", default="", help="Optional CSV output path")

    # update
    p_update = sub.add_parser("update", help="Update metadata for an existing reciter")
    p_update.add_argument("--name", required=True, help="Reciter name to update")
    p_update.add_argument("--arabic", required=True, help="Arabic name to set")

    # remove
    p_rm = sub.add_parser("remove", help="Remove a reciter from the database")
    p_rm.add_argument("name", help="Reciter name to remove")

    # listen
    p_listen = sub.add_parser("listen", help="Record from microphone and identify the reciter")
    p_listen.add_argument("--duration", type=int, default=25,
                          help="How many seconds to record (default: 10)")

    # history
    p_hist = sub.add_parser("history", help="Show recent identification history")
    p_hist.add_argument("--limit", type=int, default=20, help="Number of entries to show")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "enroll": cmd_enroll,
        "update": cmd_update,
        "list": cmd_list,
        "identify": cmd_identify,
        "listen": cmd_listen,
        "batch": cmd_batch,
        "remove": cmd_remove,
        "history": cmd_history,
    }
    dispatch[args.command](args)
