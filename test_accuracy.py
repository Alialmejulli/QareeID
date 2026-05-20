import os
import sys
from collections import Counter
from pathlib import Path

from core.audio import load_audio, slice_audio
from core.embedder import get_embedding, average_embedding
from core import database as db
from core import matcher

SAMPLES_DIR = Path("data/samples")

CONF_LABEL = {
    "high":    "HIGH   ",
    "low":     "LOW    ",
    "unknown": "UNKNOWN",
}


def test_reciter(name: str, enrolled: list[dict], folder: Path = None) -> dict:
    if folder is None:
        folder = SAMPLES_DIR / name
    mp3_files = sorted(folder.glob("*.mp3"))
    if not mp3_files:
        raise FileNotFoundError(f"No .mp3 files in {folder}")

    MIN_BYTES = 200 * 1024
    large_files = [f for f in mp3_files if f.stat().st_size >= MIN_BYTES]
    pool = large_files if large_files else mp3_files
    n = len(pool)
    indices = sorted({0, n // 2, n - 1})
    test_files = [pool[i] for i in indices]

    scores = []
    top_names = []

    for test_file in test_files:
        audio, sr = load_audio(str(test_file))
        segments = slice_audio(audio, sr)
        embeddings = [get_embedding(seg, sr) for seg in segments]
        query_emb = average_embedding(embeddings)

        ranked = matcher.match(query_emb, enrolled)
        if ranked:
            scores.append(ranked[0]["score"])
            top_names.append(ranked[0]["name"])
        else:
            scores.append(0.0)
            top_names.append(None)

    avg_score = sum(scores) / len(scores)
    n_files = len(test_files)
    correct_votes = sum(1 for t in top_names if t == name)

    if avg_score >= matcher.CONFIDENT_THRESHOLD:
        confidence = "high"
    elif avg_score >= matcher.UNCERTAIN_THRESHOLD:
        confidence = "low"
    else:
        confidence = "unknown"

    if confidence != "unknown":
        counts = Counter(t for t in top_names if t is not None)
        top = counts.most_common(1)[0][0] if counts else None
    else:
        top = None

    return {
        "name": name,
        "score": avg_score,
        "confidence": confidence,
        "top": top,
        "correct_votes": correct_votes,
        "n_files": n_files,
    }


def main():
    os.chdir(Path(__file__).parent)
    db.init_db()

    reciters = db.list_reciters()
    if not reciters:
        print("No enrolled reciters found in the database.")
        sys.exit(1)

    enrolled = db.get_all_embeddings()

    print(f"\nQuranID Accuracy Test — {len(reciters)} enrolled reciters\n")

    results = []

    for reciter in reciters:
        name = reciter["name"]
        folder = SAMPLES_DIR / name
        if not folder.is_dir():
            folder = SAMPLES_DIR / name.replace(" ", "-")
        if not folder.is_dir():
            folder = SAMPLES_DIR / name.replace("-", " ")
        if not folder.is_dir():
            print(f"  [SKIP] {name} — samples folder not found")
            continue

        try:
            result = test_reciter(name, enrolled, folder)
            results.append(result)
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")

    if not results:
        print("No results to report.")
        sys.exit(1)

    # Sort ascending by score so weakest reciters appear first
    results.sort(key=lambda r: r["score"])

    correct = 0
    n_high = 0
    n_low = 0
    n_unknown = 0
    failed = []

    for r in results:
        name = r["name"]
        score = r["score"]
        conf = r["confidence"]
        top = r["top"]
        conf_str = CONF_LABEL.get(conf, "UNKNOWN")
        majority = (r["n_files"] + 1) // 2
        is_correct = r["correct_votes"] >= majority

        if conf == "unknown":
            symbol = "⚠️ "
            n_unknown += 1
            status = "below threshold"
            failed.append((name, None, score))
        elif is_correct:
            symbol = "✅"
            correct += 1
            status = "correct"
            if conf == "high":
                n_high += 1
            else:
                n_low += 1
        else:
            symbol = "❌"
            status = f"got: {top}"
            failed.append((name, top, score))
            if conf == "high":
                n_high += 1
            else:
                n_low += 1

        print(f"  {symbol} {name:<30} {score:.3f}  {conf_str}  {status}")

    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0.0
    avg_score = sum(r["score"] for r in results) / total if total > 0 else 0.0

    print()
    print("─" * 53)
    print(f"  Accuracy  : {correct} / {total}  ({accuracy:.1f}%)")
    print(f"  Avg score : {avg_score:.3f}")
    print(f"  HIGH conf : {n_high}")
    print(f"  LOW conf  : {n_low}")
    print(f"  UNKNOWN   : {n_unknown}")
    print("─" * 53)

    if failed:
        print("  Failed reciters (need more samples):")
        for name, got, score in failed:
            if got is None:
                print(f"    - {name:<28} → unknown               ({score:.3f})")
            else:
                print(f"    - {name:<28} → got: {got:<20} ({score:.3f})")
    print()


if __name__ == "__main__":
    main()
