"""
QuranID — mp3quran.net Downloader
===================================
Downloads full surah audio from mp3quran.net using their REST API.
Unlike EveryAyah (per ayah), mp3quran gives full surahs — we download
short surahs and the enroll pipeline slices them into 5-second segments.

Files are saved into: data/samples/{reciter_name}/

Usage:
    python download_mp3quran.py --list                        # list all reciters from API
    python download_mp3quran.py --search "bandar"             # search by name
    python download_mp3quran.py --reciter "Bandar Baleela"    # download one reciter
    python download_mp3quran.py --reciter "Bandar Baleela" --surahs 10
"""

import time
import argparse
import requests
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "https://mp3quran.net/api/v3/reciters?language=eng"
SAMPLES_DIR = Path("data/samples")
REQUEST_DELAY = 0.7

# Short surahs to download — gives good voice variety without huge file sizes
# These are ordered by length (shortest first)
DEFAULT_SURAHS = [
    112,  # Al-Ikhlas      (~30 sec)
    108,  # Al-Kawthar     (~20 sec)
    103,  # Al-Asr         (~25 sec)
    110,  # An-Nasr        (~30 sec)
    114,  # An-Nas         (~40 sec)
    113,  # Al-Falaq       (~35 sec)
    111,  # Al-Masad       (~35 sec)
    105,  # Al-Fil         (~40 sec)
    107,  # Al-Maun        (~40 sec)
    109,  # Al-Kafirun     (~40 sec)
    106,  # Quraysh        (~35 sec)
    104,  # Al-Humaza      (~45 sec)
    102,  # At-Takathur    (~40 sec)
    101,  # Al-Qaria       (~50 sec)
    99,   # Az-Zalzalah    (~50 sec)
    1,    # Al-Fatiha      (~60 sec)
    67,   # Al-Mulk        (~5 min — lots of segments)
    78,   # An-Naba        (~4 min)
    55,   # Ar-Rahman      (~8 min — very long, many segments)
    36,   # Ya-Sin         (~10 min — most segments)
]

DEFAULT_SURAH_COUNT = 10  # default number of surahs to download


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_reciters() -> list[dict]:
    """Fetch full reciter list from mp3quran API."""
    print("Fetching reciter list from mp3quran.net...")
    try:
        resp = requests.get(API_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("reciters", [])
    except Exception as e:
        print(f"✗ Failed to fetch reciter list: {e}")
        return []


def find_reciter(reciters: list[dict], name: str) -> dict | None:
    """Find a reciter by name (case-insensitive partial match)."""
    name_lower = name.lower()
    # exact match first
    for r in reciters:
        if r["name"].lower() == name_lower:
            return r
    # partial match
    matches = [r for r in reciters if name_lower in r["name"].lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"\nMultiple matches for '{name}':")
        for m in matches:
            print(f"  - {m['name']}")
        print("Be more specific with --reciter")
        return None
    return None


def get_hafs_moshaf(reciter: dict) -> dict | None:
    """Get the Hafs A'n Assem Murattal moshaf (preferred) or first available."""
    moshafs = reciter.get("moshaf", [])
    # prefer Hafs Murattal (moshaf_type 11)
    for m in moshafs:
        if m.get("moshaf_type") == 11:
            return m
    # fallback to first
    return moshafs[0] if moshafs else None


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 200:
            content = resp.content
            if len(content) > 1000:  # ignore tiny/empty files
                dest.write_bytes(content)
                return True
        return False
    except requests.RequestException as e:
        print(f"    ✗ Network error: {e}")
        return False


def download_reciter(reciter: dict, surah_count: int) -> dict:
    """Download surahs for one reciter."""
    name = reciter["name"]
    moshaf = get_hafs_moshaf(reciter)

    if not moshaf:
        print(f"  ✗ No moshaf found for {name}")
        return {"name": name, "downloaded": 0, "skipped": 0, "failed": 0}

    server = moshaf["server"]
    available_surahs = [int(s) for s in moshaf["surah_list"].split(",") if s.strip()]

    # pick surahs from our preferred list that are actually available
    surahs_to_download = [s for s in DEFAULT_SURAHS if s in available_surahs][:surah_count]

    if not surahs_to_download:
        print(f"  ✗ No matching surahs available for {name}")
        return {"name": name, "downloaded": 0, "skipped": 0, "failed": 0}

    out_dir = SAMPLES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = failed = 0

    print(f"\n{'─'*55}")
    print(f"  Reciter : {name}")
    print(f"  Server  : {server}")
    print(f"  Style   : {moshaf['name']}")
    print(f"  Surahs  : {surahs_to_download}")
    print(f"{'─'*55}")

    for surah in surahs_to_download:
        filename = f"{surah:03d}.mp3"
        dest = out_dir / filename

        if dest.exists() and dest.stat().st_size > 1000:
            size_kb = dest.stat().st_size // 1024
            print(f"  → {filename}  [already exists {size_kb}KB, skipping]")
            skipped += 1
            continue

        url = f"{server}{filename}"
        success = download_file(url, dest)

        if success:
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓ {filename}  ({size_kb} KB)")
            downloaded += 1
        else:
            print(f"  ✗ {filename}  FAILED — {url}")
            failed += 1

        time.sleep(REQUEST_DELAY)

    return {"name": name, "downloaded": downloaded, "skipped": skipped, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def cmd_list(reciters: list[dict]):
    print(f"\n{'─'*60}")
    print(f"  {'#':<5} {'Name':<35} {'Surahs'}")
    print(f"{'─'*60}")
    for i, r in enumerate(reciters, 1):
        moshaf = get_hafs_moshaf(r)
        surah_count = moshaf["surah_total"] if moshaf else "?"
        print(f"  {i:<5} {r['name']:<35} {surah_count}")
    print(f"{'─'*60}")
    print(f"  Total: {len(reciters)} reciters\n")


def cmd_search(reciters: list[dict], query: str):
    query_lower = query.lower()
    matches = [r for r in reciters if query_lower in r["name"].lower()]
    if not matches:
        print(f"\n  No reciters found matching '{query}'\n")
        return
    print(f"\n  Found {len(matches)} match(es) for '{query}':\n")
    for r in matches:
        moshaf = get_hafs_moshaf(r)
        server = moshaf["server"] if moshaf else "N/A"
        print(f"  → {r['name']}")
        print(f"     Server: {server}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Qari audio from mp3quran.net for QuranID"
    )
    parser.add_argument("--list", action="store_true", help="List all available reciters")
    parser.add_argument("--search", type=str, help="Search reciters by name")
    parser.add_argument("--reciter", type=str, help="Download a specific reciter by name")
    parser.add_argument("--surahs", type=int, default=DEFAULT_SURAH_COUNT,
                        help=f"Number of surahs to download (default: {DEFAULT_SURAH_COUNT})")
    args = parser.parse_args()

    reciters = fetch_reciters()
    if not reciters:
        return

    if args.list:
        cmd_list(reciters)
        return

    if args.search:
        cmd_search(reciters, args.search)
        return

    if args.reciter:
        reciter = find_reciter(reciters, args.reciter)
        if not reciter:
            print(f"\n✗ Reciter '{args.reciter}' not found.")
            print(f"  Try: python download_mp3quran.py --search \"{args.reciter.split()[0]}\"\n")
            return

        print(f"\n{'═'*55}")
        print(f"  QuranID — mp3quran.net Downloader")
        print(f"{'═'*55}")

        result = download_reciter(reciter, args.surahs)

        print(f"\n{'═'*55}")
        print(f"  SUMMARY")
        print(f"{'═'*55}")
        print(f"  Downloaded : {result['downloaded']}")
        print(f"  Skipped    : {result['skipped']}")
        print(f"  Failed     : {result['failed']}")
        print(f"{'═'*55}\n")

        if result["downloaded"] > 0:
            print(f"  Next step → enroll into QuranID:")
            print(f"    python main.py enroll --name \"{reciter['name']}\" "
                  f"--folder \"data/samples/{reciter['name']}\" "
                  f"--style Murattal --nationality Saudi\n")
        return

    print("\nUsage:")
    print("  python download_mp3quran.py --list")
    print("  python download_mp3quran.py --search \"bandar\"")
    print("  python download_mp3quran.py --reciter \"Bandar Baleela\" --surahs 10\n")


if __name__ == "__main__":
    main()
