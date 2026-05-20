"""
QuranID — EveryAyah Sample Downloader
======================================
Downloads audio samples for multiple reciters from everyayah.com.
Files are saved into: data/samples/{reciter_name}/

Usage:
    python download_samples.py                        # download all configured reciters
    python download_samples.py --reciter "Al-Husary" # download one specific reciter
    python download_samples.py --list                 # print all available reciters
    python download_samples.py --ayahs 20            # download 20 ayahs per reciter
"""

import os
import time
import argparse
import requests
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://everyayah.com/data"

# Output folder (relative to this script — put the script in your quranID/ root)
SAMPLES_DIR = Path("data/samples")

# How many ayahs to download per reciter (more = better accuracy, more disk space)
# Recommended minimum: 15-20 for decent embeddings
DEFAULT_AYAH_COUNT = 20

# Delay between requests in seconds — be respectful to the server
REQUEST_DELAY = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# RECITERS DATABASE
# Each entry: display_name, folder_name_on_everyayah, style, nationality
# ─────────────────────────────────────────────────────────────────────────────

RECITERS = [
    # Your current 3
    {
        "name": "Al-Minshawy",
        "folder": "Minshawy_Murattal_128kbps",
        "style": "Murattal",
        "nationality": "Egyptian",
    },
    {
        "name": "Muhammad-Ayyoub",
        "folder": "Muhammad_Ayyoub_128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Yasser-Al-Dossary",
        "folder": "Yasser_Ad-Dussary_128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    # Additional popular reciters — uncomment to include
    {
        "name": "Al-Husary",
        "folder": "Husary_128kbps",
        "style": "Murattal",
        "nationality": "Egyptian",
    },
    {
        "name": "Abdul-Basit-Abdus-Samad",
        "folder": "Abdul_Basit_Murattal_192kbps",
        "style": "Murattal",
        "nationality": "Egyptian",
    },
    {
        "name": "Abdul-Basit-Abdus-Samad-Mujawwad-temp",
        "folder": "Abdul_Basit_Mujawwad_128kbps",
        "style": "Mujawwad",
        "nationality": "Egyptian",
    },
    {
        "name": "As-Sudais",
        "folder": "Abdurrahmaan_As-Sudais_192kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Alafasy",
        "folder": "Alafasy_128kbps",
        "style": "Murattal",
        "nationality": "Kuwaiti",
    },
    {
        "name": "Abu-Bakr-Ash-Shaatree",
        "folder": "Abu_Bakr_Ash-Shaatree_128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Maher-Al-Muaiqly",
        "folder": "MaherAlMuaiqly128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Hani-Rifai",
        "folder": "Hani_Rifai_192kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Hudhaify",
        "folder": "Hudhaify_128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Ash-Shuraym",
        "folder": "Saood_ash-Shuraym_128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Muhammad-Jibreel",
        "folder": "Muhammad_Jibreel_128kbps",
        "style": "Murattal",
        "nationality": "Egyptian",
    },
    {
        "name": "Ghamadi",
        "folder": "Ghamadi_40kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {
        "name": "Abdullah Awwad Al-juhaynee",
        "folder": "Abdullaah_3awwaad_Al-Juhaynee_128kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
    {   
        "name": "Nasser-Al-Qatami",
        "folder": "Nasser_Alqatami_128kbps",
        "style": "Murattal",
        "nationality": "Kuwaiti"
    },
    {       
        "name": "Salah-Al-Budair",
        "folder": "Salah_Al_Budair_128kbps",
        "style": "Murattal",
        "nationality": "Saudi"
    },
    {
        "name": "Muhsin-Al-Qasim",
        "folder": "Muhsin_Al_Qasim_192kbps",
        "style": "Murattal",
        "nationality": "Saudi"
    },
    {
        "name": "Khalid-Al-Qahtani",
        "folder": "Khaalid_Abdullaah_al-Qahtaanee_192kbps",
        "style": "Murattal",
        "nationality": "Saudi",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# AYAH SELECTION
# A spread of different surahs and styles gives more voice diversity
# than downloading the same surah 20 times
# ─────────────────────────────────────────────────────────────────────────────

# Format: (surah_number, ayah_number)
# Selected for audio length variety and natural speech patterns
AYAH_SELECTION = [
    # Al-Fatiha
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),

    # Al-Baqarah — mix of short and long
    (2, 1), (2, 2), (2, 255), (2, 256), (2, 257), (2, 285), (2, 286),

    # Al-Imran
    (3, 18), (3, 26), (3, 27),

    # Ya-Sin
    (36, 1), (36, 2), (36, 3), (36, 4), (36, 5),

    # Ar-Rahman
    (55, 1), (55, 2), (55, 3), (55, 4), (55, 13), (55, 26), (55, 27),

    # Al-Mulk
    (67, 1), (67, 2), (67, 3),

    # An-Naba
    (78, 1), (78, 2), (78, 3),

    # Al-Kahf
    (18, 1), (18, 2), (18, 3),

    # Maryam, Ta-Ha
    (19, 1), (19, 2), (20, 1), (20, 2),

    # Short surahs at the end of the Quran
    (112, 1), (113, 1), (113, 2), (114, 1), (114, 2),

    # Al-Insan, Al-Qiyamah
    (76, 1), (75, 1),
]


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

def build_url(folder: str, surah: int, ayah: int) -> str:
    """Build the EveryAyah MP3 URL for a given reciter/surah/ayah."""
    filename = f"{surah:03d}{ayah:03d}.mp3"
    return f"{BASE_URL}/{folder}/{filename}"


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    try:
        resp = requests.get(url, timeout=15, stream=True)
        if resp.status_code == 200 and int(resp.headers.get("content-length", 1)) > 500:
            dest.write_bytes(resp.content)
            return True
        else:
            return False
    except requests.RequestException as e:
        print(f"    ✗ Network error: {e}")
        return False


def download_reciter(reciter: dict, ayah_count: int) -> dict:
    """
    Download samples for one reciter.
    Returns a summary dict with counts.
    """
    name = reciter["name"]
    folder = reciter["folder"]
    out_dir = SAMPLES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ayahs_to_download = AYAH_SELECTION[:ayah_count]
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\n{'─'*55}")
    print(f"  Reciter : {name}")
    print(f"  Source  : {BASE_URL}/{folder}/")
    print(f"  Saving  : {out_dir}/")
    print(f"  Ayahs   : {len(ayahs_to_download)}")
    print(f"{'─'*55}")

    for surah, ayah in ayahs_to_download:
        filename = f"{surah:03d}{ayah:03d}.mp3"
        dest = out_dir / filename

        if dest.exists() and dest.stat().st_size > 500:
            print(f"  → {filename}  [already exists, skipping]")
            skipped += 1
            continue

        url = build_url(folder, surah, ayah)
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


def print_reciter_list():
    print("\nAvailable reciters in this script:\n")
    for i, r in enumerate(RECITERS, 1):
        print(f"  {i:2}. {r['name']:<30} {r['style']:<12} {r['nationality']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Qari audio samples from EveryAyah for QuranID"
    )
    parser.add_argument("--reciter", type=str, help="Download only this reciter by name")
    parser.add_argument("--ayahs", type=int, default=DEFAULT_AYAH_COUNT,
                        help=f"Number of ayahs to download per reciter (default: {DEFAULT_AYAH_COUNT})")
    parser.add_argument("--list", action="store_true", help="List all configured reciters and exit")
    args = parser.parse_args()

    if args.list:
        print_reciter_list()
        return

    # Filter reciters if --reciter was specified
    if args.reciter:
        targets = [r for r in RECITERS if r["name"].lower() == args.reciter.lower()]
        if not targets:
            print(f"\n✗ Reciter '{args.reciter}' not found. Run with --list to see options.\n")
            return
    else:
        targets = RECITERS

    print(f"\n{'═'*55}")
    print(f"  QuranID — EveryAyah Sample Downloader")
    print(f"{'═'*55}")
    print(f"  Reciters : {len(targets)}")
    print(f"  Ayahs ea : {args.ayahs}")
    print(f"  Output   : {SAMPLES_DIR.resolve()}")
    print(f"{'═'*55}")

    all_results = []
    for reciter in targets:
        result = download_reciter(reciter, args.ayahs)
        all_results.append(result)

    # Summary
    print(f"\n{'═'*55}")
    print("  SUMMARY")
    print(f"{'═'*55}")
    total_dl = total_skip = total_fail = 0
    for r in all_results:
        status = f"✓ {r['downloaded']} new"
        if r["skipped"]:
            status += f"  (skipped {r['skipped']})"
        if r["failed"]:
            status += f"  ✗ {r['failed']} failed"
        print(f"  {r['name']:<30} {status}")
        total_dl += r["downloaded"]
        total_skip += r["skipped"]
        total_fail += r["failed"]

    print(f"{'─'*55}")
    print(f"  Total downloaded : {total_dl}")
    print(f"  Total skipped    : {total_skip}")
    print(f"  Total failed     : {total_fail}")
    print(f"{'═'*55}\n")

    if total_dl > 0:
        print("  Next step → enroll them into QuranID:")
        for r in targets:
            print(f"    python main.py enroll --name \"{r['name']}\" --folder data/samples/{r['name']} --style {r.get('style','Murattal')} --nationality {r.get('nationality','Unknown')}")
    print()


if __name__ == "__main__":
    main()
