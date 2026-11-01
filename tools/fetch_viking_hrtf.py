"""Download the Viking HRTF dataset (CC BY 4.0) into assets/hrtf/.

Source : https://sofacoustics.org/data/database/viking/
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
Credit : v2 -- Simone Spagnol, Riccardo Miccini, Runar Unnthorsson,
         "The Viking HRTF dataset v2" (2020),
         https://doi.org/10.5281/zenodo.4160401
         v1 -- Simone Spagnol, Kristjan Bjarki Purkhus, Sverrir Karl
         Bjornsson, Runar Unnthorsson, "The Viking HRTF dataset" (SMC 2019)

CC BY permits commercial use provided attribution is given, so these
files may ship inside software that is sold.  Keep ATTRIBUTION.md next to
them and reproduce it in any product that embeds the data.

Measurement: KEMAR mannequin with 20 different pairs of artificial
silicone pinnae (subj_A .. subj_T) plus a pinna-less condition (subj_Z),
1513 directions over the full sphere.

Files already present are skipped, so re-running is cheap.
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://sofacoustics.org/data/database/viking/"
OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "hrtf" / "viking"
SUBJECTS = [chr(c) for c in range(ord("A"), ord("T") + 1)] + ["Z"]
EXTRA = ["documentation.pdf"]
RETRIES = 3
TIMEOUT = 120


def fetch(name: str, dest: Path) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"skip (already {dest.stat().st_size / 1024:.0f} KB)"
    url = BASE + name
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "audiospace-dataset-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True, f"{len(data) / 1024:.0f} KB"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == RETRIES:
                return False, f"failed after {RETRIES} tries: {exc!r}"
            time.sleep(2 * attempt)
    return False, "unreachable"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [f"subj_{s}.sofa" for s in SUBJECTS] + EXTRA
    total = 0
    failures: list[str] = []

    for i, name in enumerate(targets, start=1):
        dest = OUT_DIR / name
        ok, detail = fetch(name, dest)
        print(f"[{i:>2}/{len(targets)}] {name:<22} {detail}", flush=True)
        if ok:
            total += dest.stat().st_size
        else:
            failures.append(f"{name}: {detail}")

    print(f"\nsaved to {OUT_DIR}")
    print(f"total {total / 1024 / 1024:.1f} MB")
    if failures:
        print("[エラー] 取得できなかったファイル:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
