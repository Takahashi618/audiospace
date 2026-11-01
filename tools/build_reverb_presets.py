"""Bake room impulse responses so a game never simulates at startup.

Simulating a cathedral takes several seconds; loading the baked .npz takes
milliseconds.  Run this once per project and ship the assets folder.

    python -X utf8 tools\\build_reverb_presets.py --rate 48000
    python -X utf8 tools\\build_reverb_presets.py --preset hall cave

Out: assets/reverb/<name>_<rate>.npz  plus a plain-text report.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiospace import (BUILTIN_PRESETS, RoomIr, describe_presets,  # noqa: E402
                        preset_names, render_room_ir)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "reverb"
REPORT = ROOT / "tools" / "build_reverb_report.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-render room impulse responses.")
    parser.add_argument("--preset", nargs="*", default=None,
                        help=f"names to build; default all "
                             f"({', '.join(preset_names())})")
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--list", action="store_true",
                        help="print the presets and exit")
    args = parser.parse_args()

    if args.list:
        print(describe_presets())
        return 0

    names = args.preset or preset_names()
    unknown = [n for n in names if n not in BUILTIN_PRESETS]
    if unknown:
        print(f"[エラー] 未知のプリセット: {', '.join(unknown)}")
        print(describe_presets())
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"reverb presets rendered at {args.rate} Hz",
             "=" * 70]
    for name in names:
        preset = BUILTIN_PRESETS[name]
        started = time.perf_counter()
        room = render_room_ir(preset, args.rate, args.max_seconds)
        elapsed = time.perf_counter() - started
        path = room.save(OUT_DIR / f"{name}_{args.rate}.npz")
        size_kb = path.stat().st_size / 1024
        line = (f"{name:<12} RT60={room.rt60:5.2f}s  "
                f"length={room.duration_s:5.2f}s  "
                f"file={size_kb:7.1f} KB  "
                f"rendered in {elapsed:5.2f}s")
        print(line)
        lines.append(line)
        lines.append(f"   {preset.description}")
        # sanity: an IR that is all zeros means the simulation failed
        if float(np.max(np.abs(room.ir))) <= 0:
            lines.append("   [警告] インパルス応答が無音です")

    lines.append("")
    lines.append(f"saved to {OUT_DIR}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {REPORT}")
    return 0


def load_all(rate: int = 48000) -> dict[str, RoomIr]:
    """後方互換の別名。本体はライブラリ側 `audiospace.load_all` に移した
    （ライブラリ単体から焼き込み済みIRへ届くようにするため）。"""
    from audiospace import load_all as _load_all

    return _load_all(rate, OUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
