"""用意した空間（SPACE_PRESETS）の精密版の残響の尾を焼き込む。

`set_space(..., late="precise")` は焼き込み済みの尾をすぐ使う。焼き込みが
無い空間は別のプロセスで計算する（数秒〜数十秒）ので、用意した空間は
ここで先に焼いておく。拡散した残響の無い空間（何もない道）は飛ばす。

出力: assets/space_ir/<鍵>_<レート>.npz と、一覧の index.json（名前と鍵）。
pyroomacoustics が要る。

Run: python -X utf8 tools\\build_space_irs.py [--rate 48000] [--only 名前 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace.acoustics import SPACE_PRESETS, space_preset  # noqa: E402
from audiospace.precise import (SPACE_IR_DIR, render_space_ir,  # noqa: E402
                                space_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    SPACE_IR_DIR.mkdir(parents=True, exist_ok=True)
    index_path = SPACE_IR_DIR / "index.json"
    index = (json.loads(index_path.read_text(encoding="utf-8"))
             if index_path.exists() else {})
    for name in SPACE_PRESETS:
        if args.only and name not in args.only:
            continue
        space = space_preset(name)
        key = space_key(space)
        t0 = time.perf_counter()
        made = render_space_ir(space, args.rate)
        took = time.perf_counter() - t0
        if made is None:
            print(f"{name}: 開いた面があるか拡散した残響が無い（飛ばす）",
                  flush=True)
            index.pop(name, None)
            stale = SPACE_IR_DIR / f"{key}_{args.rate}.npz"
            if stale.exists():
                stale.unlink()              # 前に焼いた分は使わせない
            continue
        out = SPACE_IR_DIR / f"{key}_{args.rate}.npz"
        made.save(out)
        index[name] = {"key": key, "label": space.name,
                       "rate": args.rate,
                       "rt60_bands": [round(float(v), 3)
                                      for v in made.rt60_bands],
                       "eyring_rt60_bands": [round(float(v), 3)
                                             for v in space.rt60_bands()],
                       "seconds": round(made.tail.shape[1] / args.rate, 2),
                       "render_s": round(took, 1)}
        print(f"{name}: {took:.1f}秒で計算、尾 {made.tail.shape[1] / args.rate:.2f}秒、"
              f"{out.stat().st_size / 1024:.0f}KB", flush=True)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    # 飛ばした空間を一覧から外した分も残す
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
