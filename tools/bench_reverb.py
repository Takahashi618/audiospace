"""残響の処理時間を測る（デバイスは開かない）。

部屋のプリセット7種それぞれについて、2つの残響を1ブロックずつ処理させて
時間を測る。

- FDN（`player.set_reverb(名前)`）: 遅延線8本の計算で作る軽い残響
- IR 畳み込み（`player.use_room_ir(名前)`）: 焼き込み済みの部屋の
  インパルス応答（最長 5秒以上）を畳み込む

各組を交互に REPEAT 回測り、中央値を取る（PC の他の負荷で値が揺れるため）。

Run: python -X utf8 tools\\bench_reverb.py
Out: tools/bench_reverb_report.txt (UTF-8)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import (HrirSet, SpatialEngine, SpatialPlayer,  # noqa: E402
                        preset_names)

REPORT = Path(__file__).with_name("bench_reverb_report.txt")
BLOCK = 512
SR = 48000
BLOCKS = 120
REPEAT = 5

_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg, flush=True)


def make(kind: str, name: str):
    """プレイヤーと同じ作り方で残響を作り、(残響, IR の長さ) を返す。"""
    hrir = HrirSet.synthetic(sample_rate=SR)
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=2)
    player = SpatialPlayer(engine)          # choice=None: デバイスなし
    if kind == "fdn":
        player.set_reverb(name)
    else:
        player.use_room_ir(name)
    reverb = engine.reverb
    taps = getattr(reverb, "taps", 0)
    return reverb, taps


def measure(reverb, blocks: int = BLOCKS) -> tuple[float, float]:
    rng = np.random.default_rng(1)
    feed = (rng.standard_normal((blocks, BLOCK)) * 0.1).astype(np.float32)
    times = []
    for i in range(blocks):
        t0 = time.perf_counter()
        reverb.process(feed[i])
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times[10:])
    return float(np.mean(arr)), float(np.percentile(arr, 99))


def main() -> int:
    budget = BLOCK / SR * 1000.0
    log("残響の処理時間（1ブロックあたり、デバイスなし）")
    log(f"block={BLOCK} @ {SR} Hz、予算 {budget:.2f} ms。"
        f"交互に{REPEAT}回測った中央値（平均と99パーセンタイル）")
    log("=" * 72)
    log(f"{'部屋':<12} {'IR長':>8} {'FDN平均':>8} {'FDN99%':>8} "
        f"{'IR平均':>8} {'IR99%':>8}")
    for name in preset_names():
        fdn, _ = make("fdn", name)
        conv, taps = make("ir", name)
        runs = {"fdn": [], "ir": []}
        for _ in range(REPEAT):
            runs["fdn"].append(measure(fdn))
            runs["ir"].append(measure(conv))
        f_mean = float(np.median([m for m, _ in runs["fdn"]]))
        f_p99 = float(np.median([p for _, p in runs["fdn"]]))
        i_mean = float(np.median([m for m, _ in runs["ir"]]))
        i_p99 = float(np.median([p for _, p in runs["ir"]]))
        log(f"{name:<12} {taps / SR:>7.2f}s {f_mean:>8.3f} {f_p99:>8.3f} "
            f"{i_mean:>8.3f} {i_p99:>8.3f}")
    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
