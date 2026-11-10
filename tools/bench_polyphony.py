"""同時発音数と処理時間の関係を測る（デバイスは開かない）。

エンジンの render() だけを直接呼ぶので、音は一切出ない。

測る条件:
- HRTF: 全天球の Viking（1513方向、アプリが使う）と水平面のみ（216方向）
- 同時発音数: 16 / 64 / 128 / 256（容量 = 発音数）
- 動き: 全音源が毎ブロック動き、首振りも有効（向きが毎ブロック変わる、
  いちばん重い状態）
- 容量の効き: 容量256で16音だけ鳴らす（コストが容量に比例するかを見る）

最後に cProfile で、128音源・全天球のときの内訳を出す。

Run: python -X utf8 tools\\bench_polyphony.py
Out: tools/bench_polyphony_report.txt (UTF-8)
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import (WOBBLE_PRESETS, BufferSource, HrirSet,  # noqa: E402
                        SpatialEngine)

SOFA = ROOT / "assets" / "hrtf" / "viking" / "subj_E.sofa"
REPORT = Path(__file__).with_name("bench_polyphony_report.txt")
BLOCK = 512
SR = 48000
BLOCKS = 150

_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg, flush=True)


def build(hrir: HrirSet, n_active: int, capacity: int,
          wobble: bool) -> SpatialEngine:
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=capacity)
    if wobble:
        engine.set_head_wobble(WOBBLE_PRESETS["clear"])
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.02
    for _ in range(n_active):
        engine.add_source(BufferSource(noise, loop=True), reverb_send=0.0,
                          position=(float(rng.uniform(-8, 8)),
                                    float(rng.uniform(-8, 8)),
                                    float(rng.uniform(-2, 2))))
    return engine


def step(engine: SpatialEngine, handles: list[int], i: int,
         moving: bool) -> None:
    if not moving:
        return
    angle = i * 0.05
    for k, h in enumerate(handles):
        a = angle + k
        engine.set_source_position(
            h, (6.0 * float(np.cos(a)), 6.0 * float(np.sin(a)),
                float(np.sin(a * 0.5))))


def measure(engine: SpatialEngine, moving: bool = True,
            blocks: int = BLOCKS) -> tuple[float, float]:
    handles = list(engine._slots)          # noqa: SLF001 - benchmark only
    times = []
    for i in range(blocks):
        step(engine, handles, i, moving)
        t0 = time.perf_counter()
        engine.render()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times[10:])             # 立ち上がりを捨てる
    return float(np.mean(arr)), float(np.percentile(arr, 99))


def main() -> int:
    budget = BLOCK / SR * 1000.0
    log("同時発音数と処理時間（render() のみ、デバイスなし）")
    log(f"block={BLOCK} @ {SR} Hz、予算 {budget:.2f} ms／ブロック")
    log("値は平均と 99パーセンタイル（ms）。全音源が毎ブロック動き、首振りあり")
    log("=" * 72)
    full = HrirSet.from_sofa(SOFA, sample_rate=SR)
    flat = full.horizontal()
    sets = (("全天球 1513方向", full), ("水平面 216方向", flat))

    for name, hrir in sets:
        log("")
        log(name)
        log(f"{'音源':>6} {'平均':>8} {'99%':>8} {'予算比':>8}")
        for n in (16, 64, 128, 256):
            mean, p99 = measure(build(hrir, n, n, wobble=True))
            log(f"{n:>6} {mean:>8.2f} {p99:>8.2f} {mean / budget * 100:>7.0f}%")

    log("")
    log("容量の効き（全天球、首振りあり、16音だけ鳴らす）")
    for cap in (16, 64, 256):
        mean, p99 = measure(build(full, 16, cap, wobble=True))
        log(f"  容量{cap:>4}: 平均 {mean:.2f} ms、99% {p99:.2f} ms")

    log("")
    log("向きの変化の効き（全天球、128音源、音源は静止）")
    for wobble in (False, True):
        mean, p99 = measure(build(full, 128, 128, wobble=wobble),
                            moving=False)
        log(f"  首振り{'あり' if wobble else 'なし'}: "
            f"平均 {mean:.2f} ms、99% {p99:.2f} ms")

    log("")
    log("内訳（cProfile、全天球、128音源、動きあり、首振りあり、100ブロック）")
    engine = build(full, 128, 128, wobble=True)
    handles = list(engine._slots)          # noqa: SLF001
    prof = cProfile.Profile()
    prof.enable()
    for i in range(100):
        step(engine, handles, i, True)
        engine.render()
    prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("tottime")
    stats.print_stats(18)
    for line in buf.getvalue().splitlines():
        if line.strip():
            log("  " + line.rstrip())

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
