"""首振りと HRTF 差し替えの実機コスト。

首振りはブロックごとに全音源の方位を計算し直させるため、静止時より
必ず重くなる（毎ブロック HRIR の索引が変わる＝クロスフェードが常時走る）。
実用になる範囲か確かめる。出力は無音。

Out: tools/bench_wobble_report.txt (UTF-8)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("AUDIOSPACE_SILENT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiospace import (WOBBLE_PRESETS, BufferSource, HeadWobble,  # noqa: E402
                        HrirSet, SpatialEngine)

SOFA = (Path(__file__).resolve().parents[1] / "assets" / "hrtf" / "viking"
        / "subj_A.sofa")
REPORT = Path(__file__).with_name("bench_wobble_report.txt")
BLOCK = 512
SR = 48000
_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg)


def build(hrir: HrirSet, n: int, wobble: HeadWobble) -> SpatialEngine:
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=n)
    engine.set_head_wobble(wobble)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.1
    for _ in range(n):
        engine.add_source(BufferSource(noise, loop=True), reverb_send=0.0,
                          position=(float(rng.uniform(-8, 8)),
                                    float(rng.uniform(-8, 8)), 0.0))
    return engine


def measure(engine: SpatialEngine, blocks: int = 150) -> tuple[float, float]:
    times = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        engine.render()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times[10:])
    return float(np.mean(arr)), float(np.max(arr))


def main() -> int:
    budget = BLOCK / SR * 1000.0
    log("首振りのコスト測定")
    log(f"block={BLOCK} @ {SR}Hz、予算 {budget:.2f} ms")
    log("=" * 72)

    sets: list[tuple[str, HrirSet]] = [
        ("合成 72方向", HrirSet.synthetic(n_azimuth=72, taps=128,
                                          sample_rate=SR))]
    if SOFA.exists():
        full = HrirSet.from_sofa(SOFA, sample_rate=SR)
        sets.append(("実測 全天球", full))
        sets.append(("実測 水平面のみ", full.horizontal()))

    for name, hrir in sets:
        log("")
        log(f"--- {name}: {hrir.n_directions}方向、"
            f"スペクトル表 {hrir.table_bytes / 1024 / 1024:.1f} MB ---")
        log(f"{'音源':>6} {'首振り':<10} {'平均ms':>8} {'最悪ms':>8} {'予算比':>7}")
        for n in (8, 32, 64):
            for key in ("off", "clear", "strong"):
                engine = build(hrir, n, WOBBLE_PRESETS[key])
                mean, worst = measure(engine)
                log(f"{n:>6} {key:<10} {mean:>8.3f} {worst:>8.3f} "
                    f"{mean / budget * 100:>6.1f}%")

    if SOFA.exists():
        log("")
        log("--- HRTF 差し替えのコスト（耳介の選定で使う） ---")
        hrir_a = HrirSet.from_sofa(SOFA, sample_rate=SR)
        engine = build(hrir_a, 8, WOBBLE_PRESETS["clear"])
        engine.render()
        t0 = time.perf_counter()
        engine.set_hrir(hrir_a)
        swap_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        HrirSet.from_sofa(SOFA, sample_rate=SR)
        load_ms = (time.perf_counter() - t0) * 1000.0
        log(f"  SOFA 読み込み: {load_ms:.0f} ms（キャッシュすれば1回だけ）")
        log(f"  読み込み済みの差し替え: {swap_ms:.0f} ms")
        log("  差し替えは無音のタイミングで行うこと（畳み込み状態が切れる）")

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
