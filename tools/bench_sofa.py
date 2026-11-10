"""Does the engine still run in real time with a measured 1513-direction set?

The synthetic set has 72 directions; Viking has 1513, which makes the
pre-computed HRIR spectrum table 21x larger and the nearest-direction
lookup 21x wider.  Neither should matter -- the table is indexed, not
scanned, per block -- but that is worth proving rather than assuming.

Out: tools/bench_sofa_report.txt (UTF-8)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiospace import BufferSource, HrirSet, SpatialEngine  # noqa: E402

SOFA = (Path(__file__).resolve().parents[1] / "assets" / "hrtf" / "viking"
        / "subj_A.sofa")
REPORT = Path(__file__).with_name("bench_sofa_report.txt")
BLOCK = 512
SR = 48000

_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg)


def build(hrir: HrirSet, n_sources: int) -> SpatialEngine:
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=n_sources)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.1
    for _ in range(n_sources):
        engine.add_source(BufferSource(noise, loop=True), reverb_send=0.0,
                          position=(float(rng.uniform(-8, 8)),
                                    float(rng.uniform(-8, 8)), 0.0))
    return engine


def measure(engine: SpatialEngine, blocks: int = 120) -> tuple[float, float]:
    handles = list(engine._slots)          # noqa: SLF001 - benchmark only
    times = []
    for i in range(blocks):
        angle = i * 0.05
        for k, h in enumerate(handles):
            a = angle + k
            engine.set_source_position(
                h, (6.0 * float(np.cos(a)), 6.0 * float(np.sin(a)), 0.0))
        t0 = time.perf_counter()
        engine.render()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times[5:])              # drop warm-up
    return float(np.mean(arr)), float(np.max(arr))


def main() -> int:
    budget = BLOCK / SR * 1000.0
    log("measured vs synthetic HRIR — engine cost")
    log(f"block={BLOCK} @ {SR} Hz, budget {budget:.2f} ms per callback")
    log("=" * 72)

    if not SOFA.exists():
        log(f"[エラー] {SOFA} がありません。"
            "先に tools\\fetch_viking_hrtf.py を実行してください")
        REPORT.write_text("\n".join(_log), encoding="utf-8")
        return 2

    t0 = time.perf_counter()
    measured = HrirSet.from_sofa(SOFA, sample_rate=SR)
    load_ms = (time.perf_counter() - t0) * 1000.0
    synthetic = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)

    for name, hrir in (("synthetic", synthetic), ("Viking subj_A", measured)):
        table_mb = (hrir.n_directions * 2 * (BLOCK + 1) * 8) / 1024 / 1024
        log("")
        log(f"{name}: {hrir.n_directions} directions, {hrir.taps} taps, "
            f"spectrum table ~{table_mb:.1f} MB")
        log(f"{'sources':>8} {'mean ms':>9} {'worst ms':>9} {'budget':>8}")
        for n in (8, 32, 64):
            engine = build(hrir, n)
            mean, worst = measure(engine)
            log(f"{n:>8} {mean:>9.3f} {worst:>9.3f} "
                f"{mean / budget * 100:>7.1f}%")

    log("")
    log(f"SOFA load time: {load_ms:.0f} ms "
        "(one-off at startup; the table is built once)")

    log("")
    log("--- lookup cost alone ---")
    for name, hrir in (("synthetic", synthetic), ("Viking", measured)):
        az = np.random.default_rng(1).uniform(0, 360, 64)
        el = np.zeros(64)
        hrir.indices_for(az, el)
        t0 = time.perf_counter()
        for _ in range(500):
            hrir.indices_for(az, el)
        per_call = (time.perf_counter() - t0) / 500 * 1000.0
        log(f"  {name:<10} {hrir.n_directions:>5} dirs: "
            f"{per_call:.4f} ms for 64 sources")

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
