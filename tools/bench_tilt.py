# -*- coding: utf-8 -*-
"""上下の強調をエンジンで音源ごとにかけたときの重さを測る。

    python -X utf8 tools/bench_tilt.py

エンジンの後ろのローパス（`SpatialEngine._apply_damping`）と同じ作りで、
上下の強調の棚形フィルタ（x + k・高域通過(x)、3kHz、2次）を段階ごとに
まとめてかける。段階は ±12dB を 1dB 刻みの 25 段。音源の仰角はばらばらに
散らし、いちばん重い（段階が多く分かれる）場合を測る。

比べるもの（どれも 1 ブロック = 512 サンプル、48kHz の予算 10.67ms に対して）:

- 今の後ろのローパス（24段の1次フィルタ）
- 上下の強調（25段の2次フィルタ）
- 両方

後半は実際のエンジン（`SpatialEngine.elevation_emphasis`）で、全天球の
実測 HRTF・全音源が上下にも動く・首振りありの条件で、強調なしとありを
交互に測る。

音は鳴らさない（配列を計算するだけ）。結果は tools/bench_tilt_report.txt。
性能コアと高効率コアで速さが2倍違うので、同じ実行の中の比で読むこと。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import signal

OUT = Path(__file__).with_name("bench_tilt_report.txt")
SR = 48000
BLOCK = 512
BUDGET_MS = BLOCK / SR * 1000
LP_BUCKETS = 24
TILT_STEPS = np.arange(-12, 13)            # dB、25 段
REPEAT = 300


def make_tilt_sos() -> dict[int, np.ndarray]:
    """段階ごとの係数。x + k・HP(x) を1つの2次の節にまとめる。"""
    b, a = signal.butter(2, 3000.0, btype="highpass", fs=SR)
    table = {}
    for db in TILT_STEPS:
        if db == 0:
            continue
        k = 10 ** (db / 20) - 1
        # (1 + k·B/A) = (A + k·B)/A
        num = a + k * b
        table[int(db)] = np.concatenate([num, a])[None, :]
    return table


def bench(rows: int, rng: np.random.Generator) -> dict[str, float]:
    blocks = rng.standard_normal((rows, BLOCK)).astype(np.float32)
    lp_bucket = rng.integers(-1, LP_BUCKETS, rows)
    tilt_db = rng.choice(TILT_STEPS, rows)
    lp_zi = np.zeros((rows, 1))
    tilt_zi = np.zeros((1, rows, 2))
    sos = make_tilt_sos()

    def damping() -> None:
        active = lp_bucket > 0
        for bucket in np.unique(lp_bucket[active]):
            idx = np.flatnonzero(lp_bucket == bucket)
            c = max(0.05, 1.0 - bucket / (LP_BUCKETS - 1))
            out, zf = signal.lfilter([c], [1.0, -(1.0 - c)], blocks[idx],
                                     axis=1, zi=lp_zi[idx])
            blocks[idx] = out.astype(np.float32)
            lp_zi[idx] = zf
        lp_zi[~active] = 0.0

    def tilt() -> None:
        active = tilt_db != 0
        for db in np.unique(tilt_db[active]):
            idx = np.flatnonzero(tilt_db == db)
            out, zf = signal.sosfilt(sos[int(db)], blocks[idx], axis=1,
                                     zi=tilt_zi[:, idx])
            blocks[idx] = out.astype(np.float32)
            tilt_zi[:, idx] = zf
        tilt_zi[:, ~active] = 0.0

    def timed(fn) -> float:
        fn()                                   # 初回の準備を除く
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            fn()
            np.clip(blocks, -4, 4, out=blocks)  # 値が育ちすぎないように
        return (time.perf_counter() - t0) / REPEAT * 1000

    return {"後ろのローパス": timed(damping),
            "上下の強調": timed(tilt),
            "両方": timed(lambda: (damping(), tilt()))}


def bench_engine(rows: int, rounds: int = 3, blocks: int = 150
                 ) -> dict[str, tuple[float, float]]:
    """実エンジンで、上下の強調なし（0）とあり（0.5）を交互に測る。"""
    import math
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from audiospace import HrirSet
    from audiospace.engine import SpatialEngine
    from audiospace.motion import WOBBLE_PRESETS
    from audiospace.sources import BufferSource

    sofa = (Path(__file__).resolve().parents[1] / "assets" / "hrtf"
            / "viking" / "subj_E.sofa")
    hrir = HrirSet.from_sofa(sofa, sample_rate=SR)
    noise = (np.random.default_rng(3).standard_normal(SR) * 0.05
             ).astype(np.float32)
    times: dict[str, list[float]] = {"なし": [], "あり": []}
    for _ in range(rounds):
        for label, amount in (("なし", 0.0), ("あり", 0.5)):
            engine = SpatialEngine(hrir, block=BLOCK, max_sources=rows,
                                   elevation_emphasis=amount)
            engine.set_head_wobble(WOBBLE_PRESETS["clear"])
            handles = [engine.add_source(BufferSource(noise, loop=True),
                                         position=(0.0, 4.0, 0.0),
                                         reverb_send=0.0)
                       for _ in range(rows)]
            for b in range(blocks + 10):
                for i, h in enumerate(handles):
                    az = math.radians((i * 37 + b * 3) % 360)
                    el = math.radians(-45 + (i * 53 + b * 2) % 135)
                    engine.set_source_position(
                        h, (-4 * math.cos(el) * math.sin(az),
                            4 * math.cos(el) * math.cos(az),
                            4 * math.sin(el)))
                t0 = time.perf_counter()
                engine.render()
                if b >= 10:                     # 立ち上がりを除く
                    times[label].append((time.perf_counter() - t0) * 1000)
    return {k: (float(np.mean(v)), float(np.percentile(v, 99)))
            for k, v in times.items()}


def main() -> int:
    rng = np.random.default_rng(1)
    lines = ["上下の強調をエンジンでかけたときの重さ（1ブロックあたり、ミリ秒）",
             f"1ブロック {BLOCK} サンプル、予算 {BUDGET_MS:.2f}ms、各 {REPEAT} 回の平均",
             "音源の仰角と方位はばらばら（段階が最も多く分かれる場合）", ""]
    for rows in (16, 64, 128, 256):
        r = bench(rows, rng)
        lines.append(
            f"  {rows}音源: 後ろのローパス {r['後ろのローパス']:.3f}、"
            f"上下の強調 {r['上下の強調']:.3f}、両方 {r['両方']:.3f}"
            f"（予算の {r['両方'] / BUDGET_MS * 100:.1f}%）")
    lines += ["", "実エンジン（全天球の耳介 E、全音源が上下にも動く、首振りあり、"
              "残響なし）。平均 / 99%、ミリ秒",
              "強調なし（0）とあり（0.5）を交互に3回ずつ、各150ブロック"]
    for rows in (64, 128, 256):
        r = bench_engine(rows)
        (m0, p0), (m1, p1) = r["なし"], r["あり"]
        lines.append(f"  {rows}音源: なし {m0:.2f} / {p0:.2f}、"
                     f"あり {m1:.2f} / {p1:.2f}（平均の増加 {m1 - m0:+.2f}）")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
