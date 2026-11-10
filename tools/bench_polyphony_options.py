"""軽量化の候補を、エンジンを書き換えずに部品ごとに試算する。

bench_polyphony.py の内訳で重かった部分ごとに、今のやり方と候補を同じ
データで並べて測る。デバイスは開かない。PC の他の負荷で値が揺れるので、
各組を交互に REPEAT 回測って中央値を取る。

候補:
  A. 畳み込みの和を「鳴っている音源だけ」で計算する（容量ではなく実数に比例）
  B. FFT 長を 1024 から 640（ブロック512＋HRIR128）に縮める
  C. クロスフェードを「方向が変わった音源だけ」二重に計算する
  D. 音源ごとの方向計算（relative_spherical）をまとめて配列で計算する
  E. 方向表を間引く（全天球 1513方向 → 約10度格子）

Run: python -X utf8 tools\\bench_polyphony_options.py
Out: tools/bench_polyphony_options_report.txt (UTF-8)
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import HrirSet, Listener  # noqa: E402
from audiospace.geometry import relative_spherical  # noqa: E402

SOFA = ROOT / "assets" / "hrtf" / "viking" / "subj_E.sofa"
REPORT = Path(__file__).with_name("bench_polyphony_options_report.txt")
BLOCK = 512
SR = 48000
REPEAT = 7
LOOPS = 40

_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg, flush=True)


def timed(fn, loops: int = LOOPS) -> float:
    """fn を loops 回呼んだ1回あたりの ms。"""
    fn()
    t0 = time.perf_counter()
    for _ in range(loops):
        fn()
    return (time.perf_counter() - t0) / loops * 1000.0


def compare(cases: dict) -> dict[str, float]:
    """複数の関数を交互に REPEAT 回測り、中央値を返す。"""
    runs: dict[str, list[float]] = {k: [] for k in cases}
    for _ in range(REPEAT):
        for name, fn in cases.items():
            runs[name].append(timed(fn))
    return {k: float(np.median(v)) for k, v in runs.items()}


def spectra(hrir: HrirSet, nfft: int) -> np.ndarray:
    padded = np.zeros((hrir.n_directions, 2, nfft), dtype=np.float32)
    padded[..., :hrir.taps] = hrir.data
    return np.fft.rfft(padded, nfft, axis=-1).astype(np.complex64)


def conv_step(H: np.ndarray, blocks: np.ndarray, idx: np.ndarray,
              nfft: int, double: bool) -> np.ndarray:
    """今の BinauralConvolver と同じ計算（パーティション1つの場合）。"""
    X = np.fft.rfft(blocks, nfft, axis=-1)
    acc = np.einsum("sb,seb->eb", X, H[idx], optimize=True)
    out = np.fft.irfft(acc, nfft, axis=-1)
    if double:
        acc2 = np.einsum("sb,seb->eb", X, H[idx[::-1]], optimize=True)
        out = out + np.fft.irfft(acc2, nfft, axis=-1)
    return out


def main() -> int:
    budget = BLOCK / SR * 1000.0
    log("軽量化の候補の試算（部品ごと、中央値 ms／ブロック）")
    log(f"block={BLOCK} @ {SR} Hz、予算 {budget:.2f} ms。"
        f"各組を交互に{REPEAT}回測った中央値")
    log("=" * 72)
    full = HrirSet.from_sofa(SOFA, sample_rate=SR)
    rng = np.random.default_rng(7)

    # -- A と B ---------------------------------------------------------
    log("")
    log("A／B. 畳み込みの和（128音源分の容量、全天球）")
    H1024 = spectra(full, 1024)
    H640 = spectra(full, 640)
    cap = 128
    blocks = rng.standard_normal((cap, BLOCK)).astype(np.float32)
    idx = rng.integers(0, full.n_directions, cap)
    for active in (16, 128):
        b = blocks.copy()
        b[active:] = 0.0
        act = np.arange(active)
        r = compare({
            "今（容量ぶん、FFT1024）":
                lambda: conv_step(H1024, b, idx, 1024, False),
            "A 鳴っている分だけ":
                lambda: conv_step(H1024, b[act], idx[act], 1024, False),
            "B FFT640":
                lambda: conv_step(H640, b, idx, 640, False),
            "A＋B":
                lambda: conv_step(H640, b[act], idx[act], 640, False),
        })
        log(f"  {active}音が鳴っているとき:")
        for k, v in r.items():
            log(f"    {k:<22} {v:6.2f} ms")

    # -- C --------------------------------------------------------------
    log("")
    log("C. クロスフェード（128音源、全天球、FFT1024）")
    # 方向が変わる音源の割合を、首振り（標準8度、0.45Hz）だけのときで試算
    az = rng.uniform(0, 360, cap)
    el = rng.uniform(-30, 60, cap)
    changed = []
    prev = full.indices_for(az, el)
    for i in range(1, 200):
        off = 8.0 * math.sin(2 * math.pi * 0.45 * i * BLOCK / SR)
        cur = full.indices_for((az + off) % 360, el)
        changed.append(np.mean(cur != prev))
        prev = cur
    frac = float(np.mean(changed))
    n_changed = max(1, int(round(cap * frac)))
    ch = np.arange(n_changed)
    r = compare({
        "今（全音源を二重）":
            lambda: conv_step(H1024, blocks, idx, 1024, True),
        "C 変わった分だけ二重":
            lambda: (conv_step(H1024, blocks, idx, 1024, False),
                     conv_step(H1024, blocks[ch], idx[ch], 1024, False)),
        "クロスフェードなし":
            lambda: conv_step(H1024, blocks, idx, 1024, False),
    })
    log(f"  首振りだけのとき、1ブロックで方向が変わる音源は平均"
        f"{frac * 100:.1f}パーセント（{n_changed}個）")
    for k, v in r.items():
        log(f"    {k:<22} {v:6.2f} ms")

    # -- D --------------------------------------------------------------
    log("")
    log("D. 方向の計算（128音源）")
    pos = rng.uniform(-8, 8, (cap, 3))
    pos_list = [tuple(map(float, p)) for p in pos]
    listener = Listener((0.0, 0.0, 0.0), 30.0, 10.0, 5.0)

    def per_source() -> None:
        for p in pos_list:
            relative_spherical(listener, p)

    def vectorised() -> None:
        d = pos - np.asarray(listener.position)
        dist = np.linalg.norm(d, axis=1)
        yaw = math.radians(listener.yaw_deg)
        fwd = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        left = np.array([-fwd[1], fwd[0], 0.0])
        f = d @ fwd
        lft = d @ left
        up = d[:, 2]
        p = math.radians(listener.pitch_deg)
        hf = f * math.cos(p) + up * math.sin(p)
        hu = -f * math.sin(p) + up * math.cos(p)
        r_ = math.radians(listener.roll_deg)
        lft, hu = (lft * math.cos(r_) - hu * math.sin(r_),
                   lft * math.sin(r_) + hu * math.cos(r_))
        np.degrees(np.arctan2(lft, hf)) % 360.0
        np.degrees(np.arctan2(hu, np.hypot(hf, lft)))
        return dist

    r = compare({"今（音源ごと）": per_source, "D まとめて配列で": vectorised})
    for k, v in r.items():
        log(f"    {k:<22} {v:6.3f} ms")

    # -- E --------------------------------------------------------------
    log("")
    log("E. 方向表の大きさ（128音源、FFT1024、全音源の方向が毎ブロック散らばる）")
    dec = full.decimated(10.0)
    Hdec = spectra(dec, 1024)
    idx_dec = rng.integers(0, dec.n_directions, cap)
    mb_full = H1024.nbytes / 1024 / 1024
    mb_dec = Hdec.nbytes / 1024 / 1024
    r = compare({
        f"全天球 {full.n_directions}方向 {mb_full:.1f}MB":
            lambda: conv_step(H1024, blocks, idx, 1024, False),
        f"10度格子 {dec.n_directions}方向 {mb_dec:.1f}MB":
            lambda: conv_step(Hdec, blocks, idx_dec, 1024, False),
    })
    for k, v in r.items():
        log(f"    {k:<28} {v:6.2f} ms")

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
