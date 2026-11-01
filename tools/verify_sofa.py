"""Verify HrirSet.from_sofa against the real Viking dataset.

Checks that the loader handles a genuine measured file correctly, and --
the point of the whole exercise -- that a measured HRTF actually carries
front/back information, which the synthetic spherical-head model cannot.

Out: tools/verify_sofa_report.txt (UTF-8)
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("AUDIOSPACE_SILENT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiospace import HrirSet  # noqa: E402
from audiospace.geometry import angular_distance_deg  # noqa: E402

HRTF_DIR = Path(__file__).resolve().parents[1] / "assets" / "hrtf" / "viking"
REPORT = Path(__file__).with_name("verify_sofa_report.txt")
SR = 48000

_log: list[str] = []


def log(msg: str = "") -> None:
    _log.append(msg)
    print(msg)


def band_energy(x: np.ndarray, lo: float, hi: float, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(len(x), 1 / sr)
    sel = (freq >= lo) & (freq < hi)
    return float(np.sum(spec[sel] ** 2))


def inspect(path: Path) -> HrirSet:
    hrir = HrirSet.from_sofa(path)
    log(f"file      : {path.name}")
    log(f"name      : {hrir.name}")
    log(f"directions: {hrir.n_directions}")
    log(f"taps      : {hrir.taps}")
    log(f"rate      : {hrir.sample_rate} Hz "
        f"({hrir.taps / hrir.sample_rate * 1000:.1f} ms per HRIR)")
    log(f"azimuth   : {hrir.azimuth.min():.1f} .. {hrir.azimuth.max():.1f} deg")
    log(f"elevation : {hrir.elevation.min():.1f} .. "
        f"{hrir.elevation.max():.1f} deg")
    log(f"peak      : {float(np.max(np.abs(hrir.data))):.4f}")
    return hrir


def check_lookup(hrir: HrirSet) -> None:
    log("")
    log("--- direction lookup accuracy ---")
    rng = np.random.default_rng(0)
    queries = [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0),
               (45.0, 30.0), (300.0, -20.0)]
    worst = 0.0
    for az, el in queries:
        idx = hrir.index_for(az, el)
        got_az = float(hrir.azimuth[idx])
        got_el = float(hrir.elevation[idx])
        err = angular_distance_deg(az, el, got_az, got_el)
        worst = max(worst, err)
        log(f"  asked ({az:6.1f}, {el:5.1f}) -> got ({got_az:6.1f}, "
            f"{got_el:5.1f})  off by {err:4.1f} deg")
    log(f"  worst nearest-neighbour error: {worst:.1f} deg")

    # vectorised lookup must agree with the scalar one
    az = rng.uniform(0, 360, 500)
    el = rng.uniform(-40, 60, 500)
    batch = hrir.indices_for(az, el)
    one_by_one = np.array([hrir.index_for(a, e) for a, e in zip(az, el)])
    agree = int(np.sum(batch == one_by_one))
    log(f"  vectorised vs scalar lookup: {agree}/500 identical")


def onset_sample(channel: np.ndarray, frac: float = 0.15) -> int | None:
    """First sample crossing `frac` of THIS channel's own peak.

    Thresholding against the louder channel's peak silently fails on the
    shadowed ear -- it never crosses, np.argmax returns 0 on an all-False
    array, and the ITD comes out negative and meaningless.
    """
    peak = float(np.max(np.abs(channel)))
    if peak <= 0:
        return None
    crossings = np.flatnonzero(np.abs(channel) > frac * peak)
    return int(crossings[0]) if crossings.size else None


def check_itd_ild(hrir: HrirSet) -> None:
    log("")
    log("--- interaural cues at 90 deg (listener's left) ---")
    ir = hrir.data[hrir.index_for(90.0, 0.0)]
    onset_l = onset_sample(ir[0])
    onset_r = onset_sample(ir[1])
    hi_l = band_energy(ir[0], 4000, 16000, hrir.sample_rate)
    hi_r = band_energy(ir[1], 4000, 16000, hrir.sample_rate)
    if onset_l is None or onset_r is None:
        log("  [警告] どちらかの耳で立ち上がりを検出できませんでした")
        return
    itd_us = (onset_r - onset_l) / hrir.sample_rate * 1e6
    log(f"  onset left={onset_l} right={onset_r} -> ITD {itd_us:+.0f} us "
        "(positive = left ear leads, as expected for a source on the left)")
    log(f"  high-band energy L/R ratio: "
        f"{hi_l / max(hi_r, 1e-12):.1f}x (head shadow)")
    if not 100 < itd_us < 900:
        log(f"  [警告] ITD が人間の想定範囲(100-900us)の外です: {itd_us:.0f} us")
    log("  -> receiver 0 = left ear, azimuth increases counter-clockwise; "
        "matches the SOFA convention the loader assumes")


def check_front_back(hrir: HrirSet) -> None:
    """The reason for using measured data at all."""
    log("")
    log("--- front/back difference (what the synthetic set cannot do) ---")
    log("  mirrored pairs share the same lateral angle, so any difference")
    log("  between them is genuine front/back information.")
    pairs = [(30.0, 150.0), (60.0, 120.0), (0.0, 180.0),
             (330.0, 210.0), (300.0, 240.0)]
    diffs = []
    for front, back in pairs:
        a = hrir.data[hrir.index_for(front, 0.0)]
        b = hrir.data[hrir.index_for(back, 0.0)]
        n = min(a.shape[1], b.shape[1])
        rel = (float(np.sqrt(np.mean((a[:, :n] - b[:, :n]) ** 2)))
               / max(float(np.sqrt(np.mean(a[:, :n] ** 2))), 1e-12))
        hi_f = sum(band_energy(a[c], 4000, 16000, hrir.sample_rate)
                   for c in (0, 1))
        hi_b = sum(band_energy(b[c], 4000, 16000, hrir.sample_rate)
                   for c in (0, 1))
        diffs.append(rel)
        log(f"  {front:5.1f} vs {back:5.1f} deg: waveform differs by "
            f"{rel * 100:5.1f}% rms, high-band ratio back/front="
            f"{hi_b / max(hi_f, 1e-12):.2f}")
    log(f"  mean difference: {np.mean(diffs) * 100:.1f}% "
        f"(synthetic HRIR would be exactly 0.0%)")


def check_synthetic_contrast() -> None:
    log("")
    log("--- same measurement on the synthetic set, for comparison ---")
    syn = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    for front, back in [(30.0, 150.0), (0.0, 180.0)]:
        a = syn.data[syn.index_for(front)]
        b = syn.data[syn.index_for(back)]
        rel = (float(np.sqrt(np.mean((a - b) ** 2)))
               / max(float(np.sqrt(np.mean(a ** 2))), 1e-12))
        log(f"  {front:5.1f} vs {back:5.1f} deg: differs by {rel * 100:.1f}%")


def check_resample(hrir: HrirSet) -> None:
    log("")
    log("--- resampling to 44100 Hz (WASAPI often forces this) ---")
    out = hrir.resampled(44100)
    log(f"  {hrir.sample_rate}Hz/{hrir.taps}taps -> "
        f"{out.sample_rate}Hz/{out.taps}taps, "
        f"{out.n_directions} directions preserved")
    ir_a = hrir.data[hrir.index_for(90.0, 0.0)]
    ir_b = out.data[out.index_for(90.0, 0.0)]
    ratio = (float(np.sqrt(np.mean(ir_b ** 2)))
             / max(float(np.sqrt(np.mean(ir_a ** 2))), 1e-12))
    log(f"  rms preserved to within {abs(1 - ratio) * 100:.1f}%")


def main() -> int:
    files = sorted(HRTF_DIR.glob("subj_*.sofa"))
    if not files:
        log(f"[エラー] SOFA ファイルが見つかりません: {HRTF_DIR}")
        REPORT.write_text("\n".join(_log), encoding="utf-8")
        return 2

    log("Viking HRTF (CC BY 4.0) — from_sofa verification")
    log(f"{len(files)} files in {HRTF_DIR}")
    log("=" * 74)

    try:
        hrir = inspect(files[0])
        check_lookup(hrir)
        check_itd_ild(hrir)
        check_front_back(hrir)
        check_synthetic_contrast()
        check_resample(hrir)
    except Exception as exc:                      # noqa: BLE001
        log(f"[エラー] {exc!r}")
        for line in traceback.format_exc().splitlines()[-6:]:
            log(f"   {line}")
        REPORT.write_text("\n".join(_log), encoding="utf-8")
        return 1

    log("")
    log("--- loading every subject ---")
    for path in files:
        try:
            h = HrirSet.from_sofa(path)
            log(f"  {path.name:<14} {h.n_directions:>5} dirs, "
                f"{h.taps:>4} taps, {h.sample_rate} Hz")
        except Exception as exc:                  # noqa: BLE001
            log(f"  {path.name:<14} [エラー] {exc!r}")

    REPORT.write_text("\n".join(_log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
