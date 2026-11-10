"""Domain tests for audiospace -- no audio device required.

Run:  set AUDIOSPACE_SILENT=1 && python -X utf8 tests\\run_tests.py
Out:  tests/test_result.txt (UTF-8)

These check that the spatial cues are actually present in the rendered
audio (a source on the left really is louder in the left ear, a source
behind really is duller), not merely that the code runs.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
# 表示の言語を日本語に固定する（テストが期待する文は日本語）
os.environ["AUDIOSPACE_LANG"] = "ja"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiospace import (BufferSource, CallbackSource, HrirSet,  # noqa: E402
                        Listener, PushSource, SpatialEngine, SpatialPlayer,
                        StereoBufferSource, VectorFdnReverb, distance_gain,
                        front_back_gain, is_stereo_file, load_room_ir,
                        relative_spherical, resampled_by, rt60_of)
from audiospace.convolver import BinauralConvolver  # noqa: E402

OUT = Path(__file__).with_name("test_result.txt")
SR = 48000
BLOCK = 512

_log: list[str] = []
_passed = 0
_failed = 0


def log(msg: str = "") -> None:
    _log.append(msg)


def test(name: str):
    def deco(fn):
        def run() -> None:
            global _passed, _failed
            try:
                detail = fn()
                _passed += 1
                log(f"[PASS] {name}")
                for line in str(detail or "").splitlines():
                    if line.strip():
                        log(f"         {line}")
            except AssertionError as exc:
                _failed += 1
                log(f"[FAIL] {name}")
                log(f"         {exc}")
            except Exception as exc:  # noqa: BLE001
                _failed += 1
                log(f"[ERROR] {name}: {exc!r}")
                for line in traceback.format_exc().splitlines()[-4:]:
                    log(f"         {line}")
        return run
    return deco


def tone(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_energy(x: np.ndarray, lo: float, hi: float,
                sr: int = SR) -> float:
    """Total energy in [lo, hi) Hz.

    Always measure per channel and sum the energies -- summing the two ears
    into mono first comb-filters at any non-zero ITD and destroys exactly
    the difference being measured.
    """
    spec = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(len(x), 1 / sr)
    band = (freq >= lo) & (freq < hi)
    return float(np.sum(spec[band] ** 2))


# -- geometry --------------------------------------------------------------
@test("geometry: azimuth convention (0=front, 90=left, 270=right)")
def t_azimuth() -> str:
    listener = Listener((0.0, 0.0, 0.0), yaw_deg=0.0)   # facing north
    front = relative_spherical(listener, (0.0, 5.0, 0.0))[0]
    left = relative_spherical(listener, (-5.0, 0.0, 0.0))[0]
    right = relative_spherical(listener, (5.0, 0.0, 0.0))[0]
    behind = relative_spherical(listener, (0.0, -5.0, 0.0))[0]
    assert abs(front) < 1e-6, f"front should be 0, got {front}"
    assert abs(left - 90.0) < 1e-6, f"left(west) should be 90, got {left}"
    assert abs(right - 270.0) < 1e-6, f"right(east) should be 270, got {right}"
    assert abs(behind - 180.0) < 1e-6, f"behind should be 180, got {behind}"
    return (f"front={front:.1f} left={left:.1f} "
            f"right={right:.1f} behind={behind:.1f}")


@test("geometry: listener rotation moves the world correctly")
def t_rotation() -> str:
    source = (0.0, 5.0, 0.0)          # due north
    facing_north = relative_spherical(Listener(yaw_deg=0.0), source)[0]
    turned_left = relative_spherical(Listener(yaw_deg=90.0), source)[0]
    assert abs(facing_north) < 1e-6
    # after turning 90 deg counter-clockwise, north is on the right
    assert abs(turned_left - 270.0) < 1e-6, \
        f"expected 270 after turning left, got {turned_left}"
    return (f"facing north: az={facing_north:.1f}; "
            f"after yaw=90: az={turned_left:.1f}")


@test("geometry: elevation and distance")
def t_elevation() -> str:
    az, el, dist = relative_spherical(Listener(), (0.0, 3.0, 3.0))
    assert abs(el - 45.0) < 1e-6, f"expected 45 deg elevation, got {el}"
    assert abs(dist - np.sqrt(18)) < 1e-6
    return f"az={az:.1f} el={el:.1f} distance={dist:.3f}"


@test("geometry: head pitch tilts the world the right way")
def t_pitch() -> str:
    """Looking up brings a raised source to straight ahead and pushes an
    ear-level source down; pitch=0 must stay bit-identical to before."""
    up45 = (0.0, 3.0, 3.0)                      # north, 45 deg above
    level = (0.0, 5.0, 0.0)
    left = (-5.0, 0.0, 0.0)
    behind = (0.0, -5.0, 0.0)

    az, el, _ = relative_spherical(Listener(pitch_deg=45.0), up45)
    assert abs(el) < 1e-6 and min(az, 360 - az) < 1e-6, (az, el)
    _, el, _ = relative_spherical(Listener(pitch_deg=45.0), level)
    assert abs(el + 45.0) < 1e-6, f"looking up should lower a level source: {el}"
    az, el, _ = relative_spherical(Listener(pitch_deg=30.0), left)
    assert abs(az - 90.0) < 1e-6 and abs(el) < 1e-6, \
        f"pitch must not move a source on the tilt axis: {az}, {el}"
    _, el, _ = relative_spherical(Listener(pitch_deg=-15.0), behind)
    assert abs(el + 15.0) < 1e-6, \
        f"looking down raises the back of the head, so behind goes below: {el}"
    # yaw then pitch: turned left and looking up 30 faces a source there
    src = (-5.0 * np.cos(np.radians(30)), 0.0, 5.0 * np.sin(np.radians(30)))
    az, el, _ = relative_spherical(Listener(yaw_deg=90.0, pitch_deg=30.0), src)
    assert abs(el) < 1e-6 and min(az, 360 - az) < 1e-6, (az, el)
    # pitch=0 keeps the original code path exactly
    for s in (up45, level, left, behind, (1.0, 2.0, -0.5)):
        assert (relative_spherical(Listener(pitch_deg=0.0), s)
                == relative_spherical(Listener(), s))
    return ("up 45: raised source ahead, level source at -45; "
            "left stays left; yaw+pitch compose; pitch 0 unchanged")


@test("geometry: distance attenuation is monotonic and clamped")
def t_distance() -> str:
    gains = [distance_gain(d, "inverse_clamped", 1.0, 60.0)
             for d in (0.0, 0.5, 1.0, 2.0, 10.0, 60.0, 200.0)]
    assert gains[0] == gains[1] == gains[2] == 1.0, \
        f"inside the reference distance gain must be 1.0, got {gains[:3]}"
    assert all(a >= b for a, b in zip(gains[2:], gains[3:])), \
        f"gain must not increase with distance: {gains}"
    assert gains[-1] == gains[-2], "beyond max_distance gain must be clamped"
    return "gains at 0/0.5/1/2/10/60/200 m: " + \
        ", ".join(f"{g:.4f}" for g in gains)


@test("geometry: front/back cue is strongest ahead")
def t_front_back() -> str:
    ahead = front_back_gain(0.0, 0.35)
    side = front_back_gain(90.0, 0.35)
    behind = front_back_gain(180.0, 0.35)
    assert ahead > side > behind, f"{ahead} {side} {behind}"
    assert abs(ahead - 1.0) < 1e-9
    return f"ahead={ahead:.3f} side={side:.3f} behind={behind:.3f}"


@test("geometry: 前後の手がかりは上で cos(仰角)、下は45度まで全量、真上と真下では効かない")
def t_front_back_elevation() -> str:
    """真上の音は方位を変えても同じ点。以前は計算上の方位が残り、方位180度で
    指定した真上の音だけ 0.65 倍に下がってこもっていた（CUES.md）。

    下側は下45度まで水平と同じ強さ。cos で弱めると、上下の強調で暗くした
    後ろの下の音が後ろの水平の音と同じ明るさになった。"""
    from audiospace.geometry import front_back_gain_many
    depth = 0.35
    zenith = [front_back_gain(az, depth, 90.0) for az in (0.0, 90.0, 180.0)]
    assert all(abs(g - 1.0) < 1e-12 for g in zenith), zenith
    nadir = front_back_gain(180.0, depth, -90.0)
    assert abs(nadir - 1.0) < 1e-12, nadir
    flat = front_back_gain(180.0, depth, 0.0)
    assert abs(flat - front_back_gain(180.0, depth)) < 1e-15
    assert abs(flat - (1.0 - depth)) < 1e-12, flat
    up45 = front_back_gain(180.0, depth, 45.0)
    down45 = front_back_gain(180.0, depth, -45.0)
    expect = 1.0 - depth * math.cos(math.radians(45.0))
    assert abs(up45 - expect) < 1e-12, (up45, expect)
    assert abs(down45 - flat) < 1e-12, (down45, flat)
    down60 = front_back_gain(180.0, depth, -60.0)
    expect60 = 1.0 - depth * math.sqrt(2.0) * math.cos(math.radians(60.0))
    assert abs(down60 - expect60) < 1e-12, (down60, expect60)
    down30 = front_back_gain(180.0, depth, -30.0)
    assert abs(down30 - flat) < 1e-12, (down30, flat)
    rng = np.random.default_rng(5)
    azs = rng.uniform(0, 360, 100)
    els = rng.uniform(-90, 90, 100)
    many = front_back_gain_many(azs, depth, els)
    assert all(abs(many[i] - front_back_gain(azs[i], depth, els[i])) < 1e-12
               for i in range(100))
    # ローパスの段階も同じ規則（真上は -1 = かけない）
    engine = make_engine(front_back_depth=depth)
    assert engine._bucket_for(180.0, 0.0, 90.0) == -1
    assert engine._bucket_for(180.0) > 0
    buckets = engine._buckets_for(azs, np.zeros(100), els)
    assert all(buckets[i] == engine._bucket_for(azs[i], 0.0, els[i])
               for i in range(100))
    return (f"真上は方位0/90/180度とも {zenith[0]:.3f}、真後ろ: 水平 {flat:.3f}、"
            f"上45度 {up45:.3f}、下45度 {down45:.3f}、下60度 {down60:.3f}。"
            "ローパスの段階も一致")


# -- HRTF ------------------------------------------------------------------
@test("hrtf: synthetic set has the expected shape and lookup")
def t_hrir_shape() -> str:
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    assert hrir.data.shape == (72, 2, 128), hrir.data.shape
    assert hrir.index_for(0.0) == 0
    assert hrir.index_for(90.0) == 18, hrir.index_for(90.0)
    assert hrir.index_for(359.9) == 0, hrir.index_for(359.9)
    return (f"{hrir.name}: {hrir.n_directions} directions, "
            f"{hrir.taps} taps, lookup at 90 deg -> index "
            f"{hrir.index_for(90.0)}")


@test("hrtf: left source arrives at the left ear first (ITD)")
def t_itd() -> str:
    hrir = HrirSet.synthetic(sample_rate=SR)
    left_source = hrir.data[hrir.index_for(90.0)]     # 90 deg = listener left
    onset_l = int(np.argmax(np.abs(left_source[0]) > 0.05))
    onset_r = int(np.argmax(np.abs(left_source[1]) > 0.05))
    itd_us = (onset_r - onset_l) / SR * 1e6
    assert onset_l < onset_r, \
        f"left ear must lead for a left source: {onset_l} vs {onset_r}"
    assert 100 < itd_us < 900, f"implausible ITD {itd_us:.0f} us"
    return (f"left ear onset={onset_l}, right ear onset={onset_r}, "
            f"ITD={itd_us:.0f} us (human range is up to ~700)")


@test("hrtf: contralateral ear is darker (head shadow / ILD)")
def t_ild() -> str:
    hrir = HrirSet.synthetic(sample_rate=SR)
    ir = hrir.data[hrir.index_for(90.0)]
    hi_l = band_energy(ir[0], 4000, 16000)
    hi_r = band_energy(ir[1], 4000, 16000)
    assert hi_l > hi_r * 2, \
        f"shadowed ear should lose high frequencies: {hi_l:.4g} vs {hi_r:.4g}"
    return (f"high-band energy left={hi_l:.4g} right={hi_r:.4g} "
            f"(ratio {hi_l / max(hi_r, 1e-12):.1f}x)")


@test("hrtf: resampling preserves direction count")
def t_resample() -> str:
    hrir = HrirSet.synthetic(n_azimuth=36, taps=128, sample_rate=48000)
    out = hrir.resampled(44100)
    assert out.n_directions == 36
    assert out.sample_rate == 44100
    assert abs(out.taps - 128 * 44100 / 48000) <= 2, out.taps
    return f"48000Hz/128taps -> {out.sample_rate}Hz/{out.taps}taps"


# -- convolver -------------------------------------------------------------
@test("convolver: matches direct convolution")
def t_conv_accuracy() -> str:
    hrir = HrirSet.synthetic(n_azimuth=36, taps=128, sample_rate=SR)
    conv = BinauralConvolver(hrir, block=BLOCK, max_sources=1,
                             crossfade=False)
    rng = np.random.default_rng(7)
    sig = rng.standard_normal(BLOCK * 8).astype(np.float32) * 0.2
    idx = np.array([11], dtype=np.intp)
    out = np.concatenate(
        [conv.process(sig[i * BLOCK:(i + 1) * BLOCK][None, :], idx)
         for i in range(8)])
    ref_l = np.convolve(sig, hrir.data[11, 0])[:len(out)]
    err = float(np.max(np.abs(out[:, 0] - ref_l)))
    assert err < 1e-4, f"error {err:.3e} too large"
    return f"max abs error vs np.convolve = {err:.3e}"


@test("convolver: crossfade removes the click when direction jumps")
def t_crossfade() -> str:
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    sig = tone(220.0, BLOCK * 6 / SR)

    def run(crossfade: bool) -> float:
        conv = BinauralConvolver(hrir, block=BLOCK, max_sources=1,
                                 crossfade=crossfade)
        blocks = []
        for i in range(6):
            # 0 -> 90 deg.  Deliberately NOT a 180 deg jump: the synthetic
            # HRIR models only the left/right (sin) component, so index 0 and
            # index 36 are numerically identical and nothing would change.
            idx = np.array([0 if i < 3 else 18], dtype=np.intp)
            blocks.append(conv.process(
                sig[i * BLOCK:(i + 1) * BLOCK][None, :], idx))
        out = np.concatenate(blocks)[:, 0]
        seam = out[3 * BLOCK - 4:3 * BLOCK + 4]
        return float(np.max(np.abs(np.diff(seam))))

    hard = run(False)
    faded = run(True)
    assert faded < hard, \
        f"crossfade did not reduce the discontinuity: {faded} vs {hard}"
    return (f"max sample-to-sample jump at the seam: "
            f"without crossfade={hard:.5f}, with crossfade={faded:.5f} "
            f"({hard / max(faded, 1e-12):.1f}x smoother)")


def _reference_render(hrir: HrirSet, sig: np.ndarray,
                      idx_per_block: np.ndarray, crossfade: bool
                      ) -> np.ndarray:
    """Slow, obvious reference: every source, every block, both renders.

    For each block, each source's output is its signal convolved with the
    current HRIR (and, when crossfading, with the previous one, blended by a
    linear ramp over the block).  Overlap tails carry the new HRIR only --
    the same definition the convolver has always used.
    """
    n_src, total = sig.shape
    n_blocks = total // BLOCK
    taps = hrir.taps
    out = np.zeros((total + taps, 2))
    ramp = np.linspace(0.0, 1.0, BLOCK, endpoint=False)
    for b in range(n_blocks):
        seg = sig[:, b * BLOCK:(b + 1) * BLOCK].astype(np.float64)
        for s in range(n_src):
            cur = idx_per_block[b, s]
            prev = idx_per_block[b - 1, s] if b > 0 else cur
            for ear in range(2):
                new = np.convolve(seg[s], hrir.data[cur, ear])
                y = new.copy()
                if crossfade and prev != cur:
                    old = np.convolve(seg[s], hrir.data[prev, ear])
                    y[:BLOCK] = old[:BLOCK] * (1 - ramp) + new[:BLOCK] * ramp
                out[b * BLOCK:b * BLOCK + len(y), ear] += y
    return out[:total]


@test("convolver: per-source crossfade and short FFT match a direct reference")
def t_conv_partial_crossfade() -> str:
    """2026-10: only sources whose direction changed are rendered twice, and
    with a 128-tap HRIR the FFT is 640 long instead of 1024.  Both must give
    the same result as rendering every source both ways by direct
    convolution.  Some sources move, some stay, so both paths are mixed in
    the same block."""
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    n_src, n_blocks = 6, 10
    rng = np.random.default_rng(21)
    sig = (rng.standard_normal((n_src, BLOCK * n_blocks)) * 0.2
           ).astype(np.float32)
    idx = np.tile(rng.integers(0, 72, n_src), (n_blocks, 1))
    for b in range(1, n_blocks):
        idx[b, :3] = (idx[b - 1, :3] + 5 * (b % 2 + 1)) % 72   # 3 sources move
    conv = BinauralConvolver(hrir, block=BLOCK, max_sources=n_src)
    assert conv.nfft == 640, f"expected the short FFT, got {conv.nfft}"
    got = np.concatenate([
        conv.process(sig[:, b * BLOCK:(b + 1) * BLOCK], idx[b])
        for b in range(n_blocks)])
    ref = _reference_render(hrir, sig, idx, crossfade=True)
    err = float(np.max(np.abs(got - ref)))
    assert err < 1e-4, f"error {err:.3e} vs direct reference"
    return (f"6 sources (3 moving) x 10 blocks, FFT {conv.nfft}: "
            f"max abs error vs direct reference {err:.2e}")


@test("convolver: only active rows are computed, with the same result")
def t_conv_active_rows() -> str:
    """Passing `active` must give exactly what passing every row gives when
    the other rows are silent -- for one partition and for several."""
    lines = []
    for taps in (128, 1500):
        hrir = HrirSet.synthetic(n_azimuth=72, taps=taps, sample_rate=SR)
        cap = 12
        rng = np.random.default_rng(5)
        a = BinauralConvolver(hrir, block=BLOCK, max_sources=cap)
        b = BinauralConvolver(hrir, block=BLOCK, max_sources=cap)
        worst = 0.0
        for blk in range(12):
            act = np.array([1, 4, 7] if blk < 6 else [4, 9], dtype=np.intp)
            x = np.zeros((cap, BLOCK), dtype=np.float32)
            x[act] = rng.standard_normal((len(act), BLOCK)) * 0.2
            idx = rng.integers(0, 72, cap)
            ya = a.process(x, idx, active=act)
            yb = b.process(x, idx)
            worst = max(worst, float(np.max(np.abs(ya - yb))))
        assert worst < 1e-5, f"taps {taps}: {worst:.3e}"
        lines.append(f"taps {taps} ({a.nparts} part, FFT {a.nfft}): "
                     f"max diff {worst:.1e}")
    return "\n".join(lines)


@test("convolver: a newly started source begins at its own direction")
def t_conv_new_source_no_fade() -> str:
    """Until 2026-10 a row's first block was crossfaded from whatever
    direction the row last held (index 0 for a fresh row), so every new
    sound began ~11 ms from the wrong place."""
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((1, BLOCK)).astype(np.float32) * 0.2
    idx = np.array([18], dtype=np.intp)                      # 90 deg left
    faded = BinauralConvolver(hrir, block=BLOCK, max_sources=1)
    plain = BinauralConvolver(hrir, block=BLOCK, max_sources=1,
                              crossfade=False)
    diff = float(np.max(np.abs(faded.process(x, idx) - plain.process(x, idx))))
    assert diff < 1e-6, f"first block was crossfaded: {diff:.3e}"
    return f"first block equals a direct render at 90 deg (diff {diff:.1e})"


@test("engine: a source reusing a just-freed row is not faded from the old one's direction")
def t_engine_row_reuse_no_fade() -> str:
    """Stop-then-play puts the new sound in the row the old one just left.
    The convolver must treat it as a new start (no crossfade from the old
    occupant's direction), exactly as with crossfade switched off.  The old
    occupant's tail is shared output state and is identical in both."""
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    rng = np.random.default_rng(13)
    noise = (rng.standard_normal(SR) * 0.2).astype(np.float32)

    def run(crossfade: bool) -> np.ndarray:
        engine = SpatialEngine(hrir, block=BLOCK, max_sources=4,
                               crossfade=crossfade)
        a = engine.add_source(BufferSource(noise, loop=True),
                              position=(-5.0, 0.0, 0.0), reverb_send=0.0)
        engine.render()                                  # A on the left
        engine.remove_source(a)
        engine.add_source(BufferSource(noise[100:], loop=True),
                          position=(5.0, 0.0, 0.0), reverb_send=0.0)
        return engine.render()                           # B on the right

    faded, plain = run(True), run(False)
    diff = float(np.max(np.abs(faded - plain)))
    assert diff < 1e-6, (f"the new source was crossfaded from the old "
                         f"occupant's direction (diff {diff:.3e})")
    return f"row reused left -> right: first block equals no-crossfade ({diff:.1e})"


@test("geometry: linear distance model with maximum == reference does not divide by zero")
def t_linear_model_no_div0() -> str:
    import warnings

    from audiospace.geometry import distance_gain_many
    d = np.array([0.0, 0.5, 1.0, 2.0, 50.0])
    for maximum in (1.0, 0.5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            many = distance_gain_many(d, "linear", reference=1.0,
                                      maximum=maximum)
        scalar = [distance_gain(x, "linear", reference=1.0, maximum=maximum)
                  for x in d]
        assert np.allclose(many, scalar), (maximum, many, scalar)
    return "maximum == reference and maximum < reference: no warning, same as scalar"


@test("geometry: many-source direction math matches the per-source version")
def t_geometry_many() -> str:
    from audiospace.geometry import (distance_gain_many,
                                     front_back_gain_many,
                                     relative_spherical_many)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-10, 10, (300, 3))
    pts[0] = (1.0, 2.0, 3.0)                 # exactly at one listener
    worst = 0.0
    for lst in (Listener((1.0, 2.0, 3.0)), Listener((0.0, 0.0, 0.0), 40.0),
                Listener((0.5, -1.0, 0.2), 200.0, 25.0, -12.0)):
        az, el, dist = relative_spherical_many(lst, pts)
        for i, p in enumerate(pts):
            a, e, d = relative_spherical(lst, tuple(p))
            worst = max(worst, abs((az[i] - a + 180) % 360 - 180),
                        abs(el[i] - e), abs(dist[i] - d))
    assert worst < 1e-9, f"direction mismatch {worst:.3e}"
    dists = np.concatenate([[0.0, 0.5, 1.0], rng.uniform(0, 80, 200)])
    for model in ("inverse", "inverse_clamped", "linear", "none"):
        many = distance_gain_many(dists, model)
        for i, d in enumerate(dists):
            assert abs(many[i] - distance_gain(d, model)) < 1e-12, (model, d)
    azs = rng.uniform(0, 360, 100)
    fb = front_back_gain_many(azs, 0.35)
    assert all(abs(fb[i] - front_back_gain(a, 0.35)) < 1e-12
               for i, a in enumerate(azs))
    return (f"300 points x 3 listeners: max direction diff {worst:.1e} deg; "
            "distance and front/back gains identical")


# -- spaces (materials, early reflections, band reverb) --------------------
def _band_rt60(ir: np.ndarray, fc: float) -> float:
    from scipy import signal as sg
    sos = sg.butter(4, [fc / np.sqrt(2), fc * np.sqrt(2)], btype="band",
                    fs=SR, output="sos")
    y = sg.sosfilt(sos, ir.astype(np.float64))
    e = np.cumsum(y[::-1] ** 2)[::-1]
    db = 10 * np.log10(np.maximum(e / e[0], 1e-12))
    s, t = int(np.argmax(db <= -5)), int(np.argmax(db <= -25))
    return float(-60 / np.polyfit(np.arange(s, t) / SR, db[s:t], 1)[0])


@test("space: material table and per-band decay times order as expected")
def t_space_materials() -> str:
    from audiospace import absorption, band_centers, material_names
    from audiospace.acoustics import space_preset
    names = material_names()
    assert "open" in names and "rough_concrete" in names and len(names) > 80
    assert len(absorption("wooden_lining")) == len(band_centers()) == 7
    try:
        absorption("no_such_material")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown material name was accepted")
    k = band_centers().index(1000)
    rt = {n: space_preset(n).rt60_bands()[k] for n in
          ("carpet_room", "wood_room", "wood_floor_concrete_walls",
           "concrete_room")}
    order = sorted(rt, key=rt.get)
    assert order == ["carpet_room", "wood_room", "wood_floor_concrete_walls",
                     "concrete_room"], rt
    low, high = space_preset("wood_room").rt60_bands()[[1, 5]]
    assert low < high, "wood absorbs lows: its low band should decay faster"
    assert not space_preset("open_road").rt60_bands().any(), \
        "open ground must have no diffuse reverb"
    return ("1 kHz RT60: " + ", ".join(f"{n} {rt[n]:.2f}s" for n in order)
            + f"; wood room 250 Hz {low:.2f}s < 4 kHz {high:.2f}s")


@test("space: reflection paths follow the walls; open sides return nothing")
def t_space_paths() -> str:
    from audiospace import Space
    from audiospace.acoustics import SPEED_OF_SOUND, space_preset
    room = Space.room((5.0, 4.0, 2.7), floor="wood_16mm",
                      walls="rough_concrete", ceiling="plasterboard",
                      at=(0.2, 0.5), ear_height=1.6)
    paths = room.reflection_paths((0.0, 0.0, 0.0))
    first = {p.faces[0]: p for p in paths if p.order == 1}
    assert set(first) == {"west", "east", "south", "north", "floor",
                          "ceiling"}, set(first)
    # listener 1.0 m from the west wall (20% of 5 m): path 2.0 m, from the west
    west = first["west"]
    assert abs(west.distance - 2.0) < 1e-9 and west.direction == (-1, 0, 0)
    assert abs(first["floor"].distance - 3.2) < 1e-9
    assert abs(west.delay_s - 2.0 / SPEED_OF_SOUND) < 1e-12
    concrete = np.mean(west.bands)
    wood_floor = np.mean(first["floor"].bands)
    assert concrete > wood_floor, "concrete should reflect more than wood"
    road = space_preset("open_road").reflection_paths((0.0, 0.0, 0.0))
    assert [p.faces for p in road] == [("floor",)], [p.faces for p in road]
    street = space_preset("street").reflection_paths((0.0, 0.0, 0.0))
    walls = sorted((p for p in street if p.faces[0] in ("west", "east")),
                   key=lambda p: p.distance)
    sides = [p.direction[0] for p in walls]
    assert len(walls) >= 4 and min(sides) < 0 < max(sides), sides
    assert all(p.faces[-1] != "ceiling" for p in street)
    outside = room.reflection_paths((50.0, 0.0, 0.0))
    assert outside == [], "a listener outside the room must get no paths"
    return (f"room: 6 first-order paths, west 2.0 m from the left; "
            f"open road: ground only; street: {len(walls)} flutter paths "
            f"between the facades")


@test("space: band FDN reverb decays at the target time in each band")
def t_band_fdn() -> str:
    from audiospace import BandFdnReverb
    from audiospace.acoustics import space_preset
    lines = []
    for name in ("wood_room", "concrete_room"):
        target = space_preset(name).rt60_bands()
        rev = BandFdnReverb(SR, target, room_scale=0.3)
        n_blocks = int(min(10.0, target.max() * 1.6) * SR / BLOCK)
        x = np.zeros((n_blocks, BLOCK), dtype=np.float32)
        x[0, 0] = 1.0
        ir = np.concatenate([rev.process(b) for b in x])[:, 0]
        for k, fc in ((1, 250), (3, 1000), (5, 4000)):
            got = _band_rt60(ir, fc)
            ratio = got / target[k]
            assert 0.65 < ratio < 1.5, (f"{name} {fc} Hz: {got:.2f}s "
                                        f"vs target {target[k]:.2f}s")
            lines.append(f"{name} {fc}Hz {got:.2f}s (target {target[k]:.2f}s)")
    return "\n".join(lines)


@test("space: an early reflection arrives on time and from the wall's side")
def t_early_reflection() -> str:
    """Listener 1 m from the west wall, facing north: the wall is on the left.
    A click in front reaches the listener directly, then 2 m / c later again
    from the left."""
    from audiospace import BufferSource, Space
    from audiospace.acoustics import SPEED_OF_SOUND
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    room = Space.room((5.0, 4.0, 2.7), floor="open", walls={
        "west": "rough_concrete", "east": "open", "south": "open",
        "north": "open"}, ceiling="open", at=(0.2, 0.5))
    click = np.zeros(SR // 2, dtype=np.float32)
    # not at sample 0: a new source fades in over its first block, which
    # would silence a click there (and with it the reflection)
    click[BLOCK + 100] = 1.0

    def run(space) -> np.ndarray:
        engine = SpatialEngine(hrir, block=BLOCK, max_sources=2)
        if space is not None:
            engine.set_space(space, late=False)
        engine.add_source(BufferSource(click), position=(0.0, 1.0, 0.0),
                          reverb_send=0.0)
        return engine.render_seconds(0.1)

    plain, spaced = run(None), run(room)
    refl = spaced - plain
    delay = int(round(2.0 / SPEED_OF_SOUND * SR))
    direct_at = int(np.argmax(np.abs(plain).sum(axis=1)))
    refl_at = int(np.argmax(np.abs(refl).sum(axis=1)))
    # The reflection comes from the left, so it reaches the (louder) left ear
    # up to one interaural delay (~12 samples here) before a frontal sound
    # would: allow 16 samples (0.33 ms) on the early side.
    lag = refl_at - direct_at
    assert delay - 16 <= lag <= delay + 2, \
        f"reflection {lag} samples after direct, want about {delay}"
    left = float(np.sum(refl[:, 0] ** 2))
    right = float(np.sum(refl[:, 1] ** 2))
    assert left > right * 1.5, f"reflection not from the left: L={left} R={right}"
    level = 20 * np.log10(np.max(np.abs(refl)) / np.max(np.abs(plain)))
    return (f"arrives {refl_at - direct_at} samples after the direct sound "
            f"(2 m path = {delay}); L/R energy {left / right:.1f}; "
            f"{level:.1f} dB relative to the direct sound")


def _switch_jumps(transition: float) -> tuple[float, float]:
    """A steady tone in the concrete room, switched to the wood room after
    60 blocks.  Returns (largest sample jump near the switch, steady)."""
    from audiospace import BufferSource
    from audiospace.acoustics import space_preset
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    t = np.arange(SR * 3) / SR
    sig = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=2)
    engine.set_space(space_preset("concrete_room"))
    engine.add_source(BufferSource(sig, loop=True), position=(0.0, 2.0, 0.0),
                      reverb_send=0.3)
    out = []
    for i in range(160):
        if i == 60:
            engine.set_space(space_preset("wood_room"),
                             transition_s=transition)
        out.append(engine.render())
    jump = np.abs(np.diff(np.concatenate(out)[:, 0]))
    return (float(jump[60 * BLOCK - 1024:60 * BLOCK + 30000].max()),
            float(jump[40 * BLOCK:50 * BLOCK].max()))


@test("space: switching spaces with a transition does not click")
def t_space_switch_smooth() -> str:
    """Without a transition the reflections and the reverb input change
    abruptly, which shows as a sample jump ~7x the steady one.  With the
    0.3 s transition the largest jump stays at the steady level."""
    instant, steady = _switch_jumps(0.0)
    smooth, steady2 = _switch_jumps(0.3)
    assert instant > 3 * steady, (instant, steady)
    assert smooth <= 1.2 * steady2, (f"transition still clicks: jump "
                                     f"{smooth:.4f} vs steady {steady2:.4f}")
    return (f"largest sample jump: instant switch {instant:.4f}, "
            f"0.3 s transition {smooth:.4f}, steady playback {steady2:.4f}")


@test("space: the room you left rings out; release_s cuts it short")
def t_space_tail() -> str:
    from audiospace import BufferSource
    from audiospace.acoustics import space_preset
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    noise = (np.random.default_rng(6).standard_normal(SR) * 0.2
             ).astype(np.float32)

    def levels(release):
        engine = SpatialEngine(hrir, block=BLOCK, max_sources=2)
        engine.set_space(space_preset("concrete_room"))
        h = engine.add_source(BufferSource(noise, loop=True),
                              position=(0.0, 2.0, 0.0), reverb_send=0.3)
        for _ in range(60):
            engine.render()
        engine.remove_source(h)
        engine.set_space(space_preset("open_road"), release_s=release)
        rms = []
        for _ in range(400):
            y = engine.render()
            rms.append(float(np.sqrt(np.mean(y ** 2))))
        return np.array(rms), engine

    def db_at(rms, sec):
        return 20 * np.log10(rms[int(sec * SR / BLOCK)] + 1e-12)

    natural, eng_n = levels(None)
    cut, eng_c = levels(0.2)
    assert db_at(natural, 0.5) > -45, "the room's tail vanished at once"
    assert db_at(natural, 0.5) > db_at(natural, 1.5) + 10, "tail not decaying"
    assert db_at(cut, 0.5) < -120, "release_s did not cut the tail"
    assert not eng_c._tails, "the cut tail was not dropped"
    return (f"natural tail: {db_at(natural, 0.5):.0f} dB at 0.5 s, "
            f"{db_at(natural, 1.5):.0f} dB at 1.5 s; with release_s=0.2: "
            f"{db_at(cut, 0.5):.0f} dB at 0.5 s and dropped")


@test("space: set_spaces follows the listener (room inside a street, then open)")
def t_space_map() -> str:
    from audiospace import Space
    room = Space.room((5.0, 4.0, 2.7), floor="wood_16mm",
                      walls="wooden_lining", ceiling="plasterboard",
                      name="room")
    street = Space.street(width=12.0, length=80.0, name="street")
    road = Space.open_ground(name="road")
    engine = make_engine()
    engine.set_spaces([road, street, room])
    path = [((0.0, 0.0, 0.0), "room"),       # inside the room (and street)
            ((5.0, 0.0, 0.0), "street"),     # out of the room, in the street
            ((5.0, 1.0, 0.0), "street"),
            ((1.0, 0.5, 0.0), "room"),       # back into the smaller room
            ((0.0, 60.0, 0.0), "road")]      # past the end of the street
    seen = []
    for pos, want in path:
        engine.set_listener(pos)
        engine.render()
        seen.append(engine.space.name)
        assert engine.space.name == want, (pos, engine.space.name, want)
    # standing still on the boundary must not flip back and forth
    engine.set_listener((5.0, 0.0, 0.0))
    names = set()
    for _ in range(20):
        engine.render()
        names.add(engine.space.name)
    assert names == {"street"}, names
    engine.set_space(None)
    engine.set_listener((0.0, 0.0, 0.0))
    engine.render()
    assert engine.space is None, "set_space did not clear the registered map"
    return " -> ".join(seen) + "; steady on the boundary; set_space clears it"


def _has_pyroomacoustics() -> bool:
    import importlib.util
    return importlib.util.find_spec("pyroomacoustics") is not None


@test("precise: tail matches the fast reverb's energy, fades in, saves and loads")
def t_precise_tail() -> str:
    if not _has_pyroomacoustics():
        return "pyroomacoustics が無いため省略"
    import tempfile

    from audiospace import Space
    from audiospace.acoustics import space_preset
    from audiospace.precise import (SpaceIr, _fast_tail_energy,
                                    render_space_ir, space_key)
    space = space_preset("carpet_room")
    made = render_space_ir(space, SR)
    energy = float(np.sum(made.tail.astype(np.float64) ** 2))
    target = _fast_tail_energy(space, SR)
    assert abs(energy / target - 1.0) < 0.02, (energy, target)
    # 光線追跡は毎回ランダムで、先頭の値は尖りの 2.6e-4〜5.5e-4 倍あたりを
    # 揺れ、たまに 1e-3 を超えた（判定が不安定だった）。立ち上げが無ければ
    # 先頭は尖りの 1 割前後になるので、1e-2 でも取り違えない
    assert np.max(np.abs(made.tail[:, :20])) < 1e-2 * np.max(np.abs(made.tail)), \
        "the tail does not fade in"
    with tempfile.TemporaryDirectory() as tmp:
        path = made.save(Path(tmp) / "x.npz")
        back = SpaceIr.load(path)
        assert np.array_equal(back.tail, made.tail) and back.key == made.key
    moved = Space(tuple(v + 5.0 for v in space.origin), space.size,
                  space.materials, space.name)
    assert space_key(moved) == space_key(space), "placement changed the key"
    other = Space(space.origin, space.size,
                  dict(space.materials, floor="wood_16mm"), space.name)
    assert space_key(other) != space_key(space), "material ignored by the key"
    assert render_space_ir(space_preset("open_road"), SR) is None
    # 開いた面のある空間は、計算の尾が途切れるので精密版を作らない
    assert render_space_ir(space_preset("street"), SR) is None
    end = made.tail[:, -10:]
    assert np.max(np.abs(end)) < 1e-3 * np.max(np.abs(made.tail)), \
        "the tail must fade out at its end"
    return (f"carpet room: tail {made.tail.shape[1] / SR:.2f} s, energy ratio "
            f"{energy / target:.3f} to the fast reverb; RT60 1 kHz "
            f"{made.rt60_bands[3]:.2f} s (Eyring {space.rt60_bands()[3]:.2f} s)")


@test("precise: the closed preset rooms are baked and used at once")
def t_precise_baked() -> str:
    """The shipped tails (assets/space_ir) must match the current presets:
    if a preset's materials change, its key changes and the bake is stale."""
    import audiospace.precise as precise
    from audiospace import ConvolutionReverb, space_labels
    from audiospace.acoustics import space_preset
    precise._MEMORY.clear()
    closed, open_ = [], []
    for name, _label in space_labels():
        space = space_preset(name)
        if space.open_fraction > 0.0 or space.late_level() <= 0.0:
            open_.append(name)
            continue
        found = precise.find_space_ir(space, SR)
        assert found is not None, f"{name}: no baked tail (re-run the bake)"
        engine = make_engine()
        engine.set_space(space, late="precise")
        assert isinstance(engine.reverb, ConvolutionReverb), name
        closed.append(name)
    engine = make_engine()
    engine.set_space(space_preset("street"), late="precise")
    assert engine.late_status.startswith("開いた面のある空間"), \
        engine.late_status
    return (f"baked and used: {', '.join(closed)}; "
            f"fast for open spaces: {', '.join(open_)}")


@test("precise: ConvolutionReverb.clone renders like a fresh copy")
def t_conv_clone() -> str:
    from audiospace import ConvolutionReverb
    rng = np.random.default_rng(23)
    ir = (rng.standard_normal((2, 4000)) * 0.05).astype(np.float32)
    base = ConvolutionReverb(ir, block=BLOCK, normalise=False)
    x = (rng.standard_normal(BLOCK * 12) * 0.2).astype(np.float32)
    for i in range(5):                           # dirty the original's state
        base.process(x[i * BLOCK:(i + 1) * BLOCK])
    twin = base.clone()
    fresh = ConvolutionReverb(ir, block=BLOCK, normalise=False)
    a = np.concatenate([twin.process(x[i * BLOCK:(i + 1) * BLOCK])
                        for i in range(12)])
    b = np.concatenate([fresh.process(x[i * BLOCK:(i + 1) * BLOCK])
                        for i in range(12)])
    assert np.array_equal(a, b), "a clone must start empty and match a new one"
    assert twin._conv._H is base._conv._H, "the kernel should be shared"
    return "clone starts empty, matches a fresh reverb exactly, shares the kernel"


@test("precise: engine uses a cached tail at once, or falls back to fast")
def t_precise_engine() -> str:
    if not _has_pyroomacoustics():
        return "pyroomacoustics が無いため省略"
    import tempfile

    import audiospace.precise as precise
    from audiospace import ConvolutionReverb
    from audiospace.acoustics import space_preset
    from audiospace.reverb import BandFdnReverb
    space = space_preset("carpet_room")
    with tempfile.TemporaryDirectory() as tmp:
        made = precise.render_space_ir(space, SR)
        made.save(Path(tmp) / f"{made.key}_{SR}.npz")
        precise._MEMORY.clear()
        engine = make_engine()
        engine.space_ir_cache = tmp
        engine.set_space(space, late="precise")
        assert isinstance(engine.reverb, ConvolutionReverb), engine.reverb
        assert engine.late_status == "精密版の残響", engine.late_status
        cached = engine.late_status
    # 計算できない環境では速い版に落ちる
    precise._MEMORY.clear()
    saved = precise.can_render_in_background
    saved_dir = precise.SPACE_IR_DIR
    try:
        precise.can_render_in_background = lambda: False
        precise.SPACE_IR_DIR = Path(tempfile.gettempdir()) / "no_such_dir_x"
        engine = make_engine()
        engine.set_space(space, late="precise")
        assert isinstance(engine.reverb, BandFdnReverb), engine.reverb
        assert engine.late_status.startswith("精密版は使えません"), \
            engine.late_status
    finally:
        precise.can_render_in_background = saved
        precise.SPACE_IR_DIR = saved_dir
    return f"cached: {cached}; without a way to compute: {engine.late_status}"


@test("precise: background computation does not stall audio, then swaps in")
def t_precise_background() -> str:
    """pyroomacoustics holds the GIL; in a thread it stalled rendering for
    up to 12.8 s.  The engine computes in a separate process instead."""
    if not _has_pyroomacoustics():
        return "pyroomacoustics が無いため省略"
    import tempfile
    import time

    import audiospace.precise as precise
    from audiospace import BufferSource, ConvolutionReverb
    from audiospace.acoustics import space_preset
    space = space_preset("carpet_room")
    precise._MEMORY.clear()
    saved_dir = precise.SPACE_IR_DIR
    with tempfile.TemporaryDirectory() as tmp:
        try:
            precise.SPACE_IR_DIR = Path(tmp) / "none"   # 焼き込みを見せない
            engine = make_engine()
            engine.space_ir_cache = tmp
            engine.add_source(BufferSource(tone(300.0, 1.0), loop=True),
                              position=(0.0, 2.0, 0.0), reverb_send=0.3)
            engine.set_space(space, late="precise")
            pending = engine.late_status
            assert pending.startswith("精密版を計算中"), pending
            worst = 0.0
            deadline = time.perf_counter() + 90.0
            while not isinstance(engine.reverb, ConvolutionReverb):
                t0 = time.perf_counter()
                engine.render()
                worst = max(worst, (time.perf_counter() - t0) * 1000.0)
                time.sleep(0.005)
                assert time.perf_counter() < deadline, \
                    f"no precise reverb after 90 s: {engine.late_status}"
        finally:
            precise.SPACE_IR_DIR = saved_dir
    assert worst < 100.0, f"rendering stalled {worst:.0f} ms during the job"
    assert engine.late_status == "精密版の残響", engine.late_status
    return (f"{pending} -> {engine.late_status}; slowest block while "
            f"computing {worst:.1f} ms")


@test("space: removing the space leaves the engine as if it was never set")
def t_space_removed() -> str:
    from audiospace import BufferSource
    from audiospace.acoustics import space_preset
    noise = (np.random.default_rng(8).standard_normal(SR) * 0.1
             ).astype(np.float32)

    def run(touch: bool) -> np.ndarray:
        engine = make_engine()
        if touch:
            engine.set_space(space_preset("concrete_room"))
            engine.set_space(None)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(2.0, 3.0, 0.0), reverb_send=0.0)
        return engine.render_seconds(0.2)

    assert np.array_equal(run(False), run(True))
    return "set_space(space) then set_space(None): output identical"


# -- engine ----------------------------------------------------------------
def make_engine(**kwargs) -> SpatialEngine:
    hrir = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    return SpatialEngine(hrir, block=BLOCK, max_sources=8, **kwargs)


@test("engine: a source on the left is louder in the left ear")
def t_engine_pan() -> str:
    engine = make_engine()
    engine.add_source(BufferSource(tone(500.0, 1.0), loop=True),
                      position=(-5.0, 0.0, 0.0), reverb_send=0.0)  # west
    out = engine.render_seconds(0.3)
    energy_l = float(np.sum(out[:, 0] ** 2))
    energy_r = float(np.sum(out[:, 1] ** 2))
    assert energy_l > energy_r * 1.5, \
        f"expected left-dominant, got L={energy_l:.4g} R={energy_r:.4g}"
    return (f"left energy={energy_l:.4g}, right energy={energy_r:.4g} "
            f"({energy_l / max(energy_r, 1e-12):.2f}x)")


@test("engine: mirroring the source mirrors the ears")
def t_engine_symmetry() -> str:
    def energies(x: float) -> tuple[float, float]:
        engine = make_engine()
        engine.add_source(BufferSource(tone(500.0, 1.0), loop=True),
                          position=(x, 0.0, 0.0), reverb_send=0.0)
        out = engine.render_seconds(0.3)
        return float(np.sum(out[:, 0] ** 2)), float(np.sum(out[:, 1] ** 2))

    west_l, west_r = energies(-5.0)
    east_l, east_r = energies(5.0)
    ratio_w = west_l / max(west_r, 1e-12)
    ratio_e = east_r / max(east_l, 1e-12)
    assert abs(ratio_w - ratio_e) / max(ratio_w, ratio_e) < 0.05, \
        f"asymmetric panning: west ratio {ratio_w:.3f}, east {ratio_e:.3f}"
    return (f"west L/R={ratio_w:.3f}, east R/L={ratio_e:.3f} "
            "(should match)")


@test("engine: distance reduces level")
def t_engine_distance() -> str:
    def level(dist: float) -> float:
        engine = make_engine()
        engine.add_source(BufferSource(tone(400.0, 1.0), loop=True),
                          position=(0.0, dist, 0.0), reverb_send=0.0)
        out = engine.render_seconds(0.2)
        return float(np.sqrt(np.mean(out ** 2)))

    near, mid, far = level(1.0), level(5.0), level(20.0)
    assert near > mid > far, f"{near:.5f} {mid:.5f} {far:.5f}"
    return (f"rms at 1m={near:.5f}, 5m={mid:.5f}, 20m={far:.5f} "
            f"(1m/20m = {near / max(far, 1e-12):.1f}x)")


def _emphasis_render(el: float, emphasis: float, pitch: float = 0.0,
                     seconds: float = 0.5) -> np.ndarray:
    """仰角 el・距離 4m の正面の音を、上下の強調 emphasis で鳴らす（無音）。"""
    engine = make_engine(front_back_depth=0.0, elevation_emphasis=emphasis)
    engine.set_listener(pitch_deg=pitch)
    noise = (np.random.default_rng(21).standard_normal(SR) * 0.2
             ).astype(np.float32)
    a = math.radians(el)
    engine.add_source(BufferSource(noise, loop=True),
                      position=(0.0, 4.0 * math.cos(a), 4.0 * math.sin(a)),
                      reverb_send=0.0)
    return engine.render_seconds(seconds)


def _band_db(out: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(out[4800:, 0])) ** 2
    freqs = np.fft.rfftfreq(len(out) - 4800, 1.0 / SR)
    return float(10 * np.log10(spec[(freqs >= lo) & (freqs < hi)].mean()))


@test("engine: 上下の強調は上の音ほど高域を明るく、下ほど暗くし、増減の半分を全体で打ち消す")
def t_elevation_emphasis() -> str:
    """強さ 1 で、3kHz より上を真上 +12dB・下45度 -6dB。全体の音量を増減の
    半分だけ逆に動かすので、低域は真上で約 -6dB、下45度で約 +3dB。"""
    rows = []
    for el, high_expect, low_expect in ((90.0, 6.0, -6.0), (45.0, 3.0, -3.0),
                                        (-45.0, -3.0, 3.0)):
        on, off = _emphasis_render(el, 1.0), _emphasis_render(el, 0.0)
        high = _band_db(on, 8000, 14000) - _band_db(off, 8000, 14000)
        low = _band_db(on, 100, 600) - _band_db(off, 100, 600)
        assert abs(high - high_expect) < 0.8, (el, high, high_expect)
        assert abs(low - low_expect) < 0.8, (el, low, low_expect)
        rows.append(f"{el:+.0f}度 高域 {high:+.1f}dB・低域 {low:+.1f}dB")
    # 水平は変わらない。強さ 0（既定）も変わらない
    flat_on, flat_off = _emphasis_render(0.0, 1.0), _emphasis_render(0.0, 0.0)
    assert np.array_equal(flat_on, flat_off), "水平の音が変わりました"
    default = make_engine(front_back_depth=0.0)
    assert default.elevation_emphasis == 0.0
    # 段は 1dB 刻み、0〜1 に丸める
    engine = make_engine(elevation_emphasis=1.7)
    assert engine.effective_elevation_emphasis == 1.0
    steps = make_engine(elevation_emphasis=0.35)._emphasis_steps(
        np.array([90.0, 45.0, 10.0, 0.0, -45.0]))
    assert steps.tolist() == [4, 2, 0, 0, -2], steps.tolist()
    return "、".join(rows) + "。水平と強さ0は出力がそのまま"


@test("engine: 上下の強調は顔から見た仰角で決まる（上を向くと水平の音が暗くなる）")
def t_elevation_emphasis_pitch() -> str:
    """HRTF と同じく聞き手の頭から見た方向で決める。上を45度向くと、
    水平の音は顔から見て下45度になる。"""
    off = _emphasis_render(0.0, 0.0, pitch=45.0)
    on = _emphasis_render(0.0, 1.0, pitch=45.0)
    high = _band_db(on, 8000, 14000) - _band_db(off, 8000, 14000)
    assert abs(high - (-3.0)) < 0.8, high
    return f"上を45度向いたときの水平の音: 高域 {high:+.1f}dB（下45度と同じ）"


@test("engine: 上下に動く音でも、上下の強調の段の変わり目に段差が出ない")
def t_elevation_emphasis_moving() -> str:
    """段はブロックごとに変わる。変わり目（ブロックの境）で波形が飛べば
    「プツッ」と聞こえる。境での1サンプルの変化を、ブロックの中と比べる。"""
    engine = make_engine(front_back_depth=0.0, elevation_emphasis=1.0)
    noise = (np.random.default_rng(22).standard_normal(SR * 2) * 0.2
             ).astype(np.float32)
    handle = engine.add_source(BufferSource(noise, loop=True),
                               position=(0.0, 4.0, 0.0), reverb_send=0.0)
    blocks = []
    n = 160                                   # 約1.7秒で下45度から真上へ
    for i in range(n):
        el = math.radians(-45.0 + 135.0 * i / (n - 1))
        engine.set_source_position(
            handle, (0.0, 4.0 * math.cos(el), 4.0 * math.sin(el)))
        blocks.append(engine.render())
    out = np.concatenate(blocks)[BLOCK * 4:, 0]
    jumps = np.abs(np.diff(out))
    edge = np.arange(BLOCK - 1, len(jumps), BLOCK)
    inner = np.setdiff1d(np.arange(len(jumps)), edge)
    ratio = float(jumps[edge].mean() / jumps[inner].mean())
    worst = float(jumps[edge].max() / np.percentile(jumps[inner], 99.9))
    assert ratio < 1.3, f"境の変化が大きい: 平均の比 {ratio:.2f}"
    assert worst < 1.5, f"境に飛び: 最大の比 {worst:.2f}"
    return f"境と中の変化の比: 平均 {ratio:.2f}、最大/99.9% {worst:.2f}"


@test("player: set_emphasis で前後と上下の強調を変えられる")
def t_player_emphasis() -> str:
    from audiospace.player import SpatialPlayer
    player = SpatialPlayer(make_engine())
    player.set_emphasis(front_back=0.5, elevation=0.35)
    assert player.engine.front_back_depth == 0.5
    assert player.engine.elevation_emphasis == 0.35
    player.set_emphasis(elevation=0.0)
    assert player.engine.front_back_depth == 0.5, "None は変えない"
    assert player.engine.elevation_emphasis == 0.0
    return "前後 0.5・上下 0.35 を設定、None の項目は変わらない"


@test("engine: sources behind are duller than sources ahead")
def t_engine_front_back() -> str:
    def spectrum(y: float) -> float:
        engine = make_engine(front_back_depth=0.35)
        noise = (np.random.default_rng(3)
                 .standard_normal(SR).astype(np.float32) * 0.2)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(0.0, y, 0.0), reverb_send=0.0)
        out = engine.render_seconds(0.3)
        mono = out.mean(axis=1)
        return band_energy(mono, 3000, 16000) / max(
            band_energy(mono, 100, 800), 1e-12)

    ahead = spectrum(5.0)
    behind = spectrum(-5.0)
    assert behind < ahead * 0.7, \
        (f"behind must be clearly duller, not marginally: "
         f"ahead={ahead:.4g} behind={behind:.4g}")
    return (f"high/low band ratio ahead={ahead:.4g}, "
            f"behind={behind:.4g} ({behind / ahead:.2f}x — "
            "a timbre difference, not just level)")


@test("engine: front/back damping is consistent at every lateral angle")
def t_engine_orbit() -> str:
    """Compare mirrored front/back pairs at the same lateral offset.

    Measuring a full orbit would be misleading: summing the two ears to mono
    comb-filters at non-zero ITD, and the head shadow removes highs from the
    contralateral ear at the sides.  Both effects are identical for a
    front/back mirror pair, so comparing (theta) against (180 - theta)
    isolates the front/back cue itself.  Band energies are summed per
    channel rather than mixed, so no comb filtering is introduced.
    """
    def brightness(theta_deg: float, behind: bool) -> float:
        engine = make_engine(front_back_depth=0.35)
        noise = (np.random.default_rng(9)
                 .standard_normal(SR).astype(np.float32) * 0.2)
        rad = np.radians(theta_deg)
        north = 5.0 * float(np.cos(rad)) * (-1.0 if behind else 1.0)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(5.0 * float(np.sin(rad)), north, 0.0),
                          reverb_send=0.0)
        out = engine.render_seconds(0.25)
        hi = sum(band_energy(out[:, c], 3000, 16000) for c in (0, 1))
        lo = sum(band_energy(out[:, c], 100, 800) for c in (0, 1))
        return hi / max(lo, 1e-12)

    lines = []
    for theta in (0.0, 30.0, 60.0):
        front = brightness(theta, behind=False)
        back = brightness(theta, behind=True)
        assert back < front * 0.85, (
            f"at {theta:g} deg lateral the rear source is not duller: "
            f"front={front:.4g} back={back:.4g}")
        lines.append(f"lateral {theta:g} deg: front={front:.3g} "
                     f"back={back:.3g} ({back / front:.2f}x)")
    return "\n".join(lines)


@test("engine: non-spatialised UI sound is centred and listener-independent")
def t_engine_ui() -> str:
    engine = make_engine()
    engine.add_source(BufferSource(tone(880.0, 1.0), loop=True),
                      spatialise=True, position=(9.0, 0.0, 0.0),
                      reverb_send=0.0)
    spat = engine.render_seconds(0.2)
    engine2 = make_engine()
    engine2.add_source(BufferSource(tone(880.0, 1.0), loop=True),
                       spatialise=False, reverb_send=0.0)
    engine2.set_listener((30.0, 40.0, 0.0), yaw_deg=137.0)
    flat = engine2.render_seconds(0.2)
    imbalance = abs(np.sum(flat[:, 0] ** 2) - np.sum(flat[:, 1] ** 2))
    scale = max(np.sum(flat[:, 0] ** 2), 1e-12)
    assert imbalance / scale < 1e-6, "UI sound must be perfectly centred"
    return (f"UI channel imbalance={imbalance / scale:.2e} (centred); "
            f"spatialised source stays panned "
            f"(L/R={np.sum(spat[:, 0]**2) / max(np.sum(spat[:, 1]**2), 1e-12):.2f})")


@test("engine: slot exhaustion raises instead of silently dropping")
def t_engine_slots() -> str:
    engine = make_engine()
    for _ in range(8):
        engine.add_source(BufferSource(tone(300.0, 0.1)))
    try:
        engine.add_source(BufferSource(tone(300.0, 0.1)))
    except RuntimeError as exc:
        return f"raised as expected: {exc}"
    raise AssertionError("expected RuntimeError when all slots are used")


@test("engine: finished sources are reaped and slots reused")
def t_engine_reap() -> str:
    engine = make_engine()
    short = engine.add_source(BufferSource(tone(300.0, BLOCK / SR)))
    engine.render()
    engine.render()
    reaped = engine.reap_finished()
    assert short in reaped, f"expected {short} to be reaped, got {reaped}"
    assert engine.active_sources == 0
    engine.add_source(BufferSource(tone(300.0, 0.5)))
    return f"reaped {reaped}, slot reused, active={engine.active_sources}"


@test("engine: output stays within headroom (no clipping with many sources)")
def t_engine_headroom() -> str:
    engine = make_engine()
    rng = np.random.default_rng(11)
    for i in range(8):
        engine.add_source(
            BufferSource(tone(200.0 + 90 * i, 1.0), loop=True),
            position=(float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)), 0.0),
            reverb_send=0.0)
    out = engine.render_seconds(0.4)
    peak = float(np.max(np.abs(out)))
    assert peak < 1.0, f"clipped: peak={peak:.3f}"
    return f"peak with 8 simultaneous sources = {peak:.3f} (must stay < 1.0)"


@test("engine: PushSource lets the caller feed blocks (low-level API)")
def t_push_source() -> str:
    engine = make_engine()
    push = PushSource(capacity=BLOCK * 4)
    engine.add_source(push, position=(0.0, 2.0, 0.0), reverb_send=0.0)
    push.write(tone(440.0, BLOCK * 2 / SR))
    out = engine.render_seconds(BLOCK * 2 / SR)
    assert float(np.max(np.abs(out))) > 1e-4, "pushed audio did not render"
    # the HRIR tail keeps ringing for `taps` samples, so compare rms rather
    # than expecting instant silence
    active = float(np.sqrt(np.mean(out ** 2)))
    engine.render_seconds(BLOCK * 2 / SR)     # let the HRIR tail ring out
    silent = engine.render_seconds(BLOCK * 2 / SR)
    quiet = float(np.sqrt(np.mean(silent ** 2)))
    assert quiet < active * 0.01, \
        f"buffer should drain: active rms={active:.5f}, after={quiet:.5f}"
    return (f"pushed {BLOCK * 2} samples, rms while playing={active:.5f}, "
            f"after the buffer drains={quiet:.5f} "
            f"({active / max(quiet, 1e-12):.0f}x quieter)")


@test("engine: listener yaw changes which ear hears the source")
def t_engine_yaw() -> str:
    def energies(yaw: float) -> tuple[float, float]:
        engine = make_engine()
        engine.add_source(BufferSource(tone(500.0, 1.0), loop=True),
                          position=(0.0, 5.0, 0.0), reverb_send=0.0)  # north
        engine.set_listener((0.0, 0.0, 0.0), yaw_deg=yaw)
        out = engine.render_seconds(0.3)
        return float(np.sum(out[:, 0] ** 2)), float(np.sum(out[:, 1] ** 2))

    fl, fr = energies(0.0)      # facing the source
    tl, tr = energies(90.0)     # turned left -> source now on the right
    assert abs(fl - fr) / max(fl, fr) < 0.05, "facing the source must be even"
    assert tr > tl * 1.5, f"after turning left, right ear should lead: {tl} {tr}"
    return (f"facing source L={fl:.4g} R={fr:.4g} (balanced); "
            f"after yaw=90 L={tl:.4g} R={tr:.4g}")


# -- reverb ----------------------------------------------------------------
@test("reverb: FDN decays and roughly honours the requested RT60")
def t_fdn() -> str:
    rev = VectorFdnReverb(SR, rt60=1.2, damping=0.2)
    impulse = np.zeros(BLOCK, dtype=np.float32)
    impulse[0] = 1.0
    tail = [rev.process(impulse)]
    for _ in range(int(3.0 * SR / BLOCK)):
        tail.append(rev.process(np.zeros(BLOCK, dtype=np.float32)))
    ir = np.concatenate(tail).T
    measured = rt60_of(ir, SR)
    early = float(np.sqrt(np.mean(ir[:, :SR // 4] ** 2)))
    late = float(np.sqrt(np.mean(ir[:, -SR // 4:] ** 2)))
    assert late < early, f"reverb is not decaying: {early:.5f} -> {late:.5f}"
    assert 0.3 < measured < 4.0, f"implausible RT60 {measured:.2f}s"
    return (f"requested RT60=1.2s, measured={measured:.2f}s; "
            f"rms first 0.25s={early:.5f} -> last 0.25s={late:.5f}")


@test("reverb: vectorised FDN matches the per-sample reference")
def t_fdn_equivalence() -> str:
    from audiospace.reverb import FdnReverb

    # 40 blocks: every delay line (1104..4848 samples) wraps several times,
    # which is where the block-wise reads and writes split in two
    n_blocks = 40
    rng = np.random.default_rng(5)
    x = rng.standard_normal(BLOCK * n_blocks).astype(np.float32) * 0.1
    ref = FdnReverb(SR, rt60=1.0)
    fast = VectorFdnReverb(SR, rt60=1.0)
    a = np.concatenate([ref.process(x[i * BLOCK:(i + 1) * BLOCK])
                        for i in range(n_blocks)])
    b = np.concatenate([fast.process(x[i * BLOCK:(i + 1) * BLOCK])
                        for i in range(n_blocks)])
    err = float(np.max(np.abs(a - b)))
    assert err < 1e-5, f"vectorised FDN diverged: max error {err:.3e}"
    return (f"{n_blocks} blocks: max difference between reference and "
            f"vectorised = {err:.3e}")


@test("reverb: IR convolution (ring delay line) matches direct convolution")
def t_conv_reverb_direct() -> str:
    """The delay line is a ring since 2026-10; run past several wraps."""
    from audiospace import ConvolutionReverb

    rng = np.random.default_rng(17)
    ir = (rng.standard_normal((2, 3000)) * 0.05).astype(np.float32)
    rev = ConvolutionReverb(ir, block=BLOCK, normalise=False)
    n_blocks = 20                              # 3000 taps = 6 partitions
    x = (rng.standard_normal(BLOCK * n_blocks) * 0.2).astype(np.float32)
    got = np.concatenate([rev.process(x[i * BLOCK:(i + 1) * BLOCK])
                          for i in range(n_blocks)])
    worst = 0.0
    for ear in range(2):
        ref = np.convolve(x.astype(np.float64), ir[ear])[:len(got)]
        worst = max(worst, float(np.max(np.abs(got[:, ear] - ref))))
    assert worst < 1e-4, f"error {worst:.3e} vs np.convolve"
    rev.reset()
    again = np.concatenate([rev.process(x[i * BLOCK:(i + 1) * BLOCK])
                            for i in range(n_blocks)])
    assert np.array_equal(again, got), "reset() did not restore the start state"
    return (f"6 partitions x {n_blocks} blocks: max error vs np.convolve "
            f"{worst:.1e}; reset() repeats exactly")


@test("reverb: send level controls the wet amount")
def t_reverb_send() -> str:
    def wetness(send: float) -> float:
        engine = make_engine()
        engine.set_reverb(VectorFdnReverb(SR, rt60=1.5))
        engine.add_source(BufferSource(tone(440.0, BLOCK / SR)),
                          position=(0.0, 2.0, 0.0), reverb_send=send)
        engine.render_seconds(0.1)            # dry hit plus its HRIR tail
        tail = engine.render_seconds(0.4)     # only reverb can remain here
        return float(np.sqrt(np.mean(tail ** 2)))

    dry, wet = wetness(0.0), wetness(0.8)
    assert dry < 1e-6, f"with send=0 the tail must be silent, got {dry:.3e}"
    assert wet > 1e-4, f"with send=0.8 there should be an audible tail, "\
                       f"got {wet:.3e}"
    return (f"tail rms with send=0.0 is {dry:.3e} (silent), "
            f"with send=0.8 is {wet:.5f}")


# -- measured HRTF (skipped when the dataset has not been fetched) ---------
VIKING = (Path(__file__).resolve().parents[1] / "assets" / "hrtf" / "viking"
          / "subj_A.sofa")


def _skip_no_sofa() -> str | None:
    if not VIKING.exists():
        return (f"SOFA データ未取得のためスキップ "
                f"(tools\\fetch_viking_hrtf.py で取得)")
    return None


@test("sofa: loads a real measured file with the expected geometry")
def t_sofa_load() -> str:
    if (msg := _skip_no_sofa()):
        return msg
    hrir = HrirSet.from_sofa(VIKING)
    assert hrir.n_directions > 1000, hrir.n_directions
    assert hrir.data.shape[1] == 2
    assert hrir.sample_rate in (44100, 48000, 96000), hrir.sample_rate
    assert hrir.elevation.min() < -10 and hrir.elevation.max() > 45, \
        "expected a full-sphere set with real elevation coverage"
    return (f"{hrir.name}: {hrir.n_directions} directions, {hrir.taps} taps, "
            f"{hrir.sample_rate} Hz, elevation "
            f"{hrir.elevation.min():.0f}..{hrir.elevation.max():.0f} deg")


@test("sofa: interaural cues match the SOFA azimuth convention")
def t_sofa_convention() -> str:
    if (msg := _skip_no_sofa()):
        return msg
    hrir = HrirSet.from_sofa(VIKING)
    ir = hrir.data[hrir.index_for(90.0, 0.0)]   # 90 deg = listener's left

    def onset(ch: np.ndarray) -> int:
        peak = float(np.max(np.abs(ch)))
        hits = np.flatnonzero(np.abs(ch) > 0.15 * peak)
        assert hits.size, "no onset found"
        return int(hits[0])

    itd_us = (onset(ir[1]) - onset(ir[0])) / hrir.sample_rate * 1e6
    hi_l = band_energy(ir[0], 4000, 16000, hrir.sample_rate)
    hi_r = band_energy(ir[1], 4000, 16000, hrir.sample_rate)
    assert 100 < itd_us < 900, (
        f"ITD {itd_us:.0f} us is outside the human range; receiver order or "
        "azimuth direction may be reversed for this dataset")
    assert hi_l > hi_r, "the near ear must not be the darker one"
    return (f"at 90 deg: ITD {itd_us:+.0f} us (left leads), "
            f"high-band L/R {hi_l / max(hi_r, 1e-12):.0f}x")


@test("sofa: measured data carries real front/back information")
def t_sofa_front_back() -> str:
    if (msg := _skip_no_sofa()):
        return msg
    hrir = HrirSet.from_sofa(VIKING)
    syn = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    lines = []
    for front, back in ((30.0, 150.0), (0.0, 180.0)):
        a = hrir.data[hrir.index_for(front, 0.0)]
        b = hrir.data[hrir.index_for(back, 0.0)]
        rel = (float(np.sqrt(np.mean((a - b) ** 2)))
               / max(float(np.sqrt(np.mean(a ** 2))), 1e-12))
        sa = syn.data[syn.index_for(front)]
        sb = syn.data[syn.index_for(back)]
        srel = (float(np.sqrt(np.mean((sa - sb) ** 2)))
                / max(float(np.sqrt(np.mean(sa ** 2))), 1e-12))
        assert rel > 0.3, (
            f"measured set should differ front/back, got {rel:.3f}")
        assert srel < 1e-6, (
            "the synthetic set is expected to be exactly front/back "
            f"symmetric; got {srel:.3e} -- the model changed")
        lines.append(f"{front:g} vs {back:g} deg: measured differs "
                     f"{rel * 100:.0f}%, synthetic {srel * 100:.1f}%")
    return "\n".join(lines)


@test("sofa: a measured set drives the engine and pans correctly")
def t_sofa_engine() -> str:
    if (msg := _skip_no_sofa()):
        return msg
    hrir = HrirSet.from_sofa(VIKING, sample_rate=SR)
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=4)
    engine.add_source(BufferSource(tone(500.0, 1.0), loop=True),
                      position=(-5.0, 0.0, 0.0), reverb_send=0.0)  # west/left
    out = engine.render_seconds(0.3)
    el = float(np.sum(out[:, 0] ** 2))
    er = float(np.sum(out[:, 1] ** 2))
    assert el > er * 1.5, f"expected left-dominant, L={el:.4g} R={er:.4g}"
    assert float(np.max(np.abs(out))) < 1.0, "clipped"
    return (f"{hrir.n_directions}-direction measured set: "
            f"L/R energy ratio {el / max(er, 1e-12):.2f}, "
            f"peak {float(np.max(np.abs(out))):.3f}")


@test("sofa: 真上の音は指定した方位によらず同じ音になる")
def t_sofa_zenith_same() -> str:
    """前後の処理（音量とローパス）が仰角を見ていなかったので、方位180度で
    指定した真上の音だけ小さくこもっていた。上下の聴き取り（Shift+F3）は
    真上を「方位を変えても同じ音」として扱っている。"""
    if (msg := _skip_no_sofa()):
        return msg
    hrir = HrirSet.from_sofa(VIKING, sample_rate=SR)
    noise = np.random.default_rng(9).standard_normal(SR).astype(np.float32)
    noise *= 0.2

    def render(az: float, el: float) -> np.ndarray:
        engine = SpatialEngine(hrir, block=BLOCK, max_sources=2,
                               front_back_depth=0.35)
        a, e = math.radians(az), math.radians(el)
        pos = (-4.0 * math.cos(e) * math.sin(a), 4.0 * math.cos(e) * math.cos(a),
               4.0 * math.sin(e))
        engine.add_source(BufferSource(noise, loop=True), position=pos,
                          reverb_send=0.0)
        return engine.render_seconds(0.3)

    front, back = render(0.0, 90.0), render(180.0, 90.0)
    diff = float(np.max(np.abs(front - back)))
    assert diff < 1e-6, f"真上の音が方位で変わる: 最大差 {diff:.3g}"
    # 水平の真後ろには従来どおり効く
    flat_front, flat_back = render(0.0, 0.0), render(180.0, 0.0)
    ratio = float(np.sum(flat_back ** 2) / np.sum(flat_front ** 2))
    assert ratio < 0.8, f"水平の真後ろの手がかりが消えた: {ratio:.3f}"
    return (f"真上: 方位0度と180度の最大差 {diff:.1e}。"
            f"水平の真後ろ/正面のエネルギー比 {ratio:.3f}")


@test("sofa: 水平面のみに絞ると表が小さくなり定位は保たれる")
def t_sofa_horizontal() -> str:
    """2Dゲームでは仰角を使わない。全天球のまま使うとスペクトル表が
    12MB になりキャッシュに乗らず、実測で64音源 4.91ms→2.07ms の差が出る。"""
    if (msg := _skip_no_sofa()):
        return msg
    full = HrirSet.from_sofa(VIKING, sample_rate=SR)
    flat = full.horizontal()
    assert flat.n_directions < full.n_directions / 3, (
        f"間引けていません {full.n_directions} -> {flat.n_directions}")
    assert float(np.max(np.abs(flat.elevation))) <= 5.0, \
        f"仰角が残っています: {flat.elevation.min()}..{flat.elevation.max()}"
    assert flat.table_bytes < 2 * 1024 * 1024, flat.table_bytes
    # 水平方向の定位は保たれているか
    ir = flat.data[flat.index_for(90.0, 0.0)]
    hi_l = band_energy(ir[0], 4000, 16000, flat.sample_rate)
    hi_r = band_energy(ir[1], 4000, 16000, flat.sample_rate)
    assert hi_l > hi_r * 10, f"左右の差が失われました {hi_l / max(hi_r, 1e-12):.1f}"
    return (f"{full.n_directions}方向 "
            f"({full.table_bytes / 1024 / 1024:.1f}MB) → "
            f"{flat.n_directions}方向 "
            f"({flat.table_bytes / 1024 / 1024:.1f}MB)、"
            f"90度での高域L/R比 {hi_l / max(hi_r, 1e-12):.0f}倍を維持")


@test("sofa: 方位の間引きが効く")
def t_sofa_decimate() -> str:
    if (msg := _skip_no_sofa()):
        return msg
    full = HrirSet.from_sofa(VIKING, sample_rate=SR)
    coarse = full.horizontal().decimated(15.0)
    assert coarse.n_directions <= 24 + 2, coarse.n_directions
    # 間引いても最寄り探索の誤差は刻み幅の半分以内
    worst = 0.0
    for az in range(0, 360, 7):
        idx = coarse.index_for(float(az), 0.0)
        err = abs(((float(coarse.azimuth[idx]) - az + 180) % 360) - 180)
        worst = max(worst, err)
    assert worst <= 8.0, f"最寄り探索の誤差が大きすぎます {worst:.1f}度"
    return (f"水平面216方向 → 15度刻み{coarse.n_directions}方向、"
            f"最寄り探索の最大誤差{worst:.1f}度")


@test("既定アセット: numpy だけで読めて定位が保たれる")
def t_default_asset() -> str:
    """配布物には SOFA ではなくこれを同梱する。sofar と netCDF が不要で、
    2754KB が205KBになる。"""
    from audiospace import DEFAULT_PINNA, default_hrir, default_is_measured
    from audiospace.defaults import DEFAULT_ASSET

    if not default_is_measured():
        return (f"既定アセット未作成のためスキップ "
                f"(tools\\bake_hrtf_asset.py で作成)")
    hrir = default_hrir(SR)
    assert hrir.n_directions > 100, hrir.n_directions
    assert hrir.table_bytes < 2 * 1024 * 1024, hrir.table_bytes
    size_kb = DEFAULT_ASSET.stat().st_size / 1024
    assert size_kb < 600, f"配布物には大きすぎます: {size_kb:.0f} KB"

    # 左右と前後が生きているか
    ir = hrir.data[hrir.index_for(90.0, 0.0)]
    hi_l = band_energy(ir[0], 4000, 16000, hrir.sample_rate)
    hi_r = band_energy(ir[1], 4000, 16000, hrir.sample_rate)
    assert hi_l > hi_r * 10, f"左右差が失われています {hi_l / max(hi_r, 1e-12):.1f}"
    a = hrir.data[hrir.index_for(30.0, 0.0)]
    b = hrir.data[hrir.index_for(150.0, 0.0)]
    rel = (float(np.sqrt(np.mean((a - b) ** 2)))
           / max(float(np.sqrt(np.mean(a ** 2))), 1e-12))
    assert rel > 0.3, f"前後差が失われています {rel:.3f}"

    meta = HrirSet.read_npz_meta(DEFAULT_ASSET)
    assert meta.get("license") == "CC BY 4.0", meta
    assert meta.get("attribution"), "表示義務の文面がありません"
    return (f"耳介{DEFAULT_PINNA}、{hrir.n_directions}方向、{size_kb:.0f} KB、"
            f"高域L/R比{hi_l / max(hi_r, 1e-12):.0f}倍、前後差{rel * 100:.0f}%\n"
            f"ライセンス: {meta.get('license')}")


@test("既定アセット: レート変換しても壊れない")
def t_default_resample() -> str:
    from audiospace import default_hrir, default_is_measured

    if not default_is_measured():
        return "既定アセット未作成のためスキップ"
    a = default_hrir(48000)
    b = default_hrir(44100)
    assert b.sample_rate == 44100 and b.n_directions == a.n_directions
    engine = SpatialEngine(b, block=BLOCK, max_sources=2)
    engine.add_source(BufferSource(tone(500.0, 1.0, 44100), loop=True),
                      position=(-5.0, 0.0, 0.0), reverb_send=0.0)
    out = engine.render_seconds(0.2)
    el = float(np.sum(out[:, 0] ** 2))
    er = float(np.sum(out[:, 1] ** 2))
    assert el > er * 1.5, f"44100Hz で定位が壊れました L/R={el / max(er, 1e-12):.2f}"
    return (f"48000Hz {a.taps}タップ → 44100Hz {b.taps}タップ、"
            f"L/R比 {el / max(er, 1e-12):.2f}")


# -- regression tests (2026-08-26 review fixes) ----------------------------
def wait_until(cond, timeout: float = 3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@test("sources: multichannel files downmix instead of collapsing")
def t_buffer_multichannel() -> str:
    rng = np.random.default_rng(0)
    six = rng.standard_normal((1000, 6)).astype(np.float32)
    b = BufferSource(six)
    assert b.frames == 1000, \
        f"(1000, 6ch) must give 1000 frames, got {b.frames}"
    x = rng.standard_normal(500).astype(np.float32)
    cancel = BufferSource(np.stack([x, -x], axis=1))
    assert float(np.max(np.abs(cancel.read(500)))) < 1e-6, \
        "stereo downmix should average the channels"
    try:
        BufferSource(np.zeros((2, 100), dtype=np.float32))
    except ValueError as exc:
        return (f"6ch->1000frames OK, stereo average OK, "
                f"channels-first rejected: {exc}")
    raise AssertionError("channels-first (2, 100) must be rejected, "
                         "not folded into 2 samples")


@test("sources: PushSource overflow keeps sample order (drop-oldest)")
def t_push_overflow_order() -> str:
    ps = PushSource(capacity=10)
    ps.write(np.arange(1, 9, dtype=np.float32))      # 1..8
    ps.write(np.arange(9, 14, dtype=np.float32))     # 9..13, 3 overflow
    got = ps.read(10).tolist()
    want = [float(v) for v in range(4, 14)]
    assert got == want, f"order scrambled: {got}"

    ps2 = PushSource(capacity=8)
    ps2.write(np.arange(100, dtype=np.float32))
    assert ps2.available == 8, ps2.available
    got2 = ps2.read(8).tolist()
    assert got2 == [float(v) for v in range(92, 100)], got2

    ps3 = PushSource(capacity=8)
    ps3.write(np.ones(4, dtype=np.float32))
    ps3.close()
    assert not ps3.finished, "unread samples remain"
    ps3.read(4)
    assert ps3.finished
    return (f"lapped read returns {got[0]:.0f}..{got[-1]:.0f} in order; "
            "oversized write keeps the newest; finished only when drained")


@test("geometry: no distance model ever amplifies (headphone safety)")
def t_distance_no_amplify() -> str:
    worst = 0.0
    for model in ("inverse", "inverse_clamped", "linear", "none"):
        for d in (0.0, 1e-4, 0.001, 0.5, 1.0, 5.0, 100.0):
            g = distance_gain(d, model)
            worst = max(worst, g)
            assert 0.0 <= g <= 1.0, f"{model} at {d}m gives gain {g}"
    assert distance_gain(0.001, "inverse") == 1.0
    # inside the singularity (negative denominator) must also cap, not flip
    assert distance_gain(0.4, "inverse", rolloff=2.0) == 1.0
    return f"max gain over all models/distances = {worst:.3f} (was 1000x)"


@test("engine: front_back_depth=1.0 muffles but never mutes")
def t_depth_full_not_mute() -> str:
    noise = (np.random.default_rng(3)
             .standard_normal(SR).astype(np.float32) * 0.2)

    def peak(depth: float) -> float:
        engine = make_engine(front_back_depth=depth)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(0.0, -5.0, 0.0), reverb_send=0.0)
        return float(np.max(np.abs(engine.render_seconds(0.3))))

    full = peak(1.0)
    normal = peak(0.35)
    assert full > 1e-3, f"depth=1.0 fully mutes the rear source: {full}"
    assert full < normal, "depth=1.0 should still be duller than 0.35"
    return f"peak behind at depth=1.0: {full:.4f} (was 0.0), at 0.35: {normal:.4f}"


@test("engine: limiter keeps a loud mix under full scale, transparent when quiet")
def t_limiter() -> str:
    engine = make_engine()
    for i in range(8):
        engine.add_source(BufferSource(tone(200.0 + 61 * i, 1.0), loop=True),
                          position=(0.0, 1.0, 0.0), gain=4.0,
                          reverb_send=0.0)
    out = engine.render_seconds(0.3)
    peak = float(np.max(np.abs(out)))
    assert peak <= 1.0, f"limiter failed: peak={peak:.3f}"
    assert engine.limited_blocks > 0, "limiter never engaged on a loud mix"

    def quiet(limiter: bool) -> np.ndarray:
        e = make_engine(limiter=limiter)
        e.add_source(BufferSource(tone(500.0, 1.0), loop=True),
                     position=(0.0, 3.0, 0.0), gain=0.3, reverb_send=0.0)
        return e.render_seconds(0.2)

    assert np.array_equal(quiet(True), quiet(False)), \
        "limiter must be bit-transparent below the threshold"
    return (f"8 sources at gain 4.0: peak {peak:.3f}, "
            f"{engine.limited_blocks} limited blocks; quiet mix untouched")


@test("engine: fade-out stop is click-free and releases the slot")
def t_fade_out() -> str:
    steady = np.full(SR, 0.4, dtype=np.float32)   # DC: cuts click hardest

    def jump(fade_ms: float) -> float:
        engine = make_engine()
        h = engine.add_source(BufferSource(steady, loop=True),
                              position=(0.0, 2.0, 0.0), reverb_send=0.0)
        pre = engine.render_seconds(0.15)
        engine.remove_source(h, fade_ms=fade_ms)
        post = engine.render_seconds(0.15)
        seam = np.concatenate([pre, post])[:, 0]
        lo = pre.shape[0] - BLOCK
        return float(np.max(np.abs(np.diff(seam[lo:]))))

    hard = jump(0.0)
    faded = jump(40.0)
    assert faded < hard * 0.2, \
        f"fade did not smooth the cut: hard={hard:.5f} faded={faded:.5f}"

    engine = make_engine()
    h = engine.add_source(BufferSource(steady, loop=True),
                          position=(0.0, 2.0, 0.0), reverb_send=0.0)
    engine.render_seconds(0.1)
    engine.remove_source(h, fade_ms=40.0)
    out = engine.render_seconds(0.25)
    assert engine.active_sources == 0, "faded slot was not released"
    assert float(np.max(np.abs(out[-BLOCK:]))) < 1e-5, "fade left residue"
    return (f"max sample jump at the cut: hard={hard:.5f}, "
            f"fade 40ms={faded:.5f} ({hard / max(faded, 1e-9):.0f}x smoother); "
            "slot auto-released")


@test("engine: flat (UI) sources ramp in instead of stepping")
def t_flat_ramp() -> str:
    engine = make_engine()
    engine.add_source(BufferSource(np.full(SR, 0.5, dtype=np.float32),
                                   loop=True),
                      spatialise=False, reverb_send=0.0)
    first = engine.render()
    second = engine.render()
    steady = 0.5 * engine.master_gain
    assert abs(first[0, 0]) < 0.02, \
        f"UI sound starts with a step: first sample {first[0, 0]:.3f}"
    assert abs(second[0, 0] - steady) < 0.02, \
        f"did not reach steady level: {second[0, 0]:.3f} vs {steady:.3f}"
    return (f"first sample {first[0, 0]:.4f} -> steady {second[0, 0]:.4f} "
            f"(target {steady:.4f})")


@test("engine: pause freezes everything and resume continues in place")
def t_pause_resume() -> str:
    engine = make_engine()
    src = BufferSource(tone(400.0, 1.0), loop=True)
    engine.add_source(src, position=(0.0, 3.0, 0.0), reverb_send=0.0)
    engine.render_seconds(0.1)
    pos = src.position
    engine.set_paused(True)
    out = engine.render_seconds(0.1)
    assert float(np.max(np.abs(out))) == 0.0, "paused output must be silent"
    assert src.position == pos, "paused source advanced"
    engine.set_paused(False)
    out2 = engine.render_seconds(0.1)
    assert float(np.max(np.abs(out2))) > 1e-4, "did not resume"
    return f"position held at {pos} while paused, audio resumed after"


@test("engine: one source can be paused and resumed without a click")
def t_source_pause() -> str:
    engine = make_engine()
    src = BufferSource(tone(400.0, 1.0), loop=True)
    h = engine.add_source(src, position=(0.0, 3.0, 0.0), reverb_send=0.0)
    engine.render_seconds(0.1)
    engine.set_source_active(h, False)
    pos = src.position
    out = engine.render_seconds(0.1)
    assert float(np.max(np.abs(out[-BLOCK:]))) < 1e-6, "pause not silent"
    assert src.position == pos, "paused source advanced"
    engine.set_source_active(h, True)
    back = engine.render()
    assert abs(back[0, 0]) < 0.05, \
        f"resume should ramp in, first sample {back[0, 0]:.3f}"
    assert float(np.max(np.abs(engine.render()))) > 1e-4
    return f"held at sample {pos}, resumed with ramp-in"


@test("engine: gain can be set in decibels")
def t_gain_db() -> str:
    def rms_at(db: float) -> float:
        engine = make_engine()
        h = engine.add_source(BufferSource(tone(400.0, 1.0), loop=True),
                              position=(0.0, 2.0, 0.0), reverb_send=0.0)
        engine.set_source_gain_db(h, db)
        engine.render_seconds(0.1)              # let the gain ramp settle
        out = engine.render_seconds(0.2)
        return float(np.sqrt(np.mean(out ** 2)))

    ratio = rms_at(-20.0) / max(rms_at(0.0), 1e-12)
    assert abs(ratio - 0.1) < 0.02, f"-20 dB should be 0.1x, got {ratio:.3f}"
    return f"-20 dB / 0 dB rms ratio = {ratio:.3f} (expected 0.100)"


@test("player: play() reaps finished one-shots (no slot exhaustion)")
def t_play_reaps() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=8, reverb=None)
    try:
        shot = np.zeros(BLOCK, dtype=np.float32)
        shot[0] = 0.1
        for _ in range(30):
            player.play(shot, position=(1.0, 1.0, 0.0))
            player.engine.render()      # play the shot out
    finally:
        player.stop()
    return "30 one-shot plays through 8 slots without RuntimeError"


@test("player: decoded files are cached across plays")
def t_sound_cache() -> str:
    try:
        import soundfile as sf
    except ImportError:
        return "soundfile 未導入のためスキップ"
    import audiospace.sources as sources_mod

    wav = Path(tempfile.mkdtemp()) / "cache_probe.wav"
    sf.write(str(wav), tone(440.0, 0.2, 44100), 44100)

    calls = {"n": 0}
    real = sources_mod._decode

    def counting(path):
        calls["n"] += 1
        return real(path)

    player = SpatialPlayer.open(block=BLOCK, max_sources=8, reverb=None)
    sources_mod._decode = counting
    try:
        player.play(wav, position=(1.0, 0.0, 0.0))
        player.play(wav, position=(-1.0, 0.0, 0.0))
        player.play(str(wav), position=(0.0, 1.0, 0.0))
    finally:
        sources_mod._decode = real
        player.stop()
    assert calls["n"] == 1, f"decoded {calls['n']} times for 3 plays"
    assert len(player._sound_cache) == 1
    return "3 plays (Path and str), decoded once, one cache entry"


@test("player: use_room_ir loads baked IRs without simulating")
def t_room_ir_baked() -> str:
    room = load_room_ir("hall", 48000)
    if room is None:
        return "焼き込みIRが無いためスキップ (tools/build_reverb_presets.py)"
    assert room.sample_rate == 48000
    r44 = load_room_ir("hall", 44100)
    assert r44 is not None and r44.sample_rate == 44100, \
        "rate fallback (resample) failed"

    import audiospace.player as player_mod
    real = player_mod.render_room_ir

    def boom(*a, **k):
        raise AssertionError("simulated live despite a baked IR")

    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    player_mod.render_room_ir = boom
    try:
        player.use_room_ir("hall")
        from audiospace import ConvolutionReverb
        assert isinstance(player.engine.reverb, ConvolutionReverb)
        try:
            player.use_room_ir("no_such_room")
            raise AssertionError("unknown preset must raise ValueError")
        except ValueError:
            pass
    finally:
        player_mod.render_room_ir = real
        player.stop()
    return (f"hall baked at 48000 loads directly, resamples to 44100 "
            f"({r44.ir.shape[1]} taps); unknown name raises ValueError")


@test("player: stop() then start() works again (silent mode)")
def t_restart_silent() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    try:
        assert wait_until(lambda: player.callbacks > 0), "did not start"
        player.stop()
        seen = player.callbacks
        player.start()
        assert wait_until(lambda: player.callbacks > seen), \
            "no callbacks after restart -- the stop event was never cleared"
    finally:
        player.stop()
    return f"restarted; callbacks {seen} -> {player.callbacks}"


@test("player: a crashing source cannot kill the audio loop")
def t_callback_error_guard() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    try:
        def boom(n: int) -> np.ndarray:
            raise RuntimeError("wired to fail")

        player.play(CallbackSource(boom))
        assert wait_until(lambda: player.render_errors > 0), \
            "error was never recorded"
        seen = player.callbacks
        assert wait_until(lambda: player.callbacks > seen + 3), \
            "the loop died after the exception"
        assert "RuntimeError" in player.last_error
        assert "レンダリングエラー" in player.status()
    finally:
        player.stop()
    return (f"loop survived {player.render_errors} errors, "
            f"status reports: {player.last_error}")


# -- 遮蔽のこもり・音量の滑らかな変化・定位させない音 -----------------------------------------------
def brightness(x: np.ndarray) -> float:
    """高域/低域のエネルギー比。こもり具合の指標。"""
    if x.ndim == 2:
        return sum(band_energy(x[:, c], 3000, 16000) for c in (0, 1)) / max(
            sum(band_energy(x[:, c], 100, 800) for c in (0, 1)), 1e-12)
    return band_energy(x, 3000, 16000) / max(band_energy(x, 100, 800), 1e-12)


@test("遮蔽: set_source_lowpass が高域を連続的に落とす")
def t_occlusion_lowpass() -> str:
    """壁越しの音は「小さい」ではなく「こもる」。音量だけの
    合図は距離と区別がつかないので、ローパスでなければならない。"""
    noise = (np.random.default_rng(4)
             .standard_normal(SR).astype(np.float32) * 0.2)

    def measure(amount: float) -> tuple[float, float]:
        engine = make_engine(front_back_depth=0.0)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(0.0, 5.0, 0.0), reverb_send=0.0,
                          lowpass=amount)
        out = engine.render_seconds(0.3)
        return brightness(out.mean(axis=1)), float(np.max(np.abs(out)))

    clear, clear_peak = measure(0.0)
    half, _ = measure(0.5)
    full, full_peak = measure(1.0)
    assert half < clear * 0.6, f"0.5 で暗くなっていません {clear} -> {half}"
    assert full < half, f"1.0 が 0.5 より暗くありません {half} -> {full}"
    assert full_peak > 1e-4, "最大でも無音にしてはいけません"
    assert clear_peak > 0, "遮蔽なしが鳴っていません"
    return (f"高低比 0.0={clear:.2f} 0.5={half:.2f} 1.0={full:.3f}、"
            f"最大遮蔽でもピーク{full_peak:.4f}で残る")


@test("遮蔽: 前後の暗さと合成され、互いを打ち消さない")
def t_occlusion_combines() -> str:
    noise = (np.random.default_rng(9)
             .standard_normal(SR).astype(np.float32) * 0.2)

    def measure(y: float, amount: float) -> float:
        engine = make_engine(front_back_depth=0.35)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(0.0, y, 0.0), reverb_send=0.0,
                          lowpass=amount)
        return brightness(engine.render_seconds(0.3))

    front = measure(5.0, 0.0)
    back = measure(-5.0, 0.0)
    front_occ = measure(5.0, 0.6)
    back_occ = measure(-5.0, 0.6)
    assert back < front, "前後の暗さが失われました"
    assert front_occ < front, "前方で遮蔽が効いていません"
    assert back_occ < back, "後方で遮蔽が上乗せされていません"
    assert back_occ < front_occ, "遮蔽が前後の差を消しています"
    return (f"前{front:.2f} 後{back:.2f} / 遮蔽0.6で 前{front_occ:.2f} "
            f"後{back_occ:.2f}（両方の手がかりが残る）")


@test("遮蔽: 変化はランプされ、状態は音源ごとに独立")
def t_occlusion_ramp() -> str:
    steady = np.full(SR, 0.4, dtype=np.float32)
    engine = make_engine(front_back_depth=0.0)
    h = engine.add_source(BufferSource(steady, loop=True),
                          position=(0.0, 3.0, 0.0), reverb_send=0.0)
    quiet = engine.add_source(BufferSource(steady, loop=True),
                              position=(0.0, 3.0, 0.0), reverb_send=0.0)
    engine.set_source_active(quiet, False)
    engine.render_seconds(0.1)
    engine.set_source_lowpass(h, 1.0, ramp_ms=120.0)
    slot = engine.get_source(h)
    engine.render()
    after_one = slot._occl_now
    assert 0.0 < after_one < 1.0, f"1ブロックで飛んでいます: {after_one}"
    engine.render_seconds(0.2)
    assert abs(slot._occl_now - 1.0) < 1e-6, slot._occl_now
    engine.set_source_lowpass(h, 0.0, ramp_ms=0.0)
    assert slot._occl_now == 0.0, "ramp_ms=0 が即時になっていません"
    assert engine.get_source(quiet).lowpass == 0.0, \
        "ほかの音源に漏れています"
    return (f"120msランプ: 1ブロック後{after_one:.3f} → 収束1.000、"
            "ramp_ms=0 は即時、他音源は無影響")


@test("ゲインランプ: 音声スレッド側で進み、指定時間で終わる")
def t_gain_ramp() -> str:
    """UI スレッドがモーダルループで止まってもフェードは
    約束どおり終わる必要がある（ゲームティック駆動だと凍る）。"""
    engine = make_engine()
    h = engine.add_source(BufferSource(np.full(SR, 0.5, dtype=np.float32),
                                       loop=True),
                          position=(0.0, 2.0, 0.0), reverb_send=0.0)
    engine.render_seconds(0.05)
    engine.set_source_gain(h, 0.0, ramp_ms=200.0)
    levels = [float(np.sqrt(np.mean(engine.render() ** 2)))
              for _ in range(int(0.30 * SR / BLOCK))]
    slot = engine.get_source(h)
    mid = levels[len(levels) // 3]
    assert levels[0] > mid > levels[-1], f"単調に下がっていません: {levels[:5]}"
    assert levels[-1] < levels[0] * 0.01, f"下がりきっていません {levels[-1]}"
    assert slot.gain == 0.0 and slot._gain_target is None
    # 半分の時点でまだ鳴っていること（1ブロックで飛んでいない）
    assert levels[3] > levels[0] * 0.3, "ランプが速すぎます"
    engine.set_source_gain(h, 0.8)          # ramp なしは即時のまま
    assert slot.gain == 0.8
    return (f"200msフェード: {levels[0]:.4f} → {mid:.4f} → {levels[-1]:.5f}、"
            "ramp_ms 省略時は即時")


@test("ゲインランプ: クロスフェードで両方が同時に動く")
def t_gain_crossfade() -> str:
    engine = make_engine()
    a = engine.add_source(BufferSource(tone(400.0, 1.0), loop=True),
                          spatialise=False, reverb_send=0.0)
    b = engine.add_source(BufferSource(tone(700.0, 1.0), loop=True),
                          spatialise=False, gain=0.0, reverb_send=0.0)
    engine.render_seconds(0.05)
    engine.set_source_gain(a, 0.0, ramp_ms=150.0)
    engine.set_source_gain(b, 1.0, ramp_ms=150.0)
    engine.render_seconds(0.2)
    ga = engine.get_source(a).gain
    gb = engine.get_source(b).gain
    assert ga == 0.0 and gb == 1.0, (ga, gb)
    return f"150msクロスフェード後: A={ga:.2f} B={gb:.2f}"


@test("flat ステレオ: 1音源のまま2ch出力される")
def t_flat_stereo() -> str:
    """BGM はすべてステレオなので、L/R を2音源に割る回避策は
    常設になってしまう。1スロットで鳴れば枠も畳み込みも半分。"""
    rng = np.random.default_rng(11)
    left = rng.standard_normal(SR).astype(np.float32) * 0.3
    right = rng.standard_normal(SR).astype(np.float32) * 0.3
    stereo = np.stack([left, right], axis=1)

    engine = make_engine()
    engine.add_source(StereoBufferSource(stereo, loop=True),
                      spatialise=False, reverb_send=0.0)
    out = engine.render_seconds(0.2)
    corr = abs(float(np.corrcoef(out[:, 0], out[:, 1])[0, 1]))
    assert engine.active_sources == 1, "1スロットで鳴っていません"
    assert corr < 0.2, f"左右が混ざっています（相関 {corr:.3f}）"
    assert float(np.max(np.abs(out[:, 0]))) > 1e-3
    assert float(np.max(np.abs(out[:, 1]))) > 1e-3

    # モノラルを渡しても両chに複製されて鳴る
    mono_src = StereoBufferSource(left)
    assert mono_src.read(64).shape == (64, 2), "常に (n, 2) を返すこと"

    # 空間化には使えない。重要なのは拒否の「場所」: add_source で失敗
    # しなければならない。以前は render 内で毎ブロック例外になり、誤った
    # 登録1回で正常な音源まで全部無音になった。
    engine2 = make_engine()
    ok = engine2.add_source(BufferSource(np.full(SR, 0.4, np.float32),
                                         loop=True),
                            position=(0.0, 3.0, 0.0), reverb_send=0.0)
    engine2.render()
    try:
        engine2.add_source(StereoBufferSource(stereo, loop=True),
                           spatialise=True)
        raise AssertionError("空間化でステレオ音源が登録できてしまいました")
    except ValueError as exc:
        message = str(exc)
    # 拒否のあとも、いた音源は普通に鳴り続けること
    after = float(np.max(np.abs(engine2.render_seconds(0.1))))
    assert after > 1e-3, f"拒否の巻き添えで無音になりました: {after}"
    assert engine2.get_source(ok) is not None
    return (f"1スロット・L/R相関{corr:.3f}で維持、"
            f"登録時に拒否({message[:32]}...)、他の音源は継続 "
            f"(ピーク{after:.3f})")


@test("再生レート: ピッチとテンポが一緒に動く")
def t_playback_rate() -> str:
    """足音のピッチ揺らしのために毎回 ndarray を作って渡すと、
    デコードキャッシュが効かず移植側に二重のキャッシュができる。"""
    base = tone(440.0, 0.5)
    lines = []
    for rate in (1.25, 0.8):
        out = resampled_by(base, rate)
        spec = np.abs(np.fft.rfft(out))
        freq = np.fft.rfftfreq(len(out), 1 / SR)
        peak = float(freq[int(np.argmax(spec))])
        want_len = int(round(len(base) / rate))
        assert abs(len(out) - want_len) <= 2, (len(out), want_len)
        assert abs(peak - 440.0 * rate) < 12.0, (peak, 440.0 * rate)
        lines.append(f"rate={rate}: {len(out)}サンプル, ピーク{peak:.0f}Hz")
    assert resampled_by(base, 1.0) is base, "1.0 は素通しであるべきです"
    try:
        resampled_by(base, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("rate=0 が通ってしまいました")
    # ステレオでもフレーム軸で変換される
    st = np.stack([base, base], axis=1)
    out2 = resampled_by(st, 1.25)
    assert out2.ndim == 2 and out2.shape[1] == 2, out2.shape
    return "、".join(lines) + f"、ステレオ {st.shape} -> {out2.shape}"


@test("player: レートつき再生がキャッシュされ、ステレオを保つ")
def t_play_rate_and_stereo() -> str:
    try:
        import soundfile as sf
    except ImportError:
        return "soundfile 未導入のためスキップ"
    import audiospace.sources as sources_mod

    folder = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(5)
    st_path = folder / "bed.wav"
    sf.write(str(st_path), np.stack([rng.standard_normal(4410) * 0.2,
                                     rng.standard_normal(4410) * 0.2],
                                    axis=1).astype(np.float32), 44100)
    fake_path = folder / "fakestereo.wav"
    same = (rng.standard_normal(4410) * 0.2).astype(np.float32)
    sf.write(str(fake_path), np.stack([same, same], axis=1), 44100)

    assert is_stereo_file(st_path), "本物のステレオを判定できていません"
    assert not is_stereo_file(fake_path), \
        "同一chのステレオ容器を本物と誤判定しています"

    calls = {"n": 0}
    real = sources_mod._decode

    def counting(path):
        calls["n"] += 1
        return real(path)

    player = SpatialPlayer.open(block=BLOCK, max_sources=8, reverb=None)
    sources_mod._decode = counting
    try:
        h1 = player.play(st_path, spatialise=False)
        player.play(st_path, spatialise=False)          # キャッシュ命中
        decodes_same_rate = calls["n"]
        player.play(st_path, spatialise=False, rate=1.05)
        h_mono = player.play(fake_path, spatialise=False)
        src = player.engine.get_source(h1).source
        assert getattr(src, "channels", 1) == 2, "ステレオで再生されていません"
        mono_src = player.engine.get_source(h_mono).source
        assert getattr(mono_src, "channels", 1) == 1, \
            "偽ステレオに2ch使っています"
    finally:
        sources_mod._decode = real
        player.stop()
    # load_flat は1回のデコードで判定と読み込みを兼ねる。以前は
    # is_stereo_file で全曲をもう一度デコードしていた
    assert decodes_same_rate == 1, f"同一条件で{decodes_same_rate}回デコード"
    return (f"初回1デコードのみ（判定込み）、2回目はキャッシュ、"
            "本物ステレオは2ch・偽ステレオは1ch")


@test("デバイス喪失: 検知され status() で読み上げられる")
def t_device_lost() -> str:
    """沈黙は目の見えない利用者が唯一気づけない失敗なので、
    「なぜか音が消えた」を「デバイスが外れました」に変える。"""
    class DeadStream:
        active = False

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    try:
        assert player.check_device() is False, "無音モードで誤検知しました"
        player.silent = False                 # 実機を模す
        player._stream = DeadStream()
        assert player.check_device() is True, "停止したストリームを見逃しました"
        assert player.device_lost is True
        text = player.status()
        assert "デバイス" in text and "reopen" in text, text

        # 応答が途絶えた場合（active は True のまま）も検知する
        class StaleStream(DeadStream):
            active = True

        p2 = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
        p2.silent = False
        p2._stream = StaleStream()
        p2._last_callback = time.perf_counter() - 30.0
        stale = p2.check_device()
        p2.silent = True
        p2._stream = None
        p2.stop()
        assert stale is True, "コールバック途絶を検知できていません"
    finally:
        player.silent = True
        player._stream = None
        player.stop()
    return f"停止と途絶の両方を検知、読み上げ文: {text[:42]}..."


# -- 0.3.0 レビュー対応の退行防止 -------------------------------------------
@test("遮蔽: ごく弱い値で高域が持ち上がらない（係数は1以下）")
def t_coeff_no_boost() -> str:
    """バケット-1で係数が1.043になり、遮蔽ランプが
    0付近を通るたびに高域が9%持ち上がっていた。"""
    engine = make_engine()
    assert engine._coeff_of(-1) == 1.0, engine._coeff_of(-1)
    assert engine._coeff_of(0) == 1.0
    assert engine._coeff_of(engine.LP_BUCKETS - 1) == 0.05

    noise = (np.random.default_rng(1)
             .standard_normal(SR).astype(np.float32) * 0.2)

    def ultra(occl: float) -> float:
        e = make_engine(front_back_depth=0.0)
        h = e.add_source(BufferSource(noise, loop=True), spatialise=False,
                         reverb_send=0.0)
        e.set_source_lowpass(h, occl, ramp_ms=0.0)
        out = e.render_seconds(0.3)
        x = out[:, 0]
        spec = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return (float(np.sum(spec[f > 12000] ** 2))
                / max(float(np.sum(spec[(f > 100) & (f < 1000)] ** 2)),
                      1e-12))

    ratio = ultra(0.01) / max(ultra(0.0), 1e-12)
    assert ratio <= 1.02, f"微小な遮蔽で高域が {ratio:.3f} 倍になっています"
    return f"係数の上限1.0を確認、occl=0.01 の高域比 {ratio:.3f}（以前は1.15）"


@test("遮蔽: ステレオ床の右chが左chの状態を引きずらない")
def t_stereo_occlusion_state() -> str:
    """フィルタ状態が音源につき1個しかなく、右chが
    毎ブロック左chの終端値から再出発してブロック境界で跳んでいた。"""
    rng = np.random.default_rng(2)
    lch = rng.standard_normal(SR).astype(np.float32) * 0.3
    rch = rng.standard_normal(SR).astype(np.float32) * 0.3

    engine = make_engine(front_back_depth=0.0)
    h = engine.add_source(StereoBufferSource(np.stack([lch, rch], axis=1),
                                             loop=True),
                          spatialise=False, reverb_send=0.0)
    engine.set_source_lowpass(h, 0.9, ramp_ms=0.0)
    out = engine.render_seconds(0.3)

    ref_engine = make_engine(front_back_depth=0.0)
    h2 = ref_engine.add_source(BufferSource(rch, loop=True),
                               spatialise=False, reverb_send=0.0)
    ref_engine.set_source_lowpass(h2, 0.9, ramp_ms=0.0)
    ref = ref_engine.render_seconds(0.3)

    diff = float(np.max(np.abs(out[:, 1] - ref[:, 0])))
    assert diff < 1e-4, \
        f"右chが単独処理と一致しません（最大差 {diff:.6f}、以前は 0.13）"
    seams = [abs(float(out[b * BLOCK, 1] - out[b * BLOCK - 1, 1]))
             for b in range(1, out.shape[0] // BLOCK)]
    inner = float(np.median(np.abs(np.diff(out[:, 1])))) + 1e-12
    assert max(seams) < inner * 6, \
        f"境界の跳び {max(seams):.5f} が内部変化 {inner:.5f} に対して大きい"
    return (f"右chと単独処理の最大差 {diff:.2e}、"
            f"境界跳び/内部変化 = {max(seams) / inner:.1f}倍（以前は10倍超）")


@test("ランプ: gain と lowpass の ramp_ms がともに「目標到達時間」")
def t_ramp_semantics() -> str:
    """lowpass 側だけ「0→1全域の走破時間」で、
    同じ引数名なのに意味が違った。到達時間に統一。"""
    engine = make_engine()
    h = engine.add_source(BufferSource(np.full(SR, 0.4, np.float32),
                                       loop=True),
                          position=(0.0, 2.0, 0.0), reverb_send=0.0)
    engine.set_source_gain(h, 0.5, ramp_ms=100.0)
    engine.set_source_lowpass(h, 0.4, ramp_ms=100.0)
    for _ in range(5):                       # 約53ms
        engine.render()
    slot = engine.get_source(h)
    mid_gain = slot.gain
    mid_occl = slot._occl_now
    assert 0.6 < mid_gain < 0.85, f"中間のゲインが変です: {mid_gain}"
    assert 0.1 < mid_occl < 0.3, \
        f"lowpass が距離比例になっていません: {mid_occl}"
    for _ in range(7):                       # 合計 ~128ms > 100ms
        engine.render()
    assert slot.gain == 0.5, slot.gain
    assert abs(slot._occl_now - 0.4) < 1e-9, slot._occl_now
    return (f"53ms時点 gain={mid_gain:.3f} / lowpass={mid_occl:.3f}、"
            "約100msで 0.5 / 0.4 に到達（意味を統一）")


@test("再生レート: ループ点が 1/rate で補正される")
def t_rate_loop_points() -> str:
    """データ長は 1/rate になるのにループ点が
    そのままで、意図と違う場所でループしていた。"""
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    try:
        arr = np.zeros(SR, dtype=np.float32)
        arr[0] = 0.1
        h = player.play(arr, loop=True, loop_start=1000, loop_end=2000,
                        rate=2.0)
        src = player.engine.get_source(h).source
        assert src.loop_start == 500, src.loop_start
        assert src._end == 1000, src._end
        h2 = player.play(arr, loop=True, loop_start=1000, loop_end=2000)
        src2 = player.engine.get_source(h2).source
        assert src2.loop_start == 1000 and src2._end == 2000
    finally:
        player.stop()
    return "rate=2.0 で loop_start 1000→500、loop_end 2000→1000。rate=1 は不変"


@test("再生レート: Source オブジェクトには黙って効かない")
def t_rate_rejects_source() -> str:
    """「変わらなかった」系の失敗は目の見えない
    利用者ほど気づけないので、無視ではなく例外にする。"""
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    try:
        push = PushSource(1024)
        try:
            player.play(push, rate=1.2)
            raise AssertionError("Source + rate が黙って通りました")
        except ValueError as exc:
            message = str(exc)
        player.play(PushSource(1024), rate=1.0)      # 1.0 は従来どおり可
    finally:
        player.stop()
    return f"rate=1.2 + PushSource は ValueError: {message[:44]}..."


@test("キャッシュ: 大きなファイルは保持しない（容量の暴走防止）")
def t_cache_byte_cap() -> str:
    """BGM（数十MB/曲）が件数上限32のLRUに滞留して
    数GBまで育ちうる。1件の上限バイトを超えるものは返すだけで保持しない。"""
    try:
        import soundfile as sf
    except ImportError:
        return "soundfile 未導入のためスキップ"
    import audiospace.sources as sources_mod

    wav = Path(tempfile.mkdtemp()) / "cap_probe.wav"
    sf.write(str(wav), tone(300.0, 0.2, 44100), 44100)

    calls = {"n": 0}
    real = sources_mod._decode

    def counting(path):
        calls["n"] += 1
        return real(path)

    player = SpatialPlayer.open(block=BLOCK, max_sources=8, reverb=None)
    player.sound_cache_max_bytes = 1000          # デコード結果より小さく
    sources_mod._decode = counting
    try:
        player.play(wav)
        player.play(wav)
        assert calls["n"] == 2, f"上限超なのにキャッシュされました: {calls}"
        assert len(player._sound_cache) == 0
        player.sound_cache_max_bytes = 10 * 1024 * 1024
        player.play(wav)
        player.play(wav)
        assert calls["n"] == 3, calls
        assert len(player._sound_cache) == 1
    finally:
        sources_mod._decode = real
        player.stop()
    return ("上限1000バイト時: 2回とも再デコード・保持ゼロ。"
            "上限拡大後: 1回デコードで以後キャッシュ")


@test("reopen: デバイス一覧を更新し、失敗しても状態が正直")
def t_reopen() -> str:
    """PortAudio は列挙を初期化時に
    キャッシュするので、再初期化なしでは挿し直したデバイスが見えない。
    また choose_output が失敗したとき device_lost が立たず、無音のまま
    正常顔になっていた。"""
    import sounddevice as sd

    import audiospace.player as player_mod
    from audiospace.device import DeviceChoice

    class FakeStream:
        active = True

        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    engine = make_engine()
    good = DeviceChoice(index=0, name="Fake USB", host_api="WASAPI",
                        sample_rate=SR, latency_ms=10.0)
    player = SpatialPlayer(engine, good)         # silent=False、start しない

    counts = {"term": 0, "init": 0}
    real_term, real_init = sd._terminate, sd._initialize
    real_choose = player_mod.choose_output
    real_open = player_mod.open_output_stream
    stream = FakeStream()
    try:
        sd._terminate = lambda: counts.__setitem__("term", counts["term"] + 1)
        sd._initialize = lambda: counts.__setitem__("init", counts["init"] + 1)
        player_mod.open_output_stream = lambda *a, **k: stream

        # 成功経路: 一覧を更新してから選び直し、鳴り直す
        player_mod.choose_output = lambda *a, **k: good
        assert player.reopen() is True
        assert counts == {"term": 1, "init": 1}, counts
        assert stream.started and player.device_lost is False

        # レート違い: 復帰できないことを例外で伝え、喪失状態を保つ
        other = DeviceChoice(index=1, name="Speakers", host_api="WASAPI",
                             sample_rate=SR + 3900, latency_ms=10.0)
        player_mod.choose_output = lambda *a, **k: other
        try:
            player.reopen()
            raise AssertionError("レート違いが通ってしまいました")
        except RuntimeError as exc:
            assert "作り直して" in str(exc)
        assert player.device_lost is True

        # デバイス皆無: 例外でも device_lost と last_error が残る
        def none_found(*a, **k):
            raise RuntimeError("no stereo output device found")

        player_mod.choose_output = none_found
        try:
            player.reopen()
            raise AssertionError("デバイス皆無が通ってしまいました")
        except RuntimeError:
            pass
        assert player.device_lost is True
        assert "デバイス" in player.last_error
        assert "デバイス" in player.status()
    finally:
        sd._terminate = real_term
        sd._initialize = real_init
        player_mod.choose_output = real_choose
        player_mod.open_output_stream = real_open
        player._stream = None
        player.silent = True
        player.stop()
    return ("再初期化1回→選択→再開を確認。レート違いと"
            "デバイス皆無の両方で device_lost が残る")


# -- entry point -----------------------------------------------------------
@test("spaces: 設計図と空間を行き来でき、用意した10種類は元と同じに戻る")
def t_space_spec_roundtrip() -> str:
    from audiospace import (space_from_spec, space_labels, space_preset,
                            space_spec_problems, space_to_spec)
    kinds = {}
    for key, _ in space_labels():
        space = space_preset(key)
        spec = space_to_spec(space)
        again = space_from_spec(spec)
        assert again.size == space.size and again.materials == space.materials
        assert np.allclose(again.origin, space.origin), (key, again.origin)
        assert (again.early_order, again.early_gain, again.late_gain) == \
            (space.early_order, space.early_gain, space.late_gain)
        kinds[spec["kind"]] = kinds.get(spec["kind"], 0) + 1
    assert kinds == {"room": 7, "street": 2, "open": 1}, kinds
    # 道は前後と空が開き、何もない場所は地面だけ
    street = space_from_spec({"kind": "street", "size": [10, 60, 15],
                              "materials": {"floor": "rough_concrete",
                                            "west": "brickwork",
                                            "east": "glass_window"},
                              "at": [0.3, 0.5]})
    assert all(street.materials[f] == "open"
               for f in ("south", "north", "ceiling"))
    assert street.materials["east"] == "glass_window"
    ground = space_from_spec({"kind": "open",
                              "materials": {"floor": "carpet_thin"}})
    assert ground.late_level() == 0.0
    # おかしな値は日本語の問題点で断る
    bad = space_spec_problems({"kind": "room", "size": [0.1, 4, 3],
                               "materials": {"floor": "no_such"},
                               "at": [0.0, 0.5], "early_order": 9,
                               "late_gain": 5.0})
    text = "、".join(bad)
    for word in ("幅は", "床の材質が分かりません", "左の壁の材質がありません",
                 "聞き手の位置", "反射の回数", "残響の強さ"):
        assert word in text, (word, text)
    try:
        space_from_spec({"kind": "cave"})
    except ValueError as exc:
        assert "種類" in str(exc)
    else:
        raise AssertionError("知らない種類を受け付けました")
    return (f"部屋{kinds['room']}・道{kinds['street']}・開けた場所{kinds['open']}を"
            f"往復。問題点の例: {bad[0]}")


@test("spaces: 材質91種類に重ならない日本語名があり、分類つきで並ぶ")
def t_material_labels() -> str:
    from audiospace import material_choices, material_label, material_names
    choices = material_choices()
    assert len(choices) == len(material_names()) == 91, len(choices)
    labels = [v for _, v in choices]
    assert len(set(labels)) == 91, "読み上げ名が重なっています"
    assert all(": " in v for v in labels)
    assert all(not any("a" <= c <= "z" for c in v.split(": ", 1)[1])
               for v in labels), "日本語名に英字が残っています"
    assert choices[0][0] == "open"
    assert material_label("wood_16mm") == "木の板（16ミリ、下地の上）"
    assert material_label("unknown_x") == "unknown_x"
    return f"91種類。例: {choices[1][1]} / {choices[-1][1]}"


@test("spaces: 響きの文（reverb_text）は帯域・家具・体積・開けた場所を言い分ける")
def t_reverb_text() -> str:
    from audiospace import furnish_counts, reverb_text, space_preset
    room = space_preset("wood_room")
    text = reverb_text(room)
    for word in ("低い音", "中ほどの音", "高い音", "立方メートル"):
        assert word in text, (word, text)
    assert "置いてあるもの" not in text
    floor = room.size[0] * room.size[1]
    furn = room.with_contents(furnish_counts(floor, "normal"))
    assert "置いてあるものが無ければ" in reverb_text(furn)
    lines = reverb_text(furn, sep="\n").split("\n")
    assert len(lines) == 3, lines
    assert reverb_text(space_preset("open_road")) == \
        "開けた場所なので、広がった残響はありません。"
    return text


@test("reverb: 中域が短く低域が長い部屋（厚いカーテン）でも残響が発散しない")
def t_band_fdn_stable() -> str:
    from scipy.signal import sosfreqz
    from audiospace import Space
    from audiospace.reverb import band_reverb_for
    # 2026-10-20 に発散した部屋。低域 2.8秒、中域 0.33秒。線ごとの1周の
    # 利得が低域で 1.07 になり、響きが増え続けて計算があふれた
    room = Space.room((5.0, 4.0, 2.7), floor="concrete_floor",
                      walls="curtains_velvet", ceiling="rough_concrete")
    rev = band_reverb_for(room, SR, 512)
    _, h = sosfreqz(rev._sos, worN=4096, fs=SR)
    loop = float(np.max(np.abs(h)) * np.max(rev._gain))
    assert loop < 1.0, loop
    x = np.zeros(512, dtype=np.float32)
    x[0] = 1.0
    out = [rev.process(x)]
    x[0] = 0.0
    out += [rev.process(x) for _ in range(int(20 * SR / 512))]
    y = np.concatenate(out)
    assert np.all(np.isfinite(y))
    first = float(np.sqrt(np.mean(y[:SR] ** 2)))
    last = float(np.sqrt(np.mean(y[-SR:] ** 2)))
    assert last < first * 1e-3, (first, last)
    return f"1周の利得の最大 {loop:.3f}。20秒後は最初の1秒の {last / first:.1e} 倍"


@test("engine: 計算が壊れたブロックは無音にして、残響を捨てる")
def t_engine_nonfinite_guard() -> str:
    class Broken:
        def __init__(self):
            self.resets = 0

        def process(self, block):
            return np.full((block.shape[0], 2), np.nan, dtype=np.float32)

        def reset(self):
            self.resets += 1

    engine = make_engine()
    broken = Broken()
    engine.set_reverb(broken)
    engine.add_source(BufferSource(np.ones(SR, dtype=np.float32) * 0.1,
                                   loop=True),
                      position=(0.0, 2.0, 0.0), reverb_send=0.3)
    out = engine.render()
    assert np.all(out == 0.0)
    assert engine.nonfinite_blocks == 1 and broken.resets == 1
    return "NaN のブロックは無音、回数を数え、残響を初期化した"


@test("reverb: 残響の設計図から作れ、用意した部屋は従来と同じ音になる")
def t_reverb_spec() -> str:
    from audiospace import (preset_reverb_spec, reverb_from_spec,
                            reverb_spec_problems)
    from audiospace.player import SpatialPlayer
    noise = (np.random.default_rng(31).standard_normal(SR) * 0.2
             ).astype(np.float32)

    def run(set_reverb) -> np.ndarray:
        engine = make_engine()
        set_reverb(engine)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(0.0, 3.0, 0.0), reverb_send=0.3)
        return engine.render_seconds(0.4)

    by_name = run(lambda e: SpatialPlayer(e).set_reverb("hall"))
    by_spec = run(lambda e: e.set_reverb(
        reverb_from_spec(preset_reverb_spec("hall"), SR)))
    assert np.array_equal(by_name, by_spec), "名前と設計図で音が違います"
    # 大きさ（level）は残響の出力だけを変える
    loud = dict(preset_reverb_spec("hall"), level=2.0)
    a = reverb_from_spec(preset_reverb_spec("hall"), SR)
    b = reverb_from_spec(loud, SR)
    x = np.zeros(BLOCK, dtype=np.float32)
    x[0] = 1.0
    ya, yb = a.process(x), b.process(x)
    for _ in range(8):
        ya = ya + a.process(np.zeros(BLOCK, np.float32))
        yb = yb + b.process(np.zeros(BLOCK, np.float32))
    assert np.allclose(yb, ya * 2.0, atol=1e-7)
    bad = reverb_spec_problems({"rt60": 20.0, "damping": 1.5})
    assert any("残響時間" in p for p in bad) and any("減衰" in p for p in bad), bad
    assert reverb_spec_problems({"name": "x"}) == ["残響時間がありません"]
    return "hall は名前でも設計図でもビット単位で同じ。level 2 で残響だけ2倍"


@test("spaces: 家具や人を置くと残響が短くなり、床の反射ほど弱まる")
def t_space_contents() -> str:
    from audiospace import (furnish_counts, space_from_spec, space_preset,
                            space_to_spec)
    from audiospace.precise import space_key
    room = space_preset("wood_room")
    same = room.with_contents({})
    assert np.array_equal(same.rt60_bands(), room.rt60_bands())
    assert [c.bands for c in same.reflection_chains()] == \
        [c.bands for c in room.reflection_chains()], "空のとき反射が変わった"
    assert space_key(same) == space_key(room), "焼き込みの鍵が変わった"
    floor = room.size[0] * room.size[1]
    normal = furnish_counts(floor, "normal")
    assert normal == {"sofa": 1, "armchair": 1, "table": 1,
                      "wooden_chair": 2, "bookshelf": 1}, normal
    assert furnish_counts(floor * 2, "normal")["wooden_chair"] == 4
    assert furnish_counts(floor, "empty") == {}
    furn = room.with_contents(normal)
    crowd = furn.with_contents(dict(normal, person=10))
    mid = 3                                       # 1kHz
    rts = [r.rt60_bands()[mid] for r in (room, furn, crowd)]
    assert rts[0] > rts[1] > rts[2], rts
    assert room.late_level() > furn.late_level() > crowd.late_level()
    # 反射: 床が最も弱まり、天井はあまり弱まらない
    before = {c.faces: c.bands[mid] for c in room.reflection_chains()}
    after = {c.faces: c.bands[mid] for c in furn.reflection_chains()}
    drop = {f: after[(f,)] / before[(f,)] for f in ("floor", "west",
                                                    "ceiling")}
    assert drop["floor"] < drop["west"] < drop["ceiling"] < 1.0, drop
    # 保存の形でも残る
    assert space_from_spec(space_to_spec(crowd)).contents == crowd.contents
    from audiospace.acoustics import Space
    assert Space.from_dict(crowd.to_dict()).contents == crowd.contents
    assert space_key(furn) != space_key(room)
    try:
        room.with_contents({"dragon": 1})
    except KeyError:
        pass
    else:
        raise AssertionError("知らない物を受け付けました")
    # 精密な残響は家具を扱えないので速い版になる
    engine = make_engine()
    engine.set_space(furn, late="precise")
    assert "家具や人" in engine.late_status, engine.late_status
    return (f"1kHz の残響時間: 空 {rts[0]:.2f}秒、普通の家具 {rts[1]:.2f}秒、"
            f"さらに10人 {rts[2]:.2f}秒。反射の残り: 床 {drop['floor']:.2f}、"
            f"壁 {drop['west']:.2f}、天井 {drop['ceiling']:.2f}")


@test("spaces: 家具を置いた部屋は、エンジンの出力でも残響が短い")
def t_space_contents_render() -> str:
    from audiospace import furnish_counts, space_preset
    room = space_preset("concrete_room")
    full = room.with_contents(dict(
        furnish_counts(room.size[0] * room.size[1], "full"), person=6))

    def tail_energy(space) -> float:
        engine = make_engine()
        engine.set_space(space, transition_s=0.0)
        click = np.zeros(SR, dtype=np.float32)
        click[2048] = 1.0      # 鳴り始めの1ブロックはフェードインで消える
        engine.add_source(BufferSource(click), position=(0.0, 2.0, 0.0),
                          reverb_send=0.3)
        out = engine.render_seconds(1.2)
        return float(np.sum(out[2048 + int(0.15 * SR):] ** 2))

    empty_e, full_e = tail_energy(room), tail_energy(full)
    ratio_db = 10 * np.log10(full_e / empty_e)
    assert ratio_db < -3.0, ratio_db
    return f"0.15秒より後の響きのエネルギー: 家具と6人で {ratio_db:+.1f}dB"


@test("spaces: 材質の一覧は、面に合うものが上、その他は「その他」付きで下")
def t_face_materials() -> str:
    from audiospace import FACE_MATERIALS, material_choices
    lines = []
    for face, group in (("floor", "floor"), ("ceiling", "ceiling"),
                        ("north", "wall"), ("west", "wall")):
        items = material_choices(face)
        keys = [k for k, _ in items]
        assert len(keys) == 91 and len(set(keys)) == 91, face
        first = FACE_MATERIALS[group]
        assert tuple(keys[:len(first)]) == first, face
        assert all(not v.startswith("その他") for _, v in items[:len(first)])
        assert all(v.startswith("その他 ") for _, v in items[len(first):])
        lines.append(f"{face}: 上に{len(first)}種類、先頭は{items[0][1]}")
    assert material_choices("north") == material_choices("east")
    assert material_choices()[0][0] == "open"          # 面なしは従来どおり
    try:
        material_choices("roof")
    except KeyError:
        pass
    else:
        raise AssertionError("知らない面を受け付けました")
    return "\n".join(lines)


@test("review: 家具や人がいる空間では精密版の計算を始めず、速い版のまま")
def t_review_precise_contents() -> str:
    from audiospace import space_preset
    from audiospace import precise
    room = space_preset("wood_room")
    furnished = room.with_contents({"person": 5})
    assert precise.supports_precise(room)
    assert not precise.supports_precise(furnished)
    assert precise.render_space_ir(furnished) is None
    started = []
    original = precise.start_render_process
    precise.start_render_process = lambda *a, **k: started.append(a) or (
        _ for _ in ()).throw(AssertionError("計算を始めました"))
    try:
        engine = make_engine()
        engine.set_space(furnished, late="precise")
    finally:
        precise.start_render_process = original
    assert not started and not engine._precise_pending, "計算を始めました"
    assert "家具や人" in engine.late_status, engine.late_status
    return f"計算は始めず「{engine.late_status}」"


@test("review: 線がブロックより短い FDN も1サンプルずつのループに落ちず、同じ出力")
def t_review_fdn_short_lines() -> str:
    from audiospace.reverb import FdnReverb, VectorFdnReverb
    rng = np.random.default_rng(41)
    x = [rng.standard_normal(BLOCK).astype(np.float32) * 0.2
         for _ in range(6)]
    scalar = FdnReverb(SR, rt60=1.0, room_scale=0.3, damping=0.4)
    vector = VectorFdnReverb(SR, rt60=1.0, room_scale=0.3, damping=0.4)
    shortest = min(line.shape[0] for line in vector._lines)
    assert shortest <= BLOCK, shortest
    want = np.concatenate([scalar.process(b) for b in x])
    original = FdnReverb.process
    FdnReverb.process = lambda self, b: (_ for _ in ()).throw(
        AssertionError("1サンプルずつのループに落ちた"))
    try:
        got = np.concatenate([vector.process(b) for b in x])
    finally:
        FdnReverb.process = original
    err = float(np.max(np.abs(got - want)))
    assert err < 1e-5, err
    return f"最短の線 {shortest} サンプル（ブロック {BLOCK}）。1サンプルずつの版との差 {err:.1e}"


@test("review: 計算した精密版の尾は、保存先を渡さなくても次に見つかる")
def t_review_precise_default_cache() -> str:
    from audiospace import space_preset
    from audiospace import precise
    room = space_preset("concrete_room").with_contents({})
    room = room.__class__(room.origin, (4.4, 3.3, 2.5), room.materials)
    key = precise.space_key(room)
    with tempfile.TemporaryDirectory() as tmp:
        original = precise.default_cache_dir
        precise.default_cache_dir = lambda: Path(tmp)
        precise._MEMORY.pop((key, SR), None)
        try:
            assert precise.find_space_ir(room, SR) is None
            fake = precise.SpaceIr(key, "test",
                                   np.zeros((2, 64), dtype=np.float32), SR)
            fake.save(Path(tmp) / f"{key}_{SR}.npz")
            found = precise.find_space_ir(room, SR)
        finally:
            precise.default_cache_dir = original
            precise._MEMORY.pop((key, SR), None)
    assert found is not None and found.key == key
    return "既定の保存先（一時フォルダ）を探して見つけた"


@test("review: 精密版の計算を始めるとき、一時の設計図ファイルを閉じて渡す")
def t_review_render_process_spec_closed() -> str:
    import subprocess
    from audiospace import space_preset
    from audiospace import precise
    seen = {}

    class FakePopen:
        def __init__(self, args, **kw):
            spec = Path(args[args.index("--space") + 1])
            seen["text"] = spec.read_text(encoding="utf-8")
            spec.unlink()                     # 開いたままなら Windows で失敗する
            seen["deleted"] = not spec.exists()

    original = subprocess.Popen
    subprocess.Popen = FakePopen
    try:
        with tempfile.TemporaryDirectory() as tmp:
            precise.start_render_process(space_preset("concrete_room"), SR, tmp)
    finally:
        subprocess.Popen = original
    assert seen.get("deleted"), seen
    assert '"materials"' in seen["text"]
    return "設計図は書き終えて閉じてあり、子の側で消せた"


@test("review: 反射の経路が24本を超える空間は、回数の少ない24本に固定する")
def t_review_early_fixed_paths() -> str:
    from audiospace import space_from_spec
    from audiospace.early import MAX_PATHS, EarlyReflections
    room = space_from_spec({"kind": "room", "size": [6.0, 5.0, 3.0],
                            "materials": {f: "rough_concrete" for f in
                                          ("west", "east", "south", "north",
                                           "floor", "ceiling")},
                            "early_order": 8})
    assert len(room.reflection_chains()) > MAX_PATHS
    early = EarlyReflections(room, SR, BLOCK)
    assert len(early._chains) == MAX_PATHS
    assert max(len(c.faces) for c in early._chains) <= 4
    dry = np.zeros(BLOCK, dtype=np.float32)
    chosen = set()
    for x in np.linspace(-1.0, 1.0, 9):
        early.process(dry, (x, 0.3, 0.0))
        chosen |= {i for i, d in enumerate(early._delay) if d >= 0}
    assert len(chosen) <= MAX_PATHS
    assert len(early.paths) <= MAX_PATHS
    return (f"経路 {len(room.reflection_chains())} 本のうち、回数4までの "
            f"{len(early._chains)} 本に固定。動いても入れ替わらない")


@test("review: 家具の量の数は四捨五入（0.5 は 1 になる）")
def t_review_furnish_rounding() -> str:
    from audiospace import furnish_counts
    sparse = furnish_counts(20.0, "sparse")
    assert sparse.get("bookshelf") == 1, sparse
    assert furnish_counts(10.0, "sparse").get("table") == 1     # 0.5 -> 1
    assert furnish_counts(9.0, "sparse").get("table") is None   # 0.45 -> 0
    return f"20m² の「少し」: {sparse}"


@test("review: 速い版の後期残響は、エンジンと精密版の音量合わせで同じ作り方")
def t_review_band_reverb_shared() -> str:
    from audiospace import space_preset
    from audiospace.reverb import band_reverb_for
    room = space_preset("stone_hall")
    engine = make_engine()
    a = engine._fast_late(room)
    b = band_reverb_for(room, SR, BLOCK)
    assert np.array_equal(a.rt60_bands, b.rt60_bands)
    assert [x.shape for x in a._lines] == [x.shape for x in b._lines]
    assert a.level == b.level
    return "engine._fast_late と band_reverb_for が同じ残響を作る"


@test("review: 上下の強調がオフなら、毎ブロックの状態の消去もしない")
def t_review_emphasis_idle() -> str:
    engine = make_engine(elevation_emphasis=0.0)
    noise = np.random.default_rng(43).standard_normal(SR).astype(np.float32)
    engine.add_source(BufferSource(noise * 0.1, loop=True),
                      position=(0.0, 3.0, 3.0), reverb_send=0.0)
    engine.render()
    assert engine._tilt_dirty is False
    engine.elevation_emphasis = 1.0
    engine.render()
    assert engine._tilt_dirty is True
    assert np.any(engine._tilt_zi != 0.0)
    engine.elevation_emphasis = 0.0
    engine.render()
    assert engine._tilt_dirty is False and not np.any(engine._tilt_zi)
    return "オフのあいだは何もせず、オフにした直後に1回だけ状態を消す"


@test("review2: 止めている音・全体を止めている間の fade 停止は、その場で外れる")
def t_review2_fade_inactive() -> str:
    engine = make_engine()
    loop = BufferSource(np.full(SR, 0.1, np.float32), loop=True)
    h = engine.add_source(loop, position=(0.0, 2.0, 0.0))
    engine.render()
    engine.set_source_active(h, False)
    engine.remove_source(h, fade_ms=30)
    for _ in range(5):
        engine.render()
    assert engine.active_sources == 0, "止めた音が枠を握ったまま"
    h2 = engine.add_source(BufferSource(np.full(SR, 0.1, np.float32), loop=True),
                           position=(0.0, 2.0, 0.0))
    engine.render()
    engine.set_paused(True)
    engine.remove_source(h2, fade_ms=30)
    engine.set_paused(False)
    engine.render()                    # 直前の音の畳み込みの尾だけは残る
    assert float(np.max(np.abs(engine.render()))) == 0.0, "再開後に鳴った"
    return "止めた音の fade 停止で枠が返り、一時停止中の fade 停止は再開後に鳴らない"


@test("review2: 音源が NaN を返しても、1ブロックだけ無音で後は元に戻る")
def t_review2_nan_recovers() -> str:
    engine = make_engine()
    count = {"n": 0}

    def bad(n):
        count["n"] += 1
        x = np.zeros(n, np.float32)
        if count["n"] == 3:
            x[10] = np.nan
        return x

    engine.add_source(CallbackSource(bad), position=(0.0, -2.0, 0.0))
    engine.add_source(BufferSource(np.full(SR * 2, 0.1, np.float32), loop=True),
                      position=(0.0, 2.0, 0.0))
    peaks = [float(np.max(np.abs(engine.render()))) for _ in range(20)]
    assert engine.nonfinite_blocks <= 2, engine.nonfinite_blocks
    assert peaks[-1] > 0.0, "その後ずっと無音"
    return f"無音にしたのは {engine.nonfinite_blocks} ブロックだけ"


@test("review2: 壁を抜けて出ても、前の空間の反射は消えるまで続く")
def t_review2_early_hold() -> str:
    from audiospace import Space
    engine = make_engine()
    room = Space.room((5.0, 4.0, 2.7), floor="concrete_floor",
                      walls="rough_concrete", ceiling="rough_concrete",
                      listener=(0, 0, 0), at=(0.5, 0.5), name="room")
    street = Space(origin=(-6.0, 2.0, -1.6), size=(12.0, 30.0, 20.0),
                   materials={"west": "brickwork", "east": "brickwork",
                              "south": "open", "north": "open",
                              "floor": "rough_concrete", "ceiling": "open"},
                   name="street")
    engine.set_spaces([street, room], late=False)
    engine.add_source(BufferSource(tone(220.0, 4.0), loop=True),
                      position=(0.0, 0.5, 0.0), relative=True)
    for _ in range(60):
        engine.set_listener((0.0, 1.9, 0.0))
        engine.render()
    engine.set_listener((0.0, 2.05, 0.0))
    engine.render()
    assert engine.space.name == "street" and engine._old_early is not None
    sig, _, _ = engine._old_early.process(np.zeros(BLOCK, np.float32),
                                          (0.0, 2.05, 0.0),
                                          hold_outside=True)
    assert len(sig) > 0, "消えていく反射が0本になった"
    return f"外に出た直後も前の反射が {len(sig)} 本続く"


@test("review2: set_space のまま壁の外へ出ても反射はその場で切れず、戻ると滑らかに入る")
def t_review2_set_space_edge() -> str:
    from audiospace import Space

    def run(with_space: bool) -> np.ndarray:
        engine = make_engine(limiter=False)
        if with_space:
            engine.set_space(Space.room(
                (5.0, 4.0, 2.7), floor="concrete_floor",
                walls="rough_concrete", ceiling="rough_concrete",
                listener=(0, 0, 0), at=(0.5, 0.5)), late=False)
        engine.add_source(BufferSource(tone(220.0, 4.0), loop=True),
                          position=(0.0, 0.5, 0.0), relative=True)
        out = []
        for i in range(140):
            y = 2.05 if 60 <= i < 100 else 1.9     # 北の壁は y=2.0
            engine.set_listener((0.0, y, 0.0))
            out.append(engine.render())
        return np.concatenate(out)[:, 0]

    refl = run(True) - run(False)

    def peak(block: int) -> float:
        return float(np.max(np.abs(refl[block * BLOCK:(block + 1) * BLOCK])))

    before = peak(59)
    assert before > 0.05, before
    assert peak(61) > 0.8 * before, ("外へ出た直後に切れた", peak(61), before)
    assert peak(95) < 1e-6, ("外にいる間も鳴り続けている", peak(95))
    assert peak(100) < 0.2 * before, ("戻った瞬間に全量で鳴った", peak(100))
    assert peak(135) > 0.8 * before, ("戻っても反射が戻らない", peak(135))
    return (f"出る直前 {before:.3f}、出た直後 {peak(61):.3f}、外 {peak(95):.1e}、"
            f"戻った直後 {peak(100):.3f}、戻って 0.4秒後 {peak(135):.3f}")


@test("review2: 小さい部屋の速い残響の遅延線は、同じ長さが並ばない")
def t_review2_distinct_lines() -> str:
    from audiospace import BandFdnReverb
    for scale, block in ((0.3, 512), (0.3, 1024), (0.5, 1024)):
        rev = BandFdnReverb(SR, [0.5] * 7, room_scale=scale, block=block)
        lens = [len(x) for x in rev._lines]
        assert len(set(lens)) == len(lens), lens
        assert min(lens) > block
    return "どの大きさでも8本とも別の長さ"


@test("review2: load_all は、ちょうどのレートのファイルを優先する")
def t_review2_load_all_exact() -> str:
    from audiospace import BUILTIN_PRESETS, RoomIr, load_all
    pre = BUILTIN_PRESETS["hall"]
    with tempfile.TemporaryDirectory() as tmp:
        RoomIr(pre, np.ones((2, 441), np.float32), 44100).save(
            Path(tmp) / "hall_44100.npz")
        RoomIr(pre, np.full((2, 480), 2.0, np.float32), 48000).save(
            Path(tmp) / "hall_48000.npz")
        got = load_all(48000, tmp)["hall"]
        assert got.meta == {} and abs(float(got.ir.mean()) - 2.0) < 1e-6
    return "48000 のファイルが選ばれた"


@test("review2: 精密版の尾の書きかけ・壊れたファイルは無いものとして扱う")
def t_review2_broken_tail() -> str:
    from audiospace import Space, precise
    sp = Space.room((3.3, 4.4, 2.2), floor="concrete_floor",
                    walls="rough_concrete", ceiling="rough_concrete")
    key = precise.space_key(sp)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / f"{key}_48000.part.npz").write_bytes(b"PK\x03\x04 x")
        (Path(tmp) / f"{key}_44100.npz").write_bytes(b"not a zip")
        assert precise.find_space_ir(sp, 48000, tmp) is None
    return "例外にならず、見つからない扱い"


@test("review2: 排他モードで開いたら、読み上げも排他モードになる")
def t_review2_exclusive_text() -> str:
    from audiospace.device import DeviceChoice
    from audiospace.player import _mark_exclusive
    wasapi = DeviceChoice(0, "Dev", "Windows WASAPI", SR, 10.0)
    assert "排他" in _mark_exclusive(wasapi, True).describe()
    assert "共有" in _mark_exclusive(wasapi, False).describe()
    mme = DeviceChoice(0, "Dev", "MME", SR, 90.0)
    assert "共有" in _mark_exclusive(mme, True).describe()
    direct = SpatialPlayer(make_engine(), wasapi, exclusive=True)
    assert "排他" in direct.status(), direct.status()
    return "WASAPI の排他だけが排他と読まれる"



@test("i18n: 表示の言語は AUDIOSPACE_LANG か OS の表示言語で決まり、日本語以外は英語")
def t_language() -> str:
    import json
    import subprocess
    from audiospace import i18n
    for value, want in (("ja", "ja"), ("ja_JP.UTF-8", "ja"), ("en-US", "en"),
                        ("EN", "en"), ("fr_FR", None), ("", None)):
        assert i18n._normalize(value) == want, (value, i18n._normalize(value))
    assert i18n.system_language() in ("ja", "en")
    # このテストは日本語に固定して走る（run_all が AUDIOSPACE_LANG=ja を渡す）
    assert i18n.language() == "ja" and i18n.tr("あ", "a") == "あ"
    # 別のプロセスで、環境変数が効くこと・ja でも en でもない値は OS に従う
    # ことを確かめる。言語は import の時点で決まるので、同じプロセスでは
    # 切り替えられない
    code = ("import json, audiospace as a; from audiospace import i18n; "
            "print(json.dumps([a.language(), i18n.system_language(), "
            "a.FURNISH_LEVELS[0][1], a.material_label('marble_floor')]))")
    seen = {}
    for value in ("en", "ja", "xx"):
        out = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=dict(os.environ, AUDIOSPACE_LANG=value),
            capture_output=True, text=True, encoding="utf-8", check=True)
        seen[value] = json.loads(out.stdout.strip().splitlines()[-1])
    assert seen["en"][0] == "en" and seen["en"][2:] == ["None", "Marble floor"], seen
    assert seen["ja"][0] == "ja" and seen["ja"][2:] == ["何も置かない", "大理石の床"], seen
    assert seen["xx"][0] == seen["xx"][1], seen
    return f"OS の表示言語は {seen['xx'][1]}。en: {seen['en'][2:]}、ja: {seen['ja'][2:]}"


TESTS = [
    t_azimuth, t_rotation, t_elevation, t_pitch, t_distance, t_front_back,
    t_front_back_elevation,
    t_hrir_shape, t_itd, t_ild, t_resample,
    t_conv_accuracy, t_crossfade, t_conv_partial_crossfade,
    t_conv_active_rows, t_conv_new_source_no_fade, t_geometry_many,
    t_engine_row_reuse_no_fade, t_linear_model_no_div0,
    t_space_materials, t_space_paths, t_band_fdn, t_early_reflection,
    t_space_removed, t_space_switch_smooth, t_space_tail, t_space_map,
    t_space_spec_roundtrip, t_material_labels, t_reverb_text, t_band_fdn_stable,
    t_language,
    t_engine_nonfinite_guard, t_reverb_spec,
    t_space_contents, t_space_contents_render, t_face_materials,
    # 2026-10-16 のコードレビューで直した点
    t_review_precise_contents, t_review_fdn_short_lines,
    t_review_precise_default_cache, t_review_render_process_spec_closed,
    t_review_early_fixed_paths, t_review_furnish_rounding,
    t_review_band_reverb_shared, t_review_emphasis_idle,
    t_precise_tail, t_precise_baked, t_conv_clone, t_precise_engine,
    t_precise_background,
    t_engine_pan, t_engine_symmetry, t_engine_distance, t_engine_front_back,
    t_elevation_emphasis, t_elevation_emphasis_pitch,
    t_elevation_emphasis_moving, t_player_emphasis,
    t_engine_orbit, t_engine_ui, t_engine_slots, t_engine_reap,
    t_engine_headroom,
    t_push_source, t_engine_yaw,
    t_fdn, t_fdn_equivalence, t_conv_reverb_direct, t_reverb_send,
    t_sofa_load, t_sofa_convention, t_sofa_front_back, t_sofa_engine,
    t_sofa_zenith_same,
    t_sofa_horizontal, t_sofa_decimate,
    t_default_asset, t_default_resample,
    # 2026-08-26 のレビューで直した点の退行防止
    t_buffer_multichannel, t_push_overflow_order, t_distance_no_amplify,
    t_depth_full_not_mute, t_limiter, t_fade_out, t_flat_ramp,
    t_pause_resume, t_source_pause, t_gain_db,
    t_play_reaps, t_sound_cache, t_room_ir_baked, t_restart_silent,
    t_callback_error_guard,
    # 遮蔽のこもり・音量の滑らかな変化・定位させない音（2026-08-26）
    t_occlusion_lowpass, t_occlusion_combines, t_occlusion_ramp,
    t_gain_ramp, t_gain_crossfade, t_flat_stereo,
    t_playback_rate, t_play_rate_and_stereo, t_device_lost,
    # 0.3.0 レビュー対応（同日）
    t_coeff_no_boost, t_stereo_occlusion_state, t_ramp_semantics,
    t_rate_loop_points, t_rate_rejects_source, t_cache_byte_cap,
    t_reopen,
    # 2026-11-10 のレビュー対応
    t_review2_fade_inactive, t_review2_nan_recovers, t_review2_early_hold,
    t_review2_distinct_lines, t_review2_load_all_exact,
    t_review2_broken_tail, t_review2_exclusive_text,
    t_review2_set_space_edge,
]


def main() -> int:
    log(f"audiospace domain tests — Python {sys.version.split()[0]}, "
        f"AUDIOSPACE_SILENT={os.environ.get('AUDIOSPACE_SILENT')}")
    log("=" * 74)
    for fn in TESTS:
        fn()
        log()
    log("=" * 74)
    log(f"passed {_passed} / {_passed + _failed}"
        + (f", FAILED {_failed}" if _failed else ""))
    OUT.write_text("\n".join(_log), encoding="utf-8")
    print(f"passed {_passed}/{_passed + _failed}; written: {OUT}")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
