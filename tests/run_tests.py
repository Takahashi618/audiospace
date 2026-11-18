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
import threading
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


def _sounddevice():
    """sounddevice を返す。読み込めない環境では、中身の無い代わりを置く。

    音声の出口が無い機械（Linux の自動テストなど）では、import した時点で
    PortAudio の初期化が失敗する。機器の一覧もストリームも偽物に差し替える
    テストなので、初期化の2つの関数だけを持つ代わりで足りる。ライブラリが
    中で import したときも、同じ代わりを受け取る。
    """
    try:
        import sounddevice as sd
    except Exception:                                 # noqa: BLE001
        import types
        sd = types.ModuleType("sounddevice")
        sd._terminate = lambda: None
        sd._initialize = lambda: None
        sys.modules["sounddevice"] = sd
    return sd

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
    # not at sample 0: until 2026-11-18 a new source faded in over its
    # first block, which silenced a click there (the position is kept)
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
    from audiospace.precise import (SpaceIr, _band_energy,
                                    _fast_tail_energy, render_space_ir,
                                    space_key)
    space = space_preset("carpet_room")
    made = render_space_ir(space, SR)
    # そろえるのは、聞こえる帯域（100〜4000Hz）のエネルギー
    energy = _band_energy(made.tail, SR)
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


@test("engine: flat (UI) sources start as their data, and ramp gain changes")
def t_flat_ramp() -> str:
    # 2026-11-18 まで、新しい音は最初の1ブロックを 0 からフェードインして
    # いた。先頭の1サンプルだけのクリックが消えるので、鳴り始めはデータの
    # とおりにした。音量の変化・再開は今までどおりブロックの中でランプする
    engine = make_engine()
    h = engine.add_source(BufferSource(np.full(SR, 0.5, dtype=np.float32),
                                       loop=True),
                          spatialise=False, reverb_send=0.0)
    first = engine.render()
    steady = 0.5 * engine.master_gain
    assert abs(first[0, 0] - steady) < 1e-6, \
        f"UI sound faded in: first sample {first[0, 0]:.4f} vs {steady:.4f}"
    engine.set_source_gain(h, 0.0)
    down = engine.render()[:, 0]
    assert abs(down[0] - steady) < 0.02 and abs(down[-1]) < 0.02, \
        f"gain change stepped: {down[0]:.3f} -> {down[-1]:.3f}"
    assert np.all(np.diff(down) <= 1e-7), "gain ramp is not monotonic"
    engine.set_source_gain(h, 1.0)
    engine.render()
    engine.set_source_active(h, False)
    engine.set_source_active(h, True)
    resumed = engine.render()[:, 0]
    assert abs(resumed[0]) < 0.02, \
        f"resume starts with a step: first sample {resumed[0]:.3f}"
    return (f"first sample {first[0, 0]:.4f} (target {steady:.4f}); gain to 0 "
            f"ramps {down[0]:.4f} -> {down[-1]:.4f}; resume starts at "
            f"{resumed[0]:.4f}")


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


@test("player: a crashing source or render cannot kill the audio loop")
def t_callback_error_guard() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None)
    try:
        def boom(n: int) -> np.ndarray:
            raise RuntimeError("wired to fail")

        # 壊れた音源は、その音源だけが外れて数えられる
        player.play(CallbackSource(boom))
        assert wait_until(lambda: player.engine.source_errors > 0), \
            "the source error was never recorded"
        assert wait_until(lambda: player.engine.active_sources == 0), \
            "the broken source was not removed"
        assert "RuntimeError" in player.engine.last_source_error
        assert "音源の異常で外した音" in player.status(), player.status()
        # render そのものの例外は、ループが受け止めて数える
        real = player.engine.render
        player.engine.render = lambda: (_ for _ in ()).throw(
            RuntimeError("render wired to fail"))
        try:
            assert wait_until(lambda: player.render_errors > 0), \
                "error was never recorded"
            seen = player.callbacks
            assert wait_until(lambda: player.callbacks > seen + 3), \
                "the loop died after the exception"
        finally:
            player.engine.render = real
        assert "RuntimeError" in player.last_error
        assert "レンダリングエラー" in player.status()
    finally:
        player.stop()
    return (f"source removed ({player.engine.last_source_error}); loop "
            f"survived {player.render_errors} render errors")


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
        # 上限を超えるファイルは既定では少しずつ読む（2026-11-18）。ここでは
        # 丸ごと読む道のキャッシュの上限を確かめるので stream=False
        player.play(wav, stream=False)
        player.play(wav, stream=False)
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
    sd = _sounddevice()

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
            # 作り直しが要ることを、種類で見分けられる（RuntimeError の一種）
            assert isinstance(exc, player_mod.DeviceRateError), type(exc)
        assert player.device_lost is True
        assert "作り直して" in player.last_error, player.last_error

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
        click[2048] = 1.0      # 2026-11-18 までは先頭だとフェードインで消えた
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
        # 線の長さはブロックの大きさによらず、最短でも MIN_LINE
        assert min(lens) >= BandFdnReverb.MIN_LINE, lens
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



@test("review3: どの空間にも入っていない所から空間へ入っても、つないで入る")
def t_review3_enter_from_nowhere() -> str:
    from audiospace import Space

    def room():
        return Space.room((5.0, 4.0, 2.7), floor="concrete_floor",
                          walls="rough_concrete", ceiling="rough_concrete",
                          listener=(0, 0, 0), at=(0.5, 0.5), name="room")

    # 何も鳴らしていないうちの最初の空間は、つながずに全量（つなぐ元が無い）
    fresh = make_engine()
    fresh.set_space(room())
    fresh.render()
    assert (fresh._early_gain, fresh._reverb_in_gain) == (1.0, 1.0)

    engine = make_engine()
    engine.add_source(BufferSource(tone(220.0, 4.0), loop=True),
                      position=(0.0, 0.5, 0.0), relative=True)
    engine.set_listener((0.0, 10.0, 0.0))
    engine.set_spaces([room()])
    for _ in range(20):
        engine.render()
    assert engine.space is None
    engine.set_listener((0.0, 1.0, 0.0))
    engine.render()
    entered = (engine._reverb_in_gain, engine._early_gain)
    assert max(entered) < 0.2, ("入った瞬間に全量で鳴った", entered)

    # 出て、消え終わる前に戻っても、前の反射はその場で切れない
    for _ in range(40):
        engine.render()
    engine.set_listener((0.0, 10.0, 0.0))
    for _ in range(5):
        engine.render()
    left = engine._old_early_gain
    assert engine._old_early is not None and 0.0 < left < 1.0, left
    engine.set_listener((0.0, 1.0, 0.0))
    engine.render()
    assert engine._old_early is not None, "戻った瞬間に前の反射が切れた"
    assert engine._old_early_gain < left
    return (f"入った直後の倍率 残響 {entered[0]:.3f}・反射 {entered[1]:.3f}。"
            f"戻ったとき、前の反射は {left:.2f} から消え続ける")


@test("review3: 壊れた音源（例外・形の違う音）は、その音源だけが外れる")
def t_review3_source_error_isolated() -> str:
    calls = {"n": 0}

    def bad(n):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("壊れた音源")
        return np.zeros(n, np.float32)

    class WrongShape:
        channels = 1
        finished = False

        def read(self, n):
            return np.zeros((n, 2), np.float32)

    engine = make_engine()
    engine.add_source(BufferSource(np.full(SR * 2, 0.1, np.float32),
                                   loop=True), position=(0.0, 2.0, 0.0))
    engine.add_source(CallbackSource(bad), position=(1.0, 2.0, 0.0))
    engine.add_source(WrongShape(), position=(-1.0, 2.0, 0.0))
    peaks = [float(np.max(np.abs(engine.render()))) for _ in range(50)]
    assert engine.source_errors == 2, engine.source_errors
    assert engine.active_sources == 1, engine.active_sources
    assert min(peaks[5:]) > 0.0, "正常な音まで無音になった"
    assert "RuntimeError" in engine.last_source_error
    return (f"外した音源 {engine.source_errors} 個、正常な音は50ブロックとも鳴る"
            f"（最後の理由: {engine.last_source_error}）")


@test("review3: 位置・音量・聞き手の NaN や無限大は、受け取るときに断る")
def t_review3_reject_nonfinite() -> str:
    engine = make_engine()
    h = engine.add_source(BufferSource(np.full(SR * 2, 0.1, np.float32),
                                       loop=True), position=(0.0, 2.0, 0.0))
    nan, inf = float("nan"), float("inf")
    one = np.zeros(10, np.float32)
    calls = [
        ("add_source の位置", lambda: engine.add_source(
            BufferSource(one), position=(nan, 0.0, 0.0))),
        ("add_source の音量", lambda: engine.add_source(
            BufferSource(one), gain=inf)),
        ("位置", lambda: engine.set_source_position(h, (nan, 0.0, 0.0))),
        ("位置の数", lambda: engine.set_source_position(h, (1.0, 2.0))),
        ("音量", lambda: engine.set_source_gain(h, nan)),
        ("音量の dB", lambda: engine.set_source_gain_db(h, nan)),
        ("こもり", lambda: engine.set_source_lowpass(h, nan)),
        ("残響の送り", lambda: engine.set_source_reverb_send(h, nan)),
        ("聞き手の位置", lambda: engine.set_listener(position=(0.0, nan, 0.0))),
        ("聞き手の向き", lambda: engine.set_listener(yaw_deg=nan)),
        ("瞬間移動", lambda: engine.teleport_listener((0.0, 0.0, inf))),
    ]
    for what, call in calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{what}の NaN が通ってしまった")
    assert engine.get_source(h).position == (0.0, 2.0, 0.0)
    peaks = [float(np.max(np.abs(engine.render()))) for _ in range(10)]
    assert engine.nonfinite_blocks == 0 and min(peaks[2:]) > 0.0, peaks
    assert engine.active_sources == 1
    return f"{len(calls)} 通りとも ValueError。音は鳴り続ける"


@test("review3: 速い版の残響は、ブロックの大きさによらず同じ音")
def t_review3_band_reverb_block() -> str:
    from audiospace import space_preset
    from audiospace.reverb import band_reverb_for
    lines = []
    for name in ("tiled_bathroom", "concrete_room"):
        sp = space_preset(name)
        outs = {}
        for block in (256, 512, 1024, 2048):
            rev = band_reverb_for(sp, SR, block)
            n = int(2.0 * SR / block) + 1
            x = np.zeros(n * block, np.float32)
            x[0] = 1.0
            outs[block] = np.concatenate(
                [rev.process(x[i * block:(i + 1) * block]) for i in range(n)])
        length = min(len(y) for y in outs.values())
        ref = outs[512][:length]
        peak = float(np.max(np.abs(ref)))
        worst = max(float(np.max(np.abs(y[:length] - ref)))
                    for y in outs.values())
        assert worst <= 1e-6 * peak, (name, worst, peak)
        lines.append(f"{name}: 512 との差の最大 {worst:.1e}（ピーク {peak:.3f}）")
    return "\n".join(lines)


@test("review3: HRIR がブロックより長いとき、鳴り終わった音の行に新しい音を入れない")
def t_review3_row_reuse() -> str:
    def run(with_new: bool, max_sources: int = 4):
        hr = HrirSet.synthetic(sample_rate=SR, taps=128)
        e = SpatialEngine(hr, block=64, max_sources=max_sources,
                          limiter=False)
        click = np.zeros(64, np.float32)
        click[10] = 0.8
        e.add_source(BufferSource(click), position=(-2.0, 0.0, 0.0))
        outs = [e.render()]
        e.reap_finished()
        if with_new:
            e.add_source(BufferSource(np.zeros(640, np.float32)),
                         position=(2.0, 0.0, 0.0))
        outs += [e.render() for _ in range(4)]
        return np.concatenate(outs), e

    plain, eng = run(False)
    with_new, _ = run(True)
    assert eng._conv.nparts == 2, eng._conv.nparts
    diff = float(np.max(np.abs(plain - with_new)))
    assert diff == 0.0, ("無音の音を足しただけで前の音の尾が変わった", diff)
    # 空いた行がどれも鳴り終わっていなければ、残りを捨ててから入る
    hr = HrirSet.synthetic(sample_rate=SR, taps=128)
    e = SpatialEngine(hr, block=64, max_sources=1, limiter=False)
    click = np.zeros(64, np.float32)
    click[10] = 0.8
    e.add_source(BufferSource(click), position=(-2.0, 0.0, 0.0))
    e.render()
    e.reap_finished()
    assert e._conv.row_sounding(0)
    e.add_source(BufferSource(np.zeros(640, np.float32)),
                 position=(2.0, 0.0, 0.0))
    assert not e._conv.row_sounding(0), "前の音の残りが新しい音の行に残った"
    return "前の音の尾は、無音の音を足しても1サンプルも変わらない"


@test("review3: 読めない音声ファイルの誤りに、ファイルの名前と理由が入る")
def t_review3_decode_error() -> str:
    from audiospace import DecodeError
    from audiospace.sources import load_mono
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "broken.wav"
        bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
        messages = []
        for without_miniaudio in (False, True):
            saved = sys.modules.get("miniaudio")
            if without_miniaudio:
                sys.modules["miniaudio"] = None      # import が失敗する
            try:
                load_mono(bad)
                raise AssertionError("壊れたファイルが読めてしまった")
            except DecodeError as exc:
                messages.append(str(exc))
            finally:
                if without_miniaudio:
                    if saved is None:
                        sys.modules.pop("miniaudio", None)
                    else:
                        sys.modules["miniaudio"] = saved
        for msg in messages:
            assert "broken.wav" in msg and "soundfile" in msg, msg
        assert "miniaudio" in messages[1], messages[1]
    return " / ".join(messages)


@test("review3: コールバックのフレーム数がブロックと違うと、数えて状態に出す")
def t_review3_block_mismatch() -> str:
    player = SpatialPlayer(make_engine(), None)     # デバイスを開かない
    out = np.ones((256, 2), np.float32)
    player._callback(out, 256, None, None)
    assert not out.any(), "無音にしていない"
    assert (player.block_mismatches, player.last_mismatch_frames) == (1, 256)
    text = player.status()
    assert "食い違い" in text, text
    player.reset_stats()
    assert player.block_mismatches == 0
    return text


@test("review3: 英語の表示のとき、残響の設計図の誤りは「; 」で区切る")
def t_review3_reverb_spec_sep() -> str:
    import audiospace.reverb as reverb_mod
    real = reverb_mod.tr
    reverb_mod.tr = lambda ja, en: en
    try:
        reverb_mod.reverb_from_spec({"rt60": 99, "width": 5}, SR)
        raise AssertionError("おかしな設計図が通ってしまった")
    except ValueError as exc:
        text = str(exc)
    finally:
        reverb_mod.tr = real
    assert "; " in text and "、" not in text, text
    return text


@test("review3: table_bytes は畳み込み器が実際に持つ表の大きさと同じ")
def t_review3_table_bytes() -> str:
    lines = []
    for taps in (128, 600):
        h = HrirSet.synthetic(n_azimuth=8, taps=taps, sample_rate=SR)
        conv = BinauralConvolver(h, block=BLOCK, max_sources=1)
        assert h.table_bytes == conv._H.nbytes, (taps, h.table_bytes,
                                                 conv._H.nbytes)
        lines.append(f"{taps}タップ: {h.table_bytes} バイト")
    return "、".join(lines)


class _FakeDevices:
    """出力デバイスの見張りのテスト用。機器の一覧とストリームを偽物にする。

    デバイスは開かない（音は出ない）。`pick` で、つなぎ直しのときに選ばれる
    機器と、機器が無い状態を決める。`opened` は開いた偽のストリーム。
    """

    class Stream:
        def __init__(self) -> None:
            self.active = True

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    def __init__(self) -> None:
        from audiospace.device import DeviceChoice
        self.good = DeviceChoice(0, "Fake USB", "WASAPI", SR, 10.0)
        self.other = DeviceChoice(1, "Speakers", "WASAPI", SR + 3900, 10.0)
        self.pick = {"choice": self.good, "fail": False}
        self.opened: list = []

    def __enter__(self) -> "_FakeDevices":
        sd = _sounddevice()

        import audiospace.player as player_mod
        self._mods = (sd, player_mod)
        self._real = (sd._terminate, sd._initialize,
                      player_mod.choose_output, player_mod.open_output_stream)
        sd._terminate = lambda: None
        sd._initialize = lambda: None

        def choose(*a, **k):
            if self.pick["fail"]:
                raise RuntimeError("no stereo output device found")
            return self.pick["choice"]

        def open_stream(*a, **k):
            stream = self.Stream()
            self.opened.append(stream)
            return stream

        player_mod.choose_output = choose
        player_mod.open_output_stream = open_stream
        return self

    def __exit__(self, *exc) -> None:
        sd, player_mod = self._mods
        (sd._terminate, sd._initialize, player_mod.choose_output,
         player_mod.open_output_stream) = self._real


@test("見張り: poll_device は失われた機器をつなぎ直し、変わったときだけ知らせる")
def t_device_poll() -> str:
    from audiospace import DEVICE_LOST, DEVICE_RATE_CHANGED, DEVICE_RESTORED

    silent = SpatialPlayer(make_engine(), None)
    assert silent.poll_device() is None, "無音モードで知らせた"
    with _FakeDevices() as dev:
        player = SpatialPlayer(make_engine(), dev.good)   # 実機を模す
        try:
            player.start()
            assert player.poll_device() is None, "健全なのに知らせた"

            dev.opened[-1].active = False                 # 抜いた → 戻った
            ev = player.poll_device()
            assert ev is not None and ev.kind == DEVICE_RESTORED, ev
            assert "Fake USB" in ev.detail and ev.message, ev
            assert len(dev.opened) == 2 and not player.device_lost

            dev.opened[-1].active = False                 # 抜いたまま
            dev.pick["fail"] = True
            ev = player.poll_device()
            assert ev is not None and ev.kind == DEVICE_LOST, ev
            assert player.poll_device() is None, "試し直す前に知らせた"
            player._retry_at = 0.0
            assert player.poll_device() is None, "失敗を2回知らせた"

            dev.pick.update(fail=False, choice=dev.other)  # レートの違う機器
            player._retry_at = 0.0
            ev = player.poll_device()
            assert ev is not None and ev.kind == DEVICE_RATE_CHANGED, ev
            player._retry_at = 0.0
            assert player.poll_device() is None, "レート違いを2回知らせた"

            dev.pick["choice"] = dev.good                 # 元の機器が戻った
            player._retry_at = 0.0
            ev = player.poll_device()
            assert ev is not None and ev.kind == DEVICE_RESTORED, ev
            assert player.poll_device() is None
        finally:
            player.stop()
    return f"戻った、失われた（1回だけ）、レート違い、戻った の順に知らせた: {ev.message}"


@test("見張り: watch_device は別のスレッドでつなぎ直し、stop で止まる")
def t_device_watch() -> str:
    from audiospace import DEVICE_LOST, DEVICE_RESTORED

    silent = SpatialPlayer(make_engine(), None)
    silent.watch_device(lambda e: None)
    assert not silent.watching_device, "無音モードで見張りを始めた"
    events: list = []

    def on_event(event) -> None:
        events.append(event)
        raise RuntimeError("呼び出し側の誤り")   # 見張りは止まらないはず

    with _FakeDevices() as dev:
        player = SpatialPlayer(make_engine(), dev.good)
        try:
            player.start()
            player.watch_device(on_event, interval_s=0.02)
            assert player.watching_device
            dev.opened[-1].active = False
            assert wait_until(lambda: len(events) >= 1), "戻ったことを知らせない"
            assert events[0].kind == DEVICE_RESTORED, events[0]
            dev.pick["fail"] = True
            dev.opened[-1].active = False
            assert wait_until(lambda: len(events) >= 2), \
                "on_event の例外の後に見張りが止まった"
            assert events[1].kind == DEVICE_LOST, events[1]
            text = player.status()
            assert "自動でつなぎ直し" in text, text
            dev.pick["fail"] = False
            player._retry_at = 0.0
            assert wait_until(lambda: len(events) >= 3), "戻らない"
            assert events[2].kind == DEVICE_RESTORED
        finally:
            player.stop()
        assert not player.watching_device, "stop の後も見張っている"
    return f"知らせ {len(events)} 回、状態の文: {text[:40]}…"


@test("既定 HRTF: 全天球は耳介E の測定値そのままで、4つの高さがある")
def t_default_sphere() -> str:
    from audiospace import attribution, default_hrir
    from audiospace import defaults as defaults_mod
    sphere = default_hrir(SR, full_sphere=True)
    flat = default_hrir(SR)
    assert sphere.n_directions == 1513, sphere.n_directions
    assert flat.n_directions == 216, "水平面の既定が変わった"
    assert float(sphere.elevation.min()) <= -44.9
    assert float(sphere.elevation.max()) >= 89.9
    for level in (90.0, 45.0, 0.0, -45.0):
        assert float(np.min(np.abs(sphere.elevation - level))) < 1e-6, level
    sofa = (Path(__file__).resolve().parents[1] / "assets" / "hrtf" / "viking"
            / "subj_E.sofa")
    same = "SOFA が無いので比べない"
    if sofa.exists():
        ref = HrirSet.from_sofa(sofa, sample_rate=SR)
        assert np.array_equal(sphere.data, ref.data), "測定値が SOFA と違う"
        assert np.array_equal(sphere.elevation, ref.elevation)
        same = "SOFA の耳介E と値が同じ"
    assert default_hrir(44100, full_sphere=True).sample_rate == 44100
    text = attribution(full_sphere=True)
    assert "unchanged" in text and "horizontal" not in text, text
    assert "horizontal" in attribution(), "水平面の文面が変わった"
    # 全天球が無いときは、水平面や合成に黙って替えない
    real = defaults_mod.DEFAULT_SPHERE_ASSET
    defaults_mod.DEFAULT_SPHERE_ASSET = real.with_name("missing_sphere.npz")
    try:
        default_hrir(SR, full_sphere=True)
        raise AssertionError("無いのに読めてしまった")
    except FileNotFoundError:
        pass
    finally:
        defaults_mod.DEFAULT_SPHERE_ASSET = real
    return (f"{sphere.n_directions}方向、仰角 {sphere.elevation.min():.0f}〜"
            f"{sphere.elevation.max():.0f}度、{same}。"
            f"表 {sphere.table_bytes / 1024 / 1024:.1f}MB")


@test("既定 HRTF: 全天球なら上と下で音色がはっきり変わる（水平面の既定は ±5度まで）")
def t_default_sphere_engine() -> str:
    from audiospace import default_hrir

    def render(hrir, elevation_deg: float) -> np.ndarray:
        engine = SpatialEngine(hrir, block=BLOCK, max_sources=2,
                               front_back_depth=0.0, limiter=False)
        noise = (np.random.default_rng(3).standard_normal(SR)
                 * 0.2).astype(np.float32)
        el = math.radians(elevation_deg)
        engine.add_source(BufferSource(noise, loop=True),
                          position=(0.0, 2.0 * math.cos(el),
                                    2.0 * math.sin(el)), reverb_send=0.0)
        return engine.render_seconds(0.5)[BLOCK * 4:]

    def tilt(x: np.ndarray) -> float:
        """6〜12kHz と 1〜3kHz のエネルギーの比（dB）。左右を別々に測って足す。"""
        high = sum(band_energy(x[:, ch], 6000, 12000) for ch in (0, 1))
        mid = sum(band_energy(x[:, ch], 1000, 3000) for ch in (0, 1))
        return 10 * math.log10(high / mid)

    lines = []
    diffs = {}
    for name, hrir in (("全天球", default_hrir(SR, full_sphere=True)),
                       ("水平面", default_hrir(SR))):
        up, level, down = (tilt(render(hrir, e)) for e in (45.0, 0.0, -45.0))
        diffs[name] = max(abs(up - level), abs(down - level))
        lines.append(f"{name}: 上 {up:+.1f}dB、水平 {level:+.1f}dB、"
                     f"下 {down:+.1f}dB")
    # 水平面の既定は仰角 ±5度までしか無いので、上下の音もそこで鳴る
    flat = default_hrir(SR)
    used = flat.elevation[flat.indices_for(np.array([0.0, 0.0]),
                                           np.array([45.0, -45.0]))]
    assert np.all(np.abs(used) <= 5.0), used
    assert diffs["全天球"] > 3.0, ("全天球で上下の音色が変わらない", diffs)
    assert diffs["全天球"] > 3.0 * diffs["水平面"], diffs
    return "\n".join(lines)


@test("既定 HRTF: SpatialPlayer.open(full_sphere=True) で全天球になる")
def t_player_full_sphere() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                full_sphere=True)
    try:
        assert player.engine.hrir.n_directions == 1513, \
            player.engine.hrir.n_directions
    finally:
        player.stop()
    try:
        SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                           full_sphere=True,
                           hrir=HrirSet.synthetic(sample_rate=SR))
        raise AssertionError("HRTF を2通りに指定できてしまった")
    except ValueError:
        pass
    return "全天球の1513方向で開き、hrir と同時の指定は ValueError"


# -- 鳴らし始めの時刻の指定（2026-11-18） ----------------------------------
def _onset(x: np.ndarray, thresh: float = 1e-6) -> int:
    """最初に音が出たサンプルの位置（無ければ -1）。"""
    idx = np.flatnonzero(np.abs(x) > thresh)
    return int(idx[0]) if idx.size else -1


def _flat_click(engine: SpatialEngine, pan: float, at: int = 0,
                **kwargs) -> int:
    """左（pan -1）か右（+1）だけに鳴る、1サンプルのクリック。"""
    click = np.zeros(64 + at, np.float32)
    click[at] = 0.5
    return engine.add_source(BufferSource(click), spatialise=False, pan=pan,
                             reverb_send=0.0, **kwargs)


@test("遅らせて鳴らす: delay_s のサンプルちょうどに、データのとおりの大きさで鳴り始める")
def t_delay_exact() -> str:
    # 0 は遅らせない音。先頭の1サンプルのクリックも消えずに鳴る
    for wait in (0, 1, 100, 511, 512, 700, 1300, 5000):
        engine = make_engine(limiter=False)
        _flat_click(engine, -1.0, delay_s=wait / SR)
        out = np.concatenate([engine.render()
                              for _ in range(wait // BLOCK + 2)])
        at = _onset(out[:, 0])
        assert at == wait, (wait, at)
        want = 0.5 * engine.master_gain
        assert abs(out[at, 0] - want) < 1e-6, (wait, out[at, 0], want)
        assert np.max(np.abs(out[:, 1])) == 0.0, "右に漏れた"
    # ループする音は、最初の1回だけ遅れる
    engine = make_engine(limiter=False)
    seg = np.zeros(300, np.float32)
    seg[0] = 0.5
    engine.add_source(BufferSource(seg, loop=True), spatialise=False,
                      pan=-1.0, reverb_send=0.0, delay_s=700 / SR)
    out = np.concatenate([engine.render() for _ in range(5)])[:, 0]
    hits = np.flatnonzero(np.abs(out) > 1e-6)[:5].tolist()
    assert hits == [700, 1000, 1300, 1600, 1900], hits
    return ("0・1・100・511・512・700・1300・5000 サンプル遅らせて、どれもちょうど"
            "その位置で、フェードインなしに鳴り始める。ループは最初だけ遅れる")


@test("遅らせて鳴らす: 空間化した音も、前に無音を足した音と同じ音になる")
def t_delay_matches_padding() -> str:
    rng = np.random.default_rng(7)
    noise = (0.3 * rng.standard_normal(SR // 2)).astype(np.float32)
    wait = 1300                      # 3ブロック目の途中から鳴る
    outs = []
    for padded in (True, False):
        engine = make_engine()
        if padded:
            data = np.concatenate([np.zeros(wait, np.float32), noise])
            engine.add_source(BufferSource(data), position=(1.0, 2.0, 0.0))
        else:
            engine.add_source(BufferSource(noise), position=(1.0, 2.0, 0.0),
                              delay_s=wait / SR)
        outs.append(np.concatenate([engine.render() for _ in range(60)]))
    diff = float(np.max(np.abs(outs[0] - outs[1])))
    peak = float(np.max(np.abs(outs[0])))
    assert peak > 0.01 and diff <= peak * 1e-5, (diff, peak)
    return f"60ブロックの差の最大 {diff:.2e}（ピーク {peak:.3f}）"


@test("遅らせて鳴らす: together() の中の音は、間に描画が入っても同じブロックから数える")
def t_delay_together() -> str:
    from contextlib import nullcontext

    wait = 451                       # 9.4ms（水平の音の床の反射の遅れ）

    def run(use_together: bool) -> tuple[int, int, float]:
        engine = make_engine(limiter=False)
        pre = []
        with engine.together() if use_together else nullcontext():
            _flat_click(engine, -1.0, at=200)
            pre.append(engine.render())        # 音声のスレッドが割り込んだ
            _flat_click(engine, 1.0, at=200, delay_s=wait / SR)
        out = np.concatenate(pre + [engine.render() for _ in range(4)])
        return (_onset(out[:, 0]), _onset(out[:, 1]),
                float(np.max(np.abs(pre[0]))))

    left, right, pre_peak = run(True)
    assert pre_peak == 0.0, "together の中で鳴り始めた"
    assert right - left == wait, (left, right)
    left2, right2, _ = run(False)
    assert right2 - left2 == wait + BLOCK, (left2, right2)
    return (f"together あり: 差 {right - left} サンプル（指定どおり）。"
            f"なし: 差 {right2 - left2} サンプル（1ブロック遅れる）")


@test("遅らせて鳴らす: together() の中で例外が出たら、その組の音は鳴らさずに外す")
def t_delay_together_error() -> str:
    engine = make_engine()
    ones = np.ones(4800, np.float32)
    try:
        with engine.together():
            engine.add_source(BufferSource(ones), position=(0.0, 2.0, 0.0))
            with engine.together():          # 入れ子は外側と一緒に扱う
                engine.add_source(BufferSource(ones),
                                  position=(1.0, 2.0, 0.0), delay_s=0.1)
            raise RuntimeError("途中で失敗")
    except RuntimeError:
        pass
    else:
        raise AssertionError("例外が外へ出なかった")
    assert engine.active_sources == 0, engine.active_sources
    peak = max(float(np.max(np.abs(engine.render()))) for _ in range(20))
    assert peak == 0.0, peak
    with engine.together():
        with engine.together():
            h = engine.add_source(BufferSource(ones), position=(0.0, 2.0, 0.0))
        assert engine.get_source(h)._held, "内側を抜けた時点で鳴り始めた"
    assert not engine.get_source(h)._held
    return "例外のときは2つとも外れて無音。入れ子は外側を抜けてから鳴り始める"


@test("遅らせて鳴らす: 待っている間の停止・一時停止・全体の停止")
def t_delay_wait_controls() -> str:
    engine = make_engine(limiter=False)
    h = _flat_click(engine, -1.0, delay_s=0.5)
    engine.render()
    engine.remove_source(h, fade_ms=30)
    assert engine.active_sources == 0, "まだ鳴っていない音がフェードで残った"
    wait = 700
    h = _flat_click(engine, -1.0, delay_s=wait / SR)
    blocks = [engine.render()]                  # ここで1ブロック数える
    engine.set_source_active(h, False)
    blocks += [engine.render() for _ in range(3)]   # 止めている間は数えない
    engine.set_source_active(h, True)
    engine.set_paused(True)
    blocks.append(engine.render())              # 全体を止めている間も数えない
    engine.set_paused(False)
    blocks += [engine.render() for _ in range(3)]
    at = _onset(np.concatenate(blocks)[:, 0])
    assert at == wait + 4 * BLOCK, (at, wait + 4 * BLOCK)
    return ("フェード付きの停止でもその場で外れる。止めていた4ブロックぶんだけ"
            "後ろへずれて鳴る")


@test("遅らせて鳴らす: play・play_ui・together と、負の値や NaN を断る")
def t_delay_player() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        noise = np.full(SR // 10, 0.1, np.float32)
        # 無音の再生でも描画は進むので、数え始めない together の中で確かめる
        with player.together():
            a = player.play(noise, position=(0.0, 2.0, 0.0), delay_s=0.25)
            b = player.play_ui(noise, delay_s=0.5)
            for h, want in ((a, 12000), (b, 24000)):
                slot = player.engine.get_source(h)
                assert slot._held and slot._wait == want, (slot._wait, want)
        assert not player.engine.get_source(a)._held
        count = player.engine.active_sources
        for bad in (-0.1, float("nan"), float("inf")):
            try:
                player.play(noise, delay_s=bad)
                raise AssertionError(f"delay_s={bad} を受け付けた")
            except ValueError:
                pass
        assert player.engine.active_sources == count
    finally:
        player.stop()
    return "play と play_ui に delay_s が届く。負・NaN・無限大は ValueError"


# -- 音のグループ（2026-11-18） --------------------------------------------
def _flat_dc(engine: SpatialEngine, pan: float, level: float = 0.5,
             seconds: float = 4.0, **kwargs) -> int:
    """左（pan -1）か右（+1）だけに鳴る一定の値の音。大きさを読みやすい。"""
    data = np.full(int(SR * seconds), level, np.float32)
    return engine.add_source(BufferSource(data), spatialise=False, pan=pan,
                             reverb_send=0.0, **kwargs)


def _levels(engine: SpatialEngine, blocks: int) -> np.ndarray:
    """各ブロックの最後のサンプルの左右の値（ランプが終わった後の値）。"""
    return np.array([engine.render()[-1] for _ in range(blocks)])


@test("グループ: 音量はグループの音だけに、音ごとの音量に重ねて掛かる")
def t_group_gain() -> str:
    engine = make_engine(limiter=False)
    mg = engine.master_gain
    engine.set_group_gain("sfx", 0.5)          # 鳴らす前に決めておける
    _flat_dc(engine, -1.0, gain=0.5, group="sfx")
    _flat_dc(engine, 1.0)
    out = _levels(engine, 2)[-1]
    assert abs(out[0] - 0.5 * 0.5 * 0.5 * mg) < 1e-6, out
    assert abs(out[1] - 0.5 * mg) < 1e-6, "グループの外の音まで変わった"
    # ランプは指定の時間で行き先に着く（100ms = 9.4ブロック → 10ブロック目）。
    # ブロックの中でもなめらかに変えるので、着いた値になるのは次のブロック
    engine.set_group_gain("sfx", 1.0, ramp_ms=100)
    assert engine.group_gain("sfx") == 1.0, "行き先を返していない"
    lv = _levels(engine, 12)[:, 0] / (0.25 * mg)
    assert 0.5 < lv[4] < 1.0 and lv[9] > 0.999, lv
    assert abs(lv[10] - 1.0) < 1e-6, lv
    assert np.all(np.diff(lv) >= -1e-7), "ランプが行き来した"
    engine.set_group_gain_db("sfx", -6.0)
    assert abs(engine.group_gain("sfx") - 10 ** (-6 / 20)) < 1e-9
    assert engine.group_gain("まだ無い") == 1.0
    for bad in ((("sfx", float("nan")), "NaN"), ((3, 1.0), "名前が数")):
        try:
            engine.set_group_gain(*bad[0])
            raise AssertionError(f"{bad[1]} を受け付けた")
        except ValueError:
            pass
    try:
        _flat_dc(engine, 0.0, group=7)
        raise AssertionError("名前でないグループを受け付けた")
    except ValueError:
        pass
    return ("0.5 × 0.5 の音が 0.25 倍で鳴り、外の音は変わらない。100ms のランプは"
            "10ブロック目で着く。NaN・名前でない値は ValueError")


@test("グループ: 一時停止はその場で止めて、再開でフェードインして続きから鳴る")
def t_group_pause() -> str:
    engine = make_engine(limiter=False)
    ramp = np.arange(SR * 2, dtype=np.float32) / (SR * 2)
    src = BufferSource(ramp)
    music = engine.add_source(src, spatialise=False, pan=-1.0,
                              reverb_send=0.0, group="music")
    _flat_dc(engine, 1.0, group="sfx")
    engine.render()
    engine.set_group_paused("music", True)
    held = src.position
    later = _flat_dc(engine, -1.0, group="music", delay_s=0.05)
    out = _levels(engine, 5)
    assert src.position == held and np.all(out[:, 0] == 0.0), "止まらない"
    assert np.all(out[:, 1] > 0.0), "ほかのグループまで止まった"
    assert engine.get_source(later)._wait == 2400, "止めている間に数えた"
    # 止めている間のフェード停止はその場で外れる
    engine.stop_group("music", fade_ms=30)
    assert engine.sources_in_group("music") == []
    # 1つだけ止めた音は、グループを再開しても止まったまま
    music = engine.add_source(BufferSource(ramp), spatialise=False,
                              pan=-1.0, reverb_send=0.0, group="music")
    engine.set_group_paused("music", False)
    engine.render()
    engine.set_source_active(music, False)
    engine.set_group_paused("music", True)
    engine.set_group_paused("music", False)
    engine.render()
    assert np.all(engine.render()[:, 0] == 0.0), "1つだけ止めた音が鳴った"
    engine.set_source_active(music, True)
    block = engine.render()[:, 0]
    assert block[0] == 0.0 and block[-1] > 0.0, "再開でフェードインしない"
    return ("止めている間は読まず、遅らせた音の待ちも減らない。ほかのグループは"
            "鳴り続ける。1つだけ止めた音は止まったまま")


@test("グループ: stop_group・group_sounds・set_sound_group")
def t_group_stop() -> str:
    engine = make_engine(limiter=False)
    a = _flat_dc(engine, -1.0, group="sfx")
    b = _flat_dc(engine, -1.0, group="sfx")
    c = _flat_dc(engine, 1.0)
    engine.render()
    assert engine.sources_in_group("sfx") == [a, b]
    engine.set_source_group(b, "voice")
    assert engine.sources_in_group("sfx") == [a]
    assert engine.sources_in_group("voice") == [b]
    stopped = engine.stop_group("sfx", fade_ms=30)
    assert stopped == [a]
    lv = _levels(engine, 6)
    assert a not in engine.sources_in_group("sfx")
    assert engine.active_sources == 2, engine.active_sources
    assert np.all(lv[:, 1] > 0.0), "グループの外の音が止まった"
    engine.set_source_group(c, None)
    assert engine.get_source(c).group is None
    return "フェードして外れるのは sfx の音だけ。移した音は新しいグループで数える"


@test("グループ: ダッキングは声の間だけ下げて戻し、決めた音量は上書きしない")
def t_group_ducking() -> str:
    engine = make_engine(limiter=False)
    mg = engine.master_gain
    block_s = BLOCK / SR
    _flat_dc(engine, -1.0, group="music")
    engine.set_group_gain("music", 0.5)
    engine.set_ducking("music", by="voice", db=-12.0, attack_ms=50.0,
                       release_ms=200.0)
    voice = _flat_dc(engine, 1.0, level=0.2, seconds=0.5, group="voice",
                     delay_s=0.2)
    lv = _levels(engine, 150)[:, 0] / (0.5 * 0.5 * mg)
    duck = 10 ** (-12 / 20)
    start = int(0.2 / block_s) + 1
    attack = int(np.ceil(0.05 / block_s))
    assert abs(lv[start - 2] - 1.0) < 1e-6, "声の前から下がった"
    assert abs(lv[start + attack] - duck) < 1e-6, (lv[start:start + 8], duck)
    end = int(0.7 / block_s) + 2
    release = int(np.ceil(0.2 / block_s))
    assert lv[end] < 1.0, "声が終わってすぐ戻った（リリースが無い）"
    assert abs(lv[end + release + 1] - 1.0) < 1e-6, lv[end:end + release + 3]
    assert engine.group_gain("music") == 0.5, "決めた音量が変わった"
    engine.reap_finished()
    assert voice not in engine.sources_in_group("voice")
    # 声のグループを止めている間は、鳴っている扱いにしない
    _flat_dc(engine, 1.0, level=0.2, seconds=1.0, group="voice")
    engine.set_group_paused("voice", True)
    _levels(engine, 10)
    assert abs(engine.group_level("music") - 0.5) < 1e-9
    engine.set_group_paused("voice", False)
    # 手で下げる（読み上げソフトの声など、エンジンを通らない声）
    engine.stop_group("voice")
    engine.hold_ducking("music", True)
    _levels(engine, 10)
    assert abs(engine.group_level("music") - 0.5 * duck) < 1e-9
    engine.clear_ducking("music")
    _levels(engine, 25)
    assert abs(engine.group_level("music") - 0.5) < 1e-9, "決まりを消しても戻らない"
    for args, kw, why in ((("music",), {"by": "music"}, "自分で自分を下げる"),
                          (("music",), {"db": 3.0}, "上げるダッキング"),
                          (("music",), {"attack_ms": -1.0}, "負の時間")):
        try:
            engine.set_ducking(*args, **kw)
            raise AssertionError(f"{why}を受け付けた")
        except ValueError:
            pass
    try:
        engine.hold_ducking("sfx")
        raise AssertionError("決まりの無いグループを手で下げられた")
    except ValueError:
        pass
    return (f"-12dB へ {attack} ブロック（50ms）で下がり、声の後 {release} ブロック"
            "（200ms）で戻る。音量 0.5 はそのまま。手で下げる・決まりを消すと戻る")


@test("グループ: プレイヤーの play・play_ui と、グループの操作")
def t_group_player() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=6, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        tone_data = np.full(SR, 0.1, np.float32)
        player.set_group_gain_db("sfx", -6.0)
        a = player.play(tone_data, position=(0.0, 2.0, 0.0), group="sfx",
                        loop=True)
        b = player.play_ui(tone_data, group="ui")
        assert player.group_sounds("sfx") == [a]
        assert player.group_sounds("ui") == [b]
        assert abs(player.group_gain("sfx") - 10 ** (-6 / 20)) < 1e-9
        player.pause_group("sfx")
        assert player.engine.group_paused("sfx")
        player.resume_group("sfx")
        assert not player.engine.group_paused("sfx")
        player.set_sound_group(b, "sfx")
        player.set_ducking("music", by=("voice", "ui"), db=-9.0)
        player.hold_ducking("music", True)
        player.clear_ducking("music")
        assert sorted(player.stop_group("sfx")) == sorted([a, b])
        assert player.engine.active_sources == 0
    finally:
        player.stop()
    return "play・play_ui の group、音量・一時停止・移す・止める・ダッキングが届く"


# -- 鳴り終わりの知らせ（2026-11-18） --------------------------------------
@test("鳴り終わり: データを読み終えたブロックで1回だけ記録し、頼んだ音だけ")
def t_end_finished() -> str:
    from audiospace import SOUND_FINISHED
    engine = make_engine()
    short = np.full(2400, 0.1, np.float32)          # 4ブロック目の途中で終わる
    a = engine.add_source(BufferSource(short), position=(0.0, 2.0, 0.0),
                          notify_end=True)
    engine.add_source(BufferSource(np.full(SR, 0.1, np.float32), loop=True),
                      position=(1.0, 2.0, 0.0), notify_end=True)
    engine.add_source(BufferSource(short), position=(-1.0, 2.0, 0.0))
    for _ in range(4):
        engine.render()
    assert engine.take_ended() == [], "鳴り終わる前に知らせた"
    engine.render()
    events = engine.take_ended()
    assert len(events) == 1 and events[0].handle == a, events
    assert events[0].reason == SOUND_FINISHED
    assert abs(events[0].at_s - 4 * BLOCK / SR) < 1e-9, events[0].at_s
    for _ in range(5):
        engine.render()
    engine.reap_finished()
    assert engine.take_ended() == [], "回収のときに二重に知らせた"
    # 鳴る前に終わっている音（空の音）は、回収のときに finished
    empty = engine.add_source(BufferSource(np.zeros(0, np.float32)),
                              notify_end=True)
    engine.reap_finished()
    events = engine.take_ended()
    assert [(e.handle, e.reason) for e in events] == [(empty, SOUND_FINISHED)]
    return (f"2400サンプルの音は 5ブロック目（{4 * BLOCK / SR * 1000:.1f}ms から"
            "のブロック）で1回。ループと、頼んでいない音は知らせない")


@test("鳴り終わり: 止めた音は stopped（フェードは終わってから）、壊れた音は error")
def t_end_stopped() -> str:
    from audiospace import SOUND_ERROR, SOUND_STOPPED
    engine = make_engine()
    loop = BufferSource(np.full(SR, 0.1, np.float32), loop=True)
    a = engine.add_source(loop, position=(0.0, 2.0, 0.0), notify_end=True)
    engine.render()
    engine.remove_source(a, fade_ms=30)
    assert engine.take_ended() == [], "フェードの途中で知らせた"
    for _ in range(6):
        engine.render()
    events = engine.take_ended()
    assert [(e.handle, e.reason) for e in events] == [(a, SOUND_STOPPED)]
    # 待っている音・グループごと・together の失敗
    b = engine.add_source(BufferSource(np.ones(100, np.float32)),
                          notify_end=True, delay_s=1.0)
    engine.remove_source(b, fade_ms=30)
    c = engine.add_source(BufferSource(np.ones(SR, np.float32)),
                          notify_end=True, group="sfx")
    engine.stop_group("sfx")
    try:
        with engine.together():
            d = engine.add_source(BufferSource(np.ones(SR, np.float32)),
                                  notify_end=True)
            raise RuntimeError("途中で失敗")
    except RuntimeError:
        pass
    got = [(e.handle, e.reason) for e in engine.take_ended()]
    assert got == [(b, SOUND_STOPPED), (c, SOUND_STOPPED),
                   (d, SOUND_STOPPED)], got

    def broken(n):
        raise RuntimeError("壊れた音源")

    e = engine.add_source(CallbackSource(broken), notify_end=True)
    engine.render()
    events = engine.take_ended()
    assert [(x.handle, x.reason) for x in events] == [(e, SOUND_ERROR)]
    assert "壊れた音源" in events[0].detail
    return "フェードが終わってから stopped。待っている音・グループ・together も stopped。壊れた音は error と理由"


@test("鳴り終わり: プレイヤーの on_end を poll_ends で、呼んだスレッドで呼ぶ")
def t_end_player_poll() -> str:
    from audiospace import SOUND_FINISHED, SOUND_STOPPED
    player = SpatialPlayer.open(block=BLOCK, max_sources=6, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    got: list = []
    me = threading.current_thread()
    try:
        def record(event):
            got.append((event.handle, event.reason,
                        threading.current_thread() is me))

        def bad(event):
            raise ValueError("関数の中の誤り")

        a = player.play(np.full(2400, 0.1, np.float32),
                        position=(0.0, 2.0, 0.0), on_end=bad)
        b = player.play_ui(np.full(2400, 0.1, np.float32), on_end=record)
        c = player.play(np.full(SR, 0.1, np.float32), loop=True,
                        position=(1.0, 2.0, 0.0), on_end=record)
        deadline = time.time() + 3.0
        while len(got) < 1 and time.time() < deadline:
            player.poll_ends()
            time.sleep(0.01)
        player.stop_sound(c)
        player.poll_ends()
        assert got == [(b, SOUND_FINISHED, True), (c, SOUND_STOPPED, True)], got
        assert player.callback_errors == 1, "誤りを数えていない"
        assert "関数の中の誤り" in player.status()
        assert a not in player.engine._slots, "鳴り終わった音を回収していない"
        try:
            player.play(np.ones(10, np.float32), on_end="呼べない")
            raise AssertionError("呼べない on_end を受け付けた")
        except ValueError:
            pass
        # poll_ends を呼ばずに放っておくと、status() が知らせる
        d = player.play(np.full(480, 0.1, np.float32), on_end=record)
        deadline = time.time() + 2.0
        while not player.engine.pending_ends() and time.time() < deadline:
            time.sleep(0.01)
        with player.engine._lock:
            player.engine._elapsed += 5.0       # 時間が経ったことにする
        assert "届けていない鳴り終わりの知らせ1件" in player.status(), \
            player.status()
        player.poll_ends()
        assert got[-1][0] == d
    finally:
        player.stop()
    return ("on_end は poll_ends を呼んだスレッドで、終わった順に1回ずつ。関数の誤りは"
            "ほかを止めずに数える。届けていない知らせは status() に出る")


@test("鳴り終わり: watch_ends は自分のスレッドで届け、dispatch で渡し先を選べる")
def t_end_player_watch() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    handed: list = []
    done = threading.Event()
    try:
        def dispatch(fn, event):
            handed.append(threading.current_thread().name)
            fn(event)

        player.watch_ends(interval_s=0.01, dispatch=dispatch)
        assert player.watching_ends
        player.play(np.full(2400, 0.1, np.float32), position=(0.0, 2.0, 0.0),
                    on_end=lambda event: done.set())
        assert done.wait(3.0), "watch_ends が届けない"
        assert handed == ["audiospace-end-notices"], handed
    finally:
        player.stop()
    assert not player.watching_ends, "stop で止まらない"
    return "見張りのスレッドから dispatch(fn, event) で届き、stop で止まる"


# -- 部屋をまたぐ音の自動の遮蔽（2026-11-18） ------------------------------
def _occ_world(east_of_a: str = "rough_concrete"):
    """部屋 A・B（東西に並ぶ）と、北の廊下 C。A と C の間に戸口。"""
    from audiospace import Opening, Space
    walls = {f: "rough_concrete" for f in ("west", "east", "south", "north",
                                           "floor", "ceiling")}
    a = Space((0, 0, 0), (5, 4, 3), dict(walls, east=east_of_a), name="A")
    b = Space((5, 0, 0), (5, 4, 3), dict(walls, west="plasterboard"),
              name="B")
    c = Space((0, 4, 0), (10, 2, 3), dict(walls), name="C")
    door = Opening((2.0, 3.9, 0.0), (3.0, 4.1, 2.0), name="doorA")
    return [a, b, c], [door]


@test("遮蔽: 壁の材質で下げ、同じ平面の両側の壁は1枚、穴と開いた面は通す")
def t_occ_geometry() -> str:
    from audiospace import Opening, Space, material_occlusion
    from audiospace.occlusion import OcclusionMap
    spaces, doors = _occ_world()
    lis = (2.5, 2.0, 1.6)
    omap = OcclusionMap(spaces, doors)
    same, nxt, corner, front, outside = omap.query(
        [(4.0, 3.0, 1.6), (7.5, 2.0, 1.6), (8.0, 5.0, 1.6),
         (2.5, 5.0, 1.6), (2.5, -5.0, 1.6)], lis)
    concrete = material_occlusion("rough_concrete")
    assert same.loss_db == 0.0 and same.walls == ()
    # A の東の壁（コンクリート）と B の西の壁（石膏ボード）は同じ平面: 強いほう1枚
    assert (nxt.loss_db, nxt.lowpass) == concrete, nxt
    assert nxt.walls == ("rough_concrete",), nxt.walls
    assert front.loss_db == 0.0, "開いた戸口の正面で下がった"
    assert outside.walls == ("rough_concrete",)
    # 角の先の廊下: まっすぐなら壁2枚。戸口から回り込む
    assert corner.via == ("doorA",) and corner.position is not None, corner
    toward = np.subtract(corner.position, lis)
    assert toward[1] > 0 and abs(np.degrees(np.arctan2(toward[0], toward[1]))) < 30, \
        f"戸口（北）の方向から聞こえない: {toward}"
    assert 6.5 < np.linalg.norm(toward) < 8.0, "道のりの距離で聞こえない"
    # 戸を閉めると戸の材質、間の部屋を通ると壁2枚の和
    closed = OcclusionMap(spaces, [Opening(doors[0].lo, doors[0].hi,
                                           "wooden_door", "doorA")])
    front_closed = closed.query([(2.5, 5.0, 1.6)], lis)[0]
    assert (front_closed.loss_db, front_closed.lowpass) == \
        material_occlusion("wooden_door"), front_closed
    walls = {f: "rough_concrete" for f in ("west", "east", "south", "north",
                                           "floor", "ceiling")}
    row = [Space((x * 5.0, 0, 0), (5, 4, 3), dict(walls)) for x in range(3)]
    two = OcclusionMap(row).query([(12.5, 2.0, 1.6)], lis)[0]
    assert abs(two.loss_db - 2 * concrete[0]) < 1e-9, two
    # 開いた面はそのまま通す。強さと材質の差し替え
    open_east = OcclusionMap([Space((0, 0, 0), (5, 4, 3),
                                    dict(walls, east="open")),
                              Space((5, 0, 0), (5, 4, 3),
                                    dict(walls, west="open"))])
    assert open_east.query([(7.5, 2.0, 1.6)], lis)[0].loss_db == 0.0
    half = OcclusionMap(spaces, doors, strength=0.5).query(
        [(7.5, 2.0, 1.6)], lis)[0]
    assert abs(half.loss_db - concrete[0] / 2) < 1e-9
    # 差し替えると、同じ壁の反対側（石膏ボード 7dB）のほうが強くなる
    custom = OcclusionMap(spaces, doors, materials={
        "rough_concrete": (3.0, 0.2)}).query([(7.5, 2.0, 1.6)], lis)[0]
    assert custom.walls == ("plasterboard",), custom
    custom = OcclusionMap(spaces, doors, materials={
        "rough_concrete": (3.0, 0.2), "plasterboard": (2.0, 0.1)}).query(
        [(7.5, 2.0, 1.6)], lis)[0]
    assert abs(custom.loss_db - 3.0) < 1e-9 and \
        abs(custom.lowpass - 0.2) < 1e-9, custom
    return (f"隣の部屋 {nxt.loss_db:g}dB（1枚）、間の部屋を通ると {two.loss_db:g}dB、"
            f"角の先は戸口から {corner.loss_db:.1f}dB、戸を閉めると "
            f"{front_closed.loss_db:g}dB")


@test("遮蔽: エンジンで下がり、鳴り始めからこもり、動くとなめらかに変わる")
def t_occ_engine() -> str:
    spaces, doors = _occ_world()
    rng = np.random.default_rng(2)
    noise = (0.2 * rng.standard_normal(SR * 2)).astype(np.float32)

    def run(occ: bool, **kw):
        engine = make_engine()
        engine.set_spaces(spaces, early=False, late=False)
        if occ:
            engine.set_auto_occlusion(openings=doors)
        engine.set_listener((2.5, 2.0, 1.6))
        h = engine.add_source(BufferSource(noise, loop=True),
                              position=(7.5, 2.0, 1.6), reverb_send=0.0, **kw)
        engine.render()
        first = engine.get_source(h)._occ_db
        out = np.concatenate([engine.render() for _ in range(30)])
        return engine, h, first, 20 * np.log10(np.sqrt(np.mean(out ** 2)))

    _, _, _, open_level = run(False)
    engine, h, first, level = run(True)
    drop = level - open_level
    assert -26.0 < drop < -18.0, f"コンクリートの壁で {drop:.1f}dB"
    assert first == engine.source_occlusion(h).loss_db, "鳴り始めがこもっていない"
    _, _, _, free = run(True, occlude=False)
    assert abs(free - open_level) < 0.01, "occlude=False でも下がった"
    # 部屋の中へ動くと、なめらかに戻る（1ブロックで飛ばない）
    engine.set_source_position(h, (4.0, 2.0, 1.6))
    engine.render()
    slot = engine.get_source(h)
    step = engine.OCCLUSION_DB_PER_100MS * BLOCK / SR / 0.1
    assert abs(slot._occ_db - (14.0 - step)) < 1e-6, slot._occ_db
    for _ in range(30):
        engine.render()
    assert slot._occ_db == 0.0 and slot._occ_lp == 0.0
    # 戸の開け閉め、切ると戻る
    engine.set_source_position(h, (2.5, 5.0, 1.6))
    engine.render()
    assert engine.source_occlusion(h).loss_db == 0.0
    engine.set_opening("doorA", "wooden_door")
    engine.render()
    assert engine.source_occlusion(h).walls == ("wooden_door",)
    engine.set_auto_occlusion(False)
    for _ in range(30):
        engine.render()
    assert not engine._occ_active and slot._occ_db == 0.0, "切っても戻らない"
    return (f"コンクリートの壁の向こうで {drop:.1f}dB。鳴り始めから {first:g}dB、"
            f"動くと1ブロック {step:.1f}dB ずつ戻る。戸を閉めると木の戸")


@test("遮蔽: 手で決めたこもりと合わさり、誤った指定は断る")
def t_occ_controls() -> str:
    from audiospace import Opening
    spaces, doors = _occ_world()
    engine = make_engine()
    engine.set_spaces(spaces, early=False, late=False)
    for bad, why in (({"openings": ["戸"]}, "Opening でない"),
                     ({"openings": doors * 2}, "名前の重なり"),
                     ({"strength": -1.0}, "負の強さ"),
                     ({"materials": {"wood_16mm": (5.0, 2.0)}}, "こもり 2")):
        try:
            engine.set_auto_occlusion(**bad)
            raise AssertionError(f"{why}を受け付けた")
        except ValueError:
            pass
    try:
        engine.set_opening("doorA")
        raise AssertionError("切っているのに戸を開け閉めできた")
    except ValueError:
        pass
    engine.set_auto_occlusion(openings=doors)
    try:
        engine.set_opening("無い戸")
        raise AssertionError("無い戸を受け付けた")
    except KeyError:
        pass
    try:
        Opening((0, 0, 0), (-1, 1, 1))
        raise AssertionError("角の逆な穴を受け付けた")
    except ValueError:
        pass
    engine.set_listener((2.5, 2.0, 1.6))
    h = engine.add_source(BufferSource(np.full(SR, 0.1, np.float32),
                                       loop=True),
                          position=(7.5, 2.0, 1.6), lowpass=0.5)
    engine.render()
    slot = engine.get_source(h)
    bucket = engine._lp_bucket[engine._slot_of_row.index(h)]
    want = 1.0 - (1.0 - 0.5) * (1.0 - slot._occ_lp)
    assert bucket == round(want * (engine.LP_BUCKETS - 1)) or bucket >= \
        round(want * (engine.LP_BUCKETS - 1)) - 1, (bucket, want)
    return "手のこもり 0.5 と壁のこもりが 1-(1-a)(1-b) で合わさる。誤りは ValueError"


@test("遮蔽: プレイヤーの set_auto_occlusion・set_opening・sound_occlusion")
def t_occ_player() -> str:
    spaces, doors = _occ_world()
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        player.set_spaces(spaces, early=False, late=False)
        player.set_auto_occlusion(openings=doors)
        player.set_listener((2.5, 2.0, 1.6))
        a = player.play(np.full(SR, 0.1, np.float32), loop=True,
                        position=(7.5, 2.0, 1.6))
        b = player.play(np.full(SR, 0.1, np.float32), loop=True,
                        position=(7.5, 2.0, 1.6), occlude=False)
        deadline = time.time() + 2.0
        while (player.sound_occlusion(a).loss_db == 0.0
               and time.time() < deadline):
            time.sleep(0.01)
        assert player.sound_occlusion(a).walls == ("rough_concrete",)
        assert player.sound_occlusion(b).loss_db == 0.0
        player.set_opening("doorA", "wooden_door")
        assert player.engine.auto_occlusion
    finally:
        player.stop()
    return "壁の向こうの音にコンクリートの壁、occlude=False の音はそのまま"


# -- 耳介の選定（2026-11-18） ----------------------------------------------
@test("耳介の選定: viking_candidates と load_pinna（npz を優先し、レートを合わせる）")
def t_pinna_candidates() -> str:
    import shutil

    from audiospace import load_pinna, viking_candidates
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in ("subj_B.sofa", "subj_A.sofa", "subj_Z.sofa"):
            (folder / name).write_bytes(b"")
        asset = Path(__file__).resolve().parents[1] / "audiospace" / "data" \
            / "default_horizontal.npz"
        shutil.copy(asset, folder / "subj_B.npz")
        found = viking_candidates(folder)
        names = [(Path(tag).name, label) for tag, label in found]
        assert names == [("subj_A.sofa", "実測 Viking 耳介A"),
                         ("subj_B.npz", "実測 Viking 耳介B"),
                         ("subj_Z.sofa", "実測 Viking 耳介なし")], names
        hrir = load_pinna(found[1][0], 44100)
        assert hrir.sample_rate == 44100 and hrir.n_directions == 216
    try:
        viking_candidates(Path(tmp) / "無い")
        raise AssertionError("無いフォルダで例外にならない")
    except FileNotFoundError:
        pass
    return "A・B（npz）・耳介なしの順。npz は 44100Hz に合わせて読める。無いフォルダは例外"


def _pinna_world():
    """読み込みの代わりに、耳介ごとに方向の数の違う合成 HRTF を返す。"""
    sizes = {"a.sofa": 36, "b.sofa": 40, "c.sofa": 44, "d.sofa": 48}
    loads: list[str] = []

    def loader(tag, rate):
        loads.append(Path(tag).name)
        return HrirSet.synthetic(n_azimuth=sizes[Path(tag).name],
                                 sample_rate=rate)

    candidates = [(f"pinnae/{n}", f"耳介{n[0].upper()}") for n in sizes]
    return candidates, loader, loads, sizes


@test("耳介の選定: PinnaSelection が切り替え・出題・記録・元に戻すまで進める")
def t_pinna_selection() -> str:
    from audiospace import Difficulty, PinnaSelection, RecordStore
    candidates, loader, loads, sizes = _pinna_world()
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    original = player.engine.hrir
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp) / "pinna.json")
            sel = PinnaSelection(player, candidates[:3], store,
                                 difficulty=Difficulty.EIGHT, seed=3,
                                 loader=loader)
            assert len(sel.lineup) == 3 and sel.anchor is None
            assert "3種類をそれぞれ16問ずつ" in sel.intro()
            try:
                sel.finish()
                raise AssertionError("終わる前に記録できた")
            except RuntimeError:
                pass
            switches = 0
            while not sel.finished:
                if sel.prepare():
                    switches += 1
                    n = sizes[Path(sel.session.current_tag).name]
                    assert player.engine.hrir.n_directions == n
                trial = sel.current()
                handle = sel.play(np.full(480, 0.1, np.float32))
                slot = player.engine.get_source(handle)
                want = np.array(position_for_test(trial.truth_deg, 4.0))
                assert slot.relative and np.allclose(slot.position, want)
                good = sel.current_label == "耳介B"
                sel.answer(trial.truth_deg if good
                           else (180.0 - trial.truth_deg) % 360.0)
            assert switches == 3 and loads == ["a.sofa", "b.sofa", "c.sofa"]
            text = sel.finish()
            assert player.engine.hrir is original, "元の HRTF に戻らない"
            assert "通算の最良は耳介B" in text, text
            assert sel.best() == ("pinnae/b.sofa", "耳介B")
            assert (Path(tmp) / "pinna.json").exists(), "保存していない"
            # 2回目は最良（B）が基準として最初に入る。途中でやめると記録しない
            sel2 = PinnaSelection(player, candidates, store, seed=4,
                                  loader=loader)
            assert sel2.anchor == "pinnae/b.sofa"
            assert sel2.lineup[0] == ("pinnae/b.sofa", "耳介B")
            before = store.records["b.sofa"].n_answered
            sel2.prepare()
            sel2.answer(sel2.current().truth_deg)
            sel2.cancel()
            assert player.engine.hrir is original
            assert store.records["b.sofa"].n_answered == before
    finally:
        player.stop()
    return ("3種類×16問で切り替え3回、Bだけ正しく答えると通算の最良がB。2回目は"
            "Bが基準で最初に入る。途中でやめると記録しない")


def position_for_test(azimuth_deg: float, distance_m: float):
    from audiospace.quiz import position_for
    return position_for(azimuth_deg, distance_m)


@test("耳介の選定: 上下の選定は高さで答え、上下の測定が無い耳介は断る")
def t_pinna_elevation() -> str:
    from audiospace import MODE_ELEVATION, PinnaSelection, RecordStore
    from audiospace.quiz import ELEVATION_LEVELS
    candidates, loader, _, _ = _pinna_world()
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = RecordStore(Path(tmp) / "elev.json")
            flat = PinnaSelection(player, candidates[:2], store,
                                  mode=MODE_ELEVATION, loader=loader)
            try:
                flat.prepare()
                raise AssertionError("水平面だけの耳介で上下の選定ができた")
            except ValueError:
                pass
            sphere = default_hrir_sphere()
            sel = PinnaSelection(player, candidates[:2], store,
                                 mode=MODE_ELEVATION, seed=2,
                                 loader=lambda tag, rate: sphere)
            assert sel.session.trials_per_candidate == 12
            while not sel.finished:
                trial = sel.current()
                handle = sel.play(np.full(480, 0.1, np.float32))
                z = player.engine.get_source(handle).position[2]
                assert abs(z - 4.0 * np.sin(np.radians(trial.truth_el))) < 1e-9
                sel.answer_elevation(trial.truth_el)
            text = sel.finish()
            assert "上下で最も成績が良かった" in text or "同成績" in text, text
            assert store.records, "上下の記録が積まれていない"
            assert {el for el, _ in ELEVATION_LEVELS} == {
                t.truth_el for t in sel.session.trials}
    finally:
        player.stop()
    return "上下は4つの高さを3回ずつ12問、高さの位置で鳴らす。水平面だけの耳介は ValueError"


def default_hrir_sphere():
    from audiospace import default_hrir
    return default_hrir(SR, full_sphere=True)


# -- 2026-11-18 のレビューの指摘 ----------------------------------------------
@test("review4: キャッシュに入った効果音は、play のたびにファイルを開かない")
def t_review4_cached_no_open() -> str:
    import soundfile as sf

    import audiospace.sources as sources_mod
    player = SpatialPlayer.open(block=BLOCK, max_sources=6, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    opens = {"n": 0}
    real = sources_mod._open_reader

    def counting(path, backend=None):
        opens["n"] += 1
        return real(path, backend)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            step = Path(tmp) / "step.wav"
            sf.write(step, np.zeros(4800, np.float32), SR, subtype="FLOAT")
            sources_mod._open_reader = counting
            for _ in range(50):
                player.stop_sound(player.play(step))
            # 相対のパスでも同じキャッシュに当たる
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                player.stop_sound(player.play("step.wav"))
            finally:
                os.chdir(cwd)
    finally:
        sources_mod._open_reader = real
        player.stop()
    assert opens["n"] == 1, f"ファイルを {opens['n']} 回開いた"
    assert len(player._sound_cache) == 1, player._sound_cache.keys()
    return "51回鳴らして、ファイルを開いたのは最初の1回だけ（相対のパスも同じキャッシュ）"


@test("review4: 少しずつ読む音も、ループの始めが終わり以降なら終わりの手前に直す")
def t_review4_stream_loop_clamp() -> str:
    import soundfile as sf

    from audiospace import FileStreamSource
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        seg = np.zeros((1000, 1), np.float32)
        seg[::100] = 0.5
        seg[499] = 0.25
        path = Path(tmp) / "loop.wav"
        sf.write(path, seg, SR, subtype="FLOAT")
        stream = FileStreamSource(path, SR, loop=True, loop_start=900,
                                  loop_end=500)
        whole = BufferSource.from_file(path, loop=True, loop_start=900,
                                       loop_end=500)
        got = _drain(stream, limit=8)
        want = np.concatenate([whole.read(BLOCK) for _ in range(8)])
        stream.close()
    assert not stream.finished or got.size, "鳴り終わった"
    assert np.array_equal(got[:len(want)], want), "BufferSource と違う"
    return "loop_start=900・loop_end=500 で、BufferSource と同じ音（499 から繰り返す）"


@test("review4: stream=True はファイルにだけ使え、配列や音源には ValueError")
def t_review4_stream_non_file() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        for sound in (np.zeros(480, np.float32),
                      BufferSource(np.zeros(480, np.float32))):
            try:
                player.play(sound, stream=True)
                raise AssertionError(f"{type(sound).__name__} に stream=True")
            except ValueError:
                pass
        player.stop_sound(player.play(np.zeros(480, np.float32),
                                      stream=False))
    finally:
        player.stop()
    return "配列と BufferSource は ValueError、stream=False は今までどおり"


@test("review4: 押し込み音源は、中身があるときだけダッキングのきっかけになる")
def t_review4_push_ducking() -> str:
    engine = make_engine(limiter=False)
    _flat_dc(engine, -1.0, group="music")
    push = PushSource(SR)
    engine.add_source(push, spatialise=False, group="voice")
    engine.set_ducking("music", by="voice", db=-12.0, attack_ms=50.0,
                       release_ms=100.0)
    _levels(engine, 30)
    idle = engine.group_level("music")
    assert abs(idle - 1.0) < 1e-9, f"何も書いていないのに下がった: {idle}"
    push.write(np.full(SR // 2, 0.1, np.float32))
    _levels(engine, 10)
    duck = 10 ** (-12 / 20)
    assert abs(engine.group_level("music") - duck) < 1e-9
    _levels(engine, 60)                 # 0.5秒の声を鳴らし終えて、戻る
    assert abs(engine.group_level("music") - 1.0) < 1e-9
    return "空の間は 1.0、書くと -12dB、鳴らし終えると 1.0 に戻る"


@test("review4: set_space だけでも自動の遮蔽が効き、空間が無ければ status() が言う")
def t_review4_set_space_occlusion() -> str:
    from audiospace import Space
    walls = {f: "rough_concrete" for f in ("west", "east", "south", "north",
                                           "floor", "ceiling")}
    room = Space((0, 0, 0), (5, 4, 3), dict(walls))
    engine = make_engine()
    engine.set_space(room)
    engine.set_auto_occlusion()
    engine.set_listener((2.5, 2.0, 1.6))
    h = engine.add_source(BufferSource(np.full(SR, 0.1, np.float32),
                                       loop=True), position=(9.0, 2.0, 1.6))
    engine.render()
    assert engine.occlusion_spaces == 1
    assert engine.source_occlusion(h).walls == ("rough_concrete",)
    engine.set_space(None)               # 空間を外すと、遮蔽も外れる
    engine.render()
    assert engine.occlusion_spaces == 0
    assert engine.source_occlusion(h).loss_db == 0.0
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        player.set_auto_occlusion()
        assert "空間がありません" in player.status(), player.status()
    finally:
        player.stop()
    return "set_space の部屋の外の音はコンクリートの壁。空間が無いと status() が知らせる"


@test("review5: close() は少しずつ読む音のファイルを閉じる（stop() は残す）")
def t_review5_close_releases_stream() -> str:
    player = SpatialPlayer.open(block=BLOCK, max_sources=4, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        wav = Path(tmp) / "bgm.wav"
        _write_noise(wav, 3.0, SR)
        ended = []
        h = player.play(wav, spatialise=False, loop=True, stream=True,
                        on_end=ended.append)
        thread = player.engine.get_source(h).source._thread
        assert thread is not None and thread.is_alive()
        player.stop()
        assert player.engine.active_sources == 1, "stop() で音が外れた"
        player.start()                       # stop() のあと、続きから鳴らせる
        player.close()
        assert player.engine.active_sources == 0
        assert not thread.is_alive(), "読み込みのスレッドが残った"
        wav.unlink()                         # 開いたままなら PermissionError
        player.poll_ends()
        assert [e.reason for e in ended] == ["stopped"], ended
        player.close()                       # 2回呼んでもよい
    return "close() のあと、スレッドは終わり、ファイルを消せた。stop() は音を残す"


@test("review5: 関数の音源は、無音を返している間はダッキングのきっかけにならない")
def t_review5_callback_ducking() -> str:
    from audiospace import CallbackSource
    engine = make_engine()
    level = {"v": 0.0}
    src = CallbackSource(lambda n: np.full(n, level["v"], np.float32))
    engine.add_source(BufferSource(np.full(SR, 0.1, np.float32), loop=True),
                      position=(0.0, 2.0, 0.0), group="music")
    engine.add_source(src, position=(0.0, 2.0, 0.0), group="voice")
    engine.set_ducking("music", by="voice", db=-12, attack_ms=20,
                       release_ms=20)

    def gain() -> float:
        for _ in range(20):
            engine.render()
        return engine.group_level("music")

    quiet = gain()
    assert abs(quiet - 1.0) < 1e-6, f"無音なのに下げた: {quiet}"
    level["v"] = 0.2
    loud = gain()
    assert abs(loud - 10 ** (-12 / 20)) < 1e-3, loud
    level["v"] = 0.0
    assert abs(gain() - 1.0) < 1e-6
    level["v"] = float("nan")            # 壊れた値で例外にならない
    src.read(64)
    return f"無音の間は 1.000、鳴っている間は {loud:.3f}"


@test("review5: 瞬間移動の直後、自動の遮蔽はその場で行き先の値になる")
def t_review5_teleport_occlusion_jumps() -> str:
    from audiospace import Space
    walls = {f: "rough_concrete" for f in ("west", "east", "south", "north",
                                           "floor", "ceiling")}
    engine = SpatialEngine(HrirSet.synthetic(sample_rate=SR), block=BLOCK,
                           max_sources=24)
    engine.set_spaces([Space((0, 0, 0), (5, 4, 3), dict(walls))],
                      early=False, late=False)
    engine.set_auto_occlusion()
    engine.set_listener((2.5, 2.0, 1.6))
    handles = [engine.add_source(
        BufferSource(np.full(SR, 0.1, np.float32), loop=True),
        position=(1.0 + 0.1 * i, 2.0, 1.6)) for i in range(20)]
    for _ in range(5):
        engine.render()
    slots = [engine.get_source(h) for h in handles]
    assert all(s._occ_db == 0.0 for s in slots)
    engine.teleport_listener((9.0, 2.0, 1.6))        # 部屋の外へ
    engine.render()
    want = slots[0]._occ.loss_db
    assert want > 10.0
    # 1ブロックに調べる数（16）より多い音も、全部その場で変わる
    assert all(s._occ_db == want for s in slots), [s._occ_db for s in slots]
    engine.set_listener((9.5, 2.0, 1.6))
    engine.teleport_listener((2.5, 2.0, 1.6))        # 中へ戻る
    engine.render()
    assert all(s._occ_db == 0.0 for s in slots)
    # 歩いて出るときは、今までどおりなめらかに変わる
    engine.set_listener((9.0, 2.0, 1.6))
    engine.render()
    assert 0.0 < slots[0]._occ_db < want
    return f"20音とも1ブロックで {want:.1f}dB。歩いたときはなめらか"


@test("review5: 閉じた戸の近道があっても、開いた戸口を回る道を使う")
def t_review5_route_avoids_closed_door() -> str:
    from audiospace import Opening, Space
    from audiospace.occlusion import OcclusionMap
    walls = {f: "rough_concrete" for f in ("west", "east", "south", "north",
                                           "floor", "ceiling")}
    spaces = [Space((0, 0, 0), (5, 4, 3), dict(walls), name="A"),
              Space((5, 0, 0), (5, 4, 3), dict(walls), name="B"),
              Space((0, 4, 0), (10, 2, 3), dict(walls), name="C")]
    shut = Opening((4.9, 0.5, 0), (5.1, 1.5, 2), "rough_concrete", "AB")
    ways = [Opening((2.0, 3.9, 0), (3.0, 4.1, 2), None, "AC"),
            Opening((7.0, 3.9, 0), (8.0, 4.1, 2), None, "BC")]
    src, lis = (7.5, 3.0, 1.6), (2.5, 3.0, 1.6)
    both = OcclusionMap(spaces, [shut] + ways).query([src], lis)[0]
    only = OcclusionMap(spaces, ways).query([src], lis)[0]
    assert both.via == ("BC", "AC"), both
    assert abs(both.loss_db - only.loss_db) < 1e-9
    # 戸を開けると近道になり、閉めるとまた回る（道の覚えを捨てている）
    m = OcclusionMap(spaces, [shut] + ways)
    m.query([src], lis)
    m.set_opening_material("AB", None)
    opened = m.query([src], lis)[0]
    assert opened.via == ("AB",), opened
    m.set_opening_material("AB", "rough_concrete")
    again = m.query([src], lis)[0]
    assert again.via == ("BC", "AC"), again
    return (f"閉じた戸があっても {both.loss_db:.1f}dB（開いた遠回り）。"
            f"戸を開けると {opened.loss_db:.1f}dB")

# -- 2026-11-18 の古いモジュールのレビュー -----------------------------------
@test("review6: レートを変換した HRIR と残響の IR は、フィルタとしての利得が変わらない")
def t_review6_resample_gain() -> str:
    from audiospace.precise import SpaceIr
    from audiospace.presets import BUILTIN_PRESETS, RoomIr
    from audiospace.reverb import ConvolutionReverb
    rng = np.random.default_rng(3)
    # 低い周波数に寄せた応答（高い側はレートを下げると切れるので比べない）
    base = np.cumsum(rng.standard_normal((2, 400)), axis=1)
    base *= np.hanning(400)[None, :]
    base = base.astype(np.float32)
    want = base.sum(axis=1)
    hrir = HrirSet(base[None], np.zeros(1), np.zeros(1), 48000)
    tail = SpaceIr("k", "n", base, 48000)
    room = RoomIr(BUILTIN_PRESETS["hall"], base, 48000)
    worst = 0.0
    for rate in (44100, 96000, 192000):
        for got in (hrir.resampled(rate).data[0], tail.resampled(rate).tail,
                    room.resampled(rate).ir):
            ratio = got.sum(axis=1) / want
            worst = max(worst, float(np.max(np.abs(ratio - 1.0))))
    assert worst < 0.02, f"係数の和が変わった: {worst:.3f}"
    assert hrir.resampled(48000) is hrir          # 同じレートは何もしない
    # 正規化: レートを渡すと、48kHz で 1、96kHz で半分のエネルギーになる
    for rate, energy in ((48000, 1.0), (96000, 0.5), (44100, 48000 / 44100)):
        rev = ConvolutionReverb(room.resampled(rate).ir, block=BLOCK,
                                sample_rate=rate)
        got = _reverb_energy(rev)
        assert abs(got / energy - 1.0) < 0.02, (rate, got)
    plain = _reverb_energy(ConvolutionReverb(room.resampled(96000).ir,
                                             block=BLOCK))
    assert abs(plain - 1.0) < 0.02                # 渡さなければ今までどおり
    return f"3つのレートで、係数の和のずれは最大 {worst * 100:.1f}%"


def _reverb_energy(rev) -> float:
    """残響にインパルスを入れたときの応答のエネルギー。"""
    x = np.zeros((40, BLOCK), dtype=np.float32)
    x[0, 0] = 1.0
    y = np.concatenate([rev.process(b) for b in x])
    return float(np.sum(y.astype(np.float64) ** 2))


@test("review6: 精密版の尾は、聞こえる帯域で速い版と大きさがそろう（前の版の尾は読むときに直す）")
def t_review6_tail_level_band() -> str:
    from audiospace.acoustics import space_preset
    from audiospace.precise import (LEVEL_MATCH, SPACE_IR_DIR, SpaceIr,
                                    _band_energy, _fast_tail_energy,
                                    space_key)
    space = space_preset("concrete_room")
    target = _fast_tail_energy(space, SR)
    rng = np.random.default_rng(5)
    raw = (rng.standard_normal((2, SR)) * np.exp(-np.arange(SR) / 9000.0)
           ).astype(np.float32)
    old = SpaceIr(space_key(space), "x", raw, SR, meta={"n_rays": 1})
    new = old.matched(space)
    assert new.meta["level_match"] == LEVEL_MATCH and "level_match" not in old.meta
    assert abs(_band_energy(new.tail, SR) / target - 1.0) < 0.01
    assert new.matched(space) is new, "印のある尾をもう一度直した"
    off = SpaceIr("k", "x", raw, SR, meta={"level_match": "none"})
    assert off.matched(space) is off, "そろえない指定の尾を直した"
    baked = SPACE_IR_DIR / f"{space_key(space)}_{SR}.npz"
    note = "焼き込みは無い"
    if baked.exists():
        tail = SpaceIr.load(baked)
        assert tail.meta.get("level_match") == LEVEL_MATCH, "焼き込みに印が無い"
        ratio = _band_energy(tail.tail, SR) / target
        assert abs(ratio - 1.0) < 0.01, ratio
        note = f"焼き込みの尾と速い版の比 {ratio:.3f}"
    return f"印の無い尾は読むときにそろえ直す。{note}"


@test("review6: 意味のない値（大きさの NaN、基準の距離 0 など）は受け取るときに断る")
def t_review6_reject_bad_values() -> str:
    from audiospace import Space
    walls = {f: "rough_concrete" for f in ("west", "east", "south", "north",
                                           "floor", "ceiling")}
    nan, inf = float("nan"), float("inf")
    bad_spaces = [dict(size=(nan, 4, 3)), dict(size=(5, inf, 3)),
                  dict(size=(5, 4)), dict(origin=(0, nan, 0)),
                  dict(early_gain=nan), dict(late_gain=-1.0),
                  dict(scattering=inf)]
    for kw in bad_spaces:
        args = dict(origin=(0, 0, 0), size=(5, 4, 3), materials=dict(walls))
        args.update(kw)
        try:
            Space(**args)
            raise AssertionError(f"受け付けた: {kw}")
        except ValueError:
            pass
    Space((0, 0, 0), (5, 4, 3), dict(walls), early_gain=0.0, late_gain=0.0)
    hrir = HrirSet.synthetic(sample_rate=SR)
    bad_engines = [dict(reference_distance=0.0), dict(reference_distance=-1.0),
                   dict(reference_distance=nan), dict(max_distance=0.5),
                   dict(max_distance=nan)]
    for kw in bad_engines:
        try:
            SpatialEngine(hrir, block=BLOCK, max_sources=2, **kw)
            raise AssertionError(f"受け付けた: {kw}")
        except ValueError:
            pass
    SpatialEngine(hrir, block=BLOCK, max_sources=2, reference_distance=2.0,
                  max_distance=2.0)
    return (f"空間 {len(bad_spaces)} 通りとエンジン {len(bad_engines)} 通りを断り、"
            "境目の値は通す")


@test("review6: ちょうどのレートが無いとき、求めるレート以上でいちばん近い IR を元にする")
def t_review6_nearest_rate() -> str:
    from audiospace.presets import (BUILTIN_PRESETS, RoomIr, by_nearest_rate,
                                    load_room_ir)
    names = [Path(f"hall_{r}.npz") for r in (44100, 48000, 96000)]
    order = lambda rate: [p.name for p in by_nearest_rate(names, rate)]  # noqa: E731
    assert order(96000)[0] == "hall_96000.npz"
    assert order(88200)[0] == "hall_96000.npz"
    assert order(46000) == ["hall_48000.npz", "hall_96000.npz",
                            "hall_44100.npz"]
    assert order(192000)[0] == "hall_96000.npz"      # 上が無ければいちばん高い
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ir = np.zeros((2, 64), dtype=np.float32)
        ir[:, 0] = 1.0
        for rate in (44100, 48000):
            RoomIr(BUILTIN_PRESETS["hall"], ir, rate).save(
                Path(tmp) / f"hall_{rate}.npz")
        got = load_room_ir("hall", 96000, tmp)
        assert got.meta["resampled_from"] == 48000, got.meta
        assert got.sample_rate == 96000
    return "44.1k と 48k があれば、96k は 48k から作る"


@test("review6: linear の距離の減衰は、rolloff が 1 より大きくても負にならない")
def t_review6_linear_not_negative() -> str:
    from audiospace.geometry import distance_gain, distance_gain_many
    ds = np.array([0.5, 1.0, 10.0, 30.5, 40.0, 59.0, 60.0, 80.0])
    many = distance_gain_many(ds, "linear", 1.0, 60.0, 2.0)
    one = np.array([distance_gain(float(d), "linear", 1.0, 60.0, 2.0)
                    for d in ds])
    assert np.all(many >= 0.0) and np.all(one >= 0.0), (many, one)
    assert np.allclose(many, one)
    assert many[0] == 1.0 and many[1] == 1.0 and many[-1] == 0.0
    assert many[2] > 0.0 and many[4] == 0.0       # 30.5m で 0 に届く
    # rolloff が 1 のときは今までどおり
    plain = distance_gain_many(ds, "linear", 1.0, 60.0, 1.0)
    assert abs(plain[4] - (1.0 - 39.0 / 59.0)) < 1e-12
    return f"40m で {many[4]:.3f}（直す前は -0.322）。rolloff 1 は変わらない"


@test("review6: 精密版の尾と核の覚えは上限までで、長く使っていないものから捨てる")
def t_review6_precise_memory_bound() -> str:
    import audiospace.precise as precise
    from audiospace.precise import SpaceIr
    from audiospace.reverb import ConvolutionReverb
    saved = dict(precise._MEMORY)
    precise._MEMORY.clear()
    try:
        tail = np.zeros((2, 64), dtype=np.float32)
        tail[:, 0] = 1.0
        n = precise.MEMORY_MAX
        for i in range(n):
            precise._remember((f"k{i}", SR), SpaceIr(f"k{i}", "", tail, SR))
        assert len(precise._MEMORY) == n
        precise._MEMORY.move_to_end(("k0", SR))     # k0 を使った
        precise._remember(("new", SR), SpaceIr("new", "", tail, SR))
        assert len(precise._MEMORY) == n
        assert ("k0", SR) in precise._MEMORY, "使ったばかりの尾を捨てた"
        assert ("k1", SR) not in precise._MEMORY, "いちばん古い尾が残った"
    finally:
        precise._MEMORY.clear()
        precise._MEMORY.update(saved)
    engine = make_engine()
    cap = engine.PRECISE_KERNELS_MAX
    kernel = ConvolutionReverb(tail, block=BLOCK, normalise=False)
    for i in range(cap):
        engine._keep_kernel(f"k{i}", kernel)
    engine._keep_kernel("k0", kernel)               # k0 を使った
    engine._keep_kernel("new", kernel)
    keys = list(engine._precise_kernels)
    assert len(keys) == cap and "k0" in keys and "k1" not in keys, keys
    return f"尾は {n} 個、核は {cap} 個まで。使ったものは残る"

@test("review4: 遮蔽の地図はロックの外で作り、戸の開け閉めは作り直さない")
def t_review4_occlusion_map_outside_lock() -> str:
    import audiospace.engine as engine_mod
    from audiospace import Opening
    from audiospace.occlusion import OcclusionMap
    spaces, doors = _occ_world()
    engine = make_engine()
    engine.set_spaces(spaces, early=False, late=False)
    held = []
    calls = {"n": 0}

    class Spy(OcclusionMap):
        def __init__(self, *args, **kwargs):
            # ほかのスレッドからエンジンのロックを取れるなら、持っていない
            got = []

            def probe():
                ok = engine._lock.acquire(timeout=1.0)
                got.append(ok)
                if ok:
                    engine._lock.release()      # 取ったスレッドで返す

            t = threading.Thread(target=probe)
            t.start()
            t.join()
            held.append(not (got and got[0]))
            calls["n"] += 1
            if calls["n"] == 1:
                # 作っている途中で戸を閉める。作り終えたら作り直すはず
                engine.set_opening("doorA", "wooden_door")
            super().__init__(*args, **kwargs)

    real = engine_mod.OcclusionMap
    engine_mod.OcclusionMap = Spy
    try:
        engine.set_auto_occlusion(openings=doors)
    finally:
        engine_mod.OcclusionMap = real
    assert held and not any(held), "地図をロックの中で作った"
    assert calls["n"] == 2, f"途中の変更で作り直していない: {calls['n']}"
    closed = engine._occ_map.openings[0].material
    assert closed == "wooden_door", closed
    # 戸の開け閉めは地図を作り直さず、作り直した地図と同じ答えになる
    before = engine._occ_map
    engine.set_opening("doorA")
    assert engine._occ_map is before
    lis = (2.5, 2.0, 1.6)
    points = [(2.5, 5.0, 1.6), (8.0, 5.0, 1.6), (7.5, 2.0, 1.6)]
    for material in ("wooden_door", None):
        engine.set_opening("doorA", material)
        fresh = OcclusionMap(spaces, [Opening(doors[0].lo, doors[0].hi,
                                              material, "doorA")])
        assert engine._occ_map.query(points, lis) == fresh.query(points, lis)
    return ("地図を作る間、ほかのスレッドがロックを取れた。途中の戸の変更で作り直し、"
            "戸の開け閉めは値だけ替えて、作り直したのと同じ答え")


# -- 長いファイルを少しずつ読む（2026-11-18） ------------------------------
def _drain(src, block: int = BLOCK, limit: int = 20000) -> np.ndarray:
    """読み込みが追いつくのを待ちながら、最後まで読む（途切れを出さない）。"""
    out = []
    for _ in range(limit):
        st = src._state
        deadline = time.time() + 2.0
        while (st.written - st.consumed < block and not st.drained
               and time.time() < deadline):
            time.sleep(0.0005)
        out.append(src.read(block))
        if src.finished:
            break
    return np.concatenate(out)


def _write_noise(path: Path, seconds: float, rate: int, channels: int = 2,
                 seed: int = 0) -> np.ndarray:
    import soundfile as sf
    x = (0.1 * np.random.default_rng(seed).standard_normal(
        (int(seconds * rate), channels))).astype(np.float32)
    sf.write(path, x, rate, subtype="FLOAT")
    return x


@test("少しずつ読む: 丸ごと読んだ音と同じ（レートの変換も、miniaudio でも）")
def t_stream_matches_decode() -> str:
    from audiospace import FileStreamSource, load_mono, load_stereo
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        f44 = Path(tmp) / "a44.wav"
        f48 = Path(tmp) / "a48.wav"
        _write_noise(f44, 3.0, 44100)
        x48 = _write_noise(f48, 2.0, SR, seed=1)
        diffs = []
        for channels, loader in ((2, load_stereo), (1, load_mono)):
            src = FileStreamSource(f44, SR, channels=channels)
            got = _drain(src)
            want, _ = loader(f44, SR)
            n = min(len(got), len(want)) - 200
            diff = float(np.max(np.abs(got[:n] - want[:n])))
            assert diff < 1e-6 and src.underruns == 0, (channels, diff)
            diffs.append(diff)
        for backend in (None, "miniaudio"):
            src = FileStreamSource(f48, SR, channels=2, _backend=backend)
            got = _drain(src)
            assert np.array_equal(got[:len(x48)], x48), backend
            assert not np.any(got[len(x48):]), "終わりの後に音がある"
    return (f"44.1kHz→48kHz の差 ステレオ {diffs[0]:.1e}・モノラル {diffs[1]:.1e}。"
            "同じレートは soundfile・miniaudio ともビット単位で同じ")


@test("少しずつ読む: ループの区間、捨てるとスレッドが終わる、間に合わないと数える")
def t_stream_lifecycle() -> str:
    import gc

    import soundfile as sf

    from audiospace import FileStreamSource

    def streams() -> int:
        return sum(1 for t in threading.enumerate()
                   if t.name == "audiospace-stream" and t.is_alive())

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        seg = np.zeros((1000, 1), np.float32)
        seg[0] = 0.5
        seg[600] = 0.25
        sf.write(Path(tmp) / "loop.wav", seg, SR, subtype="FLOAT")
        src = FileStreamSource(Path(tmp) / "loop.wav", SR, loop=True,
                               loop_start=100, loop_end=800)
        got = _drain(src, limit=10)
        hits = np.flatnonzero(np.abs(got) > 1e-6)[:6].tolist()
        assert hits == [0, 600, 1300, 2000, 2700, 3400], hits
        before = streams()
        del src
        gc.collect()
        deadline = time.time() + 2.0
        while streams() >= before and time.time() < deadline:
            time.sleep(0.02)
        assert streams() < before, "捨てた音源の読み込みのスレッドが残った"
        long_file = Path(tmp) / "long.wav"
        _write_noise(long_file, 3.0, SR)
        fast = FileStreamSource(long_file, SR, buffer_seconds=0.1)
        for _ in range(100):
            fast.read(BLOCK)
        assert fast.underruns > 0, "間に合わなかったのに数えない"
        fast.close()
        assert fast.finished and not np.any(fast.read(BLOCK))
    return (f"700フレームの区間を正しく繰り返す。捨てるとスレッドが終わる。小さい"
            f"バッファで急いで読むと途切れ {fast.underruns} 回")


@test("少しずつ読む: 途中で読めなくなったら、その音だけ error で外れる")
def t_stream_error() -> str:
    import audiospace.sources as sources_mod
    from audiospace import SOUND_ERROR, FileStreamSource

    class Broken:
        rate, channels, frames = SR, 1, SR * 10

        def __init__(self) -> None:
            self.reads = 0

        def read(self, n):
            self.reads += 1
            if self.reads > 12:
                raise OSError("ディスクが外れた")
            return np.full((n, 1), 0.1, np.float32)

        def seek(self, frame):
            pass

        def close(self):
            pass

    real = sources_mod._open_reader
    sources_mod._open_reader = lambda path, backend=None: Broken()
    try:
        src = FileStreamSource(Path(__file__), SR)
    finally:
        sources_mod._open_reader = real
    engine = make_engine()
    h = engine.add_source(src, position=(0.0, 2.0, 0.0), notify_end=True)
    for _ in range(400):
        engine.render()
        time.sleep(0.001)
        if h not in engine._slots:
            break
    events = engine.take_ended()
    assert [e.reason for e in events] == [SOUND_ERROR], events
    assert "ディスクが外れた" in events[0].detail
    return f"読めた分を鳴らした後に外れ、理由は「{events[0].detail}」"


@test("少しずつ読む: play は大きいファイルだけ少しずつ読む（stream で選べる）")
def t_stream_player() -> str:
    from audiospace import SOUND_FINISHED, FileStreamSource
    player = SpatialPlayer.open(block=BLOCK, max_sources=6, reverb=None,
                                hrir=HrirSet.synthetic(sample_rate=SR))
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            big = Path(tmp) / "big.wav"
            _write_noise(big, 3.0, 44100)
            small = Path(tmp) / "small.wav"
            _write_noise(small, 0.3, SR)

            def kind(**kw) -> str:
                h = player.play(big, spatialise=False, **kw)
                name = type(player.engine.get_source(h).source).__name__
                player.stop_sound(h)
                return name

            assert kind() == "StereoBufferSource", "16MB 未満なのに少しずつ読んだ"
            player.sound_cache_max_bytes = 100_000      # 大きいファイルの扱い
            # キャッシュに入った音は、そのまま使う（ファイルを開いて調べない）。
            # 入っていなければ少しずつ読む
            assert kind() == "StereoBufferSource"
            player.clear_sound_cache()
            assert kind() == "FileStreamSource"
            assert kind(stream=False) == "StereoBufferSource"
            spatial = player.play(big, position=(0.0, 2.0, 0.0))
            assert player.engine.get_source(spatial).source.channels == 1
            player.stop_sound(spatial)
            try:
                player.play(big, stream=True, rate=1.2)
                raise AssertionError("stream=True と rate を受け付けた")
            except ValueError:
                pass
            ended = []
            h = player.play(small, spatialise=False, stream=True,
                            on_end=ended.append)
            assert isinstance(player.engine.get_source(h).source,
                              FileStreamSource)
            deadline = time.time() + 3.0
            while not ended and time.time() < deadline:
                player.poll_ends()
                time.sleep(0.01)
            assert [e.reason for e in ended] == [SOUND_FINISHED], ended
            looping = player.play(big, spatialise=False, loop=True)
            src = player.engine.get_source(looping).source
            src._state.underruns = 3
            assert "ファイルの読み込みが間に合わず無音3ブロック" in player.status()
            player.stop_sound(looping)
            src.close()
    finally:
        player.stop()
    return ("既定は 16MB を超えるファイルだけ。stream=True/False で選べ、rate と"
            "一緒は ValueError。鳴り終わりの知らせも届く")


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
    # 2026-11-17 のレビュー対応
    t_review3_enter_from_nowhere, t_review3_source_error_isolated,
    t_review3_reject_nonfinite, t_review3_band_reverb_block,
    t_review3_row_reuse, t_review3_decode_error, t_review3_block_mismatch,
    t_review3_reverb_spec_sep, t_review3_table_bytes,
    # 出力デバイスの見張り（2026-11-17）
    t_device_poll, t_device_watch,
    # 全天球の既定 HRTF（2026-11-17）
    t_default_sphere, t_default_sphere_engine, t_player_full_sphere,
    # 鳴らし始めの時刻の指定（2026-11-18）
    t_delay_exact, t_delay_matches_padding, t_delay_together,
    t_delay_together_error, t_delay_wait_controls, t_delay_player,
    # 音のグループ（2026-11-18）
    t_group_gain, t_group_pause, t_group_stop, t_group_ducking,
    t_group_player,
    # 鳴り終わりの知らせ（2026-11-18）
    t_end_finished, t_end_stopped, t_end_player_poll, t_end_player_watch,
    # 部屋をまたぐ音の自動の遮蔽（2026-11-18）
    t_occ_geometry, t_occ_engine, t_occ_controls, t_occ_player,
    # 耳介の選定（2026-11-18）
    t_pinna_candidates, t_pinna_selection, t_pinna_elevation,
    # 長いファイルを少しずつ読む（2026-11-18）
    t_stream_matches_decode, t_stream_lifecycle, t_stream_error,
    t_stream_player,
    # 2026-11-18 のレビューの指摘
    t_review4_cached_no_open, t_review4_stream_loop_clamp,
    t_review4_stream_non_file, t_review4_push_ducking,
    t_review4_set_space_occlusion, t_review4_occlusion_map_outside_lock,
    # 2026-11-18 の全体のレビューの指摘
    t_review5_close_releases_stream, t_review5_teleport_occlusion_jumps,
    t_review5_route_avoids_closed_door, t_review5_callback_ducking,
    # 2026-11-18 の古いモジュールのレビュー
    t_review6_resample_gain, t_review6_tail_level_band,
    t_review6_reject_bad_values, t_review6_nearest_rate,
    t_review6_linear_not_negative, t_review6_precise_memory_bound,
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
