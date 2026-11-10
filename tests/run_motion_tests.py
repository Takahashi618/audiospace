"""首振り（動的手がかり）のテスト。

肝心なのは「前の音源と後ろの音源で、首振りに対する音の動き方が逆になる」
こと。これが成立していれば、耳介の個人差と無関係に前後が分かる。

Run: set AUDIOSPACE_SILENT=1 && python -X utf8 tests\\run_motion_tests.py
Out: tests/motion_test_result.txt (UTF-8)
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

os.environ["AUDIOSPACE_SILENT"] = "1"
# 表示の言語を日本語に固定する（テストが期待する文は日本語）
os.environ["AUDIOSPACE_LANG"] = "ja"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import (WOBBLE_PRESETS, BufferSource, HeadWobble,  # noqa: E402
                        HrirSet, SpatialEngine, wobble_labels)

OUT = Path(__file__).with_name("motion_test_result.txt")
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


def hrir() -> HrirSet:
    return HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * seconds)).astype(np.float32)
    return x * (0.4 / float(np.max(np.abs(x))))


@test("HeadWobble: 無効時はオフセット0、有効時は振り幅内に収まる")
def t_range() -> str:
    off = HeadWobble(0.0)
    assert not off.enabled
    assert all(off.offset_at(t) == 0.0 for t in np.linspace(0, 5, 50))

    w = HeadWobble(8.0, 0.5)
    vals = [w.offset_at(t) for t in np.linspace(0, 4, 400)]
    assert max(vals) <= 8.0 + 1e-9 and min(vals) >= -8.0 - 1e-9, \
        f"振り幅を超えました: {min(vals)}..{max(vals)}"
    assert max(vals) > 7.9 and min(vals) < -7.9, "振り幅に届いていません"
    return (f"無効時は常に0、振り幅8度で実際の範囲は "
            f"{min(vals):.2f}..{max(vals):.2f}度")


@test("HeadWobble: 周期が指定どおり")
def t_period() -> str:
    w = HeadWobble(10.0, 0.5)      # 2秒周期
    for t in (0.0, 2.0, 4.0, 6.0):
        assert abs(w.offset_at(t) - w.offset_at(0.0)) < 1e-9, \
            f"t={t} で位相がずれています"
    # 半周期で符号が反転する
    assert w.offset_at(0.5) > 5.0 and w.offset_at(1.5) < -5.0
    return "2秒周期で一致、半周期(1秒)ごとに符号反転を確認"


@test("HeadWobble: 三角波は角速度が一定")
def t_triangle() -> str:
    w = HeadWobble(10.0, 0.5, "triangle")
    ts = np.linspace(0.05, 0.45, 40)
    speeds = np.diff([w.offset_at(float(t)) for t in ts])
    spread = float(np.std(speeds) / max(abs(np.mean(speeds)), 1e-12))
    assert spread < 0.02, f"角速度が一定ではありません (ばらつき{spread:.3f})"
    return f"上り区間の角速度のばらつき {spread * 100:.2f}パーセント"


@test("HeadWobble: 不正な値を弾く")
def t_validation() -> str:
    lines = []
    for kwargs, why in (({"amplitude_deg": -1.0}, "負の振り幅"),
                        ({"rate_hz": 0.0}, "周波数0"),
                        ({"shape": "square"}, "未知の波形")):
        try:
            HeadWobble(**{"amplitude_deg": 5.0, **kwargs})
        except ValueError as exc:
            lines.append(f"{why}: {exc}")
            continue
        raise AssertionError(f"{why} が通ってしまいました")
    return "\n".join(lines)


@test("プリセットがすべて有効で、順に強くなる")
def t_presets() -> str:
    labels = dict(wobble_labels())
    amps = []
    for key, label in wobble_labels():
        assert key in WOBBLE_PRESETS, f"{key} がありません"
        amps.append(WOBBLE_PRESETS[key].amplitude_deg)
    assert amps == sorted(amps), f"強さが単調ではありません: {amps}"
    assert amps[0] == 0.0, "先頭は無効であるべきです"
    return "、".join(f"{labels[k]}={WOBBLE_PRESETS[k].amplitude_deg:g}度"
                     for k, _ in wobble_labels())


@test("エンジン: 首振りで実際に定位が揺れる")
def t_engine_moves() -> str:
    def run(wobble: HeadWobble) -> np.ndarray:
        engine = SpatialEngine(hrir(), block=BLOCK, max_sources=2)
        engine.set_head_wobble(wobble)
        engine.add_source(BufferSource(noise(3.0), loop=True),
                          position=(0.0, 5.0, 0.0), reverb_send=0.0)  # 正面
        out = engine.render_seconds(2.0)
        # ブロックごとの左右差の推移を見る
        n = out.shape[0] // BLOCK
        return np.array([
            float(np.sum(out[i * BLOCK:(i + 1) * BLOCK, 0] ** 2)
                  - np.sum(out[i * BLOCK:(i + 1) * BLOCK, 1] ** 2))
            for i in range(n)])

    still = run(HeadWobble(0.0))
    moving = run(WOBBLE_PRESETS["clear"])
    assert np.std(still) < np.std(moving) * 0.1, \
        f"首振りなしでも揺れています: 静止{np.std(still):.4g} 首振り{np.std(moving):.4g}"
    assert np.std(moving) > 0, "首振りが効いていません"
    return (f"左右差の標準偏差: 静止={np.std(still):.4g}、"
            f"首振りあり={np.std(moving):.4g}")


@test("【核心】前と後ろで首振りに対する動き方が逆になる")
def t_front_back_opposite() -> str:
    """これが成立すれば、耳介の個人差と無関係に前後が分かる。

    首を左に振ると、前方の音源は相対的に右へ動き、後方の音源は左へ動く。
    合成HRIRは前後のスペクトルが完全に同一なので、この差は純粋に
    幾何によるものになる。つまり誰にでも効く手がかりである。
    """
    wobble = HeadWobble(10.0, 0.5)

    def lr_trace(north: float) -> np.ndarray:
        engine = SpatialEngine(hrir(), block=BLOCK, max_sources=2)
        engine.set_head_wobble(wobble)
        engine.add_source(BufferSource(noise(4.0), loop=True),
                          position=(0.0, north, 0.0), reverb_send=0.0)
        out = engine.render_seconds(2.0)          # ちょうど1周期
        n = out.shape[0] // BLOCK
        trace = np.array([
            float(np.sum(out[i * BLOCK:(i + 1) * BLOCK, 0] ** 2)
                  - np.sum(out[i * BLOCK:(i + 1) * BLOCK, 1] ** 2))
            for i in range(n)])
        return trace - trace.mean()

    front = lr_trace(5.0)     # 正面 5m
    back = lr_trace(-5.0)     # 真後ろ 5m
    corr = float(np.corrcoef(front, back)[0, 1])
    assert corr < -0.9, (
        f"前後で動きが逆になっていません (相関 {corr:+.3f})。"
        "これが負の強い値でないと、首振りは前後の手がかりにならない")
    return (f"左右差の時間変化の相関 = {corr:+.3f} "
            "(-1に近いほど完全に逆位相。前が右へ動くとき後ろは左へ動く)")


@test("エンジン: effective_yaw_deg が首振りを含む")
def t_effective_yaw() -> str:
    engine = SpatialEngine(hrir(), block=BLOCK, max_sources=1)
    engine.set_listener((0.0, 0.0, 0.0), yaw_deg=30.0)
    assert engine.effective_yaw_deg == 30.0, "首振りなしでは素の値のはず"
    engine.set_head_wobble(HeadWobble(10.0, 0.5))
    seen = []
    for _ in range(200):
        engine.render()
        seen.append(engine.effective_yaw_deg)
    lo, hi = min(seen), max(seen)
    assert 19.0 < lo < 21.0 and 39.0 < hi < 41.0, \
        f"30度を中心に±10度のはずが {lo:.1f}..{hi:.1f}"
    return f"基準30度に対し実際の範囲 {lo:.1f}..{hi:.1f}度"


@test("エンジン: 個別パラメータだけ差し替えられる")
def t_partial_update() -> str:
    engine = SpatialEngine(hrir(), block=BLOCK, max_sources=1)
    engine.set_head_wobble(HeadWobble(8.0, 0.5, "sine"))
    engine.set_head_wobble(amplitude_deg=12.0)
    w = engine.head_wobble
    assert (w.amplitude_deg, w.rate_hz, w.shape) == (12.0, 0.5, "sine"), w
    engine.set_head_wobble(shape="triangle")
    assert engine.head_wobble.shape == "triangle"
    assert engine.head_wobble.amplitude_deg == 12.0, "他の値が消えました"
    return f"振り幅のみ変更 → {engine.head_wobble.describe()}"


@test("エンジン: 音を止めずに HRTF を差し替えられる")
def t_swap_hrir() -> str:
    a = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=SR)
    b = HrirSet.synthetic(n_azimuth=36, taps=128, sample_rate=SR)
    engine = SpatialEngine(a, block=BLOCK, max_sources=2)
    engine.add_source(BufferSource(noise(3.0), loop=True),
                      position=(-4.0, 2.0, 0.0), reverb_send=0.0)
    before = engine.render_seconds(0.2)
    engine.set_hrir(b)
    after = engine.render_seconds(0.2)
    assert engine.hrir.n_directions == 36, engine.hrir.n_directions
    assert float(np.max(np.abs(after))) > 1e-4, "差し替え後に無音になりました"
    assert float(np.max(np.abs(after))) < 1.0, "差し替え後にクリップしました"
    # 左右の偏りは保たれているはず（どちらも左寄りの音源）
    def ratio(x: np.ndarray) -> float:
        return float(np.sum(x[:, 0] ** 2) / max(np.sum(x[:, 1] ** 2), 1e-12))
    assert ratio(after) > 1.2, f"差し替え後に定位が壊れました {ratio(after):.2f}"
    return (f"72方向→36方向に差し替え、L/R比 {ratio(before):.2f}→"
            f"{ratio(after):.2f}、ピーク{float(np.max(np.abs(after))):.3f}")


@test("エンジン: サンプルレートの違う HRTF も差し替えで吸収する")
def t_swap_resample() -> str:
    engine = SpatialEngine(hrir(), block=BLOCK, max_sources=1)
    other = HrirSet.synthetic(n_azimuth=72, taps=128, sample_rate=44100)
    engine.set_hrir(other)
    assert engine.hrir.sample_rate == SR, engine.hrir.sample_rate
    return f"44100Hz の HRTF を渡して {engine.hrir.sample_rate}Hz に変換"


TESTS = [t_range, t_period, t_triangle, t_validation, t_presets,
         t_engine_moves, t_front_back_opposite, t_effective_yaw,
         t_partial_update, t_swap_hrir, t_swap_resample]


def main() -> int:
    log(f"motion / head-wobble tests — Python {sys.version.split()[0]}")
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
