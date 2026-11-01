# -*- coding: utf-8 -*-
"""前後の強調と上下の強調を併用したときの、方向の聞き分けやすさを数値で比べる。

    python -X utf8 tools/study_elevation_cues.py

耳で聞く代わりに、紛らわしい方向の組ごとに「両耳の音色の差」を測る。
実測 HRTF（Viking 耳介 E）の各方向の周波数特性に、人工の手がかり
（後ろの音量とローパス、上下の強調の棚形フィルタ）の特性を掛け、
1/3 オクターブ帯域（250Hz〜16kHz）のレベルを両耳で並べる。

- 音色の差: 2つの方向の帯域レベルの差から、全体の音量差（平均）を除いた
  ものの二乗平均（dB）。音量は距離と区別がつかないので、音色だけを見る
- 音量の差: 平均の差（dB）。参考として並べる

音は鳴らさない（HRTF の数値を読むだけ）。結果は
tools/study_elevation_cues_report.txt に書く。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace.hrtf import HrirSet  # noqa: E402

SOFA = ROOT / "assets" / "hrtf" / "viking" / "subj_E.sofa"
OUT = Path(__file__).with_name("study_elevation_cues_report.txt")
SR = 48000
NFFT = 4096
LP_BUCKETS = 24          # engine.SpatialEngine.LP_BUCKETS と同じ
TILT_MAX_DB = 12.0       # engine.ELEVATION_EMPHASIS_MAX_DB と同じ
TILT_CORNER_HZ = 3000.0

freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
centers = [250 * 2 ** (i / 3) for i in range(19)]          # 250Hz〜16kHz
band_masks = [(freqs >= c / 2 ** (1 / 6)) & (freqs < c * 2 ** (1 / 6))
              for c in centers]


def rear_strength(az: float, el: float, depth: float,
                  upper_only: bool = False) -> float:
    """エンジンと同じ: 強さ × (1 − cos 方位)/2 × cos 仰角。

    upper_only なら、cos 仰角で弱めるのは上側だけにし、下側は水平と同じ
    強さのままにする（後ろの下の音が、後ろの水平の音より明るくならない
    ようにする案）。
    """
    weight = max(0.0, math.cos(math.radians(el)))
    if upper_only and el < 0:
        weight = 1.0
    return depth * (1 - math.cos(math.radians(az))) / 2 * weight


def rear_response(az: float, el: float, depth: float,
                  upper_only: bool = False) -> np.ndarray:
    """後ろの音量とローパス（24段階に丸める）の周波数特性。"""
    s = rear_strength(az, el, depth, upper_only)
    gain = 1.0 - s
    bucket = int(round(s * (LP_BUCKETS - 1)))
    if bucket <= 0:
        return np.full(len(freqs), gain, dtype=complex)
    c = max(0.05, 1.0 - bucket / (LP_BUCKETS - 1))
    _, h = signal.freqz([c], [1.0, -(1.0 - c)], worN=freqs, fs=SR)
    return gain * h


def tilt_db(el: float, amount: float, mapping: str) -> float:
    if mapping == "linear":          # アプリの今の式: 仰角に比例
        return TILT_MAX_DB * amount * el / 90.0
    return TILT_MAX_DB * amount * math.sin(math.radians(el))   # sin(仰角)


def tilt_response(el: float, amount: float, mapping: str) -> np.ndarray:
    """上下の強調: x + k・高域通過(x)。検討した時点のアプリの形。

    2026-10-16 にエンジンへ移したときは RBJ のハイシェルフに替えた
    （この形は 3kHz の手前で打ち消し合うため）。明るさの比較の結論は
    変わらないので、検討の記録としてこの形のまま残す。
    """
    db = tilt_db(el, amount, mapping)
    if db == 0.0:
        return np.ones(len(freqs), dtype=complex)
    b, a = signal.butter(2, TILT_CORNER_HZ, btype="highpass", fs=SR)
    _, hp = signal.freqz(b, a, worN=freqs, fs=SR)
    k = 10 ** (db / 20) - 1
    return 1.0 + k * hp


def features(hrir: HrirSet, az: float, el: float, cfg: dict) -> np.ndarray:
    """両耳 × 19帯域のレベル（dB）。"""
    idx = int(hrir.indices_for(np.array([az]), np.array([el]))[0])
    spec = np.fft.rfft(hrir.data[idx], n=NFFT, axis=-1)       # (2, bins)
    h = np.ones(len(freqs), dtype=complex)
    if cfg.get("depth", 0):
        h = h * rear_response(az, el, cfg["depth"],
                              cfg.get("upper_only", False))
    if cfg.get("tilt", 0):
        h = h * tilt_response(el, cfg["tilt"], cfg.get("map", "sin"))
    spec = spec * h[None, :]
    power = np.abs(spec) ** 2
    out = np.array([[10 * np.log10(power[ear, m].mean() + 1e-20)
                     for m in band_masks] for ear in range(2)])
    return out.ravel()


def diff(fa: np.ndarray, fb: np.ndarray) -> tuple[float, float]:
    d = fa - fb
    level = float(d.mean())
    timbre = float(np.sqrt(np.mean((d - level) ** 2)))
    return timbre, level


CONFIGS = [
    ("A なし（HRTF だけ）", {}),
    ("B 前後の強調35%", {"depth": 0.35}),
    ("C 上下の強調35%（仰角に比例、今のアプリ）", {"tilt": 0.35, "map": "linear"}),
    ("D 上下の強調35%（sin 仰角）", {"tilt": 0.35, "map": "sin"}),
    ("E 前後35% と 上下35%（sin）の併用", {"depth": 0.35, "tilt": 0.35}),
    ("F 前後50% と 上下50%（sin）の併用", {"depth": 0.5, "tilt": 0.5}),
    ("G 前後90% と 上下50%（sin）の併用", {"depth": 0.9, "tilt": 0.5}),
    ("H 前後35% と 上下35%（比例）の併用", {"depth": 0.35, "tilt": 0.35,
                                         "map": "linear"}),
    ("I H と同じで、後ろの処理を下側では弱めない",
     {"depth": 0.35, "tilt": 0.35, "map": "linear", "upper_only": True}),
    ("J 前後50% と 上下50%（比例）、下側では弱めない",
     {"depth": 0.5, "tilt": 0.5, "map": "linear", "upper_only": True}),
    ("K 前後90% と 上下50%（比例）、下側では弱めない",
     {"depth": 0.9, "tilt": 0.5, "map": "linear", "upper_only": True}),
    ("L 前後50% と 上下50%（比例）、今の重み（上下とも cos）",
     {"depth": 0.5, "tilt": 0.5, "map": "linear"}),
]

# (群, 説明, 方向1, 方向2)。方位は正面0度・反時計回り（左が90度）
PAIRS = [
    ("前後", "水平: 前と後ろ", (0, 0), (180, 0)),
    ("前後", "水平: 左斜め前と左斜め後ろ", (45, 0), (135, 0)),
    ("前後", "上45度: 前と後ろ", (0, 45), (180, 45)),
    ("前後", "下45度: 前と後ろ", (0, -45), (180, -45)),
    ("前後", "上45度: 左斜め前と左斜め後ろ", (45, 45), (135, 45)),
    ("上下", "前: 上45度と水平", (0, 45), (0, 0)),
    ("上下", "前: 水平と下45度", (0, 0), (0, -45)),
    ("上下", "後ろ: 上45度と水平", (180, 45), (180, 0)),
    ("上下", "後ろ: 水平と下45度", (180, 0), (180, -45)),
    ("上下", "左: 上45度と水平", (90, 45), (90, 0)),
    ("上下", "左: 水平と下45度", (90, 0), (90, -45)),
    ("上下", "真上と前の上45度", (0, 90), (0, 45)),
    ("交差", "後ろの水平と前の下45度（どちらも暗い）", (180, 0), (0, -45)),
    ("交差", "後ろの上45度と前の水平", (180, 45), (0, 0)),
    ("交差", "前の上45度と後ろの水平", (0, 45), (180, 0)),
]


AZIMUTHS = [(0, "前"), (45, "左斜め前"), (90, "左"), (135, "左斜め後ろ"),
            (180, "後ろ"), (225, "右斜め後ろ"), (270, "右"), (315, "右斜め前")]
HEIGHTS = [(-45, "下45"), (0, "水平"), (45, "上45"), (90, "真上")]
LOW = [i for i, c in enumerate(centers) if c <= 2000]
HIGH = [i for i, c in enumerate(centers) if c >= 3000]


def brightness(hrir: HrirSet, az: float, el: float, cfg: dict) -> float:
    """明るさ: 3kHz 以上の平均レベル − 2kHz 以下の平均レベル（両耳の平均、dB）。

    人工の手がかりはどちらもこの量を動かす（後ろはこもらせて下げ、
    上下の強調は上で上げて下で下げる）。だから2つがぶつかるならここに出る。
    """
    f = features(hrir, az, el, cfg).reshape(2, -1)
    return float(np.mean(f[:, HIGH]) - np.mean(f[:, LOW]))


def brightness_report(hrir: HrirSet) -> list[str]:
    lines = ["■ 明るさが高さの順に並ぶか（どの方位でも、上ほど明るいか）",
             "  逆転: 高さを1段上げたのに明るさが下がった数（8方位 × 3段 = 24か所中）",
             "  段差の最小: 1段上げたときの明るさの増え方の最小（dB、負なら逆転）",
             "  前後の差: 同じ高さで、前と後ろの明るさの差の平均（dB、正なら前が明るい）",
             "  紛れ: 別の高さ・別の方位どうしで、明るさの差が 1dB 未満の組の数",
             "        （明るさだけでは区別できない組。8方位 × 4高さ = 32方向のうち",
             "         高さの違う組だけを数える）",
             ""]
    for name, cfg in CONFIGS:
        grid = {(az, el): brightness(hrir, az, el, cfg)
                for az, _ in AZIMUTHS for el, _ in HEIGHTS}
        inversions = 0
        steps = []
        for az, _ in AZIMUTHS:
            for (lo, _), (hi, _) in zip(HEIGHTS, HEIGHTS[1:]):
                step = grid[(az, hi)] - grid[(az, lo)]
                steps.append(step)
                inversions += step <= 0
        fb = [grid[(0, el)] - grid[(180, el)] for el, _ in HEIGHTS[:3]]
        keys = list(grid)
        confusable = sum(
            1 for i, a in enumerate(keys) for b in keys[i + 1:]
            if a[1] != b[1] and abs(grid[a] - grid[b]) < 1.0)
        pairs_total = sum(1 for i, a in enumerate(keys) for b in keys[i + 1:]
                          if a[1] != b[1])
        lines.append(f"  {name}: 逆転 {inversions}/24、段差の最小 "
                     f"{min(steps):+.1f}、前後の差 {np.mean(fb):+.1f}、"
                     f"紛れ {confusable}/{pairs_total}")
    # 併用の条件 E の表を全部出す（耳で確かめる前の見取り図）
    for name, cfg in (CONFIGS[0], CONFIGS[7], CONFIGS[8]):
        lines.append("")
        lines.append(f"  明るさの表（{name}、dB）。行は高さ、列は方位")
        lines.append("    " + "、".join(n for _, n in AZIMUTHS))
        for el, en in reversed(HEIGHTS):
            vals = "、".join(f"{brightness(hrir, az, el, cfg):+.1f}"
                            for az, _ in AZIMUTHS)
            lines.append(f"    {en}: {vals}")
    return lines


def main() -> int:
    hrir = HrirSet.from_sofa(SOFA, sample_rate=SR)
    lines = ["前後の強調と上下の強調の併用: 紛らわしい方向の組の音色の差（dB）",
             f"HRTF: {hrir.name}、{len(PAIRS)}組 × {len(CONFIGS)}条件",
             "音色の差が大きいほど聞き分けやすい（音量の差は除いてある）。",
             ""]
    table: dict[str, list[float]] = {}
    for name, cfg in CONFIGS:
        lines.append(f"■ {name}")
        values = []
        for group, label, a, b in PAIRS:
            t, lv = diff(features(hrir, *a, cfg), features(hrir, *b, cfg))
            values.append(t)
            lines.append(f"  [{group}] {label}: 音色 {t:.2f}、音量 {lv:+.2f}")
        table[name] = values
        for group in ("前後", "上下", "交差"):
            vs = [v for v, p in zip(values, PAIRS) if p[0] == group]
            lines.append(f"  {group}の平均 {np.mean(vs):.2f}、最小 {min(vs):.2f}")
        lines.append("")
    # 1kHz 以上で、後ろの処理と上下の強調がどれだけ重なるか（同じ組で比較）
    lines.append("■ 条件ごとの群平均（音色の差、dB）")
    for name, values in table.items():
        parts = []
        for group in ("前後", "上下", "交差"):
            vs = [v for v, p in zip(values, PAIRS) if p[0] == group]
            parts.append(f"{group} {np.mean(vs):.2f}")
        lines.append(f"  {name}: " + "、".join(parts))
    lines.append("")
    lines += brightness_report(hrir)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[lines.index("■ 条件ごとの群平均（音色の差、dB）"):]))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
