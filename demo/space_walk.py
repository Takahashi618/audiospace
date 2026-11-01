"""場所の移動で響きが変わる様子を、音のファイルにする（デバイスは開かない）。

コンクリートの部屋 → 建物の間の道 → 何もない道 と歩いて抜ける。各場所で
立ち止まって2回手をたたき、歩くあいだは足音が鳴る。`set_spaces` で場所ごとの
空間を登録してあるので、境目を越えると響きが 0.3秒かけてつなぎ替わり、
出てきた場所の残響は自然に消えていく。

作るファイル: demo/out/space_walk.wav（48kHz、ステレオ、約30秒）。
ヘッドホンで聞く。どの時刻にどこにいるかは、実行すると表示する。

Run: python -X utf8 demo\\space_walk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audiospace import (BufferSource, HrirSet, Space,  # noqa: E402
                        SpatialEngine)
from audiospace.i18n import tr  # noqa: E402

SR = 48000
BLOCK = 512
OUT = ROOT / "demo" / "out" / "space_walk.wav"
SOFA = ROOT / "assets" / "hrtf" / "viking" / "subj_E.sofa"


def clap(rng: np.random.Generator) -> np.ndarray:
    """手をたたく音: 短い広帯域の破裂音。"""
    n = int(0.03 * SR)
    env = np.exp(-np.arange(n) / (0.004 * SR))
    return (rng.standard_normal(n) * env * 0.8).astype(np.float32)


def step(rng: np.random.Generator) -> np.ndarray:
    """足音: 低めにこもった短い音。"""
    n = int(0.06 * SR)
    env = np.exp(-np.arange(n) / (0.012 * SR))
    x = rng.standard_normal(n) * env
    # 簡単なローパス（1次）で靴底のこもった感じにする
    y = np.zeros_like(x)
    for i in range(1, n):
        y[i] = y[i - 1] + 0.25 * (x[i] - y[i - 1])
    return (y * 1.5).astype(np.float32)


def main() -> int:
    hrir = (HrirSet.from_sofa(SOFA, sample_rate=SR) if SOFA.exists()
            else HrirSet.synthetic(sample_rate=SR))
    engine = SpatialEngine(hrir, block=BLOCK, max_sources=16)

    # 場所: 部屋（原点の周り）を、北へ延びる建物の間の道の南端に置き、
    # その先は何もない道。部屋の北の壁を抜けると道に出る
    room = Space.room((5.0, 4.0, 2.7), floor="concrete_floor",
                      walls="rough_concrete", ceiling="rough_concrete",
                      listener=(0.0, 0.0, 0.0), at=(0.5, 0.5),
                      name=tr("コンクリートの部屋", "Concrete room"))
    street = Space(origin=(-6.0, 2.0, -1.6), size=(12.0, 30.0, 20.0),
                   materials={"west": "brickwork", "east": "brickwork",
                              "south": "open", "north": "open",
                              "floor": "rough_concrete", "ceiling": "open"},
                   name=tr("建物の間の道", "Street between buildings"))
    road = Space.open_ground(ground="rough_concrete",
                             listener=(0.0, 0.0, 0.0),
                             name=tr("何もない道", "Empty road"))
    engine.set_spaces([road, street, room])

    rng = np.random.default_rng(3)
    # 予定: (秒, 動き)。"clap" は立ち止まってたたく、"walk" は (行き先 y) へ歩く
    plan = [("clap", 0.0), ("clap", 0.0), ("walk", 6.0),
            ("clap", 6.0), ("clap", 6.0), ("walk", 20.0),
            ("clap", 20.0), ("clap", 20.0), ("walk", 40.0),
            ("clap", 40.0), ("clap", 40.0)]
    speed = 1.3                     # m/s
    total = 0.0
    events = []                     # (開始秒, 音, 聞き手から見た位置)
    y = 0.0
    times = []                      # (秒, y) の折れ線（聞き手の位置）
    for kind, target in plan:
        if kind == "clap":
            events.append((total + 0.2, clap(rng), (0.0, 0.4, -0.3)))
            times.append((total, y))
            total += 1.6
        else:
            dur = abs(target - y) / speed
            n_steps = int(dur / 0.55)
            for k in range(n_steps):
                events.append((total + k * 0.55, step(rng), (0.0, 0.2, -1.5)))
            times.append((total, y))
            total += dur
            y = target
    times.append((total + 2.0, y))
    total += 2.0
    t_axis = np.array([t for t, _ in times])
    y_axis = np.array([v for _, v in times])

    n_blocks = int(np.ceil(total * SR / BLOCK))
    out = []
    pending = sorted(events, key=lambda e: e[0])
    timeline = []
    last_name = None
    for b in range(n_blocks):
        now = b * BLOCK / SR
        engine.set_listener((0.0, float(np.interp(now, t_axis, y_axis)), 0.0))
        while pending and pending[0][0] <= now:
            _, sound, rel = pending.pop(0)
            engine.add_source(BufferSource(sound), position=rel,
                              relative=True, reverb_send=0.3)
        block = engine.render()
        out.append(block)
        name = (engine.space.name if engine.space is not None
                else tr("なし", "none"))
        if name != last_name:
            timeline.append((now, name))
            last_name = name
        engine.reap_finished()      # 鳴り終わった音を回収する（枠を空ける）

    audio = np.concatenate(out)
    audio = audio / max(1e-9, float(np.max(np.abs(audio)))) * 0.8
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(OUT, SR, (audio * 32767).astype(np.int16))
    print(tr(f"書き出し: {OUT}（{len(audio) / SR:.1f}秒）",
             f"Written: {OUT} ({len(audio) / SR:.1f} seconds)"))
    for t, name in timeline:
        print(tr(f"  {t:5.1f}秒〜 {name}", f"  from {t:5.1f} s: {name}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
