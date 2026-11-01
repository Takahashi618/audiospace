"""方向つきの初期反射（空間の面からの最初の反射）。

`Space.reflection_chains()` が決める反射の1本ずつを、エンジンの仮想の音源の
行として鳴らす。各行には、空間化したすべての音源を足した信号を、その反射の
行程ぶん遅らせ、当たった面の材質で音色を変え、距離で弱めたものを入れる。
方向は面の向きで、エンジンが聞き手の向き（首振り・顔の上下・首かしげを
含む）に合わせて HRTF を選ぶ。

費用は音源の数によらず、反射の本数（部屋なら最大24本）で決まる。軽くする
ための工夫:

- 経路の形（どの面にどの順で当たるか、材質による帯域ごとの減衰）は空間で
  決まり、聞き手の位置で変わるのは行程だけ（1次式）。形は最初に1回だけ
  作り、行程は毎ブロック全経路まとめて計算する
- 材質の音色のフィルタは、同じ材質の組み合わせの経路で共有し、まとめて
  1回で掛ける（1本ずつ呼ぶと、呼び出しの手間だけで 2ms かかった）
- 空気吸収は帯域ごとではなく、経路ごとの全体の倍率として掛ける
  （0.5秒以内の行程では、帯域の差は小さい）

遅れが変わる（聞き手が動いた）ブロックでは、古い遅れと新しい遅れの両方で
読んでブロックの中でつなぎ替える。いきなり切り替えると、反射の波形が
飛んで「プツッ」と鳴る。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .acoustics import (AIR_ABSORPTION, SPEED_OF_SOUND, ReflectionPath,
                        Space, shelf_filter)

#: 一度に鳴らす反射の最大本数（エンジンが行を確保する数）
MAX_PATHS = 24


class EarlyReflections:
    """聞き手の位置から決まる初期反射を、ブロックごとに作る。"""

    def __init__(self, space: Space, sample_rate: int, block: int,
                 max_delay_s: float = 0.5, min_gain: float = 0.005) -> None:
        self.space = space
        self.sample_rate = int(sample_rate)
        self.block = int(block)
        self.max_delay_s = float(max_delay_s)
        self.min_gain = float(min_gain)
        longest = int(np.ceil(max_delay_s * sample_rate)) + 2 * self.block
        self._history = np.zeros(longest, dtype=np.float32)
        self._write = 0
        self._arange = np.arange(self.block)
        self._ramp = np.linspace(0.0, 1.0, self.block, endpoint=False,
                                 dtype=np.float32)

        chains = space.reflection_chains()
        if len(chains) > MAX_PATHS:
            # 鳴らせるのは MAX_PATHS 本まで。ブロックごとに強い順で選び直すと、
            # 聞き手が動くたびに境目の経路が出入りし、入ってきた経路は
            # いきなり鳴り、出ていった経路はいきなり消えて「プツッ」となる。
            # 回数の少ない（強く、方向のはっきりした）経路を最初に決めて固定する
            chains = sorted(chains, key=lambda c: len(c.faces))[:MAX_PATHS]
        self._chains = chains
        count = len(chains)
        self._axis = np.array([c.axis for c in chains], dtype=np.intp)
        self._sign = np.array([c.sign for c in chains], dtype=np.float64)
        self._const = np.array([c.const for c in chains], dtype=np.float64)
        bands = np.array([c.bands for c in chains],
                         dtype=np.float64).reshape(count, -1)
        self._bands = bands
        # 材質の組み合わせごとに1つのフィルタ。同じ帯域振幅の経路は共有する
        self._group = np.zeros(count, dtype=np.intp)
        self._sos: list[np.ndarray] = []
        self._mid: list[float] = []
        seen: dict[tuple, int] = {}
        for i, c in enumerate(chains):
            key = tuple(round(b, 9) for b in c.bands)
            if key not in seen:
                seen[key] = len(self._sos)
                sos, mid = shelf_filter(np.asarray(c.bands), self.sample_rate)
                self._sos.append(sos)
                self._mid.append(mid)
            self._group[i] = seen[key]
        self._zi = np.zeros((count, 2, 2))
        self._delay = np.full(count, -1, dtype=np.intp)
        self._last_position: tuple | None = None

    def reset(self) -> None:
        self._history[:] = 0.0
        self._write = 0
        self._zi[:] = 0.0
        self._delay[:] = -1

    @property
    def paths(self) -> list[ReflectionPath]:
        """いまの聞き手の位置での反射（確認・読み上げ用）。"""
        if self._last_position is None:
            return []
        return self.space.reflection_paths(self._last_position,
                                           self.max_delay_s, self.min_gain,
                                           chains=self._chains)

    def _read(self, delays: np.ndarray) -> np.ndarray:
        """いま書いたブロックを、行ごとに delays サンプル遅らせて読む。"""
        size = self._history.shape[0]
        start = self._write - self.block - delays
        idx = (start[:, None] + self._arange[None, :]) % size
        return self._history[idx]

    def process(self, dry: np.ndarray, listener_position
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1ブロック進める。

        dry                 空間化した音源をすべて足した信号 (block,)
        listener_position   聞き手の world 座標

        戻り値は (信号 (本数, block), 向き (本数, 3), 行程 (本数,))。
        向きは world 座標の単位ベクトル。
        """
        size = self._history.shape[0]
        n = self.block
        end = self._write + n
        x = np.asarray(dry, dtype=np.float32)
        if end <= size:
            self._history[self._write:end] = x
        else:
            first = size - self._write
            self._history[self._write:] = x[:first]
            self._history[:n - first] = x[first:]
        self._write = end % size

        p = np.asarray(listener_position, dtype=np.float64)
        self._last_position = tuple(float(v) for v in p)
        empty = (np.zeros((0, n), dtype=np.float32), np.zeros((0, 3)),
                 np.zeros(0))
        if not len(self._chains) or not self.space.contains(p):
            self._delay[:] = -1
            self._zi[:] = 0.0
            return empty

        coord = p[self._axis]
        offset = self._sign * coord + self._const - coord
        length = np.abs(offset)
        air = np.exp(-AIR_ABSORPTION[None, :] * length[:, None] / 2.0)
        broadband_air = air.mean(axis=1)
        gain = ((self._bands * air).mean(axis=1) / (1.0 + length)
                * self.space.early_gain)
        max_len = self.max_delay_s * SPEED_OF_SOUND
        valid = (length > 1e-6) & (length <= max_len) & (gain >= self.min_gain)
        chosen = np.flatnonzero(valid)
        if chosen.size > MAX_PATHS:
            chosen = chosen[np.argsort(-gain[chosen])[:MAX_PATHS]]
        # 今回鳴らさない経路は状態を捨てる（戻ってきたら最初から）
        dropped = np.ones(len(self._chains), dtype=bool)
        dropped[chosen] = False
        self._delay[dropped] = -1
        self._zi[dropped] = 0.0
        if not chosen.size:
            return empty

        delays = np.rint(length[chosen] / SPEED_OF_SOUND
                         * self.sample_rate).astype(np.intp)
        src = self._read(delays)
        prev = self._delay[chosen]
        moved = (prev >= 0) & (prev != delays)
        if moved.any():
            old = self._read(prev[moved])
            src[moved] = (old * (1.0 - self._ramp)
                          + src[moved] * self._ramp)
        self._delay[chosen] = delays

        out = np.empty_like(src)
        groups = self._group[chosen]
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            chain_ids = chosen[rows]
            zi = np.moveaxis(self._zi[chain_ids], 0, 1)    # (2, k, 2)
            y, zf = signal.sosfilt(self._sos[g], src[rows], axis=1, zi=zi)
            self._zi[chain_ids] = np.moveaxis(zf, 1, 0)
            out[rows] = y * self._mid[g]
        amp = broadband_air[chosen] / (1.0 + length[chosen]) \
            * self.space.early_gain
        out *= amp[:, None].astype(np.float32)

        dirs = np.zeros((chosen.size, 3))
        dirs[np.arange(chosen.size), self._axis[chosen]] = np.sign(
            offset[chosen])
        return out.astype(np.float32), dirs, length[chosen]
