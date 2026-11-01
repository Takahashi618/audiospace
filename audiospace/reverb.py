"""Reverb sends: a cheap algorithmic FDN and an IR convolution reverb.

Both implement the same `Reverb` protocol so the engine can switch at
runtime:

    reverb.process(mono_block) -> (block, 2) stereo

`FdnReverb` costs almost nothing and never needs an asset, so it is the
sensible default while prototyping.  `ConvolutionReverb` plays back a real
room impulse response -- generate one with `presets.render_room_ir`, which
uses pyroomacoustics (MIT).
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy import signal

from .convolver import MonoConvolver
from .i18n import tr


class Reverb(Protocol):
    """Anything the engine can use as a reverb send."""

    def process(self, mono_block: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


# Mutually prime delay lengths in milliseconds, spread over a plausible
# range of early-reflection times for a medium room.
_BASE_DELAYS_MS = (23.0, 31.0, 41.0, 53.0, 61.0, 73.0, 89.0, 101.0)


class FdnReverb:
    """8-line feedback delay network with a Householder mixing matrix.

    The Householder matrix (I - 2/N * ones) is lossless and needs no matrix
    multiply -- mixing is one mean and one subtract -- which is why an FDN
    of this size is essentially free per block.

    Parameters
    ----------
    sample_rate  Hz
    rt60         decay time to -60 dB, seconds
    room_scale   multiplies the delay lengths; >1 = larger space
    damping      0..1 high-frequency absorption per pass (0 = bright)
    width        0..1 stereo spread of the output taps
    level        overall output gain (1 = the energy-preserving default)
    """

    def __init__(self, sample_rate: int, rt60: float = 1.4,
                 room_scale: float = 1.0, damping: float = 0.35,
                 width: float = 1.0, level: float = 1.0) -> None:
        if rt60 <= 0:
            raise ValueError("rt60 must be positive")
        self.sample_rate = int(sample_rate)
        self.rt60 = float(rt60)
        self.damping = float(np.clip(damping, 0.0, 0.95))
        self.width = float(np.clip(width, 0.0, 1.0))

        lengths = [max(8, int(round(ms * 1e-3 * sample_rate * room_scale)))
                   for ms in _BASE_DELAYS_MS]
        self._lines = [np.zeros(n, dtype=np.float32) for n in lengths]
        self._pos = [0] * len(lengths)
        # per-line feedback gain giving the requested RT60
        self._gain = np.array(
            [10.0 ** (-3.0 * n / (sample_rate * self.rt60))
             for n in lengths], dtype=np.float32)
        self._lp = np.zeros(len(lengths), dtype=np.float32)
        # alternate polarity across lines for a wide, decorrelated tail
        self._pan = np.array([1.0 if i % 2 == 0 else -1.0
                              for i in range(len(lengths))],
                             dtype=np.float32)
        # Injecting the same signal into every line with the same sign makes
        # the Householder mixing cancel most of it, which costs roughly 8x
        # of level.  A balanced sign pattern decorrelates the lines instead.
        self._inject = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
                                dtype=np.float32)[:len(lengths)]
        self._n = len(lengths)
        # energy-preserving output scale: each ear sums n/2 lines
        self._out_scale = float(1.0 / math.sqrt(len(lengths)))
        if level != 1.0:
            self._out_scale *= float(max(0.0, level))

    def reset(self) -> None:
        for line in self._lines:
            line[:] = 0
        self._pos = [0] * self._n
        self._lp[:] = 0

    def process(self, mono_block: np.ndarray) -> np.ndarray:
        x = np.asarray(mono_block, dtype=np.float32)
        out = np.zeros((x.shape[0], 2), dtype=np.float32)
        taps = np.zeros(self._n, dtype=np.float32)
        damp = self.damping

        for i in range(x.shape[0]):
            for k in range(self._n):
                taps[k] = self._lines[k][self._pos[k]]
            # Householder: y = v - 2/N * sum(v)
            mixed = taps - (2.0 / self._n) * taps.sum()
            # one-pole damping in the feedback path
            self._lp += (1.0 - damp) * (mixed * self._gain - self._lp)
            fed = self._lp + x[i] * self._inject
            for k in range(self._n):
                self._lines[k][self._pos[k]] = fed[k]
                self._pos[k] = (self._pos[k] + 1) % self._lines[k].shape[0]
            left = float(taps[self._pan > 0].sum())
            right = float(taps[self._pan < 0].sum())
            mid = 0.5 * (left + right)
            out[i, 0] = mid + self.width * (left - mid)
            out[i, 1] = mid + self.width * (right - mid)

        return out * self._out_scale


class VectorFdnReverb(FdnReverb):
    """FDN variant that processes a whole block without a per-sample loop.

    Valid only when every delay line is at least `block` samples long,
    which holds for all realistic room sizes (the shortest line here is
    23 ms = 1104 samples at 48 kHz, versus a 512-sample block).  Under that
    condition no sample written this block is read back within the same
    block, so the recursion can be unrolled across the block.
    """

    def process(self, mono_block: np.ndarray) -> np.ndarray:
        x = np.asarray(mono_block, dtype=np.float32)
        n = x.shape[0]
        shortest = min(line.shape[0] for line in self._lines)
        if shortest > n:
            return self._process_span(x)
        # 線がブロックより短い（小さい部屋）ときは、ブロックを線より短い
        # 区切りに分けて同じ計算をする。以前は1サンプルずつの Python の
        # ループ（FdnReverb.process）に落ちて、1ブロック約5ms かかった
        # （区切っても出力は同じ。区切りの中では、書いた値を読み返さない）
        if shortest < 2:
            return super().process(mono_block)
        step = shortest - 1
        return np.concatenate([self._process_span(x[i:i + step])
                               for i in range(0, n, step)], axis=0)

    def _process_span(self, x: np.ndarray) -> np.ndarray:
        """Every line is longer than len(x): process the span at once."""
        n = x.shape[0]
        # Each line is longer than the block, so a block's span wraps at
        # most once: two slices replace building an index array per line
        # for every read and write (same samples, no fancy indexing).
        taps = np.empty((self._n, n), dtype=np.float32)
        for k in range(self._n):
            line = self._lines[k]
            start = self._pos[k]
            first = min(n, line.shape[0] - start)
            taps[k, :first] = line[start:start + first]
            if first < n:
                taps[k, first:] = line[:n - first]

        mixed = taps - (2.0 / self._n) * taps.sum(axis=0, keepdims=True)
        target = mixed * self._gain[:, None]
        # One-pole damping, run in C instead of a 512-iteration Python loop.
        #   y[i] = y[i-1] + alpha*(x[i] - y[i-1])
        # is exactly b=[alpha], a=[1, -(1-alpha)]; for a first-order section
        # scipy's initial delay value is (1-alpha)*y[-1].
        alpha = 1.0 - self.damping
        lp, _ = signal.lfilter([alpha], [1.0, -(1.0 - alpha)], target,
                               axis=1, zi=((1.0 - alpha) * self._lp)[:, None])
        self._lp = lp[:, -1].astype(np.float32)

        fed = lp + x[None, :] * self._inject[:, None]
        for k in range(self._n):
            line = self._lines[k]
            length = line.shape[0]
            start = self._pos[k]
            first = min(n, length - start)
            line[start:start + first] = fed[k, :first]
            if first < n:
                line[:n - first] = fed[k, first:]
            self._pos[k] = (start + n) % length

        left = taps[self._pan > 0].sum(axis=0)
        right = taps[self._pan < 0].sum(axis=0)
        mid = 0.5 * (left + right)
        out = np.stack([mid + self.width * (left - mid),
                        mid + self.width * (right - mid)], axis=-1)
        return (out * self._out_scale).astype(np.float32)


class BandFdnReverb(VectorFdnReverb):
    """帯域ごとに違う速さで減衰する FDN（材質から決まる後期残響）。

    `rt60_bands` は 125Hz〜8kHz の7帯域の残響時間（秒）。`Space.rt60_bands()`
    の値をそのまま渡す。木の部屋は低域が短く、コンクリートは全帯域で長く、
    じゅうたんは高域が短い、という違いがここで出る。

    仕組み: 各遅延線の1周ぶんの減衰を、帯域ごとに 10^(-3・線の長さ/(fs・RT60))
    として求め、低域・高域のシェルフ2段と全体の倍率で近似する
    （`acoustics.shelf_filter`）。単一の倍率と1次ローパスの FdnReverb より、
    材質の音色をはっきり出せる。

    遅延線はブロックより長くする（ブロック単位でまとめて計算する前提）。
    小さい部屋では遅延を縮めるが、ブロック+1サンプルより短くはしない。
    """

    def __init__(self, sample_rate: int, rt60_bands, room_scale: float = 1.0,
                 width: float = 1.0, level: float = 1.0,
                 block: int = 512) -> None:
        from .acoustics import shelf_filter

        bands = np.maximum(np.asarray(rt60_bands, dtype=np.float64), 0.05)
        super().__init__(sample_rate, rt60=float(bands[3]),
                         room_scale=room_scale, damping=0.0, width=width)
        # 短すぎる線は伸ばす（ブロック単位の計算が成り立つように）
        lengths = [max(int(block) + 1, line.shape[0]) for line in self._lines]
        self._lines = [np.zeros(n, dtype=np.float32) for n in lengths]
        self._pos = [0] * len(lengths)
        self.rt60_bands = bands
        self.level = float(level)
        # 帯域の形（低域・高域のシェルフ）は、平均の長さの線で作って全線で
        # 共有し、1回のフィルタ呼び出しで済ませる（線ごとに呼ぶと手間が
        # 8倍）。線ごとの長さの違いは全体の倍率で付ける。線どうしは毎周
        # 混ざるので、ネットワーク全体は平均の速さで減衰する。
        mean_len = float(np.mean(lengths))
        shape = 10.0 ** (-3.0 * mean_len / (sample_rate * bands))
        self._sos, mean_mid = shelf_filter(shape, sample_rate)
        gains = []
        for n in lengths:
            band_gain = 10.0 ** (-3.0 * n / (sample_rate * bands))
            _, mid = shelf_filter(band_gain, sample_rate)
            gains.append(mid)
        # 共有の形は平均の長さで作るので、平均より短い線では、中域より
        # 長く響く帯域（例: 厚いカーテンの部屋の低域）の1周の利得が、
        # その線に本来あるべき値より大きくなる。中域の残響が短く低域が長い
        # 部屋では 1 を超え、響きが増え続けて発散した。線ごとに、どの周波数
        # でも「いちばん長く響く帯域での、その線の利得」を超えないよう抑える
        # （必ず 1 未満になる）。ふつうの部屋ではこの上限に届かない。
        _, response = signal.sosfreqz(self._sos, worN=2048, fs=sample_rate)
        peak = max(1e-9, float(np.max(np.abs(response))))
        slowest = float(np.max(bands))
        caps = [10.0 ** (-3.0 * n / (sample_rate * slowest)) / peak
                for n in lengths]
        self._gain = np.minimum(np.asarray(gains), np.asarray(caps)
                                ).astype(np.float32)
        self._zi = np.zeros((2, len(lengths), 2))

    def reset(self) -> None:
        super().reset()
        self._zi[:] = 0.0

    def process(self, mono_block: np.ndarray) -> np.ndarray:
        x = np.asarray(mono_block, dtype=np.float32)
        n = x.shape[0]
        taps = np.empty((self._n, n), dtype=np.float32)
        for k in range(self._n):
            line = self._lines[k]
            start = self._pos[k]
            first = min(n, line.shape[0] - start)
            taps[k, :first] = line[start:start + first]
            if first < n:
                taps[k, first:] = line[:n - first]

        mixed = taps - (2.0 / self._n) * taps.sum(axis=0, keepdims=True)
        target = mixed * self._gain[:, None]
        fed, self._zi = signal.sosfilt(self._sos, target, axis=1, zi=self._zi)
        fed = fed.astype(np.float32)
        fed += x[None, :] * self._inject[:, None]
        for k in range(self._n):
            line = self._lines[k]
            length = line.shape[0]
            start = self._pos[k]
            first = min(n, length - start)
            line[start:start + first] = fed[k, :first]
            if first < n:
                line[:n - first] = fed[k, first:]
            self._pos[k] = (start + n) % length

        left = taps[self._pan > 0].sum(axis=0)
        right = taps[self._pan < 0].sum(axis=0)
        mid = 0.5 * (left + right)
        out = np.stack([mid + self.width * (left - mid),
                        mid + self.width * (right - mid)], axis=-1)
        return (out * (self._out_scale * self.level)).astype(np.float32)


def band_reverb_for(space, sample_rate: int,
                    block: int = 512) -> BandFdnReverb:
    """材質つきの空間の、速い版の後期残響（`SpatialEngine` が使うもの）。

    精密版の尾の音量はこれのエネルギーにそろえる（`precise.py`）。作り方を
    1か所にまとめてあるので、ここを変えれば両方が一緒に変わる（別々に
    持つと、片方だけ変えたときに切り替えで音量が変わってしまう）。
    """
    areas = space.face_areas()
    mean_free_path = 4.0 * space.volume / sum(areas.values())
    return BandFdnReverb(sample_rate, space.rt60_bands(),
                         room_scale=min(3.0, max(0.3, mean_free_path / 10.0)),
                         level=space.late_level(), block=block)


class ConvolutionReverb:
    """Convolve the send bus with a measured or simulated stereo IR."""

    def __init__(self, ir: np.ndarray, block: int = 512,
                 normalise: bool = True) -> None:
        ir = np.atleast_2d(np.asarray(ir, dtype=np.float32))
        if normalise:
            energy = float(np.sqrt(np.sum(ir.astype(np.float64) ** 2)))
            if energy > 0:
                ir = ir / energy
        self._conv = MonoConvolver(ir, block=block)
        self.taps = self._conv.taps

    def reset(self) -> None:
        self._conv.reset()

    def process(self, mono_block: np.ndarray) -> np.ndarray:
        return self._conv.process(np.asarray(mono_block, dtype=np.float32))

    def clone(self) -> "ConvolutionReverb":
        """同じ IR で、状態だけ空の新しい残響を作る（安い）。

        IR のスペクトル（長い IR では数 MB、作るのに数ミリ秒〜数十ミリ秒）は
        共有し、遅延線だけ新しく持つ。音声スレッドで空間を切り替えるときに
        使う（IR から作り直すと、その間だけ処理が遅れて音が途切れうる）。
        """
        twin = object.__new__(ConvolutionReverb)
        conv = object.__new__(MonoConvolver)
        src = self._conv
        conv.block, conv.nfft, conv.taps = src.block, src.nfft, src.taps
        conv.nparts = src.nparts
        conv._H = src._H                       # 共有（読むだけ）
        conv._fdl = np.zeros_like(src._fdl)
        conv._head = 0
        conv._tail = np.zeros_like(src._tail)
        twin._conv = conv
        twin.taps = self.taps
        return twin


def rt60_of(ir: np.ndarray, sample_rate: int) -> float:
    """Estimate RT60 from an impulse response by Schroeder backward integration.

    Fits the -5 dB to -25 dB span and extrapolates, which is the usual T20
    method; returns 0.0 if the decay is too short to fit.
    """
    x = np.asarray(ir, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    energy = np.cumsum(x[::-1] ** 2)[::-1]
    if energy[0] <= 0:
        return 0.0
    db = 10.0 * np.log10(np.maximum(energy / energy[0], 1e-12))
    try:
        start = int(np.argmax(db <= -5.0))
        end = int(np.argmax(db <= -25.0))
    except ValueError:
        return 0.0
    if end <= start:
        return 0.0
    t = np.arange(start, end) / sample_rate
    slope = np.polyfit(t, db[start:end], 1)[0]
    return float(-60.0 / slope) if slope < 0 else 0.0


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def gain_to_db(gain: float) -> float:
    return float(20.0 * math.log10(max(gain, 1e-12)))


# ---------------------------------------------------------------------------
# 残響の設計図（自分で作る残響を JSON に保存するための形）
# ---------------------------------------------------------------------------
#: 設計図で決められる値の範囲
REVERB_SPEC_LIMITS = {
    "rt60": (0.1, 8.0),          # 残響時間、秒
    "room_scale": (0.3, 2.5),    # 部屋の大きさ（遅延線の長さの倍率）
    "damping": (0.0, 0.95),      # 高い音の減衰
    "width": (0.0, 1.0),         # 左右の広がり
    "level": (0.0, 3.0),         # 残響の大きさ
}


def reverb_spec_problems(spec: dict) -> list[str]:
    """残響の設計図の問題点（いまの言語の文）。問題が無ければ空。"""
    if not isinstance(spec, dict):
        return [tr("設計図の形が違います（辞書ではありません）",
                   "The design has the wrong shape (it is not a dictionary)")]
    labels = {"rt60": tr("残響時間", "Reverb time"),
              "room_scale": tr("部屋の大きさ", "Room size"),
              "damping": tr("高い音の減衰", "High damping"),
              "width": tr("広がり", "Width"),
              "level": tr("残響の大きさ", "Reverb level")}
    problems = []
    if "rt60" not in spec:
        problems.append(tr("残響時間がありません", "The reverb time is missing"))
    for key, (lo, hi) in REVERB_SPEC_LIMITS.items():
        try:
            value = float(spec.get(key, 1.0 if key != "damping" else 0.35))
        except (TypeError, ValueError, OverflowError):
            problems.append(tr(f"{labels[key]}が数ではありません",
                               f"{labels[key]} is not a number"))
            continue
        if not lo <= value <= hi:
            problems.append(tr(f"{labels[key]}は{lo:g}から{hi:g}の間にしてください",
                               f"{labels[key]} must be between {lo:g} and "
                               f"{hi:g}"))
    if len(str(spec.get("name", ""))) > 40:
        problems.append(tr("名前は40文字以内にしてください",
                           "Use a name of 40 characters or fewer"))
    return problems


def reverb_from_spec(spec: dict, sample_rate: int) -> VectorFdnReverb:
    """設計図から FDN の残響を作る。

        {"name": "...", "rt60": 秒, "room_scale": 倍率, "damping": 0〜0.95,
         "width": 0〜1, "level": 倍率}

    値がおかしければ ValueError（問題点を日本語で）。
    """
    problems = reverb_spec_problems(spec)
    if problems:
        raise ValueError("、".join(problems))
    return VectorFdnReverb(
        sample_rate, rt60=float(spec["rt60"]),
        room_scale=float(spec.get("room_scale", 1.0)),
        damping=float(spec.get("damping", 0.35)),
        width=float(spec.get("width", 1.0)),
        level=float(spec.get("level", 1.0)))
