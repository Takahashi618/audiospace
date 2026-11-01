"""HRIR sets: synthetic generation and SOFA file loading.

An `HrirSet` is a directional grid of head-related impulse responses with
shape (n_directions, 2, taps).  Direction lookup is nearest-neighbour on
the unit sphere, which is precomputed into a KD-style dot-product table so
that per-block lookups cost nothing measurable.

`sofar` (MIT) is imported lazily so that the synthetic path has no
dependency beyond numpy/scipy.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .geometry import unit_vectors
from .i18n import tr


@dataclass
class HrirSet:
    """Directional HRIR grid.

    Attributes
    ----------
    data        (n_directions, 2, taps) float32, [:, 0] = left ear
    azimuth     (n_directions,) degrees, SOFA convention
    elevation   (n_directions,) degrees
    sample_rate Hz
    name        human-readable origin, shown in diagnostics
    """

    data: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    sample_rate: int
    name: str = "unnamed"

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        if self.data.ndim != 3 or self.data.shape[1] != 2:
            raise ValueError(
                f"HRIR data must be (n, 2, taps); got {self.data.shape}")
        self.azimuth = np.asarray(self.azimuth, dtype=np.float64) % 360.0
        self.elevation = np.asarray(self.elevation, dtype=np.float64)
        if not (len(self.azimuth) == len(self.elevation)
                == self.data.shape[0]):
            raise ValueError("direction arrays must match data length")
        self._dirs = unit_vectors(self.azimuth, self.elevation)

    @property
    def taps(self) -> int:
        return int(self.data.shape[2])

    @property
    def n_directions(self) -> int:
        return int(self.data.shape[0])

    def index_for(self, azimuth_deg: float, elevation_deg: float = 0.0) -> int:
        """Nearest stored direction (single lookup)."""
        return int(self.indices_for(np.array([azimuth_deg]),
                                    np.array([elevation_deg]))[0])

    def indices_for(self, azimuth_deg: np.ndarray,
                    elevation_deg: np.ndarray | None = None) -> np.ndarray:
        """Vectorised nearest-direction lookup for many sources at once."""
        az = np.asarray(azimuth_deg, dtype=np.float64)
        el = (np.zeros_like(az) if elevation_deg is None
              else np.asarray(elevation_deg, dtype=np.float64))
        query = unit_vectors(az, el)              # (n, 3)
        # nearest on the sphere == largest dot product
        return np.argmax(query @ self._dirs.T, axis=-1).astype(np.intp)

    def resampled(self, sample_rate: int) -> "HrirSet":
        """Return a copy resampled to `sample_rate` (no-op if already equal)."""
        if sample_rate == self.sample_rate:
            return self
        ratio = sample_rate / self.sample_rate
        new_taps = int(round(self.taps * ratio))
        flat = self.data.reshape(-1, self.taps)
        out = signal.resample_poly(flat, sample_rate, self.sample_rate,
                                   axis=-1).astype(np.float32)
        out = out[:, :new_taps].reshape(self.n_directions, 2, -1)
        return HrirSet(out, self.azimuth, self.elevation, sample_rate,
                       f"{self.name}@{sample_rate}Hz")

    def horizontal(self, tolerance_deg: float = 5.0) -> "HrirSet":
        """水平面付近の方向だけを残した小さいセットを返す。

        2D のオーディオゲームは仰角を使わないので、全天球のデータを
        持つ意味がない。それどころか実測で害がある: 1513方向のスペクトル
        表は 12.4MB あり、そこから毎ブロック行を集めるとキャッシュに
        乗らない。64音源での実測が 5.35ms なのに対し、72方向の合成HRIR
        では 2.5ms しかかからないのは、主にこの差である。

        仰角が要らないなら、これを通すだけで表が 1MB 未満に収まる。
        """
        keep = np.flatnonzero(np.abs(self.elevation) <= tolerance_deg)
        if keep.size == 0:
            raise ValueError(
                tr(f"仰角±{tolerance_deg}度に方向がありません "
                   f"(このセットの範囲は {self.elevation.min():.0f}.."
                   f"{self.elevation.max():.0f}度)",
                   f"no directions within ±{tolerance_deg} degrees of "
                   f"elevation (this set covers "
                   f"{self.elevation.min():.0f}..{self.elevation.max():.0f}"
                   " degrees)"))
        return HrirSet(self.data[keep], self.azimuth[keep],
                       self.elevation[keep], self.sample_rate,
                       f"{self.name}-horizontal")

    def decimated(self, azimuth_step_deg: float = 5.0) -> "HrirSet":
        """方位を間引いて軽くする。

        各格子点に最も近い測定を1つずつ選ぶ。仰角は保つので、
        `horizontal()` と併用できる。
        """
        if azimuth_step_deg <= 0:
            raise ValueError(tr("azimuth_step_deg は正である必要があります",
                                "azimuth_step_deg must be positive"))
        keep: list[int] = []
        seen: set[tuple[int, int]] = set()
        el_step = max(azimuth_step_deg, 1.0)
        for i in range(self.n_directions):
            cell = (int(round(self.azimuth[i] / azimuth_step_deg)),
                    int(round(self.elevation[i] / el_step)))
            if cell in seen:
                continue
            seen.add(cell)
            keep.append(i)
        idx = np.asarray(keep, dtype=np.intp)
        return HrirSet(self.data[idx], self.azimuth[idx],
                       self.elevation[idx], self.sample_rate,
                       f"{self.name}-{azimuth_step_deg:g}deg")

    @property
    def table_bytes(self) -> int:
        """畳み込み器が持つスペクトル表のおよその大きさ（ブロック512想定）。

        キャッシュに収まるかどうかの目安。1MB 未満なら軽い。
        """
        return self.n_directions * 2 * 513 * 8

    # -- 配布用の軽量形式 -------------------------------------------------
    def save_npz(self, path: str | Path, **meta) -> Path:
        """.npz に保存する。

        SOFA は 1 ファイル 2.7MB あり、読み込みに sofar と netCDF が要る。
        水平面に絞ったものを .npz にすれば 200KB 台になり、numpy だけで
        読める。ゲームに同梱するのはこちらにする。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, data=self.data, azimuth=self.azimuth,
            elevation=self.elevation, sample_rate=self.sample_rate,
            name=self.name,
            meta=np.asarray(json.dumps(meta, ensure_ascii=False)))
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "HrirSet":
        """save_npz で書いたファイルを読む。依存は numpy だけ。"""
        raw = np.load(Path(path), allow_pickle=False)
        return cls(raw["data"], raw["azimuth"], raw["elevation"],
                   int(raw["sample_rate"]), str(raw["name"]))

    @staticmethod
    def read_npz_meta(path: str | Path) -> dict:
        """付随メタデータ（出典・ライセンス・選定理由など）を読む。"""
        raw = np.load(Path(path), allow_pickle=False)
        if "meta" not in raw:
            return {}
        try:
            return json.loads(str(raw["meta"]))
        except json.JSONDecodeError:
            return {}

    # -- constructors ----------------------------------------------------
    @classmethod
    def synthetic(cls, n_azimuth: int = 72, taps: int = 128,
                  sample_rate: int = 48000,
                  head_radius_m: float = 0.0875) -> "HrirSet":
        """Build a horizontal-plane HRIR set from a spherical head model.

        Not a measurement: it encodes interaural time difference (Woodworth
        formula) plus a head-shadow low-pass on the contralateral ear.  That
        is enough for reliable left/right and useful distance cues, and it
        keeps the library dependency-free.

        Important limitation: both cues depend only on sin(azimuth), so
        mirrored front/back directions are *numerically identical* -- the
        HRIR for 30 deg equals the one for 150 deg.  Front/back must
        therefore be conveyed separately, which is what
        `geometry.front_back_gain` and the engine's `front_back_depth` are
        for.  Load a measured set with `from_sofa` when true front/back
        discrimination, elevation or externalisation matter.
        """
        if n_azimuth < 4 or taps < 16:
            raise ValueError("n_azimuth >= 4 and taps >= 16 required")
        c = 343.0
        data = np.zeros((n_azimuth, 2, taps), dtype=np.float32)
        azimuth = np.arange(n_azimuth, dtype=np.float64) * 360.0 / n_azimuth
        base_delay = 16  # headroom so a negative ITD never clips the start

        for i, az_deg in enumerate(azimuth):
            az = math.radians(az_deg)
            # positive azimuth = to the left, so the left ear leads
            for ear, side in ((0, +1.0), (1, -1.0)):
                # Woodworth ITD for a spherical head: the ear on the same
                # side as the source is reached earlier.
                itd_s = -side * (head_radius_m / c) * math.sin(az)
                delay = base_delay + itd_s * sample_rate
                imp = _fractional_impulse(taps, delay)
                # ipsilateral when sin(az) has the same sign as the ear
                exposure = 0.5 + 0.5 * side * math.sin(az)
                cutoff = 0.12 + 0.85 * exposure          # normalised to Nyquist
                b, a = signal.butter(2, min(0.98, cutoff))
                imp = signal.lfilter(b, a, imp)
                imp *= 0.55 + 0.45 * exposure
                data[i, ear] = imp.astype(np.float32)

        peak = float(np.max(np.abs(data)))
        if peak > 0:
            data /= peak
        return cls(data, azimuth, np.zeros(n_azimuth), sample_rate,
                   f"synthetic-{n_azimuth}dir")

    @classmethod
    def from_sofa(cls, path: str | Path,
                  sample_rate: int | None = None) -> "HrirSet":
        """Load a SimpleFreeFieldHRIR .sofa file via `sofar` (MIT).

        Raises ImportError with an actionable message if sofar is missing.
        """
        try:
            import sofar
        except ImportError as exc:            # pragma: no cover
            raise ImportError(
                "SOFA support needs the 'sofar' package: "
                "python -m pip install sofar") from exc

        # sofar prints a "SOFA file contained custom entries" notice straight
        # to stdout for many real-world files.  Harmless, but a library must
        # not write to a game's console on every load, so it is captured.
        notice = io.StringIO()
        with contextlib.redirect_stdout(notice):
            sofa = sofar.read_sofa(str(Path(path)))
        ir = np.asarray(sofa.Data_IR, dtype=np.float32)   # (M, R, N)
        if ir.ndim != 3 or ir.shape[1] < 2:
            raise ValueError(f"unexpected Data_IR shape {ir.shape}; "
                             "expected (measurements, 2 receivers, taps)")
        ir = ir[:, :2, :]
        fs = int(np.ravel(sofa.Data_SamplingRate)[0])

        pos = np.asarray(sofa.SourcePosition, dtype=np.float64)
        units = str(getattr(sofa, "SourcePosition_Type", "spherical")).lower()
        if "cartesian" in units:
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            az = np.degrees(np.arctan2(y, x))
            el = np.degrees(np.arcsin(np.divide(z, radius,
                                                out=np.zeros_like(z),
                                                where=radius > 0)))
        else:
            az, el = pos[:, 0], pos[:, 1]

        hrir = cls(ir, az, el, fs, Path(path).stem)
        return hrir.resampled(sample_rate) if sample_rate else hrir


def _fractional_impulse(taps: int, delay: float) -> np.ndarray:
    """Unit impulse at a fractional sample position (linear interpolation)."""
    imp = np.zeros(taps, dtype=np.float64)
    delay = float(np.clip(delay, 0.0, taps - 2))
    lo = int(math.floor(delay))
    frac = delay - lo
    imp[lo] = 1.0 - frac
    imp[lo + 1] = frac
    return imp
