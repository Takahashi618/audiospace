"""Room presets and binaural impulse-response generation.

`render_room_ir` drives pyroomacoustics (MIT) with a two-microphone array
standing in for a pair of ears, so the resulting IR is already stereo and
slightly decorrelated -- which is what makes a convolution reverb sound
like a space rather than a wash.

Generated IRs are cached as .npz next to the caller's chosen directory:
simulating a cathedral takes seconds, so a game must not do it at startup.
Use `tools/build_reverb_presets.py` to bake them ahead of time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .i18n import tr


@dataclass(frozen=True)
class RoomPreset:
    """A shoebox room to simulate.

    absorption is the fraction of energy absorbed per reflection: 0.02 is a
    tiled bathroom, 0.6 is a heavily furnished living room.
    """

    name: str
    dimensions: tuple[float, float, float]
    absorption: float
    max_order: int = 17
    description: str = ""
    source_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ear_distance_m: float = 0.17


BUILTIN_PRESETS: dict[str, RoomPreset] = {
    p.name: p for p in [
        RoomPreset("bathroom", (2.5, 2.0, 2.4), 0.04, 20,
                   tr("小さく硬い部屋。強い初期反射と金属的な響き",
                      "Small hard room. Strong early reflections and a "
                      "metallic ring")),
        RoomPreset("small_room", (4.0, 3.5, 2.5), 0.25, 15,
                   tr("一般的な居室。短く自然な残響",
                      "Ordinary living room. Short, natural reverb")),
        RoomPreset("corridor", (2.0, 18.0, 2.6), 0.15, 20,
                   tr("細長い廊下。前後方向にはっきりした反射",
                      "Long narrow corridor. Clear reflections front and "
                      "back")),
        RoomPreset("hall", (18.0, 26.0, 9.0), 0.16, 20,
                   tr("コンサートホール規模。長く滑らかな残響",
                      "Concert hall size. Long, smooth reverb")),
        RoomPreset("cathedral", (24.0, 60.0, 22.0), 0.06, 22,
                   tr("石造の大聖堂。非常に長い残響",
                      "Stone cathedral. Very long reverb")),
        RoomPreset("cave", (12.0, 20.0, 6.0), 0.05, 22,
                   tr("洞窟。低域が強く残る不規則な響き",
                      "Cave. Irregular reverb with lingering lows")),
        RoomPreset("warehouse", (30.0, 40.0, 12.0), 0.20, 18,
                   tr("倉庫。広いが吸音があり明瞭",
                      "Warehouse. Large but absorbent and clear")),
    ]
}


@dataclass
class RoomIr:
    """A rendered stereo impulse response plus how it was made."""

    preset: RoomPreset
    ir: np.ndarray                      # (2, taps) float32
    sample_rate: int
    rt60: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.ir.shape[1] / self.sample_rate)

    def resampled(self, sample_rate: int) -> "RoomIr":
        """Convert to another sample rate.

        Needed more often than one would expect: WASAPI shared mode forces
        the stream to the device's configured rate, so an IR baked at 48 kHz
        is unusable on a card running at 44.1 kHz unless it is converted.
        """
        if sample_rate == self.sample_rate:
            return self
        from math import gcd

        from scipy import signal

        g = gcd(int(self.sample_rate), int(sample_rate))
        out = signal.resample_poly(self.ir, sample_rate // g,
                                   self.sample_rate // g,
                                   axis=-1).astype(np.float32)
        return RoomIr(self.preset, out, sample_rate, self.rt60,
                      dict(self.meta, resampled_from=self.sample_rate))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, ir=self.ir, sample_rate=self.sample_rate,
            rt60=self.rt60, name=self.preset.name,
            dimensions=np.asarray(self.preset.dimensions),
            absorption=self.preset.absorption)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RoomIr":
        data = np.load(Path(path), allow_pickle=False)
        name = str(data["name"])
        preset = BUILTIN_PRESETS.get(
            name, RoomPreset(name, tuple(data["dimensions"]),
                             float(data["absorption"])))
        return cls(preset, data["ir"].astype(np.float32),
                   int(data["sample_rate"]), float(data["rt60"]))


def render_room_ir(preset: RoomPreset, sample_rate: int = 48000,
                   max_seconds: float = 4.0) -> RoomIr:
    """Simulate `preset` and return a stereo IR.

    The listener is placed slightly off-centre and the source diagonally
    across the room; both are deliberately away from the exact centre,
    because a symmetric placement in a shoebox produces comb filtering that
    sounds artificial.
    """
    try:
        import pyroomacoustics as pra
    except ImportError as exc:            # pragma: no cover
        raise ImportError(
            "room simulation needs pyroomacoustics: "
            "python -m pip install pyroomacoustics") from exc

    from .reverb import rt60_of

    w, d, h = preset.dimensions
    room = pra.ShoeBox(list(preset.dimensions), fs=sample_rate,
                       max_order=preset.max_order,
                       materials=pra.Material(preset.absorption))

    listener = np.array([w * 0.42, d * 0.38, min(1.65, h * 0.55)])
    half = preset.ear_distance_m / 2.0
    ears = np.c_[listener + [0.0, 0.0, 0.0], listener + [0.0, 0.0, 0.0]]
    ears[0, 0] -= half          # left ear toward -x
    ears[0, 1] += half          # right ear toward +x
    room.add_microphone_array(pra.MicrophoneArray(ears, sample_rate))

    source = np.array([w * 0.68, d * 0.72, min(1.6, h * 0.5)])
    source = source + np.asarray(preset.source_offset, dtype=np.float64)
    source = np.clip(source, 0.2, np.asarray(preset.dimensions) - 0.2)
    room.add_source(source.tolist())

    room.compute_rir()
    limit = int(max_seconds * sample_rate)
    left = np.asarray(room.rir[0][0], dtype=np.float32)[:limit]
    right = np.asarray(room.rir[1][0], dtype=np.float32)[:limit]
    n = max(len(left), len(right))
    ir = np.zeros((2, n), dtype=np.float32)
    ir[0, :len(left)] = left
    ir[1, :len(right)] = right

    peak = float(np.max(np.abs(ir)))
    if peak > 0:
        ir /= peak

    return RoomIr(preset, ir, sample_rate, rt60_of(ir, sample_rate),
                  meta={"listener": listener.tolist(),
                        "source": source.tolist()})


#: where tools/build_reverb_presets.py bakes its IRs (repository layout).
#: Absent in an installed wheel, in which case the loaders return nothing
#: and callers fall back to the FDN or to live simulation.
REVERB_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "reverb"


def load_room_ir(name: str, sample_rate: int,
                 directory: str | Path | None = None) -> RoomIr | None:
    """Load one pre-baked IR by preset name, or None when nothing is baked.

    Prefers an IR baked at exactly `sample_rate`; otherwise takes whatever
    rate was baked and resamples it.  Without that fallback a card
    configured for 44.1 kHz would silently find no reverb at all, because
    WASAPI forces the stream to the device's own rate.
    """
    folder = Path(directory) if directory else REVERB_ASSET_DIR
    if not folder.exists():
        return None
    exact = folder / f"{name}_{int(sample_rate)}.npz"
    if exact.exists():
        return RoomIr.load(exact)
    candidates = sorted(folder.glob(f"{name}_*.npz"))
    if not candidates:
        return None
    return RoomIr.load(candidates[0]).resampled(int(sample_rate))


def load_all(sample_rate: int = 48000,
             directory: str | Path | None = None) -> dict[str, RoomIr]:
    """Load every baked IR: name -> RoomIr at `sample_rate`.

    Same rate fallback as `load_room_ir`.  Lived in
    tools/build_reverb_presets.py originally, which made the baked assets
    unreachable from the library itself and forced use_room_ir() to
    re-simulate rooms at runtime.
    """
    folder = Path(directory) if directory else REVERB_ASSET_DIR
    out: dict[str, RoomIr] = {}
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.npz")):
        room = RoomIr.load(path)
        name = room.preset.name
        if name in out and out[name].sample_rate == sample_rate:
            continue            # already have an exact-rate version
        out[name] = room.resampled(sample_rate)
    return out


def preset_names() -> list[str]:
    """Names of the built-in presets, for menus and command-line help."""
    return sorted(BUILTIN_PRESETS)


def describe_presets() -> str:
    """Plain-text listing suitable for reading aloud with a screen reader."""
    lines = []
    for i, name in enumerate(preset_names(), start=1):
        p = BUILTIN_PRESETS[name]
        w, d, h = p.dimensions
        lines.append(f"{i}. {name} — {p.description} " + tr(
            f"(幅{w:g}m 奥行{d:g}m 高さ{h:g}m 吸音率{p.absorption:g})",
            f"(width {w:g} m, depth {d:g} m, height {h:g} m, "
            f"absorption {p.absorption:g})"))
    return "\n".join(lines)


def preset_reverb_spec(name: str) -> dict:
    """用意した部屋の、FDN の残響としての設計図（`SpatialPlayer.set_reverb`
    が名前を受け取ったときと同じ値）。自分で作る残響の元にする。"""
    preset = BUILTIN_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"unknown reverb preset {name!r}; "
                         f"known: {sorted(BUILTIN_PRESETS)}")
    w, d, h = preset.dimensions
    volume = float(w * d * h)
    area = 2.0 * (w * d + w * h + d * h)
    rt60 = max(0.25, 0.161 * volume / (preset.absorption * area))
    return {"name": name, "rt60": min(rt60, 8.0),
            "room_scale": min(2.5, (volume / 40.0) ** (1 / 3)),
            "damping": 0.2 + 0.5 * preset.absorption,
            "width": 1.0, "level": 1.0}
