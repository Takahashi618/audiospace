"""Geometry helpers for spatial audio.

Coordinate system
-----------------
Right-handed, metres:  x = east, y = north, z = up.

Listener yaw is degrees counter-clockwise from north, so yaw=0 means the
listener faces north (+y) and yaw=90 means facing west (-x).  Listener
pitch tilts the head about its left-right axis after the yaw: positive
pitch looks up, negative looks down.  Listener roll then tilts the head
about its own front-back axis: positive roll lowers the left ear toward
the left shoulder.

Azimuth / elevation follow the SOFA convention so that HRTF data loaded
from a .sofa file can be used without any re-mapping:

    azimuth   0 deg = straight ahead, increasing counter-clockwise
              (90 deg = to the listener's left, 270 deg = right)
    elevation 0 deg = ear level, +90 deg = directly above

This module is pure numpy; it never touches an audio device, so it can be
unit-tested headlessly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Vec3 = tuple[float, float, float]

DistanceModel = Literal["inverse", "inverse_clamped", "linear", "none"]


@dataclass(frozen=True)
class Listener:
    """Position and facing of the listener."""

    position: Vec3 = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0      # 顔の上下。正で上を向く
    roll_deg: float = 0.0       # 首かしげ。正で左耳を左肩へ下げる

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (forward, left) unit vectors in the xy plane.

        These ignore pitch: they describe the body's facing, which is what
        callers moving the listener around a map want.
        """
        rad = math.radians(self.yaw_deg)
        forward = np.array([-math.sin(rad), math.cos(rad), 0.0])
        left = np.array([-forward[1], forward[0], 0.0])
        return forward, left


def relative_spherical(listener: Listener,
                       source: Vec3) -> tuple[float, float, float]:
    """Convert a world-space source position to listener-relative polar.

    Returns (azimuth_deg in [0, 360), elevation_deg in [-90, 90], distance_m).
    A source exactly at the listener yields azimuth 0, elevation 0, distance 0.
    """
    d = np.asarray(source, dtype=np.float64) - np.asarray(listener.position,
                                                          dtype=np.float64)
    distance = float(np.linalg.norm(d))
    if distance < 1e-9:
        return 0.0, 0.0, 0.0

    forward, left = listener.basis()
    f = float(d @ forward)
    lft = float(d @ left)
    if listener.pitch_deg == 0.0 and listener.roll_deg == 0.0:
        # 顔が水平で首も傾けていなければ従来どおりの式。ビット単位で結果を
        # 変えないため、回転の式を通さない（0度の回転でも丸め誤差が入る）。
        azimuth = math.degrees(math.atan2(lft, f)) % 360.0
        horiz = math.hypot(float(d[0]), float(d[1]))
        elevation = math.degrees(math.atan2(float(d[2]), horiz))
        return azimuth, elevation, distance

    # 体の向き（yaw）で水平の前・左に分けたあと、左右の軸まわりに顔を
    # 傾ける。上を向くと、正面にあった音は相対的に下へ動く。
    p = math.radians(listener.pitch_deg)
    up = float(d[2])
    head_f = f * math.cos(p) + up * math.sin(p)
    head_u = -f * math.sin(p) + up * math.cos(p)
    if listener.roll_deg != 0.0:
        # 首かしげ: 前後の軸まわりに傾ける。左耳を下げると頭のてっぺんは
        # 左へ傾くので、真上の音は相対的に右耳側へ、真下の音は左耳側へ
        # 動く。上と下で動く向きが逆になり、正面・水平の音は動かない。
        r = math.radians(listener.roll_deg)
        lft, head_u = (lft * math.cos(r) - head_u * math.sin(r),
                       lft * math.sin(r) + head_u * math.cos(r))
    azimuth = math.degrees(math.atan2(lft, head_f)) % 360.0
    elevation = math.degrees(math.atan2(head_u, math.hypot(head_f, lft)))
    return azimuth, elevation, distance


def relative_spherical_many(listener: Listener, sources: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`relative_spherical` for many sources at once.

    `sources` is (n, 3) world positions.  Returns three (n,) float64 arrays
    (azimuth_deg, elevation_deg, distance_m) with the same conventions,
    including (0, 0, 0) for a source exactly at the listener.

    The engine calls this once per block instead of once per source: at
    128 sources the per-call overhead of the scalar version dominated the
    whole mixer (2.4 ms against 0.06 ms for this).  The arithmetic follows
    the scalar version step by step, so results agree to rounding.
    """
    d = (np.asarray(sources, dtype=np.float64).reshape(-1, 3)
         - np.asarray(listener.position, dtype=np.float64))
    distance = np.sqrt(np.einsum("ij,ij->i", d, d))
    forward, left = listener.basis()
    f = d @ forward
    lft = d @ left
    if listener.pitch_deg == 0.0 and listener.roll_deg == 0.0:
        azimuth = np.degrees(np.arctan2(lft, f)) % 360.0
        elevation = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    else:
        p = math.radians(listener.pitch_deg)
        up = d[:, 2]
        head_f = f * math.cos(p) + up * math.sin(p)
        head_u = -f * math.sin(p) + up * math.cos(p)
        if listener.roll_deg != 0.0:
            r = math.radians(listener.roll_deg)
            lft, head_u = (lft * math.cos(r) - head_u * math.sin(r),
                           lft * math.sin(r) + head_u * math.cos(r))
        azimuth = np.degrees(np.arctan2(lft, head_f)) % 360.0
        elevation = np.degrees(np.arctan2(head_u, np.hypot(head_f, lft)))
    at = distance < 1e-9
    if at.any():
        azimuth[at] = 0.0
        elevation[at] = 0.0
        distance[at] = 0.0
    return azimuth, elevation, distance


def distance_gain_many(distance: np.ndarray,
                       model: DistanceModel = "inverse_clamped",
                       reference: float = 1.0,
                       maximum: float = 60.0,
                       rolloff: float = 1.0) -> np.ndarray:
    """`distance_gain` for an array of distances (same rules, same caps)."""
    d = np.maximum(np.asarray(distance, dtype=np.float64), 0.0)
    if model == "none":
        return np.ones_like(d)
    if model == "linear":
        if maximum <= reference:
            # No ramp between the two: beyond `maximum` is silent, inside
            # is full.  The scalar version never reaches its division here
            # either; computing it for every element would divide by zero
            # (and warn from the audio thread every block).
            return np.where(d >= maximum, 0.0, 1.0)
        lin = 1.0 - rolloff * (d - reference) / (maximum - reference)
        return np.where(d >= maximum, 0.0,
                        np.where(d > reference, lin, 1.0))
    if model == "inverse_clamped":
        d = np.minimum(np.maximum(d, reference), maximum)
    else:
        d = np.maximum(d, 1e-6)
    denom = reference + rolloff * (d - reference)
    safe = np.where(denom <= reference * 1e-9, 1.0, denom)
    return np.where(denom <= reference * 1e-9, 1.0,
                    np.minimum(1.0, reference / safe))


def front_back_gain_many(azimuth_deg: np.ndarray,
                         depth: float = 0.45,
                         elevation_deg: np.ndarray | None = None
                         ) -> np.ndarray:
    """`front_back_gain` for an array of azimuths (and elevations)."""
    forward = np.cos(np.radians(azimuth_deg))
    weight = depth * (1.0 - forward) * 0.5
    if elevation_deg is not None:
        weight = weight * elevation_weight_many(elevation_deg)
    return 1.0 - weight


#: 下側で前後の手がかりを弱め始める仰角の倍率。cos(下45度) × √2 = 1 なので、
#: 下45度までは水平と同じ強さのまま、そこから真下へ向けて弱まる
_LOWER_SCALE = math.sqrt(2.0)


def elevation_weight(elevation_deg: float) -> float:
    """How much of the front/back cue applies at this elevation.

    The cue is about where a source sits around the head *horizontally*.
    Straight overhead (or below) there is no front or back, yet the
    azimuth still survives in the arithmetic -- a zenith source specified
    at azimuth 180 used to be damped and muffled while the same point at
    azimuth 0 was not.  So the weight reaches 0 at both poles.

    Above the horizon it is cos(elevation) (0.71 at +45 degrees).  Below,
    it stays at full strength down to -45 degrees and only then fades,
    as min(1, sqrt(2) * cos(elevation)).  With the elevation emphasis on,
    a source behind and below must sound darker than one behind on the
    horizon; a plain cos weight muffled the lower one *less*, which
    cancelled the emphasis and made the two indistinguishable.
    """
    c = max(0.0, math.cos(math.radians(elevation_deg)))
    if elevation_deg < 0.0:
        return min(1.0, _LOWER_SCALE * c)
    return c


def elevation_weight_many(elevation_deg: np.ndarray) -> np.ndarray:
    """`elevation_weight` for an array of elevations."""
    el = np.asarray(elevation_deg, dtype=np.float64)
    c = np.maximum(0.0, np.cos(np.radians(el)))
    return np.where(el < 0.0, np.minimum(1.0, _LOWER_SCALE * c), c)


def distance_gain(distance: float,
                  model: DistanceModel = "inverse_clamped",
                  reference: float = 1.0,
                  maximum: float = 60.0,
                  rolloff: float = 1.0) -> float:
    """Attenuation for a source `distance` metres away, as a linear gain.

    `inverse_clamped` is the OpenAL default and the sensible choice for an
    audio game: full volume within `reference` metres, then 1/d falloff,
    never attenuating beyond `maximum` metres.

    No model ever returns a gain above 1.0.  The textbook inverse model
    amplifies inside the reference distance (unboundedly, as d approaches
    the singularity); on headphones that is an ear-safety hazard, so a
    source closer than the reference plays at full volume instead.
    """
    if model == "none":
        return 1.0
    d = max(distance, 0.0)
    if model == "linear":
        if d >= maximum:
            return 0.0
        return float(1.0 - rolloff * (d - reference) / (maximum - reference)
                     ) if d > reference else 1.0
    if model == "inverse_clamped":
        d = min(max(d, reference), maximum)
    else:
        d = max(d, 1e-6)
    denom = reference + rolloff * (d - reference)
    if denom <= reference * 1e-9:        # at or inside the model's singularity
        return 1.0
    return float(min(1.0, reference / denom))


def front_back_gain(azimuth_deg: float,
                    depth: float = 0.45,
                    elevation_deg: float = 0.0) -> float:
    """Extra damping cue for sources behind the listener.

    Synthetic HRIR sets convey front/back only weakly, which is exactly the
    confusion blind players report most.  Multiplying a source by this gain
    (and optionally low-passing it by the same amount) reinstates the cue.

    Returns 1.0 straight ahead and `1 - depth` directly behind.  Away
    from the horizontal plane the cue is scaled by `elevation_weight`
    (cos(elevation) above, full strength down to -45 degrees below), so a
    source straight overhead or below is never damped whatever azimuth it
    was given.
    """
    forward = math.cos(math.radians(azimuth_deg))   # +1 front, -1 behind
    weight = depth * (1.0 - forward) * 0.5 * elevation_weight(elevation_deg)
    return float(1.0 - weight)


def wrap_azimuth(azimuth_deg: float) -> float:
    """Normalise any azimuth into [0, 360)."""
    return float(azimuth_deg % 360.0)


def angular_distance_deg(a_az: float, a_el: float,
                         b_az: float, b_el: float) -> float:
    """Great-circle angle between two (azimuth, elevation) directions."""
    a = _unit(a_az, a_el)
    b = _unit(b_az, b_el)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def _unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return np.array([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])


def unit_vectors(azimuth_deg: np.ndarray,
                 elevation_deg: np.ndarray) -> np.ndarray:
    """Vectorised (azimuth, elevation) -> unit vectors, shape (n, 3)."""
    az = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.radians(np.asarray(elevation_deg, dtype=np.float64))
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], axis=-1)
