"""空間の音響: 材質つきの部屋・屋外と、そこから決まる響き。

オーディオゲームでは、同じ足音でも「木の床と木の壁の部屋」「コンクリートの
壁の部屋」「何もない道」「建物の間」で聞こえ方が変わることが、場所を知る
手がかりになる。ここでは場所を `Space`（箱形の空間と、6つの面それぞれの
材質）で表し、そこから2種類の響きを決める。

- 長く残る響き（後期残響）: 材質の吸音率から、帯域ごとの残響時間を Eyring の
  式（空気吸収を含む）で計算する。`reverb.BandFdnReverb` がそれを再現する。
  計算は一瞬なので、ゲーム中に場所が変わってもすぐ切り替えられる
  （部屋を精密に計算する pyroomacoustics は1部屋に数秒かかる）
- 最初の反射（初期反射）: 聞き手から各面までの距離で、反射が届く時刻・方向・
  強さ・音色（その面の材質による帯域ごとの減衰）を決める。`early.EarlyReflections`
  が、面の方向から HRTF で鳴らす。向かい合う2面の間では反射が往復する
  （建物の間の「はためくような反響」）

反射は「聞き手の位置」から計算する（音源ごとには計算しない）。ゲーム向けの
音響エンジンで広く使われる近似で、壁際に立てばその壁の側から反射が聞こえ、
費用は音源の数によらず一定になる。代わりに、聞き手から遠い音源の反射の
細かい違いは表さない。

座標は audiospace 全体と同じ（x = 東、y = 北、z = 上、メートル）。

材質の吸音率は pyroomacoustics の材質データベース（MIT）から書き出した表
（`data/materials.json`、7帯域 125Hz〜8kHz、ATTRIBUTION.md を参照）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .i18n import is_japanese, list_sep, tr

SPEED_OF_SOUND = 343.0
#: 面の名前。x 方向が east/west、y 方向が north/south、z 方向が ceiling/floor
FACES = ("west", "east", "south", "north", "floor", "ceiling")
_FACE_AXIS = {"west": (0, 0), "east": (0, 1), "south": (1, 0),
              "north": (1, 1), "floor": (2, 0), "ceiling": (2, 1)}
#: 空気吸収（エネルギー、1/m）。20℃・湿度50% 程度の概算。7帯域
AIR_ABSORPTION = np.array([0.0001, 0.0003, 0.0006, 0.001, 0.002, 0.005,
                           0.012])
#: 開いた面の割合がこれを超えたら、拡散した後期残響は無いものとする
#: （空の下では音が戻ってこない。Eyring の式は閉じた空間でしか成り立たない）
OPEN_LIMIT = 0.5

# ---------------------------------------------------------------------------
# 置いてあるもの（家具や人）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Furnishing:
    """空間に置くもの1種類。

    absorption_area  吸音の量（m²。1m² の完全に吸う面と同じ、が 1）。7帯域
    blocking_area    反射をふさぎ、散らす広さの目安（m²）

    値は文献の典型値をもとにした目安（大まかに響きを変えるためのもの）。
    """

    label: str
    absorption_area: tuple[float, ...]
    blocking_area: float


#: 置けるもの。キー -> Furnishing
FURNISHINGS: dict[str, Furnishing] = {
    "person": Furnishing(tr("立っている人", "standing person"),
                         (0.20, 0.35, 0.42, 0.46, 0.50, 0.50, 0.50), 0.5),
    "seated_person": Furnishing(tr("座っている人", "seated person"),
                                (0.25, 0.35, 0.45, 0.50, 0.55, 0.55, 0.55),
                                0.4),
    "sofa": Furnishing(tr("ソファ", "sofa"),
                       (0.60, 0.90, 1.20, 1.40, 1.50, 1.50, 1.50),
                       1.6),
    "armchair": Furnishing(tr("布張りのひじ掛け椅子", "upholstered armchair"),
                           (0.25, 0.35, 0.45, 0.50, 0.55, 0.55, 0.55), 0.6),
    "wooden_chair": Furnishing(tr("木の椅子", "wooden chair"),
                               (0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05),
                               0.3),
    "table": Furnishing(tr("テーブル", "table"),
                        (0.05, 0.06, 0.07, 0.08, 0.08, 0.08, 0.08),
                        1.2),
    "bookshelf": Furnishing(tr("本の詰まった本棚", "full bookshelf"),
                            (0.30, 0.40, 0.50, 0.55, 0.60, 0.60, 0.60), 1.5),
    "bed": Furnishing(tr("ベッド", "bed"),
                      (0.50, 0.90, 1.30, 1.50, 1.60, 1.60, 1.60),
                      3.0),
    "cabinet": Furnishing(tr("木の戸棚", "wooden cabinet"),
                          (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
                          1.0),
    "boxes": Furnishing(tr("段ボール箱", "cardboard box"),
                        (0.10, 0.15, 0.20, 0.25, 0.25, 0.25, 0.25),
                        0.5),
}

#: 1種類あたりの数の上限
CONTENTS_MAX = 1000
#: 混み具合の上限（床がすべて物で埋まっても、反射が全部消えはしない）
CLUTTER_MAX = 0.8
#: 混み具合で反射が散らされる割合の帯域ごとの形（高い音ほど散る）
CLUTTER_SCATTER = (0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7)
#: 面ごとの効き方（家具は床の近くにあるので床が最も、天井はあまり）
CLUTTER_FACE_WEIGHT = {"floor": 1.0, "west": 0.6, "east": 0.6,
                       "south": 0.6, "north": 0.6, "ceiling": 0.3}

#: 家具の量。(キー, 読み上げ名)
FURNISH_LEVELS: tuple[tuple[str, str], ...] = (
    ("empty", tr("何も置かない", "None")),
    ("sparse", tr("少し", "A little")),
    ("normal", tr("普通", "Normal")),
    ("full", tr("多い", "A lot")),
)

#: 家具の量ごとの、床 20m² あたりの数（人は含めない）
_FURNISH_PER_20M2 = {
    "empty": {},
    "sparse": {"table": 1, "wooden_chair": 2, "bookshelf": 0.5},
    "normal": {"sofa": 1, "armchair": 1, "table": 1, "wooden_chair": 2,
               "bookshelf": 1},
    "full": {"sofa": 1, "armchair": 2, "table": 1, "wooden_chair": 4,
             "bookshelf": 2, "bed": 1, "cabinet": 1, "boxes": 2},
}


def furnishing_labels() -> list[tuple[str, str]]:
    """(キー, 読み上げ名) の一覧。"""
    return [(k, f.label) for k, f in FURNISHINGS.items()]


def furnish_counts(floor_area_m2: float, level: str) -> dict[str, int]:
    """家具の量（FURNISH_LEVELS のキー）から、床の広さに合わせた数を出す。

    人は含めない（人の数は別に決める）。床 20m² あたりの数を比例させて
    四捨五入する。狭い部屋で 0 になる種類は置かない。
    """
    if level not in _FURNISH_PER_20M2:
        raise KeyError(f"unknown level {level!r}; "
                       f"known: {[k for k, _ in FURNISH_LEVELS]}")
    scale = max(0.0, float(floor_area_m2)) / 20.0
    # round() は偶数への丸め（0.5 が 0 になる）なので、四捨五入を自分で書く
    counts = {k: int(math.floor(v * scale + 0.5))
              for k, v in _FURNISH_PER_20M2[level].items()}
    return {k: min(CONTENTS_MAX, v) for k, v in counts.items() if v > 0}


def contents_text(contents: dict) -> str:
    """読み上げ用の文（例: 「ソファ 1、立っている人 3」）。空なら「なし」。"""
    parts = [f"{FURNISHINGS[k].label} {int(n)}"
             for k, n in contents.items() if int(n) > 0]
    return list_sep().join(parts) if parts else tr("なし", "none")


#: reverb_text で読む帯域と、その読み上げ名
_REVERB_TEXT_BANDS = {125: tr("低い音", "low"), 1000: tr("中ほどの音", "middle"),
                      8000: tr("高い音", "high")}


def reverb_text(space: "Space", sep: str = "") -> str:
    """空間の響きを読み上げ用の文にする。文と文の間は `sep` でつなぐ。

    例: 「残響時間は、低い音 0.9秒、中ほどの音 0.8秒、高い音 0.6秒。
    体積は60立方メートル。」。家具や人を置いた空間では、置かなかったときの
    中ほどの音の残響時間も添える。体積は閉じた部屋のときだけ言う。
    開けた場所では、残響が無いことを言う。
    """
    if space.late_level() <= 0.0:
        return tr("開けた場所なので、広がった残響はありません。",
                  "This is an open place, so there is no spread-out reverb.")
    rt = space.rt60_bands()
    centers = band_centers()
    unit = tr("秒", " seconds")
    parts = [f"{label} {rt[centers.index(c)]:.1f}{unit}"
             for c, label in _REVERB_TEXT_BANDS.items()]
    lines = [tr("残響時間は、" + "、".join(parts) + "。",
                "Reverb time: " + ", ".join(parts) + ".")]
    if space.contents:
        empty = space.with_contents({}).rt60_bands()[centers.index(1000)]
        lines.append(tr(f"置いてあるものが無ければ、中ほどの音は{empty:.1f}秒です。",
                        f"Without the things in it, the middle would be "
                        f"{empty:.1f} seconds."))
    if space.open_fraction == 0.0:
        lines.append(tr(f"体積は{space.volume:.0f}立方メートル。",
                        f"Volume {space.volume:.0f} cubic metres."))
    # 英語は文と文の間に空白が要る（日本語は詰める）
    if sep == "" and not is_japanese():
        sep = " "
    return sep.join(lines)

_DATA =Path(__file__).resolve().parent / "data" / "materials.json"


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def band_centers() -> tuple[int, ...]:
    """材質表の帯域の中心周波数（Hz）。125 から 8000 までの7つ。"""
    return tuple(_table()["center_freqs"])


def material_names() -> list[str]:
    """使える材質の名前の一覧（pyroomacoustics の名前と「open」）。"""
    return sorted(_table()["materials"])


def absorption(material: str | tuple | list | np.ndarray) -> np.ndarray:
    """材質の吸音率（7帯域）。名前、または7つの値をそのまま渡せる。

    名前が表に無いときは KeyError（黙って別の材質にしない）。
    """
    if isinstance(material, str):
        try:
            values = _table()["materials"][material]["absorption"]
        except KeyError:
            raise KeyError(f"unknown material {material!r}; "
                           "see audiospace.material_names()") from None
        return np.asarray(values, dtype=np.float64)
    values = np.asarray(material, dtype=np.float64).reshape(-1)
    if values.size != len(band_centers()):
        raise ValueError(f"absorption needs {len(band_centers())} bands "
                         f"({band_centers()} Hz), got {values.size}")
    return np.clip(values, 0.0, 1.0)


@dataclass(frozen=True)
class ReflectionPath:
    """聞き手に届く1本の反射。

    direction  world 座標の単位ベクトル（反射が来る向き）
    distance   反射の行程（m）。遅れは distance / 音速
    bands      帯域ごとの振幅（材質による減衰と空気吸収、7つ）
    gain       距離による減衰を含めた全体の振幅
    order      反射の回数
    faces      当たった面の名前（順に）
    """

    direction: tuple[float, float, float]
    distance: float
    bands: tuple[float, ...]
    gain: float
    order: int
    faces: tuple[str, ...]

    @property
    def delay_s(self) -> float:
        return self.distance / SPEED_OF_SOUND


@dataclass(frozen=True)
class ReflectionChain:
    """反射の経路の形（聞き手の位置によらない部分）。

    axis   0 = 東西、1 = 南北、2 = 上下
    sign, const   鏡像の座標 = sign・(聞き手の座標) + const
    faces  当たる面の名前（順に）
    bands  当たる面の材質による帯域ごとの振幅の積（7つ。空気吸収は含まない）
    """

    axis: int
    sign: float
    const: float
    faces: tuple[str, ...]
    bands: tuple[float, ...]

    def image(self, coordinate: float) -> float:
        return self.sign * coordinate + self.const


@dataclass(frozen=True)
class Space:
    """材質つきの箱形の空間（部屋・廊下・建物の間・開けた場所）。

    origin      箱の西・南・床の角の world 座標
    size        (東西の幅, 南北の奥行き, 高さ) m
    materials   面ごとの材質。名前か7帯域の吸音率。「open」は何も返らない面
    name        読み上げ用の名前
    scattering  後期残響の拡散の度合いの目安（今は使っていない。将来の精密版用）
    early_order 向かい合う面の間で反射を何回まで追うか
    early_gain  初期反射の全体の強さ
    late_gain   後期残響の全体の強さ（材質と大きさから決まる値に掛ける）
    contents    置いてあるもの（家具や人）。{種類: 数}。種類は FURNISHINGS。
                吸音の分だけ残響が短くなり、混み具合の分だけ反射が弱まる
                （`with_contents` で中身だけ替えた空間を作れる）
    """

    origin: tuple[float, float, float]
    size: tuple[float, float, float]
    materials: dict = field(default_factory=dict)
    name: str = ""
    scattering: float = 0.15
    early_order: int = 4
    early_gain: float = 1.0
    late_gain: float = 1.0
    contents: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.size) <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        for key, count in self.contents.items():
            if key not in FURNISHINGS:
                raise KeyError(f"unknown furnishing {key!r}; "
                               f"known: {sorted(FURNISHINGS)}")
            if not 0 <= int(count) <= CONTENTS_MAX:
                raise ValueError(f"count for {key!r} must be 0..{CONTENTS_MAX}")
        missing = [f for f in FACES if f not in self.materials]
        if missing:
            raise ValueError(f"materials missing for {missing}")
        for face in FACES:
            absorption(self.materials[face])        # 早めに名前を確かめる

    # -- 作り方 ---------------------------------------------------------
    @classmethod
    def room(cls, size: tuple[float, float, float], *, floor, walls,
             ceiling, listener: tuple[float, float, float] = (0.0, 0.0, 0.0),
             at: tuple[float, float] = (0.42, 0.38), ear_height: float = 1.6,
             name: str = "", **kw) -> "Space":
        """部屋。`walls` は4面共通、または {"east": ..., ...} で面ごと。

        聞き手（`listener`、耳の位置）が、部屋の幅・奥行きの `at` の割合の
        ところで、床から `ear_height` の高さに立つように置く。真ん中は避ける
        （箱の真ん中では左右の反射が同時に来て、不自然に聞こえる）。
        """
        w, d, h = size
        lx, ly, lz = listener
        origin = (lx - w * at[0], ly - d * at[1], lz - min(ear_height,
                                                           h - 0.1))
        if isinstance(walls, dict):
            wall = {f: walls[f] for f in ("west", "east", "south", "north")}
        else:
            wall = {f: walls for f in ("west", "east", "south", "north")}
        return cls(origin, (w, d, h),
                   dict(wall, floor=floor, ceiling=ceiling), name, **kw)

    @classmethod
    def street(cls, width: float = 12.0, length: float = 80.0,
               height: float = 20.0, *, ground="rough_concrete",
               facades="brickwork", listener=(0.0, 0.0, 0.0),
               across: float = 0.35, ear_height: float = 1.6,
               name: str = "", **kw) -> "Space":
        """建物にはさまれた道（東西が建物の壁、空と道の先は開いている）。"""
        lx, ly, lz = listener
        origin = (lx - width * across, ly - length / 2, lz - ear_height)
        mats = {"west": facades, "east": facades, "south": "open",
                "north": "open", "floor": ground, "ceiling": "open"}
        return cls(origin, (width, length, height), mats, name, **kw)

    @classmethod
    def open_ground(cls, *, ground="rough_concrete",
                    listener=(0.0, 0.0, 0.0), ear_height: float = 1.6,
                    name: str = "", **kw) -> "Space":
        """周りに何もない屋外。地面の反射だけがある。"""
        lx, ly, lz = listener
        span = 2000.0
        origin = (lx - span / 2, ly - span / 2, lz - ear_height)
        mats = {f: "open" for f in FACES}
        mats["floor"] = ground
        return cls(origin, (span, span, span), mats, name, **kw)

    # -- 性質 ---------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON にできる形（別のプロセスへ渡すとき用）。"""
        return {
            "origin": [float(v) for v in self.origin],
            "size": [float(v) for v in self.size],
            "materials": {f: (m if isinstance(m, str)
                              else [float(a) for a in absorption(m)])
                          for f, m in self.materials.items()},
            "name": self.name, "scattering": float(self.scattering),
            "early_order": int(self.early_order),
            "early_gain": float(self.early_gain),
            "late_gain": float(self.late_gain),
            "contents": {k: int(v) for k, v in self.contents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Space":
        return cls(tuple(data["origin"]), tuple(data["size"]),
                   dict(data["materials"]), data.get("name", ""),
                   data.get("scattering", 0.15), data.get("early_order", 4),
                   data.get("early_gain", 1.0), data.get("late_gain", 1.0),
                   dict(data.get("contents", {})))

    # -- 置いてあるもの ---------------------------------------------------
    def with_contents(self, contents: dict | None) -> "Space":
        """中身（家具や人）だけを替えた空間。数が 0 の種類は外す。

        ゲームで物を置いたり人が入ってきたりしたら、これで作った空間を
        `SpatialPlayer.set_space` に渡す（切り替えはなめらかにつながる）。
        """
        return replace(self, contents={k: int(v) for k, v in
                                       (contents or {}).items() if int(v) > 0})

    def contents_absorption(self) -> np.ndarray:
        """置いてあるものの吸音の量（m²、7帯域）。"""
        total = np.zeros(len(band_centers()))
        for key, count in self.contents.items():
            total += int(count) * np.asarray(FURNISHINGS[key].absorption_area)
        return total

    def clutter(self) -> float:
        """混み具合（0〜CLUTTER_MAX）。置いてあるものがふさぐ面積 ÷ 床の面積。"""
        if not self.contents:
            return 0.0
        floor_area = self.size[0] * self.size[1]
        blocked = sum(int(n) * FURNISHINGS[k].blocking_area
                      for k, n in self.contents.items())
        return float(min(CLUTTER_MAX, blocked / max(floor_area, 1e-6)))

    def face_areas(self) -> dict[str, float]:
        w, d, h = self.size
        return {"west": d * h, "east": d * h, "south": w * h,
                "north": w * h, "floor": w * d, "ceiling": w * d}

    @property
    def volume(self) -> float:
        w, d, h = self.size
        return w * d * h

    def is_open(self, face: str) -> bool:
        return bool(np.all(absorption(self.materials[face]) >= 1.0))

    @property
    def open_fraction(self) -> float:
        """開いた面の面積の割合（0 は閉じた部屋、ほぼ1 は開けた屋外）。"""
        areas = self.face_areas()
        total = sum(areas.values())
        return sum(a for f, a in areas.items() if self.is_open(f)) / total

    def rt60_bands(self) -> np.ndarray:
        """帯域ごとの残響時間（秒、7つ）。Eyring の式に空気吸収を足したもの。

        開いた面が OPEN_LIMIT を超える屋外では 0（拡散した残響は無い。
        地面や建物からの反射は初期反射で表す）。
        """
        if self.open_fraction > OPEN_LIMIT:
            return np.zeros(len(band_centers()))
        areas = self.face_areas()
        total = sum(areas.values())
        # 置いてあるもの（家具や人）の吸音は、面の吸音に面積として足す
        mean_alpha = (sum(areas[f] * absorption(self.materials[f])
                          for f in FACES)
                      + self.contents_absorption()) / total
        mean_alpha = np.minimum(mean_alpha, 0.999)
        v = self.volume
        return 0.161 * v / (-total * np.log1p(-mean_alpha)
                            + 4.0 * AIR_ABSORPTION * v)

    def late_level(self) -> float:
        """後期残響の大きさの目安（振幅）。

        同じ大きさの音を出したとき、拡散した響きのエネルギーは残響時間に
        比例し、空間の体積に反比例する（響きが長いほど大きく、空間が広いほど
        薄まる）。1kHz の残響時間 1秒・体積 200m³ を 1 とした振幅で返す。
        """
        rt = float(self.rt60_bands()[band_centers().index(1000)])
        if rt <= 0.0:
            return 0.0
        level = math.sqrt(rt / self.volume * 200.0)
        return float(min(3.0, max(0.05, level)) * self.late_gain)

    def contains(self, point: tuple[float, float, float]) -> bool:
        """点が、閉じた面の内側にあるか（開いた面の側は問わない）。"""
        lo = np.asarray(self.origin, dtype=np.float64)
        hi = lo + np.asarray(self.size, dtype=np.float64)
        p = np.asarray(point, dtype=np.float64)
        for face, (axis, side) in _FACE_AXIS.items():
            if self.is_open(face):
                continue
            if side == 0 and p[axis] < lo[axis]:
                return False
            if side == 1 and p[axis] > hi[axis]:
                return False
        return True

    def in_region(self, point: tuple[float, float, float]) -> bool:
        """点が、この空間の受け持つ範囲（箱）の中にあるか。

        `contains` と違い、開いた面の側も箱で区切る。場所ごとに空間を
        切り替える（`SpatialEngine.set_spaces`）ときの判定に使う。建物の間の
        道なら、道の端（開いた面）を出たら別の場所になる。
        """
        lo = np.asarray(self.origin, dtype=np.float64)
        hi = lo + np.asarray(self.size, dtype=np.float64)
        p = np.asarray(point, dtype=np.float64)
        return bool(np.all(p >= lo) and np.all(p <= hi))

    def reflection_paths(self, listener: tuple[float, float, float],
                         max_delay_s: float = 0.5,
                         min_gain: float = 0.005,
                         chains: list | None = None) -> list[ReflectionPath]:
        """聞き手の位置から見た反射の一覧（強い順ではなく、軸ごとの順）。

        各軸（東西・南北・上下）について、向かい合う2面の間の反射を
        `early_order` 回まで追う（鏡像法を1次元ずつ）。開いた面に当たった
        経路はそこで終わる。行程が `max_delay_s` を超えるもの、全体の振幅が
        `min_gain` を下回るものは捨てる。聞き手が閉じた面の外にいるときは空。
        `chains` を渡すとその経路だけを見る（`EarlyReflections` が鳴らす分）。
        """
        if not self.contains(listener):
            return []
        p = np.asarray(listener, dtype=np.float64)
        max_len = max_delay_s * SPEED_OF_SOUND
        paths: list[ReflectionPath] = []
        for chain in (self.reflection_chains() if chains is None else chains):
            offset = chain.image(p[chain.axis]) - p[chain.axis]
            length = abs(offset)
            if length <= 1e-6 or length > max_len:
                continue
            air = np.exp(-AIR_ABSORPTION * length / 2.0)
            bands = np.asarray(chain.bands) * air
            # 距離の減衰は、音源を聞き手の位置に置いたときの直接音（1m）との
            # 比。1/(1 + 行程) で、壁から1mなら 1/3
            gain = float(np.mean(bands) / (1.0 + length) * self.early_gain)
            if gain < min_gain:
                continue
            direction = [0.0, 0.0, 0.0]
            direction[chain.axis] = 1.0 if offset > 0 else -1.0
            paths.append(ReflectionPath(
                tuple(direction), float(length),
                tuple(float(b) for b in bands), gain,
                len(chain.faces), chain.faces))
        return paths

    def reflection_chains(self) -> list["ReflectionChain"]:
        """反射の経路の「形」の一覧。聞き手の位置によらない部分。

        各軸（東西・南北・上下）について、向かい合う2面の間の反射を
        `early_order` 回まで追う（鏡像法を1次元ずつ）。開いた面に当たった
        経路はそこで終わる。鏡像の座標は聞き手の座標 x の1次式
        （sign・x + const）なので、位置が変わっても行程は掛け算と足し算で
        求まる（`EarlyReflections` が毎ブロックまとめて計算する）。
        """
        lo = np.asarray(self.origin, dtype=np.float64)
        hi = lo + np.asarray(self.size, dtype=np.float64)
        refl = {f: np.sqrt(np.maximum(0.0, 1.0 - absorption(self.materials[f])))
                for f in FACES}
        # 置いてあるものが反射をふさぎ、散らす分。家具は床の近くにあるので
        # 床の反射を最も弱め、天井はあまり弱めない。高い音ほど散らされる
        clutter = self.clutter()
        if clutter > 0.0:
            shape = np.asarray(CLUTTER_SCATTER)
            refl = {f: r * (1.0 - clutter * CLUTTER_FACE_WEIGHT[f] * shape)
                    for f, r in refl.items()}
        names = (("west", "east"), ("south", "north"), ("floor", "ceiling"))
        chains: list[ReflectionChain] = []
        for axis in range(3):
            low_face, high_face = names[axis]
            mirrors = {low_face: lo[axis], high_face: hi[axis]}
            # (sign, const, 最後に当たった面, 帯域振幅, 面の列)。最初は聞き手自身
            frontier = [(1.0, 0.0, None, np.ones(len(band_centers())), ())]
            for _order in range(self.early_order):
                nxt = []
                for sign, const, last, amp, faces in frontier:
                    for face, plane in mirrors.items():
                        if face == last or np.all(refl[face] <= 0.0):
                            continue
                        # 2・plane - (sign・x + const) = (-sign)・x + (2・plane - const)
                        nxt.append((-sign, 2.0 * float(plane) - const, face,
                                    amp * refl[face], faces + (face,)))
                frontier = nxt
                for sign, const, _last, amp, faces in frontier:
                    chains.append(ReflectionChain(
                        axis, sign, const, faces,
                        tuple(float(a) for a in amp)))
                if not frontier:
                    break
        return chains


def shelf_filter(band_gains: np.ndarray, sample_rate: int
                 ) -> tuple[np.ndarray, float]:
    """7帯域の振幅を、低域・高域のシェルフ2段と全体の倍率で近似する。

    戻り値は (sos, 全体の倍率)。sos は scipy.signal.sosfilt の形 (2, 6)。
    中域（500Hz〜2kHz）の平均を全体の倍率にし、低域（125・250Hz）と
    高域（4k・8kHz）の差をシェルフで付ける。材質の音色（木は低域を吸い、
    じゅうたんは高域を吸う）を表すには、この粗さで十分。
    """
    g = np.maximum(np.asarray(band_gains, dtype=np.float64), 1e-6)
    mid = float(np.exp(np.mean(np.log(g[2:5]))))
    low = float(np.exp(np.mean(np.log(g[0:2])))) / mid
    high = float(np.exp(np.mean(np.log(g[5:7])))) / mid
    nyq = sample_rate / 2.0
    sections = [_rbj_shelf("low", 200.0, low, sample_rate),
                _rbj_shelf("high", min(4000.0, nyq * 0.45), high,
                           sample_rate)]
    return np.vstack(sections), mid


def _rbj_shelf(kind: str, freq: float, gain: float,
               sample_rate: int) -> np.ndarray:
    """RBJ の Audio EQ Cookbook のシェルフ（傾き1）。gain は振幅の比。"""
    a = math.sqrt(max(gain, 1e-6))
    w0 = 2.0 * math.pi * freq / sample_rate
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2.0 * math.sqrt(2.0)
    sa = 2.0 * math.sqrt(a) * alpha
    if kind == "low":
        b0 = a * ((a + 1) - (a - 1) * cw + sa)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - sa)
        a0 = (a + 1) + (a - 1) * cw + sa
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - sa
    else:
        b0 = a * ((a + 1) + (a - 1) * cw + sa)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - sa)
        a0 = (a + 1) - (a - 1) * cw + sa
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - sa
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


#: すぐ使える空間の読み上げ名
_SPACE_NAMES = {
    "wood_room": tr("木の床と木の壁の部屋", "Room with wooden floor and walls"),
    "wood_floor_concrete_walls": tr("木の床とコンクリートの壁の部屋",
                                    "Room with wooden floor and concrete walls"),
    "concrete_room": tr("コンクリートの部屋", "Concrete room"),
    "carpet_room": tr("じゅうたんと吸音天井の部屋",
                      "Room with carpet and absorbing ceiling"),
    "tiled_bathroom": tr("タイル張りの浴室", "Tiled bathroom"),
    "corridor": tr("コンクリートの廊下", "Concrete corridor"),
    "stone_hall": tr("石造りの広間", "Stone hall"),
    "open_road": tr("何もない道", "Empty road"),
    "street": tr("建物の間の道", "Street between buildings"),
    "narrow_alley": tr("狭い路地", "Narrow alley"),
}

#: すぐ使える空間。名前 -> (読み上げ名, 作り方)。聞き手の位置を渡して作る
SPACE_PRESETS: dict[str, tuple[str, object]] = {
    "wood_room": (_SPACE_NAMES["wood_room"], lambda listener: Space.room(
        (5.0, 4.0, 2.7), floor="wood_16mm", walls="wooden_lining",
        ceiling="plasterboard", listener=listener,
        name=_SPACE_NAMES["wood_room"])),
    "wood_floor_concrete_walls": (
        _SPACE_NAMES["wood_floor_concrete_walls"], lambda listener: Space.room(
            (5.0, 4.0, 2.7), floor="wood_16mm", walls="rough_concrete",
            ceiling="plasterboard", listener=listener,
            name=_SPACE_NAMES["wood_floor_concrete_walls"])),
    "concrete_room": (_SPACE_NAMES["concrete_room"], lambda listener: Space.room(
        (5.0, 4.0, 2.7), floor="concrete_floor", walls="rough_concrete",
        ceiling="rough_concrete", listener=listener,
        name=_SPACE_NAMES["concrete_room"])),
    "carpet_room": (_SPACE_NAMES["carpet_room"], lambda listener: Space.room(
        (5.0, 4.0, 2.7), floor="carpet_soft_10mm", walls="plasterboard",
        ceiling="ceiling_fibre_absorber", listener=listener,
        name=_SPACE_NAMES["carpet_room"])),
    # 全面タイルだと低域の残響が 5.8秒になる（表のタイルはほとんど吸わない）。
    # 実際の浴室は扉や天井で吸うので、木の扉1面と石膏ボードの天井にした
    # （1.4〜1.7秒）
    "tiled_bathroom": (_SPACE_NAMES["tiled_bathroom"], lambda listener: Space.room(
        (2.5, 2.0, 2.4), floor="ceramic_tiles",
        walls={"west": "ceramic_tiles", "east": "ceramic_tiles",
               "north": "ceramic_tiles", "south": "wooden_door"},
        ceiling="plasterboard", listener=listener,
        name=_SPACE_NAMES["tiled_bathroom"])),
    "corridor": (_SPACE_NAMES["corridor"], lambda listener: Space.room(
        (2.2, 20.0, 2.6), floor="linoleum_on_concrete",
        walls="rough_concrete", ceiling="plasterboard", listener=listener,
        at=(0.45, 0.5), name=_SPACE_NAMES["corridor"])),
    # 天井まで石だと低域が 22秒になる。石造りの教会に多い木の屋根にした
    # （約6秒）
    "stone_hall": (_SPACE_NAMES["stone_hall"], lambda listener: Space.room(
        (20.0, 30.0, 12.0), floor="marble_floor", walls="limestone_wall",
        ceiling="wooden_lining", listener=listener,
        name=_SPACE_NAMES["stone_hall"])),
    "open_road": (_SPACE_NAMES["open_road"], lambda listener: Space.open_ground(
        ground="rough_concrete", listener=listener,
        name=_SPACE_NAMES["open_road"])),
    "street": (_SPACE_NAMES["street"], lambda listener: Space.street(
        width=12.0, ground="rough_concrete", facades="brickwork",
        listener=listener, name=_SPACE_NAMES["street"])),
    "narrow_alley": (_SPACE_NAMES["narrow_alley"], lambda listener: Space.street(
        width=3.0, length=40.0, height=12.0, ground="rough_concrete",
        facades="rough_concrete", listener=listener, across=0.4,
        name=_SPACE_NAMES["narrow_alley"])),
}


def space_preset(name: str, listener=(0.0, 0.0, 0.0)) -> Space:
    """名前で空間を作る（聞き手の位置に合わせて置く）。"""
    try:
        _label, make = SPACE_PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown space {name!r}; "
                       f"known: {sorted(SPACE_PRESETS)}") from None
    return make(listener)


def space_labels() -> list[tuple[str, str]]:
    """(名前, 読み上げ名) の一覧。"""
    return [(k, v[0]) for k, v in SPACE_PRESETS.items()]


# ---------------------------------------------------------------------------
# 材質の読み上げ名（日本語は materials_ja.json、英語は materials_en.json）
# ---------------------------------------------------------------------------
_DATA_JA = Path(__file__).resolve().parent / "data" / "materials_ja.json"
_DATA_EN = Path(__file__).resolve().parent / "data" / "materials_en.json"


@lru_cache(maxsize=2)
def _table_lang(japanese: bool) -> dict:
    path = _DATA_JA if japanese else _DATA_EN
    return json.loads(path.read_text(encoding="utf-8"))


def _table_names() -> dict:
    """いまの言語の材質の名前の表（categories と names）。"""
    return _table_lang(is_japanese())


def material_label(name: str) -> str:
    """材質の読み上げ名。表に無い名前はそのまま返す。"""
    if not isinstance(name, str):
        return tr("吸音率を直接指定した材質",
                  "Material with absorption given directly")
    return _table_names()["names"].get(name, name)


#: 面ごとに合う材質（この順で一覧の上に出す）。壁は前後左右で共通
FACE_MATERIALS: dict[str, tuple[str, ...]] = {
    "floor": (
        "concrete_floor", "marble_floor", "ceramic_tiles", "rough_concrete",
        "unpainted_concrete", "hard_surface",
        "wood_16mm", "wood_1.6cm", "audience_floor", "stage_floor",
        "linoleum_on_concrete", "carpet_thin", "carpet_cotton", "felt_5mm",
        "carpet_rubber_5mm", "carpet_6mm_closed_cell_foam",
        "carpet_6mm_open_cell_foam", "carpet_tufted_9m",
        "carpet_tufted_9.5mm", "carpet_1.35_kg_m2", "carpet_soft_10mm",
        "carpet_hairy", "cocos_fibre_roll_29mm",
    ),
    "wall": (
        "hard_surface", "brickwork", "rough_concrete", "unpainted_concrete",
        "rough_lime_wash", "smooth_brickwork_flush_pointing",
        "smooth_brickwork_10mm_pointing", "brick_wall_rough",
        "ceramic_tiles", "limestone_wall", "reverb_chamber",
        "plasterboard", "wooden_lining", "plywood_thin", "wood_16mm",
        "wood_1.6cm", "wooden_door",
        "glass_3mm", "glass_window", "lead_glazing", "double_glazing_30mm",
        "double_glazing_10mm", "double_glazing_inside",
        "curtains_0.2", "curtains_cotton_0.33", "curtains_cotton_0.5",
        "curtains_densely_woven", "curtains_velvet", "curtains_fabric",
        "curtains_fabric_folded", "curtains_glass_mat", "studio_curtains",
        "blinds_half_open", "blinds_open",
        "panel_fabric_covered_6pcf", "panel_fabric_covered_8pcf",
        "facing_brick", "acoustical_plaster_25mm", "rockwool_50mm_80kgm3",
        "rockwool_50mm_40kgm3", "mineral_wool_50mm_40kgm3",
        "mineral_wool_50mm_70kgm3", "gypsum_board",
        "perforated_veneered_chipboard", "fibre_absorber_1",
        "fibre_absorber_2", "microperforated_foil_kaefer",
        "microperforated_glass_sheets",
        "open",
    ),
    "ceiling": (
        "plasterboard", "ceiling_plasterboard", "ceiling_fibre_absorber",
        "ceiling_fissured_tile", "ceiling_perforated_gypsum_board",
        "ceiling_melamine_foam", "ceiling_metal_panel",
        "wooden_lining", "plywood_thin", "wood_16mm",
        "hard_surface", "rough_concrete", "unpainted_concrete",
        "rough_lime_wash", "limestone_wall", "acoustical_plaster_25mm",
        "glass_window", "hanging_absorber_panels_1",
        "hanging_absorber_panels_2", "microperforated_foil_kaefer",
        "open",
    ),
}

#: 面の合う材質のあとに並べる、その他の材質の読み上げ名の頭
OTHER_PREFIX = tr("その他 ", "Other, ")


def _face_group(face: str) -> str:
    if face in ("floor", "ceiling"):
        return face
    if face in ("west", "east", "south", "north", "wall"):
        return "wall"
    raise KeyError(f"unknown face {face!r}")


def material_choices(face: str | None = None) -> list[tuple[str, str]]:
    """(名前, 読み上げ名) の一覧。読み上げ名は「分類: 名前」。

    材質は91種類あるので、選ぶ一覧では分類を頭に付けて、上下の矢印で
    たどったときに今どの辺りにいるかが分かるようにする。

    `face`（"floor"、"ceiling"、壁の面の名前か "wall"）を渡すと、その面に
    合う材質（FACE_MATERIALS）を先に並べ、残りを分類順にその後へ並べる。
    残りの読み上げ名には「その他」を頭に付ける（合う材質の外に出たことが
    耳で分かるように）。渡さなければ、全部を分類順に並べる。
    """
    table = _table()["materials"]
    ja = _table_names()
    order = list(ja["categories"])
    names = list(table)                      # 表の順（分類の中ではこの順）
    names.sort(key=lambda n: (order.index(table[n]["category"])
                              if table[n]["category"] in order
                              else len(order)))

    def label(n: str) -> str:
        cat = ja["categories"].get(table[n]["category"],
                                   table[n]["category"])
        return f"{cat}: {material_label(n)}"

    if face is None:
        return [(n, label(n)) for n in names]
    first = FACE_MATERIALS[_face_group(face)]
    rest = [n for n in names if n not in first]
    return ([(n, label(n)) for n in first]
            + [(n, OTHER_PREFIX + label(n)) for n in rest])


# ---------------------------------------------------------------------------
# 空間の設計図（自分で作る空間を JSON に保存するための形）
# ---------------------------------------------------------------------------
#: 空間の種類。(キー, 読み上げ名)
SPACE_KINDS: tuple[tuple[str, str], ...] = (
    ("room", tr("部屋", "Room")),
    ("street", tr("建物の間の道", "Street between buildings")),
    ("open", tr("周りに何もない場所", "Open place")),
)

#: 耳の高さ（床から）。部屋が低ければ天井の 10cm 下で頭打ち
EAR_HEIGHT_M = 1.6

#: 設計図で決められる値の範囲
SPEC_LIMITS = {
    "size": (0.5, 500.0),        # 幅・奥行き・高さ、m
    "at": (0.05, 0.95),          # 聞き手の位置（幅・奥行きに対する割合）
    "early_order": (1, 8),       # 向かい合う面の間の反射を何回まで追うか
    "gain": (0.0, 3.0),          # 反射の強さ・残響の強さ
    "name": 40,                  # 名前の長さ（文字）
}


def _number(value, kind=float):
    """数に直す。直せなければ None（検査で例外を出さないため）。"""
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _numbers(seq) -> list | None:
    """並び（リスト・タプル・配列など）を数のリストに直す。並びでなければ None。

    文字列と辞書は並びとして扱わない（"abc" を3つの値と取り違えない）。
    """
    if isinstance(seq, (str, bytes, dict)):
        return None
    try:
        return [_number(v) for v in seq]
    except TypeError:
        return None


def space_spec_problems(spec: dict) -> list[str]:
    """設計図の問題点（いまの言語の文）。問題が無ければ空。

    どんな形の値を渡されても例外は出さず、問題点の文で返す（手で編集した
    保存ファイルを読むときにも使うので、数でない値や形の違う値もありうる）。
    """
    if not isinstance(spec, dict):
        return [tr("設計図の形が違います（辞書ではありません）",
                   "The design has the wrong shape (it is not a dictionary)")]
    problems: list[str] = []
    kind = spec.get("kind")
    kinds = [k for k, _ in SPACE_KINDS]
    if kind not in kinds:
        return [tr(f"種類が分かりません（{kind!r}）",
                   f"Unknown kind ({kind!r})")]
    name = str(spec.get("name", ""))
    if len(name) > SPEC_LIMITS["name"]:
        problems.append(tr(f"名前は{SPEC_LIMITS['name']}文字以内にしてください",
                           f"Use a name of {SPEC_LIMITS['name']} characters "
                           "or fewer"))
    need = {"room": FACES, "street": ("west", "east", "floor"),
            "open": ("floor",)}[kind]
    mats = spec.get("materials", {})
    if not isinstance(mats, dict):
        problems.append(tr("材質の形が違います",
                           "The materials have the wrong shape"))
        mats = {}
    for face in need:
        if face not in mats:
            problems.append(tr(f"{FACE_LABELS[face]}の材質がありません",
                               f"The {FACE_LABELS[face].lower()} has no "
                               "material"))
            continue
        try:
            absorption(mats[face])
        except (KeyError, ValueError, TypeError):
            problems.append(tr(f"{FACE_LABELS[face]}の材質が分かりません"
                               f"（{mats[face]!r}）",
                               f"Unknown material for the "
                               f"{FACE_LABELS[face].lower()} "
                               f"({mats[face]!r})"))
    if kind != "open":
        lo, hi = SPEC_LIMITS["size"]
        size = spec.get("size", ())
        values = _numbers(size)
        if values is None or len(values) != 3:
            problems.append(tr("大きさは幅・奥行き・高さの3つが要ります",
                               "The size needs three values: width, depth "
                               "and height"))
        else:
            for label, v in zip((tr("幅", "Width"), tr("奥行き", "Depth"),
                                 tr("高さ", "Height")), values):
                if v is None or not lo <= v <= hi:
                    problems.append(tr(
                        f"{label}は{lo:g}メートルから{hi:g}メートルの"
                        "間にしてください",
                        f"{label} must be between {lo:g} and {hi:g} metres"))
        lo, hi = SPEC_LIMITS["at"]
        at = spec.get("at", (0.5, 0.5))
        values = _numbers(at)
        if (values is None or len(values) != 2
                or not all(v is not None and lo <= v <= hi for v in values)):
            problems.append(tr(f"聞き手の位置は{lo * 100:.0f}パーセントから"
                               f"{hi * 100:.0f}パーセントの間にしてください",
                               f"The listener position must be between "
                               f"{lo * 100:.0f} and {hi * 100:.0f} percent"))
    contents = spec.get("contents", {})
    if not isinstance(contents, dict):
        problems.append(tr("置いてあるものの形が違います",
                           "The contents have the wrong shape"))
        contents = {}
    for key, count in contents.items():
        number = _number(count, int)
        if key not in FURNISHINGS:
            problems.append(tr(f"置いてあるものの種類が分かりません（{key!r}）",
                               f"Unknown kind of contents ({key!r})"))
        elif number is None or not 0 <= number <= CONTENTS_MAX:
            problems.append(tr(f"{FURNISHINGS[key].label}の数は0から"
                               f"{CONTENTS_MAX}の間にしてください",
                               f"The number of {FURNISHINGS[key].label} must "
                               f"be between 0 and {CONTENTS_MAX}"))
    lo, hi = SPEC_LIMITS["early_order"]
    order = _number(spec.get("early_order", 4), int)
    if order is None or not lo <= order <= hi:
        problems.append(tr(f"反射の回数は{lo}から{hi}の間にしてください",
                           f"The reflection order must be between {lo} "
                           f"and {hi}"))
    lo, hi = SPEC_LIMITS["gain"]
    for key, label in (("early_gain", tr("反射の強さ", "Reflection level")),
                       ("late_gain", tr("残響の強さ", "Reverb level"))):
        value = _number(spec.get(key, 1.0))
        if value is None or not lo <= value <= hi:
            problems.append(tr(f"{label}は{lo * 100:.0f}パーセントから"
                               f"{hi * 100:.0f}パーセントの間にしてください",
                               f"{label} must be between {lo * 100:.0f} and "
                               f"{hi * 100:.0f} percent"))
    return problems


def space_from_spec(spec: dict, listener=(0.0, 0.0, 0.0)) -> Space:
    """設計図から空間を作る。聞き手（耳の位置）に合わせて置く。

    設計図の形（JSON にそのまま書ける）:

        {"kind": "room" | "street" | "open", "name": "...",
         "size": [幅, 奥行き, 高さ],          # m。open では使わない
         "materials": {"floor": ..., "ceiling": ..., "west": ...,
                       "east": ..., "south": ..., "north": ...},
         "at": [左右の位置, 前後の位置],       # 幅・奥行きに対する割合
         "early_order": 4, "early_gain": 1.0, "late_gain": 1.0,
         "contents": {"sofa": 1, "person": 3}}  # 置いてあるもの（省略可）

    聞き手は北（前）を向いている前提で、west が左、east が右、north が前、
    south が後ろの壁。street では前後と空は開いた面になり、open では床
    （地面）だけが返る。値がおかしければ ValueError（問題点をいまの言語で）。
    """
    problems = space_spec_problems(spec)
    if problems:
        raise ValueError(tr("、", "; ").join(problems))
    kind = spec["kind"]
    mats = dict(spec["materials"])
    kw = {"early_order": int(spec.get("early_order", 4)),
          "early_gain": float(spec.get("early_gain", 1.0)),
          "late_gain": float(spec.get("late_gain", 1.0)),
          "contents": {k: int(v) for k, v in
                       dict(spec.get("contents", {})).items() if int(v) > 0}}
    name = str(spec.get("name", ""))
    if kind == "open":
        return Space.open_ground(ground=mats["floor"], listener=listener,
                                 ear_height=EAR_HEIGHT_M, name=name, **kw)
    w, d, h = (float(v) for v in spec["size"])
    ax, ay = (float(v) for v in spec.get("at", (0.42, 0.38)))
    lx, ly, lz = listener
    origin = (lx - w * ax, ly - d * ay, lz - min(EAR_HEIGHT_M, h - 0.1))
    if kind == "street":
        for face in ("south", "north", "ceiling"):
            mats[face] = "open"
    return Space(origin, (w, d, h), {f: mats[f] for f in FACES}, name, **kw)


def space_to_spec(space: Space, listener=(0.0, 0.0, 0.0)) -> dict:
    """空間を設計図に戻す（用意した空間を元にして作り直すとき用）。"""
    mats = {f: (m if isinstance(m, str) else [float(a) for a in absorption(m)])
            for f, m in space.materials.items()}
    opened = {f for f, m in mats.items() if m == "open"}
    if opened >= {"west", "east", "south", "north", "ceiling"}:
        kind = "open"
    elif opened >= {"south", "north", "ceiling"}:
        kind = "street"
    else:
        kind = "room"
    w, d, h = (float(v) for v in space.size)
    ox, oy, _ = space.origin
    lx, ly, _ = listener
    spec = {"kind": kind, "name": space.name, "materials": mats,
            "early_order": int(space.early_order),
            "early_gain": float(space.early_gain),
            "late_gain": float(space.late_gain)}
    if space.contents:
        spec["contents"] = {k: int(v) for k, v in space.contents.items()}
    if kind != "open":
        spec["size"] = [w, d, h]
        spec["at"] = [round((lx - ox) / w, 4), round((ly - oy) / d, 4)]
    return spec


#: 面の読み上げ名（聞き手は北を向いている前提）
FACE_LABELS = {"floor": tr("床", "Floor"), "ceiling": tr("天井", "Ceiling"),
               "north": tr("前の壁", "Front wall"),
               "south": tr("後ろの壁", "Back wall"),
               "west": tr("左の壁", "Left wall"),
               "east": tr("右の壁", "Right wall")}
