"""部屋をまたぐ音の遮蔽: 壁の材質で、こもりと音量の下がりを決める。

`SpatialEngine.set_spaces` で場所ごとの空間（箱）を登録してあるとき、
音源と聞き手を結ぶ線が閉じた面（壁・床・天井）を通り抜けたら、その面の
材質から、どれだけこもらせ、どれだけ小さくするかを決める。

- 面の材質の分類（材質表の category）ごとに、下げる量（dB）とこもり
  （0〜1、`set_source_lowpass` と同じ尺度）を決めてある
  （`OCCLUSION_BY_CATEGORY`）。値は実際の遮音性能よりずっと小さい。
  コンクリートの壁は実際には 50dB ほど音を下げ、向こうの音はほとんど
  聞こえない。目の見えないプレイヤーには、壁の向こうの音がこもって
  聞こえること自体が手がかりなので、聞こえる範囲に留める
- 隣り合う2つの部屋の壁のように、ほぼ同じ平面にある面（間が
  `WALL_GAP_M` 以内）は、1枚の壁として数える（強いほうの値）
- 戸口・窓・吹き抜けは `Opening`（穴を囲む箱）で壁に穴をあける。開いた穴を
  通る線は遮られない。戸を閉めたら、穴の材質を戸の材質にする
- 回り込み: まっすぐの線が壁で遮られるとき、開いた面や穴を通って届く道が
  あれば、その道も調べる。回り込んだ道のほうが大きく聞こえるなら、音は
  聞き手の部屋の出入口の方向から、道のりの距離で聞こえる。曲がるほど
  小さく、こもる（回折の近似）

座標は audiospace 全体と同じ（x = 東、y = 北、z = 上、メートル）。
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .acoustics import FACES, _FACE_AXIS, _table, absorption

#: 材質の分類ごとの (下げる量 dB, こもり 0〜1)。こもり自体も音を小さくする
#: （広い帯域の雑音で、こもり 0.85 は約 8dB、0.7 は約 6dB）ので、合わせると
#: コンクリートの壁で約 22dB、木の戸で約 14dB、石膏ボードで約 11dB、カーテンで約 3dB 下がる
OCCLUSION_BY_CATEGORY = {
    "Massive constructions and hard surfaces": (14.0, 0.85),
    "Floor coverings": (12.0, 0.8),
    "Wood": (9.0, 0.7),
    "Wall absorbers": (9.0, 0.7),
    "Ceiling absorbers": (9.0, 0.7),
    "Special absorbers": (9.0, 0.7),
    "Lightweight constructions and linings": (7.0, 0.6),
    "Glazing": (7.0, 0.5),
    "Curtains": (2.0, 0.25),
    "Seating (2 seats per m2)": (1.0, 0.15),
    "Audience (unless not specified explicitly, 2 persons per m2)":
        (1.0, 0.15),
}
#: 分類の分からない材質（吸音率だけを渡した面など）
DEFAULT_OCCLUSION = (9.0, 0.7)
#: 何枚の壁を通っても、これ以上は下げない（聞こえなくしない）
MAX_LOSS_DB = 40.0
MAX_LOWPASS = 0.95
#: これより近い平行な面は、1枚の壁（両側の部屋の面）として数える
WALL_GAP_M = 0.5
#: 回り込みで1回曲がるときの、曲がる角度 90度あたりの下がりとこもり
BEND_DB_PER_90 = 6.0
BEND_LOWPASS_PER_90 = 0.4
#: 道を選ぶとき、出入口1つにつき足す目安（曲がりと遠回りのぶん）
ROUTE_STEP_DB = 3.0
_EPS = 1e-6
_TOL = 0.05


@dataclass(frozen=True)
class Opening:
    """壁・床・天井にあけた穴（戸口・窓・吹き抜け）。

    lo, hi    穴を囲む箱の角（world 座標）。壁の厚みを覆うように置く
              （両側の部屋の面がどちらも箱に入るように）。いちばん薄い
              向きが壁の向き
    material  None なら開いている。材質の名前なら、その材質の戸や窓が
              閉まっている（`SpatialEngine.set_opening` で開け閉めする）
    name      開け閉めに使う名前
    """

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    material: str | None = None
    name: str = ""

    def __post_init__(self) -> None:
        lo = [float(v) for v in self.lo]
        hi = [float(v) for v in self.hi]
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError("Opening needs lo and hi as three numbers each")
        if not all(math.isfinite(v) for v in lo + hi):
            raise ValueError("Opening corners must be finite")
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError(f"Opening lo must not exceed hi: {lo} {hi}")
        if self.material is not None:
            material_occlusion(self.material)        # 名前を早めに確かめる
        object.__setattr__(self, "lo", tuple(lo))
        object.__setattr__(self, "hi", tuple(hi))

    @property
    def axis(self) -> int:
        """壁の向き（いちばん薄い向き。0 = 東西、1 = 南北、2 = 上下）。"""
        size = np.subtract(self.hi, self.lo)
        return int(np.argmin(size))


@dataclass(frozen=True)
class Occlusion:
    """1つの音源にかかる遮蔽。

    loss_db   下げる量（dB、0 以上）
    lowpass   こもり（0〜1、`set_source_lowpass` と同じ尺度）
    walls     まっすぐの線が通り抜けた壁・戸の材質（回り込んだときは空）
    via       回り込んだ出入口の名前（穴の name、開いた面は「空間の名前:面」）。
              まっすぐ届くときは空の組
    position  回り込んだとき、音が聞こえてくる位置（world 座標）。
              まっすぐ届くときは None
    """

    loss_db: float = 0.0
    lowpass: float = 0.0
    walls: tuple[str, ...] = ()
    via: tuple[str, ...] = ()
    position: tuple[float, float, float] | None = None


NO_OCCLUSION = Occlusion()


def material_occlusion(material, overrides: dict | None = None
                       ) -> tuple[float, float]:
    """材質の (下げる量 dB, こもり 0〜1)。

    `overrides` に名前があればその値を使う（`set_auto_occlusion` の
    materials）。「open」と、吸音率がすべて 1 の面は (0, 0)。名前が材質表に
    無いときは KeyError。
    """
    if isinstance(material, str):
        if overrides and material in overrides:
            db, lp = overrides[material]
            return float(db), float(lp)
        try:
            entry = _table()["materials"][material]
        except KeyError:
            raise KeyError(f"unknown material {material!r}; "
                           "see audiospace.material_names()") from None
        if material == "open":
            return 0.0, 0.0
        return OCCLUSION_BY_CATEGORY.get(entry.get("category", ""),
                                         DEFAULT_OCCLUSION)
    if np.all(absorption(material) >= 1.0):
        return 0.0, 0.0
    return DEFAULT_OCCLUSION


@dataclass(frozen=True)
class _Portal:
    """空間どうしをつなぐ出入口（開いた面か、穴）。"""

    a: int
    b: int                 # -1 は、どの空間にも入っていない外
    axis: int
    coord: float
    lo: tuple              # 長方形の範囲（axis の成分は coord）
    hi: tuple
    loss_db: float
    lowpass: float
    name: str
    #: 穴（Opening）の番号。開いた面なら -1
    opening: int = -1

    @property
    def center(self) -> list:
        return [(a + b) * 0.5 for a, b in zip(self.lo, self.hi)]


class OcclusionMap:
    """登録した空間と穴から、音源ごとの遮蔽を求める。

    作るのは空間や穴が変わったときだけ（重い計算はここでまとめる）。
    `query` は音源の位置の配列をまとめて調べる。
    """

    def __init__(self, spaces, openings=(), strength: float = 1.0,
                 materials: dict | None = None,
                 portals: bool = True) -> None:
        self.spaces = list(spaces)
        self.openings = list(openings)
        self.strength = float(strength)
        self.use_portals = bool(portals)
        self._materials = dict(materials or {})
        k = len(self.spaces)
        self.lo = np.array([sp.origin for sp in self.spaces],
                           dtype=np.float64).reshape(k, 3)
        self.hi = self.lo + np.array([sp.size for sp in self.spaces],
                                     dtype=np.float64).reshape(k, 3)
        self.volume = np.array([sp.volume for sp in self.spaces],
                               dtype=np.float64)
        # 閉じた面
        axes, coords, rlo, rhi, loss, lp, names = [], [], [], [], [], [], []
        owners = []
        open_faces = []
        for i, sp in enumerate(self.spaces):
            for face in FACES:
                axis, side = _FACE_AXIS[face]
                coord = self.lo[i, axis] if side == 0 else self.hi[i, axis]
                flo = self.lo[i].copy()
                fhi = self.hi[i].copy()
                flo[axis] = fhi[axis] = coord
                mat = sp.materials[face]
                if sp.is_open(face):
                    open_faces.append((i, face, axis, side, coord, flo, fhi))
                    continue
                db, low = material_occlusion(mat, materials)
                axes.append(axis)
                coords.append(coord)
                lo_ = flo.copy()
                hi_ = fhi.copy()
                lo_[axis] = -np.inf               # 面の向きの成分は見ない
                hi_[axis] = np.inf
                rlo.append(lo_)
                rhi.append(hi_)
                loss.append(db)
                lp.append(low)
                names.append(mat if isinstance(mat, str) else "custom")
                owners.append(i)
        self.f_axis = np.array(axes, dtype=np.intp)
        self.f_coord = np.array(coords, dtype=np.float64)
        self.f_rlo = np.array(rlo, dtype=np.float64).reshape(-1, 3) - _TOL
        self.f_rhi = np.array(rhi, dtype=np.float64).reshape(-1, 3) + _TOL
        self.f_loss = np.array(loss, dtype=np.float64)
        self.f_lp = np.array(lp, dtype=np.float64)
        self.f_name = names
        self.f_space = np.array(owners, dtype=np.intp)
        self.groups = self._group_planes()
        self.f_group = np.zeros(self.f_axis.size, dtype=np.intp)
        for g, members in enumerate(self.groups):
            self.f_group[members] = g
        # 穴
        self.o_lo = np.array([o.lo for o in self.openings],
                             dtype=np.float64).reshape(-1, 3) - _TOL
        self.o_hi = np.array([o.hi for o in self.openings],
                             dtype=np.float64).reshape(-1, 3) + _TOL
        o_vals = [material_occlusion(o.material, materials)
                  if o.material is not None else (0.0, 0.0)
                  for o in self.openings]
        self.o_loss = np.array([v[0] for v in o_vals], dtype=np.float64)
        self.o_lp = np.array([v[1] for v in o_vals], dtype=np.float64)
        self.o_name = [o.material or "open" for o in self.openings]
        # 出入口のつながり
        self.portals: list[_Portal] = []
        self._links: dict[int, list[tuple[int, int]]] = {}
        self._routes: dict[tuple[int, int], list[int] | None] = {}
        # 出入口の材質の減りがいちばん小さい道（戸の開け閉めで変わる）
        self._easy_routes: dict[tuple[int, int], list[int] | None] = {}
        if self.use_portals:
            self._find_portals(open_faces, materials)

    # -- 作るときの計算 ------------------------------------------------
    def _group_planes(self) -> list[np.ndarray]:
        """ほぼ同じ平面の面をまとめる（1本の線は1つの平面を1回しか通らない
        ので、同じ平面の面を2つ通ったら、それは同じ壁の両側）。"""
        groups: list[np.ndarray] = []
        for axis in range(3):
            idx = np.flatnonzero(self.f_axis == axis)
            if idx.size == 0:
                continue
            order = idx[np.argsort(self.f_coord[idx])]
            start = 0
            for j in range(1, order.size + 1):
                if (j == order.size or self.f_coord[order[j]]
                        - self.f_coord[order[j - 1]] > WALL_GAP_M):
                    groups.append(order[start:j])
                    start = j
        return groups

    def space_of(self, points: np.ndarray) -> np.ndarray:
        """点ごとに、その点を含むいちばん小さい空間の番号（無ければ -1）。"""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not self.spaces:
            return np.full(p.shape[0], -1, dtype=np.intp)
        inside = np.all((p[:, None, :] >= self.lo[None]) &
                        (p[:, None, :] <= self.hi[None]), axis=2)
        vol = np.where(inside, self.volume[None, :], np.inf)
        best = np.argmin(vol, axis=1)
        return np.where(np.isfinite(vol[np.arange(p.shape[0]), best]),
                        best, -1).astype(np.intp)

    def _space_of_except(self, point: np.ndarray, skip: int) -> int:
        inside = np.all((point >= self.lo) & (point <= self.hi), axis=1)
        if skip >= 0:
            inside[skip] = False
        if not inside.any():
            return -1
        vol = np.where(inside, self.volume, np.inf)
        return int(np.argmin(vol))

    def _blocked(self, point: np.ndarray, axis: int, coord: float,
                 space: int) -> bool:
        """space の閉じた面が、この点で同じ平面（WALL_GAP_M 以内）をふさぐか。
        穴があいていればふさがない。"""
        if space < 0:
            return False
        # その空間の、同じ向きでほぼ同じ平面の面だけを、まとめて調べる
        # （全部の面を1つずつ見ると、空間40個で地図を作るのに約10ms かかり、
        # その間ほかのスレッド（音声のスレッド）が待たされた）
        near = np.flatnonzero((self.f_space == space) & (self.f_axis == axis)
                              & (np.abs(self.f_coord - coord) <= WALL_GAP_M))
        if near.size == 0:
            return False
        inside = np.all((point >= self.f_rlo[near])
                        & (point <= self.f_rhi[near]), axis=1)
        for f in near[inside]:
            on_plane = point.copy()
            on_plane[axis] = self.f_coord[f]
            if self.o_lo.size and np.any(np.all(
                    (on_plane >= self.o_lo) & (on_plane <= self.o_hi),
                    axis=1)):
                continue
            return True
        return False

    def _find_portals(self, open_faces, materials) -> None:
        # 開いた面: 面の上の9点から外へ少し出た所の空間とつなぐ
        for i, face, axis, side, coord, flo, fhi in open_faces:
            normal = -1.0 if side == 0 else 1.0
            others = [a for a in range(3) if a != axis]
            found: dict[int, None] = {}
            for u in (1 / 6, 1 / 2, 5 / 6):
                for v in (1 / 6, 1 / 2, 5 / 6):
                    p = flo.copy()
                    p[others[0]] = flo[others[0]] + u * (fhi[others[0]]
                                                         - flo[others[0]])
                    p[others[1]] = flo[others[1]] + v * (fhi[others[1]]
                                                         - flo[others[1]])
                    probe = p.copy()
                    probe[axis] = coord + normal * 0.05
                    b = self._space_of_except(probe, i)
                    if b in found:
                        continue
                    if self._blocked(probe, axis, coord, b):
                        continue
                    found[b] = None
            name = f"{self.spaces[i].name or i}:{face}"
            for b in found:
                lo = flo.copy()
                hi = fhi.copy()
                if b >= 0:
                    for a in others:
                        lo[a] = max(lo[a], self.lo[b, a])
                        hi[a] = min(hi[a], self.hi[b, a])
                    if any(lo[a] > hi[a] for a in others):
                        continue
                self._add_portal(_Portal(i, b, axis, coord, _tup(lo),
                                         _tup(hi), 0.0, 0.0, name))
        # 穴: 薄い向きの両側の空間をつなぐ
        for n, o in enumerate(self.openings):
            axis = o.axis
            lo = np.array(o.lo, dtype=np.float64)
            hi = np.array(o.hi, dtype=np.float64)
            center = (lo + hi) / 2.0
            half = (hi[axis] - lo[axis]) / 2.0 + WALL_GAP_M
            sides = []
            for sign in (-1.0, 1.0):
                probe = center.copy()
                probe[axis] = center[axis] + sign * half
                sides.append(int(self.space_of(probe)[0]))
            if sides[0] == sides[1]:
                continue
            plo, phi = lo.copy(), hi.copy()
            plo[axis] = phi[axis] = center[axis]
            self._add_portal(_Portal(sides[0], sides[1], axis,
                                     float(center[axis]), _tup(plo),
                                     _tup(phi),
                                     float(self.o_loss[n]),
                                     float(self.o_lp[n]),
                                     o.name or f"opening{n}", n))

    def set_opening_material(self, name: str, material: str | None) -> None:
        """名前の穴の材質だけを替える（戸の開け閉め）。つながりは変えない。

        戸の開け閉めは出入口のつながりを変えない（閉じた戸も、通るときに
        下がる出入口のまま）ので、作り直さずに値だけ書き換える。
        """
        for n, opening in enumerate(self.openings):
            if opening.name != name:
                continue
            db, lp = ((0.0, 0.0) if material is None
                      else material_occlusion(material, self._materials))
            self.openings[n] = replace(opening, material=material)
            self.o_loss[n] = db
            self.o_lp[n] = lp
            self.o_name[n] = material or "open"
            for i, portal in enumerate(self.portals):
                if portal.opening == n:
                    self.portals[i] = replace(portal, loss_db=float(db),
                                              lowpass=float(lp))
            self._easy_routes.clear()
            return
        raise KeyError(f"no opening named {name!r}")

    def _add_portal(self, portal: _Portal) -> None:
        index = len(self.portals)
        self.portals.append(portal)
        self._links.setdefault(portal.a, []).append((portal.b, index))
        self._links.setdefault(portal.b, []).append((portal.a, index))

    def _route(self, start: int, goal: int) -> list[int] | None:
        """出入口の数がいちばん少ない道（出入口の番号の並び）。"""
        key = (start, goal)
        if key in self._routes:
            return self._routes[key]
        prev: dict[int, tuple[int, int] | None] = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                break
            for nxt, portal in self._links.get(node, ()):
                if nxt not in prev:
                    prev[nxt] = (node, portal)
                    queue.append(nxt)
        route = None
        if goal in prev:
            route = []
            node = goal
            while prev[node] is not None:
                node, portal = prev[node]
                route.append(portal)
            route.reverse()
        self._routes[key] = route
        return route

    def _easy_route(self, start: int, goal: int) -> list[int] | None:
        """出入口の材質の減りの合計がいちばん小さい道。

        出入口の数だけで選ぶと、閉じた戸の近道があるとき、開いた戸口を回る
        道を調べない。出入口1つにつき ROUTE_STEP_DB を足して、減りが同じ
        なら出入口の少ない道にする。
        """
        key = (start, goal)
        if key in self._easy_routes:
            return self._easy_routes[key]
        best = {start: 0.0}
        prev: dict[int, tuple[int, int]] = {}
        heap = [(0.0, start)]
        while heap:
            cost, node = heapq.heappop(heap)
            if node == goal:
                break
            if cost > best.get(node, math.inf):
                continue
            for nxt, portal in self._links.get(node, ()):
                c = cost + self.portals[portal].loss_db + ROUTE_STEP_DB
                if c < best.get(nxt, math.inf):
                    best[nxt] = c
                    prev[nxt] = (node, portal)
                    heapq.heappush(heap, (c, nxt))
        route = None
        if goal in best:
            route = []
            node = goal
            while node != start:
                node, portal = prev[node]
                route.append(portal)
            route.reverse()
        self._easy_routes[key] = route
        return route

    # -- 調べる --------------------------------------------------------
    def query(self, sources, listener) -> list[Occlusion]:
        """音源ごとの遮蔽。sources は (n, 3)、listener は (3,)。"""
        src = np.asarray(sources, dtype=np.float64).reshape(-1, 3)
        lis = np.asarray(listener, dtype=np.float64).reshape(3)
        n = src.shape[0]
        if n == 0:
            return []
        if self.f_axis.size == 0:
            return [NO_OCCLUSION] * n
        d = lis[None, :] - src
        denom = d[:, self.f_axis]
        num = self.f_coord[None, :] - src[:, self.f_axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / denom
            valid = ((np.abs(denom) > 1e-12) & (t > _EPS) & (t < 1.0 - _EPS))
            t = np.where(valid, t, 0.0)
            points = src[:, None, :] + t[:, :, None] * d[:, None, :]
        inside = np.all((points >= self.f_rlo[None]) &
                        (points <= self.f_rhi[None]), axis=2)
        cross = valid & inside
        loss = np.where(cross, self.f_loss[None, :], 0.0)
        low = np.where(cross, self.f_lp[None, :], 0.0)
        through = np.full(cross.shape, -1, dtype=np.intp)
        if self.o_lo.size and cross.any():
            hit = np.all((points[:, :, None, :] >= self.o_lo[None, None]) &
                         (points[:, :, None, :] <= self.o_hi[None, None]),
                         axis=3) & cross[:, :, None]
            any_hit = hit.any(axis=2)
            if any_hit.any():
                which = np.argmax(hit, axis=2)
                loss = np.where(any_hit, self.o_loss[which], loss)
                low = np.where(any_hit, self.o_lp[which], low)
                through = np.where(any_hit, which, -1)
        total = np.zeros(n)
        keep = np.ones(n)
        for group in self.groups:
            total += np.max(loss[:, group], axis=1)
            keep *= 1.0 - np.max(low[:, group], axis=1)
        direct_lp = 1.0 - keep
        dist = np.linalg.norm(d, axis=1)
        results: list[Occlusion] = []
        src_space = lis_space = None
        for i in range(n):
            if total[i] <= 0.0 and direct_lp[i] <= 0.0:
                results.append(NO_OCCLUSION)
                continue
            # 同じ平面（同じ壁の両側）からは、強いほうの面だけを数える
            best: dict[int, int] = {}
            for f in np.flatnonzero((loss[i] > 0.0) | (low[i] > 0.0)):
                g = self.f_group[f]
                if g not in best or loss[i, f] > loss[i, best[g]]:
                    best[g] = f
            walls = []
            for f in sorted(best.values(), key=lambda f: t[i, f]):
                o = through[i, f]
                walls.append(self.o_name[o] if o >= 0 else self.f_name[f])
            direct = Occlusion(*self._scaled(total[i], direct_lp[i]),
                               walls=tuple(walls))
            if self.use_portals and self.portals:
                if src_space is None:
                    src_space = self.space_of(src)
                    lis_space = int(self.space_of(lis)[0])
                around = self._around(src[i], lis, int(src_space[i]),
                                      lis_space)
                if around is not None:
                    a_occ, length = around
                    if (a_occ.loss_db + _db_of_distance(length)
                            < direct.loss_db + _db_of_distance(dist[i])):
                        results.append(a_occ)
                        continue
            results.append(direct)
        return results

    def _scaled(self, loss_db: float, lowpass: float) -> tuple[float, float]:
        s = self.strength
        return (float(min(MAX_LOSS_DB, loss_db * s)),
                float(min(MAX_LOWPASS, max(0.0, lowpass * s))))

    def _around(self, src: np.ndarray, lis: np.ndarray, s: int, l: int):
        """出入口を通って回り込む道。(遮蔽, 道のり) か None。"""
        if s == l:
            return None
        short = self._route(s, l)
        if not short:
            return None
        found = self._along(src, lis, short)
        # 閉じた戸を通らない道が別にあれば、聞こえやすいほうを使う
        easy = self._easy_route(s, l)
        if easy and easy != short:
            other = self._along(src, lis, easy)
            if other is not None and (found is None or (
                    other[0].loss_db + _db_of_distance(other[1])
                    < found[0].loss_db + _db_of_distance(found[1]))):
                found = other
        return found

    def _along(self, src: np.ndarray, lis: np.ndarray, route: list[int]):
        """出入口の並び route を通る道。(遮蔽, 道のり) か None。"""
        portals = [self.portals[p] for p in route]
        a = [float(v) for v in src]
        z = [float(v) for v in lis]
        points = [a] + [p.center for p in portals] + [z]
        # 出入口が1つなら、両端が決まっているので1回で決まる
        for _ in range(1 if len(portals) == 1 else 4):
            for j, portal in enumerate(portals, start=1):
                points[j] = _through(points[j - 1], points[j + 1], portal)
        length = sum(math.dist(points[j], points[j + 1])
                     for j in range(len(points) - 1))
        loss = 0.0
        keep = 1.0
        for j, portal in enumerate(portals, start=1):
            v1 = [points[j][k] - points[j - 1][k] for k in range(3)]
            v2 = [points[j + 1][k] - points[j][k] for k in range(3)]
            n1 = math.sqrt(sum(v * v for v in v1))
            n2 = math.sqrt(sum(v * v for v in v2))
            angle = 0.0
            if n1 > 1e-9 and n2 > 1e-9:
                cos = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            loss += BEND_DB_PER_90 * angle / 90.0 + portal.loss_db
            keep *= ((1.0 - min(0.9, BEND_LOWPASS_PER_90 * angle / 90.0))
                     * (1.0 - portal.lowpass))
        # 聞き手の部屋の出入口の方向から、道のりの距離で聞こえる
        toward = [points[-2][k] - z[k] for k in range(3)]
        norm = math.sqrt(sum(v * v for v in toward))
        if norm < 0.05:                     # 出入口に立っている
            back = points[-3] if len(points) > 3 else a
            toward = [back[k] - z[k] for k in range(3)]
            norm = math.sqrt(sum(v * v for v in toward))
        if norm < 1e-9:
            return None
        scale = max(length, 1e-3) / norm
        position = tuple(z[k] + toward[k] * scale for k in range(3))
        loss_db, lowpass = self._scaled(loss, 1.0 - keep)
        return (Occlusion(loss_db, lowpass, (),
                          tuple(p.name for p in portals), position), length)


def _through(a: list, b: list, portal: _Portal) -> list:
    """a から b への線が、出入口の長方形を通るいちばん近い点。"""
    axis, c = portal.axis, portal.coord
    da, db = a[axis] - c, b[axis] - c
    if da * db < 0.0:
        t = da / (da - db)
        p = [a[k] + t * (b[k] - a[k]) for k in range(3)]
    else:
        p = [(a[k] + b[k]) * 0.5 for k in range(3)]
    p = [min(max(p[k], portal.lo[k]), portal.hi[k]) for k in range(3)]
    p[axis] = c
    return p


def _tup(v) -> tuple:
    return tuple(float(x) for x in v)


def _db_of_distance(distance: float) -> float:
    """距離による下がり（dB、1m で 0）。どちらの道が大きいかを比べる目安。"""
    return 20.0 * math.log10(max(1.0, float(distance)))


def with_material(openings, name: str, material: str | None) -> list:
    """名前の穴の材質だけを替えた一覧（戸の開け閉め）。無ければ KeyError。"""
    out = list(openings)
    for i, o in enumerate(out):
        if o.name == name:
            out[i] = replace(o, material=material)
            return out
    raise KeyError(f"no opening named {name!r}")
