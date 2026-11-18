"""SpatialEngine -- the low-level rendering core.

Knows nothing about audio devices: you call `render()` and get a stereo
block back.  That makes the whole spatial pipeline unit-testable without
any hardware, and lets the same engine feed a real-time callback, an
offline WAV render, or a test harness.

    engine = SpatialEngine(HrirSet.synthetic(sample_rate=48000))
    voice = engine.add_source(BufferSource(samples, loop=True),
                              position=(3.0, 4.0, 0.0))
    block = engine.render()          # (block, 2) float32

Threading
---------
`render()` is expected to run on the audio thread.  Mutating calls
(`add_source`, `set_source_position`, ...) take a lock that `render()` also
takes; the critical sections are tiny (array writes, no allocation), so
this is cheap, and it removes an entire class of intermittent bug that is
very hard to diagnose by ear.

Timing
------
`add_source(..., delay_s=)` starts a sound that many seconds after the
next rendered block, to the sample.  Sounds added inside
`with engine.together():` all count from the same block -- the one after
the with-block ends -- so their offsets stay exact even if the audio
thread renders a block between two of the calls.

Occlusion
---------
`set_auto_occlusion()` muffles and lowers a source heard through the walls
of the spaces registered with `set_spaces`, from the wall materials, and
lets it come round through open faces and `Opening` holes (doors).

Groups
------
`add_source(..., group="music")` puts a source in a named group.
`set_group_gain`, `set_group_paused`, `stop_group` and `set_ducking` act on
the whole group, on top of each source's own gain, on the audio thread.
"""
from __future__ import annotations

import collections
import contextlib
import math
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .convolver import BinauralConvolver
from .geometry import (DistanceModel, Listener, Vec3, distance_gain_many,
                       elevation_weight, elevation_weight_many,
                       front_back_gain_many, relative_spherical_many)
from .hrtf import HrirSet
from .acoustics import Space, _rbj_shelf
from .early import MAX_PATHS, EarlyReflections
from .i18n import tr
from .motion import ROLL_NOUN, HeadWobble
from .reverb import (BandFdnReverb, ConvolutionReverb, Reverb,
                     band_reverb_for, db_to_gain)
from .occlusion import (NO_OCCLUSION, Occlusion, OcclusionMap, Opening,
                        material_occlusion, with_material)
from .sources import Source

#: 残響の作り方の読み上げ（late_status）。「使えません」は文の頭で見分ける
_PRECISE_ON = tr("精密版の残響", "precise reverb")
_PRECISE_UNAVAILABLE = tr("精密版は使えません",
                          "the precise reverb is not available")

MASTER_HEADROOM_DB = -3.0

# 上下の強調（`SpatialEngine.elevation_emphasis`）。上の音ほど高い音域を
# 明るく、下の音ほど暗くする人工の手がかり。強さ 1 のとき、3kHz より上を
# 真上で +12dB、上45度で +6dB、下45度で -6dB 変える（仰角に比例）。
# sin(仰角) より比例のほうがよい: 実測 HRTF（耳介 E）は前の上45度が真上より
# 明るく、真上を強く持ち上げないと高さの順に並ばない。
ELEVATION_EMPHASIS_MAX_DB = 12.0
ELEVATION_EMPHASIS_CORNER_HZ = 3000.0
# 増減の半分を全体の音量で打ち消す（高域 +12dB なら全体 -6dB）。明るさは
# 変えるが、大きさはなるべく変えない。音量の差は距離の差と紛れるため
ELEVATION_EMPHASIS_LEVEL_SHARE = 0.5


def _finite(value, what: str) -> float:
    """有限の数に直す。NaN・無限大・数でない値は ValueError。

    位置や音量に NaN が1つ入ると、その音源がある間は出力全体が NaN になり、
    毎ブロック無音になる（壊れた値を出さない守りが働くため）。原因の音だけで
    なく全部が消えるので、受け取るところで断る。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return number


def _finite_vec(value, what: str) -> Vec3:
    """3つの有限の数の組に直す（位置）。"""
    try:
        items = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be three numbers, got {value!r}"
                         ) from None
    if len(items) != 3:
        raise ValueError(f"{what} must be three numbers, got {value!r}")
    if not all(math.isfinite(v) for v in items):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return (items[0], items[1], items[2])


@dataclass
class SourceSlot:
    """Per-source mixer state held by the engine."""

    source: Source
    position: Vec3 = (0.0, 0.0, 0.0)
    gain: float = 1.0
    reverb_send: float = 0.25
    relative: bool = False          # position is relative to the listener
    spatialise: bool = True         # False = play flat, e.g. narration
    # Where a flat source sits between the ears: -1 hard left, 0 both
    # (the default, and what every caller got before this existed),
    # +1 hard right.  Ignored when spatialise is True, where direction
    # comes from the position instead.
    #
    # This is how a stereo asset is played: one source per channel,
    # panned hard apart.  Sources are mono by protocol -- read() returns
    # one channel -- so without it there is no way to get a stereo bed
    # through the mixer, and music arrives folded to the middle.
    pan: float = 0.0
    active: bool = True
    # Extra muffling for this source, 0 = clear, 1 = maximum.  Occlusion:
    # a wall between the listener and the source removes the highs, and
    # that is the cue a level drop cannot imitate (a quiet sound is just
    # far away).  Combined with the direction-derived damping in the
    # engine, never replacing it.  Set through set_source_lowpass so the
    # change is ramped rather than stepped.
    lowpass: float = 0.0
    # 音のグループの名前（set_group_gain など）。None はどのグループにも
    # 入っていない音。グループの音量・一時停止・ダッキングが、この音の
    # 音量に重なってかかる
    group: str | None = None
    # smoothed gain, so a jump in distance never produces a click.  The
    # front/back filter state lives in the engine, not here, because it is
    # applied in row-batches rather than per source.
    _cur_gain: float = field(default=0.0, repr=False)
    # fade-out state driven by remove_source(handle, fade_ms=...): the gain
    # multiplier walks to zero by _fade_step per block, then the slot is
    # released automatically.
    _fade_gain: float = field(default=1.0, repr=False)
    _fade_step: float = field(default=0.0, repr=False)
    # gain ramp driven by set_source_gain(..., ramp_ms=...): `gain` walks
    # to _gain_target by _gain_step per block, on the audio thread.
    _gain_target: float | None = field(default=None, repr=False)
    _gain_step: float = field(default=0.0, repr=False)
    # smoothed occlusion, so a door opening does not switch timbre in one
    # block.  _occl_step is per block.
    _occl_now: float = field(default=0.0, repr=False)
    _occl_step: float = field(default=1.0, repr=False)
    # one-pole state for occlusion on the flat path, one value per channel.
    # Lives here rather than in the engine's per-row table because a stereo
    # source needs two states, and the per-row table belongs to the batched
    # spatial pass.  None until the source is first filtered.
    _flat_zi: np.ndarray | None = field(default=None, repr=False)
    # 鳴り始めを遅らせる音（add_source の delay_s）。鳴り始めるまでに残って
    # いる無音のフレーム数。描画のたびにブロックぶん減り、ブロックの途中で
    # 尽きるときは、そのサンプルの位置から鳴り始める
    _wait: int = field(default=0, repr=False)
    # together() の中で足した音。with を抜けるまで数え始めない。同じ組の音が
    # すべて同じブロックから数えるので、間に描画が入っても時間差がずれない
    _held: bool = field(default=False, repr=False)
    # 新しい音は、鳴り始めのブロックで音量を目標の値から始める（フェード
    # インしない）。0 から上げると先頭のサンプルが 0 倍になり、先頭の1サンプル
    # だけのクリックは消え、足音などの立ち上がりも約11ms 弱まっていた
    # （2026-11-18 まで）。鳴り始めはデータのとおり。一時停止からの再開は
    # 波形の途中から戻るので、今までどおりフェードインする（_cur_gain = 0）
    _gain_jump: bool = field(default=False, repr=False)
    # 鳴り終わりを記録する音か（add_source の notify_end）と、もう記録したか
    _notify: bool = field(default=False, repr=False)
    # 部屋をまたぐ音の自動の遮蔽（set_auto_occlusion）。occlude が False の音は
    # 受けない。_occ は調べた結果（行き先）、_occ_db・_occ_lp は鳴らしている
    # 値（行き先へなめらかに動く）。_occ_key は調べたときの位置
    occlude: bool = True
    _occ: Occlusion = field(default=NO_OCCLUSION, repr=False)
    _occ_key: tuple | None = field(default=None, repr=False)
    _occ_db: float = field(default=0.0, repr=False)
    _occ_lp: float = field(default=0.0, repr=False)
    _occ_fresh: bool = field(default=True, repr=False)
    _ended: bool = field(default=False, repr=False)


#: 鳴り終わりの理由（`SoundEnd.reason`）
SOUND_FINISHED = "finished"   # データを最後まで鳴らした
SOUND_STOPPED = "stopped"     # 止めた（止める・グループごと止める・フェードの後）
SOUND_ERROR = "error"         # 音源が例外を出したか、形の違う音を返したので外した


@dataclass(frozen=True)
class SoundEnd:
    """音が鳴り終わったという知らせ。

    handle  play() や add_source() が返した番号
    reason  SOUND_FINISHED、SOUND_STOPPED、SOUND_ERROR のどれか
    detail  SOUND_ERROR のときの理由（例外の文）。ほかは空
    at_s    鳴り終わったときのエンジンの時刻（`SpatialEngine.elapsed_s`）
    """

    handle: int
    reason: str
    detail: str = ""
    at_s: float = 0.0


@dataclass
class SoundGroup:
    """音のグループの状態（効果音・BGM・声など）。

    グループの音量は、そのグループの音それぞれの音量に掛かる。設定画面の
    「効果音の音量」は、音ごとの音量（距離・フェード）を作り直さずに、
    これ1つを変えれば済む。ダッキング（ほかのグループが鳴っている間だけ
    下げる）は、この音量とは別の倍率で、利用者が決めた音量を上書きしない。
    """

    #: 利用者が決めた音量（ランプの行き先ではなく、いまの値）
    gain: float = 1.0
    paused: bool = False
    _gain_target: float | None = field(default=None, repr=False)
    _gain_step: float = field(default=0.0, repr=False)
    # ダッキング。duck_by のグループのどれかが鳴っている間か、duck_hold の
    # 間は、倍率 duck_now を duck_gain へ下げる。1ブロックごとに
    # _attack（下げるとき）か _release（戻すとき）ずつ動く
    duck_by: tuple[str, ...] = ()
    duck_gain: float | None = None      # None = ダッキングの決まりが無い
    duck_hold: bool = False
    duck_now: float = 1.0
    _attack: float = field(default=1.0, repr=False)
    _release: float = field(default=1.0, repr=False)

    @property
    def level(self) -> float:
        """いまグループの音に掛かっている倍率（音量 × ダッキング）。"""
        return self.gain * self.duck_now


class SpatialEngine:
    """Binaural mixer for many moving mono sources."""

    def __init__(self, hrir: HrirSet, block: int = 512,
                 max_sources: int = 64,
                 reverb: Reverb | None = None,
                 distance_model: DistanceModel = "inverse_clamped",
                 reference_distance: float = 1.0,
                 max_distance: float = 60.0,
                 front_back_depth: float = 0.35,
                 elevation_emphasis: float = 0.0,
                 crossfade: bool = True,
                 limiter: bool = True) -> None:
        self.hrir = hrir
        self.sample_rate = hrir.sample_rate
        self.block = int(block)
        self.max_sources = int(max_sources)
        self.distance_model: DistanceModel = distance_model
        self.reference_distance = float(reference_distance)
        self.max_distance = float(max_distance)
        # 基準の距離が 0 だと、すべての音が無音になる（誤りも出ない）。
        # 負の値や、基準より近い最大の距離も意味を持たない
        if not (math.isfinite(self.reference_distance)
                and self.reference_distance > 0.0):
            raise ValueError("reference_distance must be a positive number, "
                             f"got {reference_distance}")
        if not (math.isfinite(self.max_distance)
                and self.max_distance >= self.reference_distance):
            raise ValueError("max_distance must be a number not below "
                             f"reference_distance, got {max_distance}")
        self.front_back_depth = float(front_back_depth)
        self.elevation_emphasis = float(elevation_emphasis)
        self.reverb = reverb
        self.master_gain = 10.0 ** (MASTER_HEADROOM_DB / 20.0)

        # 畳み込みの行: 音源の行（max_sources）の後ろに、空間の初期反射の行
        # （early.MAX_PATHS）を確保しておく。鳴っていない行は費用がかからない
        # 切り替えのつなぎでは、前の空間と新しい空間の反射が同時に鳴るので2組
        self._rows = self.max_sources + 2 * MAX_PATHS
        self._conv = BinauralConvolver(hrir, block=block,
                                       max_sources=self._rows,
                                       crossfade=crossfade)
        self._crossfade = bool(crossfade)
        self._lock = threading.RLock()
        self._listener = Listener()
        self._slots: dict[int, SourceSlot] = {}
        self._slot_of_row: list[int | None] = [None] * self.max_sources
        self._next_id = 1
        # 材質つきの空間（set_space）。None なら初期反射は無し
        self.space: Space | None = None
        self._early: EarlyReflections | None = None
        # 空間の切り替えのつなぎ（set_space の transition_s / release_s）
        self._early_gain = 1.0          # いまの反射の出力の倍率（入ってくる途中）
        self._early_step = 0.0
        self._old_early: EarlyReflections | None = None   # 消えていく反射
        self._old_early_gain = 0.0
        self._old_early_step = 0.0
        # 鳴り終わるまで残す前の残響。{"reverb", "gain", "step"}
        self._tails: list[dict] = []
        # いまの空間（空間なしを含む）で鳴らしたブロック数。0 なら、まだ何も
        # 鳴らしていないので切り替えをつながない。空間なしの間も数える
        # （数えないと、どの空間にも入っていない場所から部屋へ入ったとき、
        # 残響と反射がつなぎなしで全量になる）
        self._space_blocks = 0
        self._reverb_in_gain = 1.0      # いまの残響への入力の倍率（入る途中）
        self._reverb_in_step = 1.0
        # 場所ごとの空間（set_spaces）
        self._space_map: list[Space] = []
        self._map_options: dict = {}
        # 精密版の後期残響。鍵 -> 畳み込みの核（clone して使う）
        self.space_ir_cache: str | None = None   # 計算した尾の保存先
        self._precise_kernels: dict[str, ConvolutionReverb] = {}
        self._precise_pending: set[str] = set()
        self._late_mode: object = True
        self._swap_transition = 0.3
        self.late_status = ""

        self._blocks = np.zeros((self._rows, self.block), dtype=np.float32)
        self._dir_idx = np.zeros(self._rows, dtype=np.intp)
        self._send = np.zeros(self.block, dtype=np.float32)
        self._flat = np.zeros((self.block, 2), dtype=np.float32)
        # front/back damping: one filter state per row, plus the coefficient
        # bucket each row wants this block (see _apply_front_back)
        self._lp_zi = np.zeros((self._rows, 1), dtype=np.float64)
        self._lp_bucket = np.full(self._rows, -1, dtype=np.intp)
        # 上下の強調: 行ごとの 2次フィルタの状態と、このブロックで使う
        # 段（dB、整数。0 ならかけない）。係数は段ごとに作って取っておく
        self._tilt_zi = np.zeros((1, self._rows, 2), dtype=np.float64)
        self._tilt_step = np.zeros(self._rows, dtype=np.intp)
        self._tilt_sos: dict[int, np.ndarray] = {}
        self._tilt_dirty = False        # どこかの行に状態が残っているか
        self._ramp01 = np.linspace(0.0, 1.0, self.block, endpoint=False,
                                   dtype=np.float32)
        self.last_peak = 0.0
        # 首振り。ブロック単位で進む内部時計を持つので、レンダリング速度に
        # 依存せず、オフラインレンダリングでも同じ結果になる。
        self._wobble = HeadWobble(0.0)
        # 首かしげ（ロール）の揺れ。首振りと同じ時計で進む
        self._roll_wobble = HeadWobble(0.0, noun=ROLL_NOUN)
        self._elapsed = 0.0
        self._block_seconds = self.block / self.sample_rate
        self._paused = False
        # 出力リミッター。多数の音源が重なるとマスターの -3dB だけでは
        # フルスケールを超え、ヘッドホンで耳障りなクリップになる。閾値を
        # 超えたブロックはそのピークに合わせて全体を下げ、ゲインは
        # ブロック内でランプさせるので補正自体は聞こえない。
        self.limiter_enabled = bool(limiter)
        self.limit_threshold = 0.98
        self.limited_blocks = 0
        self._limiter_gain = 1.0
        #: 計算が壊れて（NaN や無限大）無音にしたブロックの数
        self.nonfinite_blocks = 0
        #: read() が例外を出したか、形の違う音を返したので外した音源の数と、
        #: 最後の理由。1つの音源の不具合で全部の音が止まらないよう、その音源
        #: だけを外す
        self.source_errors = 0
        self.last_source_error = ""
        # together() の組。スレッドごとに持つ（別のスレッドが足した音を、
        # こちらの組に入れないため）。中身は組の入れ子の一覧
        self._together = threading.local()
        # 音のグループ（効果音・BGM・声など）。名前 -> 状態
        self._sound_groups: dict[str, SoundGroup] = {}
        # 鳴り終わりの知らせ（notify_end の音だけ）。音声のスレッドが足し、
        # take_ended が取り出す
        self._ended_events: collections.deque[SoundEnd] = collections.deque()
        # 部屋をまたぐ音の自動の遮蔽。_occ_opts が None なら切っている。
        # _occ_active は、切った後も鳴っている値が 0 に戻るまで True
        self._occ_opts: dict | None = None
        self._occ_map: OcclusionMap | None = None
        self._occ_version = 0
        # 地図を作るのは1つずつ（エンジンのロックの外で作る。空間が多いと
        # 数ミリ秒かかり、その間に描画を止めないため）。_occ_gen は地図の
        # 材料（設定・空間）が変わるたびに増やし、作っている間に変わったら
        # 作り直す
        self._occ_build_lock = threading.Lock()
        self._occ_gen = 0
        self._occ_active = False
        self._occ_cursor = 0
        self._occ_jump = False          # teleport_listener の直後だけ True

    # -- 首振り ----------------------------------------------------------
    @property
    def head_wobble(self) -> HeadWobble:
        return self._wobble

    def set_head_wobble(self, wobble: HeadWobble | None = None, *,
                        amplitude_deg: float | None = None,
                        rate_hz: float | None = None,
                        shape: str | None = None) -> None:
        """微小な自動首振りを設定する。

        HeadWobble を直接渡すか、個別の値だけを差し替える。0 度で無効。
        なぜこれが要るかは motion.py を参照。
        """
        with self._lock:
            if wobble is not None:
                self._wobble = wobble
                return
            current = self._wobble
            self._wobble = HeadWobble(
                amplitude_deg if amplitude_deg is not None
                else current.amplitude_deg,
                rate_hz if rate_hz is not None else current.rate_hz,
                shape if shape is not None else current.shape)

    @property
    def effective_yaw_deg(self) -> float:
        """首振りを含めた、いまの実際の向き。"""
        return self._listener.yaw_deg + self._wobble.offset_at(self._elapsed)

    @property
    def roll_wobble(self) -> HeadWobble:
        return self._roll_wobble

    def set_roll_wobble(self, wobble: HeadWobble | None) -> None:
        """首かしげ（ロール）の自動の揺れを設定する。None か 0 度で無効。

        上下の音を左右へ逆向きに動かす。理由は motion.py の ROLL_PRESETS。
        """
        with self._lock:
            self._roll_wobble = (wobble if wobble is not None
                                 else HeadWobble(0.0, noun=ROLL_NOUN))

    # -- HRTF の差し替え --------------------------------------------------
    def set_hrir(self, hrir: HrirSet, crossfade: bool = True) -> None:
        """再生を止めずに HRTF を差し替える。

        耳介を聞き比べるには、音を止めずに切り替えられることが要る。
        止めて作り直すと直前の音の記憶が薄れて比較にならない。

        サンプルレートが違う場合は自動で変換する。畳み込みの状態は
        作り直しになるので、無音のタイミングで呼ぶのが望ましい。
        """
        if hrir.sample_rate != self.sample_rate:
            hrir = hrir.resampled(self.sample_rate)
        conv = BinauralConvolver(hrir, block=self.block,
                                 max_sources=self._rows,
                                 crossfade=crossfade)
        with self._lock:
            self.hrir = hrir
            self._conv = conv
            self._lp_zi[:] = 0.0
            self._tilt_zi[:] = 0.0

    # -- listener --------------------------------------------------------
    @property
    def listener(self) -> Listener:
        return self._listener

    def set_listener(self, position: Vec3 | None = None,
                     yaw_deg: float | None = None,
                     pitch_deg: float | None = None,
                     roll_deg: float | None = None) -> None:
        """Update the listener.  Arguments left as None keep their value.

        `pitch_deg` tilts the head up (positive) or down (negative);
        `roll_deg` lowers the left ear (positive) or the right ear.
        Raises ValueError for NaN or infinite values.
        """
        if position is not None:
            position = _finite_vec(position, "listener position")
        yaw_deg = None if yaw_deg is None else _finite(yaw_deg, "yaw_deg")
        pitch_deg = (None if pitch_deg is None
                     else _finite(pitch_deg, "pitch_deg"))
        roll_deg = None if roll_deg is None else _finite(roll_deg, "roll_deg")
        with self._lock:
            cur = self._listener
            self._listener = Listener(
                position if position is not None else cur.position,
                yaw_deg if yaw_deg is not None else cur.yaw_deg,
                pitch_deg if pitch_deg is not None else cur.pitch_deg,
                roll_deg if roll_deg is not None else cur.roll_deg)

    def teleport_listener(self, position: Vec3, yaw_deg: float = 0.0,
                          pitch_deg: float = 0.0,
                          roll_deg: float = 0.0) -> None:
        """Move without smearing: clears convolution and reverb state."""
        listener = Listener(_finite_vec(position, "listener position"),
                            _finite(yaw_deg, "yaw_deg"),
                            _finite(pitch_deg, "pitch_deg"),
                            _finite(roll_deg, "roll_deg"))
        with self._lock:
            self._listener = listener
            self._conv.reset()
            if self.reverb is not None:
                self.reverb.reset()
            if self._early is not None:
                self._early.reset()
            # 瞬間移動では前の場所の響きは残さない
            self._tails.clear()
            self._old_early = None
            # 自動の遮蔽も、次のブロックで全部の音を調べ直してその値にする
            # （なめらかに動かすと、前の場所の聞こえ方が一瞬残る）
            self._occ_jump = True

    # -- sources ---------------------------------------------------------
    def add_source(self, source: Source, position: Vec3 = (0.0, 0.0, 0.0),
                   gain: float = 1.0, reverb_send: float = 0.25,
                   relative: bool = False,
                   spatialise: bool = True, pan: float = 0.0,
                   lowpass: float = 0.0, delay_s: float = 0.0,
                   group: str | None = None,
                   notify_end: bool = False,
                   occlude: bool = True) -> int:
        """Register a source and return its handle.

        `occlude` (default True) lets the automatic occlusion of
        set_auto_occlusion act on this source; pass False for a sound that
        should never be muffled by walls (a guide beacon, say).

        `notify_end` records when the source ends -- played out, stopped,
        or removed after an error -- for take_ended() (the player turns
        these into on_end callbacks).

        `group` puts the source in a named group ("sfx", "music", "voice"
        ...) whose gain, pause and ducking apply on top of its own gain;
        see set_group_gain.  None (the default) is in no group.

        `lowpass` starts the source already muffled (see
        set_source_lowpass); a sound that begins behind a wall should not
        be heard opening up over the first few blocks.

        `delay_s` starts the source that many seconds after the next
        rendered block, to the sample (inside `together()`, after the
        with-block ends).  Until then the source is not read and its
        ramps do not advance, so a gain ramp asked for while it waits runs
        from when it starts; it holds a slot meanwhile.

        A source begins exactly as its data begins, at full gain -- there
        is no fade-in, so a click on the very first sample is heard.  Data
        that starts abruptly (cut mid-waveform) clicks as it would in any
        player; give it a short fade, or start it at gain 0 and ramp it up
        with set_source_gain inside together().

        Raises RuntimeError when every slot is taken, rather than silently
        dropping the sound -- a dropped cue is a bug the player cannot see.
        Raises ValueError for a stereo source with spatialise=True: a
        direction can only be given to a mono signal.  This must fail here,
        at registration -- failing later, inside render(), would take every
        other sound down with it on every block.  NaN or infinite numbers
        raise ValueError for the same reason.
        """
        if spatialise and getattr(source, "channels", 1) != 1:
            raise ValueError(
                "a spatialised source must be mono; play a stereo asset "
                "with spatialise=False, or downmix it")
        position = _finite_vec(position, "position")
        gain = _finite(gain, "gain")
        reverb_send = _finite(reverb_send, "reverb_send")
        pan = _finite(pan, "pan")
        amount = float(min(1.0, max(0.0, _finite(lowpass, "lowpass"))))
        delay = _finite(delay_s, "delay_s")
        if delay < 0.0:
            raise ValueError(f"delay_s must not be negative, got {delay_s!r}")
        wait = int(round(delay * self.sample_rate))
        if group is not None and not isinstance(group, str):
            raise ValueError(f"group must be a name (str) or None, "
                             f"got {group!r}")
        pending = self._open_together()
        with self._lock:
            free = [r for r, h in enumerate(self._slot_of_row) if h is None]
            if not free:
                raise RuntimeError(
                    f"all {self.max_sources} source slots are in use; "
                    "raise max_sources or release finished sources")
            # 前の音の入力がまだ畳み込みの遅延線に残っている行（HRIR が
            # ブロックより長いとき、鳴り終わってから数ブロック）は避ける。
            # 入ると、前の音の残りが新しい音の方向で鳴ってしまう
            row = next((r for r in free if not self._conv.row_sounding(r)),
                       free[0])
            handle = self._next_id
            self._next_id += 1
            slot = SourceSlot(source, position, gain, reverb_send,
                              relative, spatialise, pan, lowpass=amount)
            slot._cur_gain = 0.0
            slot._occl_now = amount          # already muffled at birth
            slot._wait = wait
            slot._gain_jump = True           # no fade-in: start as the data
            slot._notify = bool(notify_end)
            slot.occlude = bool(occlude)
            if group is not None:
                slot.group = group
                self._group(group)           # a new name starts at full gain
            if pending is not None:
                slot._held = True
                pending.append(handle)
            self._slots[handle] = slot
            self._lp_zi[row] = 0.0           # no tail from the previous tenant
            self._tilt_zi[:, row] = 0.0
            self._conv.forget(row)           # nor its direction to fade from
            self._slot_of_row[row] = handle
            return handle

    def _open_together(self) -> list[int] | None:
        """このスレッドで開いている together() の組（無ければ None）。"""
        stack = getattr(self._together, "stack", None)
        return stack[-1] if stack else None

    @contextlib.contextmanager
    def together(self):
        """Start every source added inside the with-block from one block.

            with engine.together():
                engine.add_source(step, pos)
                engine.add_source(step_echo, floor, delay_s=0.0094)

        Sources added inside (on this thread) wait, silent, until the
        with-block ends, and then all count their `delay_s` from the same
        next block.  Without it, the audio thread may render a block
        between two add_source calls, and the second sound then starts one
        block late -- 10.7 ms at 512 frames, larger than a floor
        reflection's whole delay.  The lock is not held inside the block,
        so decoding a file there cannot stall the audio.

        If the block raises, its sources are removed unplayed and the
        exception propagates.  Nested blocks join the outermost one.
        """
        stack = getattr(self._together, "stack", None)
        if stack is None:
            stack = self._together.stack = []
        pending: list[int] = []
        stack.append(pending)
        ok = False
        try:
            yield self
            ok = True
        finally:
            stack.pop()
            if ok and stack:
                stack[-1].extend(pending)   # the outer block releases them
            else:
                with self._lock:
                    for handle in pending:
                        slot = self._slots.get(handle)
                        if slot is None:
                            continue
                        if ok:
                            slot._held = False
                        else:
                            self.remove_source(handle)

    def _keep_kernel(self, key: str, kernel: ConvolutionReverb) -> None:
        """精密版の核を覚える。PRECISE_KERNELS_MAX を超えたら、長く使って
        いないものから捨てる（鳴らすのは複製なので、捨てても鳴っている
        響きは切れない。次に使うとき、尾から作り直す）。"""
        self._precise_kernels.pop(key, None)
        self._precise_kernels[key] = kernel
        while len(self._precise_kernels) > self.PRECISE_KERNELS_MAX:
            del self._precise_kernels[next(iter(self._precise_kernels))]

    def remove_all_sources(self) -> list[Source]:
        """Release every source at once (no fade) and return them.

        For shutting down: the caller can then close the sources that hold
        a file.  End notices are recorded as for remove_source.
        """
        with self._lock:
            sources = []
            for handle in list(self._slots):
                sources.append(self._slots[handle].source)
                self.remove_source(handle)
            return sources

    def remove_source(self, handle: int, fade_ms: float = 0.0) -> None:
        """Release a source, optionally fading it out over `fade_ms`.

        Cutting a sustained sound mid-sample clicks; with a fade the slot
        keeps playing while its gain walks to zero and is then released
        automatically.  30 ms is enough to be click-free without feeling
        sluggish.  A missing handle is ignored, so callers may stop a
        sound that has already been reaped.
        """
        with self._lock:
            slot = self._slots.get(handle)
            if slot is None:
                return
            # 止めている音と、全体を止めている間は描画されず、フェードが
            # 進まない（枠が返らない、再開後に鳴る）。その場で外す。
            # まだ鳴り始めていない音（遅らせている音、together() の中の音）
            # も、聞こえていないのでフェードせずに外す。グループごと止めて
            # いる音も同じ
            group_paused = (slot.group is not None
                            and self._sound_groups[slot.group].paused)
            if (fade_ms > 0.0 and slot.active and not self._paused
                    and not group_paused
                    and not slot._wait and not slot._held):
                slot._fade_step = max(
                    1e-6, self._block_seconds * 1000.0 / float(fade_ms))
                return
            if slot._notify and not slot._ended:
                try:
                    done = bool(slot.source.finished)
                except Exception:                 # noqa: BLE001
                    done = False
                self._note_end(handle, slot,
                               SOUND_FINISHED if done else SOUND_STOPPED)
            del self._slots[handle]
            row = self._slot_of_row.index(handle)
            self._slot_of_row[row] = None
            self._blocks[row] = 0.0

    # -- end notices -----------------------------------------------------
    def _note_end(self, handle: int, slot: SourceSlot, reason: str,
                  detail: str = "") -> None:
        """鳴り終わりを1回だけ記録する（notify_end の音だけ）。ロックの中で呼ぶ。"""
        if slot._notify and not slot._ended:
            slot._ended = True
            self._ended_events.append(
                SoundEnd(handle, reason, detail, self._elapsed))

    def take_ended(self) -> list[SoundEnd]:
        """Return and forget the end notices recorded so far.

        Only sources added with notify_end=True are recorded, each once:
        SOUND_FINISHED in the block where its data ran out,
        SOUND_STOPPED when it is removed before that (remove_source, after
        a fade-out has finished, stop_group, a failed together()), and
        SOUND_ERROR when it raised and was taken out.  Call it from any
        thread; the audio thread only appends.
        """
        with self._lock:
            events = list(self._ended_events)
            self._ended_events.clear()
        return events

    def pending_ends(self) -> list[SoundEnd]:
        """The end notices not yet taken (a copy; nothing is removed)."""
        with self._lock:
            return list(self._ended_events)

    @property
    def elapsed_s(self) -> float:
        """Seconds rendered so far (stops while paused); SoundEnd.at_s
        uses this clock."""
        return self._elapsed

    # -- sound groups ----------------------------------------------------
    @staticmethod
    def _group_name(name) -> str:
        if not isinstance(name, str):
            raise ValueError(f"a group name must be a str, got {name!r}")
        return name

    def _group(self, name: str) -> SoundGroup:
        """名前のグループ。無ければ音量 1 で作る。ロックの中で呼ぶ。"""
        group = self._sound_groups.get(name)
        if group is None:
            group = self._sound_groups[name] = SoundGroup()
        return group

    def set_group_gain(self, name: str, gain: float,
                       ramp_ms: float = 0.0) -> None:
        """Set the gain of a group of sounds ("sfx", "music", "voice" ...).

        It multiplies every sound in the group on top of its own gain,
        distance and fades, so a settings screen's "effects volume" is one
        call rather than a walk over every handle.  `ramp_ms` works as on
        set_source_gain (reaches the target in that time, on the audio
        thread).  A name not used yet is created, so the volume can be set
        before the first sound of the group plays.
        """
        name = self._group_name(name)
        target = _finite(gain, "gain")
        ramp_ms = _finite(ramp_ms, "ramp_ms")
        with self._lock:
            group = self._group(name)
            if ramp_ms <= 0.0:
                group.gain = target
                group._gain_target = None
                return
            blocks = max(1.0, float(ramp_ms) / (self._block_seconds * 1000.0))
            group._gain_target = target
            group._gain_step = abs(target - group.gain) / blocks

    def set_group_gain_db(self, name: str, db: float,
                          ramp_ms: float = 0.0) -> None:
        """Set a group's gain in decibels (0 dB = unchanged, -6 = half)."""
        self.set_group_gain(name, db_to_gain(_finite(db, "db")), ramp_ms)

    def group_gain(self, name: str) -> float:
        """The gain set for a group (where a ramp is heading).

        1.0 for a name never used.  Ducking is not included -- it is a
        separate factor (see group_level).
        """
        with self._lock:
            group = self._sound_groups.get(name)
            if group is None:
                return 1.0
            if group._gain_target is not None:
                return group._gain_target
            return group.gain

    def group_level(self, name: str) -> float:
        """The factor on the group's sounds right now: gain x ducking."""
        with self._lock:
            group = self._sound_groups.get(name)
            return 1.0 if group is None else group.level

    def set_group_paused(self, name: str, paused: bool) -> None:
        """Pause (True) or resume (False) every sound in a group.

        Like set_source_active for each of them: a paused sound is not
        pulled and holds its place, a delayed one holds its countdown, and
        on resume the sound fades back in over one block.  A sound paused
        on its own (set_source_active) stays paused when the group
        resumes.  Sounds added to a paused group wait for the resume.
        """
        name = self._group_name(name)
        with self._lock:
            group = self._group(name)
            if group.paused and not paused:
                for slot in self._slots.values():
                    if slot.group == name and not slot._gain_jump:
                        slot._cur_gain = 0.0     # 波形の途中から戻る
            group.paused = bool(paused)

    def group_paused(self, name: str) -> bool:
        with self._lock:
            group = self._sound_groups.get(name)
            return bool(group is not None and group.paused)

    def sources_in_group(self, name: str) -> list[int]:
        """Handles of the sounds now in a group, in the order added."""
        with self._lock:
            return [h for h, s in self._slots.items() if s.group == name]

    def stop_group(self, name: str, fade_ms: float = 0.0) -> list[int]:
        """Stop every sound in a group (see remove_source); returns them."""
        with self._lock:
            handles = [h for h, s in self._slots.items() if s.group == name]
            for handle in handles:
                self.remove_source(handle, fade_ms)
        return handles

    def set_source_group(self, handle: int, name: str | None) -> None:
        """Move a sound to another group (None = no group)."""
        if name is not None:
            name = self._group_name(name)
        with self._lock:
            slot = self._slots[handle]
            was_paused = (slot.group is not None
                          and self._sound_groups[slot.group].paused)
            now_paused = name is not None and self._group(name).paused
            if was_paused and not now_paused and not slot._gain_jump:
                slot._cur_gain = 0.0             # 止まっていた所から戻る
            slot.group = name

    def set_ducking(self, name: str, by: str | tuple[str, ...] = (),
                    db: float = -12.0, attack_ms: float = 80.0,
                    release_ms: float = 400.0) -> None:
        """Lower a group while another group is sounding (ducking).

            engine.set_ducking("music", by="voice", db=-12)

        While any sound of a group in `by` is playing, the group is lowered
        by `db`, reaching it in `attack_ms`; when they have all stopped it
        returns in `release_ms`.  This is a factor of its own: the gain set
        with set_group_gain (the player's volume setting) is kept.  `by`
        may be empty; then only hold_ducking lowers the group -- for speech
        that does not go through the engine, such as a screen reader.
        Calling it again replaces the rule.
        """
        name = self._group_name(name)
        if isinstance(by, str):
            by = (by,)
        by = tuple(self._group_name(b) for b in by)
        if name in by:
            raise ValueError(f"group {name!r} cannot duck under itself")
        depth = _finite(db, "db")
        if depth > 0.0:
            raise ValueError(f"ducking lowers a group; db must be 0 or "
                             f"less, got {db!r}")
        attack_ms = _finite(attack_ms, "attack_ms")
        release_ms = _finite(release_ms, "release_ms")
        if attack_ms < 0.0 or release_ms < 0.0:
            raise ValueError("attack_ms and release_ms must not be negative")
        duck = db_to_gain(depth)
        span = 1.0 - duck
        block_ms = self._block_seconds * 1000.0
        with self._lock:
            group = self._group(name)
            for other in by:
                self._group(other)
            group.duck_by = by
            group.duck_gain = duck
            group._attack = max(1e-9, span / max(1.0, attack_ms / block_ms))
            group._release = max(1e-9, span / max(1.0, release_ms / block_ms))

    def clear_ducking(self, name: str) -> None:
        """Remove a group's ducking rule; a lowered group returns at the
        rule's release speed."""
        with self._lock:
            group = self._sound_groups.get(name)
            if group is not None:
                group.duck_by = ()
                group.duck_gain = None
                group.duck_hold = False

    def hold_ducking(self, name: str, on: bool = True) -> None:
        """Lower (True) or release (False) a ducked group by hand.

        For a voice the engine cannot see -- a screen reader, or speech
        played some other way -- call this around it.  The depth and
        speed come from set_ducking, which must be called first.
        """
        with self._lock:
            group = self._sound_groups.get(name)
            if group is None or group.duck_gain is None:
                raise ValueError(f"group {name!r} has no ducking; call "
                                 "set_ducking first")
            group.duck_hold = bool(on)

    @staticmethod
    def _is_sounding(source) -> bool:
        """ダッキングのきっかけになる「鳴っている」か。音源が `sounding` を
        持てばそれ（押し込み音源は中身があるとき、関数の音源は直前に返した
        音が無音でないとき）、無ければ鳴り終わっていないこと。"""
        live = getattr(source, "sounding", None)
        if live is not None:
            return bool(live)
        return not source.finished

    def _advance_groups(self) -> tuple[dict[str, float], set[str]]:
        """グループの音量とダッキングを1ブロック進め、倍率と止めている
        グループの名前を返す。render の中（ロックの中）で呼ぶ。"""
        groups = self._sound_groups
        sounding: set[str] = set()
        if any(g.duck_by and g.duck_gain is not None
               for g in groups.values()):
            for slot in self._slots.values():
                name = slot.group
                if (name is not None and name not in sounding
                        and slot.active and not groups[name].paused
                        and not slot._held and not slot._wait
                        and self._is_sounding(slot.source)):
                    sounding.add(name)
        levels: dict[str, float] = {}
        paused: set[str] = set()
        for name, group in groups.items():
            if group.paused:
                paused.add(name)
            if group._gain_target is not None:
                remaining = group._gain_target - group.gain
                if abs(remaining) <= group._gain_step:
                    group.gain = group._gain_target
                    group._gain_target = None
                else:
                    group.gain += math.copysign(group._gain_step, remaining)
            want = 1.0
            if group.duck_gain is not None and (
                    group.duck_hold
                    or any(b in sounding for b in group.duck_by)):
                want = group.duck_gain
            if group.duck_now != want:
                delta = want - group.duck_now
                step = group._attack if delta < 0.0 else group._release
                group.duck_now += math.copysign(min(abs(delta), step), delta)
            levels[name] = group.gain * group.duck_now
        return levels, paused

    def set_source_position(self, handle: int, position: Vec3) -> None:
        """Move a source.  Raises ValueError for NaN or infinite values."""
        position = _finite_vec(position, "position")
        with self._lock:
            self._slots[handle].position = position

    def set_source_gain(self, handle: int, gain: float,
                        ramp_ms: float = 0.0) -> None:
        """Set a source's gain, optionally ramping to it over `ramp_ms`.

        The ramp is advanced by the audio thread, one step per rendered
        block, so it finishes on time no matter what the application
        thread is doing.  A game that ramps from its own tick loop stalls
        the fade whenever the UI blocks -- dragging a window or opening a
        modal menu -- and the sound sits at half volume until the user
        lets go.
        """
        target = _finite(gain, "gain")
        ramp_ms = _finite(ramp_ms, "ramp_ms")
        with self._lock:
            slot = self._slots[handle]
            if ramp_ms <= 0.0:
                slot.gain = target
                slot._gain_target = None
                return
            blocks = max(1.0, float(ramp_ms) / (self._block_seconds * 1000.0))
            slot._gain_target = target
            slot._gain_step = abs(target - slot.gain) / blocks

    def set_source_gain_db(self, handle: int, db: float,
                           ramp_ms: float = 0.0) -> None:
        """Set a source's gain in decibels (0 dB = unchanged, -6 = half)."""
        self.set_source_gain(handle, db_to_gain(db), ramp_ms)

    def set_source_lowpass(self, handle: int, amount: float,
                           ramp_ms: float = 60.0) -> None:
        """Muffle one source: 0 = untouched, 1 = maximum.

        This is the occlusion control -- "there is a wall in the way".
        Level alone cannot say that, because a quieter sound is also what
        distance sounds like; losing the highs is what a listener reads as
        an obstruction.  The same filter carries the engine's front/back
        darkening, and the two combine rather than override each other.

        The amount is ramped so that it *reaches the target* in `ramp_ms`
        -- the same meaning the parameter has on set_source_gain --
        because occlusion changes as the player walks, and stepping the
        coefficient makes the timbre jump.  Pass ramp_ms=0 to apply it at
        the next block.
        """
        amount = float(min(1.0, max(0.0, _finite(amount, "amount"))))
        ramp_ms = _finite(ramp_ms, "ramp_ms")
        with self._lock:
            slot = self._slots[handle]
            slot.lowpass = amount
            if ramp_ms <= 0.0:
                slot._occl_now = amount
                slot._occl_step = 1.0
                return
            blocks = max(1.0, float(ramp_ms) / (self._block_seconds * 1000.0))
            slot._occl_step = max(1e-6,
                                  abs(amount - slot._occl_now) / blocks)

    def set_source_active(self, handle: int, active: bool) -> None:
        """Pause (False) or resume (True) one source without releasing it.

        A paused source is not pulled at all, so a BufferSource holds its
        position.  On resume the gain ramps back in from silence, so the
        edit never clicks.
        """
        with self._lock:
            slot = self._slots[handle]
            if active and not slot.active:
                slot._cur_gain = 0.0
            slot.active = bool(active)

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Pause the whole engine: render() returns silence, no source or
        reverb state advances, and everything resumes exactly where it was.
        Intended for a game's pause menu."""
        with self._lock:
            self._paused = bool(paused)

    def set_source_reverb_send(self, handle: int, send: float) -> None:
        send = float(np.clip(_finite(send, "send"), 0.0, 1.0))
        with self._lock:
            self._slots[handle].reverb_send = send

    def get_source(self, handle: int) -> SourceSlot:
        return self._slots[handle]

    @property
    def active_sources(self) -> int:
        return sum(1 for h in self._slot_of_row if h is not None)

    def reap_finished(self) -> list[int]:
        """Drop sources that have played out.  Call from the game loop."""
        with self._lock:
            done = [h for h, s in self._slots.items() if s.source.finished]
        for handle in done:
            self.remove_source(handle)
        return done

    def set_reverb(self, reverb: Reverb | None) -> None:
        with self._lock:
            self.reverb = reverb

    def set_space(self, space: Space | None, early: bool = True,
                  late: bool = True, transition_s: float = 0.3,
                  release_s: float | None = None) -> None:
        """材質つきの空間を設定する。None で空間の響きを外す。

        - late: 材質から決まる帯域ごとの残響時間で後期残響（BandFdnReverb）を
          作り、残響として設定する（今の残響は置き換わる）。屋外のように
          拡散した残響が無い空間では残響を外す
        - early: 聞き手の位置から決まる面の反射を、方向つきで鳴らす

        初期反射の入力は、空間化したすべての音源を足した信号（距離による
        減衰の後）。残響の送り量（reverb_send）には左右されない。直接音に
        対する反射の大きさを、物理的な比（壁から1mなら約 -10dB）に保つため。

        切り替えのつなぎ（すでに空間の響きを鳴らしているとき）:

        - 前の空間の残響は、新しい音を受け取らなくなり、自然に鳴り終わるまで
          残る（部屋を出たあと、その部屋の響きが消えていくのが聞こえる）。
          `release_s` を渡すと、その秒数で打ち切る（フェードで消す）
        - 反射は `transition_s` 秒かけて、前の空間から新しい空間へ
          クロスフェードする。いきなり切り替えると反射の波形が途切れて
          プツッと鳴る。0 なら即座に切り替える

        後期残響の作り方（late）:

        - True または "fast": 材質から求めた残響時間の FDN（速い版、一瞬で作れる）
        - "precise": pyroomacoustics で精密に計算した尾（`precise.py`）。
          焼き込み済み・保存済みならすぐ使う。無ければ別のプロセスで計算を
          始め、その間は速い版で鳴らし、できたら transition_s でつなぎ替える。
          計算できない環境（pyroomacoustics が無い、固めた exe）では速い版
        - False: 後期残響なし

        いまどの残響で鳴っているかは `late_status`（読み上げ用の文）。
        set_spaces で登録した場所ごとの空間は、これを呼ぶと外れる。
        """
        if late == "precise" and space is not None:
            self._prepare_precise(space)
        with self._lock:
            self._space_map = []
            self._swap_transition = transition_s
            self._switch_space(space, early, late, transition_s, release_s)
            self._occ_gen += 1
            occluding = self._occ_opts is not None
        if occluding:
            # set_spaces の登録が無いときは、この空間で遮蔽する
            self._refresh_occlusion()

    # -- 部屋をまたぐ音の遮蔽 ----------------------------------------------
    #: 1ブロックで遮蔽を調べ直す音源の数の上限（残りは次のブロック）
    #: 覚えておく精密版の核の数の上限（1つ数 MB）
    PRECISE_KERNELS_MAX = 16
    OCCLUSION_BATCH = 16
    #: 遮蔽が変わるときの速さ（100ms あたり）
    OCCLUSION_DB_PER_100MS = 30.0
    OCCLUSION_LP_PER_100MS = 1.0

    def set_auto_occlusion(self, enabled: bool = True, *,
                           strength: float = 1.0, openings=(),
                           materials: dict | None = None,
                           portals: bool = True) -> None:
        """Muffle and lower sounds heard through walls, automatically.

        Uses the spaces registered with set_spaces, or the space given to
        set_space when there is no such registration (with neither, nothing
        is occluded, and `SpatialPlayer.status()` says so).  When the line from a
        source to the listener passes a closed face (wall, floor, ceiling),
        that face's material decides how much it is lowered and muffled
        (see audiospace.occlusion.OCCLUSION_BY_CATEGORY; walls of two
        neighbouring spaces in nearly one plane count once).  The values
        are far below real sound insulation on purpose: hearing a muffled
        sound through a wall is itself a cue, so it stays audible.

        openings   `Opening` boxes that cut doors, windows or stairwells
                   into faces.  A closed door is an Opening with a
                   material; set_opening opens and closes it
        materials  {material name: (dB, lowpass 0..1)} to override the
                   table
        strength   scales the whole effect (0 = off, 1 = the table)
        portals    when the straight line is blocked, also try the way
                   round through open faces and openings; if that is
                   louder, the sound comes from the listener's doorway,
                   at the length of the way round

        The result for each source can be read with source_occlusion.
        It adds to set_source_lowpass rather than replacing it.  Off by
        default; calling it with enabled=False eases every source back.
        """
        if not enabled:
            with self._lock:
                self._occ_opts = None
                self._occ_map = None
                self._occ_version += 1
                self._occ_gen += 1
            return
        strength = _finite(strength, "strength")
        if strength < 0.0:
            raise ValueError(f"strength must not be negative, got {strength}")
        openings = list(openings)
        for o in openings:
            if not isinstance(o, Opening):
                raise ValueError(f"openings must be Opening objects, got {o!r}")
        names = [o.name for o in openings if o.name]
        if len(names) != len(set(names)):
            raise ValueError("opening names must be unique")
        table = {}
        for name, value in (materials or {}).items():
            try:
                db, lp = value
            except (TypeError, ValueError):
                raise ValueError(f"materials[{name!r}] must be (dB, lowpass)"
                                 ) from None
            db = _finite(db, f"materials[{name!r}] dB")
            lp = _finite(lp, f"materials[{name!r}] lowpass")
            if db < 0.0 or not 0.0 <= lp <= 1.0:
                raise ValueError(f"materials[{name!r}] needs dB >= 0 and "
                                 "lowpass 0..1")
            table[name] = (db, lp)
        opts = dict(openings=openings, strength=strength, materials=table,
                    portals=bool(portals))
        with self._lock:
            self._occ_opts = opts
            self._occ_gen += 1
        self._refresh_occlusion()

    def _occlusion_spaces(self) -> list[Space]:
        """遮蔽に使う空間: set_spaces の登録、無ければ set_space の空間。"""
        if self._space_map:
            return list(self._space_map)
        return [self.space] if self.space is not None else []

    def _refresh_occlusion(self) -> None:
        """遮蔽の地図を作り直して差し替える。エンジンのロックを持たずに呼ぶ。

        地図はロックの外で作る（空間40個・戸40個で約10ms。1ブロックの予算に
        近い）。差し替えだけをロックの中で行う。作っている間に設定や空間が
        変わったら（_occ_gen が進んだら）、新しい材料で作り直す。
        """
        with self._occ_build_lock:
            while True:
                with self._lock:
                    opts = self._occ_opts
                    spaces = self._occlusion_spaces()
                    gen = self._occ_gen
                if opts is None:
                    return
                omap = OcclusionMap(spaces, **opts)
                with self._lock:
                    if self._occ_gen != gen:
                        continue
                    self._occ_map = omap
                    self._occ_version += 1
                    self._occ_active = True
                    return

    @property
    def occlusion_spaces(self) -> int:
        """自動の遮蔽が使っている空間の数（切っていれば 0）。"""
        with self._lock:
            return len(self._occ_map.spaces) if self._occ_map else 0

    @property
    def auto_occlusion(self) -> bool:
        """set_auto_occlusion で自動の遮蔽を入れているか。"""
        return self._occ_opts is not None

    def set_opening(self, name: str, material: str | None = None) -> None:
        """Open (material None) or close an Opening by its name.

            engine.set_opening("kitchen_door", "wooden_door")   # close
            engine.set_opening("kitchen_door")                  # open
        """
        if material is not None:
            material_occlusion(material)          # 名前を確かめる
        with self._lock:
            if self._occ_opts is None:
                raise ValueError("auto occlusion is off; call "
                                 "set_auto_occlusion first")
            openings = with_material(self._occ_opts["openings"], name,
                                     material)
            # 設定は新しい辞書にする（作りかけの地図が古い設定を読み続ける
            # ことのないように。作りかけは _occ_gen で気づいて作り直す）
            self._occ_opts = dict(self._occ_opts, openings=openings)
            self._occ_gen += 1
            # 戸の材質だけなので、地図のつながりは作り直さず、値だけ替える
            # （空間が多くても、描画をほとんど止めない）
            if self._occ_map is not None:
                self._occ_map.set_opening_material(name, material)
            self._occ_version += 1

    def source_occlusion(self, handle: int) -> Occlusion:
        """The occlusion found for a source (what it is heading to).

        loss_db and lowpass, the materials passed through (walls), the
        openings it came round through (via) and where it is heard from
        then (position).  Useful to say "behind a concrete wall".
        """
        with self._lock:
            return self._slots[handle]._occ

    def _update_occlusion(self, slots: list[SourceSlot]) -> None:
        """遮蔽を調べ直し、鳴らす値を行き先へ1ブロックぶん動かす。"""
        omap = self._occ_map
        lis = self._listener.position
        lkey = (round(lis[0] * 10.0), round(lis[1] * 10.0),
                round(lis[2] * 10.0), self._occ_version)
        jump = self._occ_jump
        self._occ_jump = False
        stale = []
        for i, slot in enumerate(slots):
            if omap is None or not slot.occlude or slot.relative:
                slot._occ = NO_OCCLUSION
                slot._occ_key = None
                continue
            pos = slot.position
            key = (round(pos[0] * 10.0), round(pos[1] * 10.0),
                   round(pos[2] * 10.0), lkey)
            if key != slot._occ_key:
                stale.append((not slot._occ_fresh,
                              (i - self._occ_cursor) % len(slots), slot, key))
        if stale:
            # 鳴り始めの音を先に（壁の向こうで鳴り始める音は、最初から
            # こもっているべき）。残りは順番に回す
            stale.sort(key=lambda item: item[:2])
            # 瞬間移動の直後は、分けずに全部を調べる
            batch = stale if jump else stale[:self.OCCLUSION_BATCH]
            found = omap.query([item[2].position for item in batch], lis)
            for (_, _, slot, key), occ in zip(batch, found):
                slot._occ = occ
                slot._occ_key = key
            self._occ_cursor += len(batch)
        step_db = self.OCCLUSION_DB_PER_100MS * self._block_seconds / 0.1
        step_lp = self.OCCLUSION_LP_PER_100MS * self._block_seconds / 0.1
        busy = omap is not None
        for slot in slots:
            want_db, want_lp = slot._occ.loss_db, slot._occ.lowpass
            if jump or (slot._occ_fresh and (
                    slot._occ_key is not None or omap is None
                    or not slot.occlude or slot.relative)):
                slot._occ_db, slot._occ_lp = want_db, want_lp
                slot._occ_fresh = False
            else:
                delta = want_db - slot._occ_db
                slot._occ_db += math.copysign(min(abs(delta), step_db), delta)
                delta = want_lp - slot._occ_lp
                slot._occ_lp += math.copysign(min(abs(delta), step_lp), delta)
            if slot._occ_db or slot._occ_lp:
                busy = True
        self._occ_active = busy

    def set_spaces(self, spaces: list[Space], early: bool = True,
                   late: bool = True, transition_s: float = 0.3,
                   release_s: float | None = None) -> None:
        """場所ごとの空間を登録する。聞き手のいる場所で自動で切り替わる。

        毎ブロック、聞き手を範囲（`Space.in_region`）に含む空間のうち、
        いちばん小さいもの（体積）を選ぶ。建物の間の道の中に部屋を置けば、
        部屋に入ると部屋の響きに、出ると道の響きになる。いまの空間から
        出るまでは、ほかの空間へは移らない（ただし、より小さい空間に
        入ったときは移る）。境目で行ったり来たりしないため。
        どの空間にも入っていなければ空間の響きは無し。
        切り替えのつなぎは set_space と同じ（transition_s, release_s）。
        空のリストで登録を外す（空間の響きも外す）。
        """
        if late == "precise":
            for sp in spaces:
                self._prepare_precise(sp)
        with self._lock:
            self._space_map = list(spaces)
            self._swap_transition = transition_s
            self._map_options = dict(early=early, late=late,
                                     transition_s=transition_s,
                                     release_s=release_s)
            self._occ_gen += 1
            occluding = self._occ_opts is not None
            if not self._space_map:
                self._switch_space(None, early, late, transition_s,
                                   release_s)
            else:
                self._follow_space_map(self._listener.position)
        if occluding:
            self._refresh_occlusion()

    def _follow_space_map(self, position) -> None:
        """登録した場所ごとの空間から、聞き手のいる空間を選んで切り替える。"""
        inside = [sp for sp in self._space_map if sp.in_region(position)]
        best = min(inside, key=lambda sp: sp.volume) if inside else None
        current = self.space
        if current is not None and any(current is sp for sp in inside):
            if best is None or best is current or best.volume >= current.volume:
                return
        if best is current:
            return
        self._switch_space(best, **self._map_options)

    def _switch_space(self, space: Space | None, early: bool, late: bool,
                      transition_s: float, release_s: float | None) -> None:
        """空間を切り替える（ロックを持った状態で呼ぶ）。"""
        # まだ1ブロックも鳴らしていない空間からの切り替えは、つなぐものが無い
        fade = self._space_blocks > 0 and transition_s > 0.0
        # 残響の入力（送り）も transition_s でつなぐ。前の残響への入力を
        # いきなり止めたり、新しい残響へ全量をいきなり入れたりすると、
        # 遅延線を一巡した後（最短で1ブロック後）に響きの頭や終わりが角張って
        # 出て、プツッと聞こえた。前の残響の入力は 1→0、新しい残響は 0→1。
        in_step = self._block_seconds / transition_s if fade else 1.0
        if self.reverb is not None and self._space_blocks > 0:
            step = (self._block_seconds / release_s
                    if release_s and release_s > 0 else 0.0)
            self._tails.append({"reverb": self.reverb, "gain": 1.0,
                                "step": step, "in_gain": 1.0,
                                "in_step": in_step})
            del self._tails[:-4]        # 残す尾は多くて4つ
        self._reverb_in_gain = 0.0 if fade else 1.0
        self._reverb_in_step = in_step
        if fade and self._early is not None:
            self._old_early = self._early
            self._old_early_gain = self._early_gain
            self._old_early_step = self._block_seconds / transition_s
        elif not (fade and self._old_early is not None):
            # いまの反射が無く（空間なしの間）、前の反射がまだ消えていく
            # 途中なら、そのまま消し終える。部屋を出て、消え終わる前に
            # 戻ったときに、その場で切らない
            self._old_early = None
        self.space = space
        self._space_blocks = 0
        self.reverb = None
        self._early = None
        self._late_mode = late
        if space is None:
            self.late_status = ""
            return
        if early:
            self._early = EarlyReflections(space, self.sample_rate,
                                           self.block)
            self._early_gain = 0.0 if fade else 1.0
            self._early_step = (self._block_seconds / transition_s
                                if fade else 0.0)
        self.reverb = self._late_for(space, late)

    def _fast_late(self, space: Space) -> BandFdnReverb:
        return band_reverb_for(space, self.sample_rate, self.block)

    def _late_for(self, space: Space, late) -> Reverb | None:
        """空間の後期残響を作る（ロックを持った状態で呼ぶ。重い計算はしない）。"""
        if not late:
            self.late_status = tr("後期残響なし", "no late reverb")
            return None
        if space.late_level() <= 0.0:
            self.late_status = tr("拡散した残響なし（開けた場所）",
                                  "no diffuse reverb (open place)")
            return None
        if late == "precise" and space.open_fraction > 0.0:
            # 開いた面のある空間は精密な計算の尾が途切れる（precise.py）
            self.late_status = tr("開いた面のある空間のため速い版の残響",
                                  "fast reverb, because the space has "
                                  "open sides")
            return self._fast_late(space)
        if late == "precise" and space.contents:
            # 精密な計算（pyroomacoustics）は面の材質しか扱えず、置いてある
            # もの（家具や人）の吸音が入らない。速い版は入るのでそちらにする
            self.late_status = tr("家具や人がいるため速い版の残響",
                                  "fast reverb, because there is furniture "
                                  "or people")
            return self._fast_late(space)
        if late == "precise":
            from .precise import space_key
            key = space_key(space)
            kernel = self._precise_kernels.get(key)
            if kernel is not None:
                self._keep_kernel(key, kernel)      # 使ったので後ろへ回す
                self.late_status = _PRECISE_ON
                return kernel.clone()
            if key in self._precise_pending:
                self.late_status = tr("精密版を計算中（いまは速い版）",
                                      "computing the precise reverb "
                                      "(fast reverb for now)")
            elif not self.late_status.startswith(_PRECISE_UNAVAILABLE):
                self.late_status = tr("精密版が無いため速い版",
                                      "fast reverb, because there is no "
                                      "precise one")
        else:
            self.late_status = tr("速い版の残響", "fast reverb")
        return self._fast_late(space)

    def _prepare_precise(self, space: Space) -> None:
        """精密版の尾を用意する（呼び出し側のスレッドで。音声スレッドでは呼ばない）。

        保存済みなら読み込んで畳み込みの核を作る。無ければ別のプロセスで計算を
        始め、終わるのを待つスレッドを立てる。そのスレッドはプロセスの終了を
        待つだけなので GIL を使わない（同じプロセスで計算すると描画が止まる）。
        """
        from .precise import (SpaceIr, can_render_in_background,
                              find_space_ir, space_key, start_render_process,
                              supports_precise)
        if not supports_precise(space):
            # 精密版を使わない空間。家具や人がいる空間で計算を始めると、
            # 物の吸音を含まない尾ができ、それに差し替わってしまう
            return
        key = space_key(space)
        if key in self._precise_kernels or key in self._precise_pending:
            return
        found = find_space_ir(space, self.sample_rate, self.space_ir_cache)
        if found is not None:
            kernel = ConvolutionReverb(found.tail, block=self.block,
                                       normalise=False)
            with self._lock:
                self._keep_kernel(key, kernel)
            return
        if not can_render_in_background():
            self.late_status = (_PRECISE_UNAVAILABLE + tr(
                "（pyroomacoustics が無いか、固めた exe のため）。"
                "速い版で鳴らします",
                " (pyroomacoustics is missing, or this is a frozen exe). "
                "Playing the fast reverb"))
            return
        proc, out = start_render_process(space, self.sample_rate,
                                         self.space_ir_cache)
        with self._lock:
            self._precise_pending.add(key)

        def watch() -> None:
            # wait() してから stderr を読むと、子が大量に書いたときに互いに
            # 待ち合って止まる。communicate() は読みながら待つ
            _out, raw = proc.communicate()
            code = proc.returncode
            err = (raw or b"").decode("utf-8", "replace")
            kernel = None
            if code == 0 and out.exists():
                try:
                    found = (SpaceIr.load(out).matched(space)
                             .resampled(self.sample_rate))
                    kernel = ConvolutionReverb(found.tail, block=self.block,
                                               normalise=False)
                except Exception as exc:          # noqa: BLE001
                    err = f"{err}\n{exc}"
            with self._lock:
                self._precise_pending.discard(key)
                if kernel is None:
                    tail = err.strip().splitlines()[-1:] or [f"code {code}"]
                    self.late_status = tr(
                        f"精密版の計算に失敗（{tail[0]}）。速い版で鳴らします",
                        f"computing the precise reverb failed ({tail[0]}). "
                        "Playing the fast reverb")
                    return
                self._keep_kernel(key, kernel)
                current = self.space
                if (current is not None and self._late_mode == "precise"
                        and space_key(current) == key):
                    self._swap_late(kernel.clone(), self._swap_transition)
                    self.late_status = _PRECISE_ON

        threading.Thread(target=watch, name="audiospace-precise",
                         daemon=True).start()

    def _swap_late(self, new: Reverb, transition_s: float) -> None:
        """いまの空間のまま、後期残響だけを差し替える（ロックを持って呼ぶ）。

        前の残響は尾として鳴り終わるまで残し、入力を transition_s で
        前から新しいほうへ移す（空間の切り替えと同じつなぎ方）。
        """
        fade = self._space_blocks > 0 and transition_s > 0.0
        in_step = self._block_seconds / transition_s if fade else 1.0
        if self.reverb is not None and self._space_blocks > 0:
            self._tails.append({"reverb": self.reverb, "gain": 1.0,
                                "step": 0.0, "in_gain": 1.0,
                                "in_step": in_step})
            del self._tails[:-4]
        self.reverb = new
        self._reverb_in_gain = 0.0 if fade else 1.0
        self._reverb_in_step = in_step

    # -- rendering -------------------------------------------------------
    def render(self) -> np.ndarray:
        """Produce one stereo block, shape (block, 2) float32."""
        with self._lock:
            if self._paused:
                self.last_peak = 0.0
                return np.zeros((self.block, 2), dtype=np.float32)
            if self._space_map:
                self._follow_space_map(self._listener.position)
            self._blocks[:] = 0.0
            self._send[:] = 0.0
            self._flat[:] = 0.0
            self._lp_bucket[:] = -1
            self._tilt_step[:] = 0
            listener = self._listener
            if self._wobble.enabled or self._roll_wobble.enabled:
                # ブロックの中央の時刻で評価する。ブロック先頭で評価すると
                # 実際の音より半ブロック分だけ首振りが遅れる。
                mid = self._elapsed + self._block_seconds * 0.5
                listener = Listener(
                    listener.position,
                    listener.yaw_deg + self._wobble.offset_at(mid),
                    listener.pitch_deg,
                    listener.roll_deg + self._roll_wobble.offset_at(mid))
            sends: list[tuple[int, float]] = []
            finished_fades: list[int] = []
            sp_rows: list[int] = []
            sp_slots: list[SourceSlot] = []
            sp_fades: list[float] = []
            sp_jumps: list[int] = []
            if self._sound_groups:
                levels, paused_groups = self._advance_groups()
            else:
                levels, paused_groups = {}, set()

            for row, handle in enumerate(self._slot_of_row):
                if handle is None:
                    continue
                slot = self._slots[handle]
                if not slot.active or (paused_groups
                                       and slot.group in paused_groups):
                    if slot._fade_step:
                        # フェード中に止められた音。描画されないので外す
                        finished_fades.append(handle)
                    continue
                if slot._held or slot._wait >= self.block:
                    # まだ鳴り始めない音。読まず、ランプも進めない
                    if not slot._held:
                        slot._wait -= self.block
                    continue
                try:
                    if slot._wait:
                        mono = self._read_late(slot)
                    else:
                        mono = self._read_source(slot)
                    if (slot._notify and not slot._ended
                            and slot.source.finished):
                        # データを読み終えたブロック。鳴り終わりを記録する
                        self._note_end(handle, slot, SOUND_FINISHED)
                except Exception as exc:          # noqa: BLE001
                    # この音源だけを外す。外さないと毎ブロック同じ例外で
                    # render 全体が止まり、ほかの音も全部聞こえなくなる
                    self.source_errors += 1
                    self.last_source_error = f"{type(exc).__name__}: {exc}"
                    self._note_end(handle, slot, SOUND_ERROR,
                                   self.last_source_error)
                    finished_fades.append(handle)
                    continue
                fade = slot._fade_gain
                if slot._fade_step:
                    if fade <= 0.0 and slot._cur_gain <= 1e-5:
                        # the fade finished last block (its gain ramped to
                        # zero inside that block), so releasing now is silent
                        finished_fades.append(handle)
                        continue
                    slot._fade_gain = max(0.0, fade - slot._fade_step)
                if slot._gain_target is not None:
                    # multi-block gain ramp, advanced here on the audio
                    # thread rather than by the caller's tick loop
                    remaining = slot._gain_target - slot.gain
                    if abs(remaining) <= slot._gain_step:
                        slot.gain = slot._gain_target
                        slot._gain_target = None
                    else:
                        slot.gain += math.copysign(slot._gain_step, remaining)
                if slot._occl_now != slot.lowpass:
                    delta = slot.lowpass - slot._occl_now
                    slot._occl_now += math.copysign(
                        min(abs(delta), slot._occl_step), delta)

                if not slot.spatialise:
                    # pan 0 leaves both ears at full gain, which is what
                    # this did before panning existed.
                    left = min(1.0, 1.0 - slot.pan)
                    right = min(1.0, 1.0 + slot.pan)
                    # same per-block gain ramp as spatialised sources, so
                    # a gain change, a fade-out or a resume never clicks
                    target = slot.gain * fade
                    if slot.group is not None:
                        target *= levels[slot.group]
                    if slot._gain_jump:
                        # 鳴り始め: フェードインせず、データのとおりに鳴らす
                        slot._cur_gain = target
                        slot._gain_jump = False
                    gains = (slot._cur_gain
                             + (target - slot._cur_gain) * self._ramp01)
                    slot._cur_gain = target
                    if slot._occl_now > 0.0:
                        # Rare on the flat path (music is seldom occluded),
                        # so this pays a per-source filter call instead of
                        # joining the batched pass.  State lives on the
                        # slot, one value per channel; the per-row table
                        # belongs to the batched spatial pass.
                        mono = self._filter_flat(slot, mono, slot._occl_now)
                    else:
                        slot._flat_zi = None
                    if getattr(slot.source, "channels", 1) == 2:
                        self._flat[:, 0] += mono[:, 0] * (gains * left)
                        self._flat[:, 1] += mono[:, 1] * (gains * right)
                        if slot.reverb_send > 0:
                            self._send += (mono.mean(axis=1)
                                           * (gains * slot.reverb_send))
                    else:
                        self._flat[:, 0] += mono * (gains * left)
                        self._flat[:, 1] += mono * (gains * right)
                        if slot.reverb_send > 0:
                            self._send += mono * (gains * slot.reverb_send)
                    continue

                # 方向・距離・前後の計算はここでは行わず、ループの後で
                # 全音源ぶんをまとめて配列で行う。音源ごとに呼ぶと、128音源で
                # 呼び出しの手間だけで 2.4ms かかっていた（まとめると 0.06ms）。
                # ここでは音をそのまま置き、音量はあとでまとめて掛ける。
                self._blocks[row] = mono
                if slot._gain_jump:
                    sp_jumps.append(len(sp_slots))
                    slot._gain_jump = False
                sp_rows.append(row)
                sp_slots.append(slot)
                # グループの倍率は、フェードと同じく音量に掛ける
                sp_fades.append(fade if slot.group is None
                                else fade * levels[slot.group])
                if slot.reverb_send > 0:
                    sends.append((row, slot.reverb_send))

            for handle in finished_fades:
                self.remove_source(handle)

            rows = np.asarray(sp_rows, dtype=np.intp)
            if self._occ_active and sp_slots:
                self._update_occlusion(sp_slots)
            if rows.size:
                self._place_spatial(listener, rows, sp_slots, sp_fades,
                                    sp_jumps)

            self._apply_damping()
            self._apply_elevation_emphasis()
            for row, amount in sends:
                self._send += self._blocks[row] * amount

            active = rows
            if self._early is not None or self._old_early is not None:
                active = self._place_early(listener, rows)

            # 鳴っている音源の行だけを畳み込みに渡す。容量（max_sources）
            # ぶんを毎回計算すると、16音を鳴らすのに容量256で22msかかった
            out = self._conv.process(self._blocks, self._dir_idx,
                                     active=active)
            out += self._flat
            if self.reverb is not None:
                send = self._send
                if self._reverb_in_gain < 1.0:
                    g0 = self._reverb_in_gain
                    g1 = min(1.0, g0 + self._reverb_in_step)
                    self._reverb_in_gain = g1
                    send = send * (g0 + (g1 - g0) * self._ramp01)
                out += self.reverb.process(send)
            if self._tails:
                out += self._ring_out_tails()
            if not np.all(np.isfinite(out)):
                # 計算が壊れた（残響の発散など）ブロックは無音にし、残響の
                # 中身を捨てる。壊れた値をそのまま出すと、出力機器によっては
                # 大きな雑音になる。回数は nonfinite_blocks で分かる
                out[:] = 0.0
                self.nonfinite_blocks += 1
                # 壊れた値が残ったままだと、次のブロックも壊れて無音が続く。
                # 行ごとのフィルタ・畳み込みの尾・反射の状態も消す
                self._lp_zi[:] = 0.0
                self._tilt_zi[:] = 0.0
                self._conv.reset()
                for s in self._slots.values():
                    s._flat_zi = None
                for early in (self._early, self._old_early):
                    if early is not None:
                        early.reset()
                if self.reverb is not None and hasattr(self.reverb, "reset"):
                    self.reverb.reset()
                self._tails = []
            self._space_blocks += 1
            out *= self.master_gain
            if self.limiter_enabled:
                out = self._apply_limiter(out)
            self._elapsed += self._block_seconds

        self.last_peak = float(np.max(np.abs(out))) if out.size else 0.0
        return out

    def _read_source(self, slot: SourceSlot) -> np.ndarray:
        """音源から1ブロック読む。形が違えば ValueError。

        形の違う音（長さが違う、モノラルのはずが2列など）をそのまま行に
        書くと、その場で例外になって render 全体が止まる。ここで確かめて、
        その音源だけを外せるようにする。
        """
        return self._read_frames(slot, self.block)

    def _read_frames(self, slot: SourceSlot, n: int) -> np.ndarray:
        data = np.asarray(slot.source.read(n), dtype=np.float32)
        want = ((n, 2) if getattr(slot.source, "channels", 1) == 2
                else (n,))
        if data.shape != want:
            raise ValueError(f"read() returned shape {data.shape}, "
                             f"expected {want}")
        return data

    def _read_late(self, slot: SourceSlot) -> np.ndarray:
        """遅らせた音が鳴り始めるブロック。前に無音を置いて1ブロックにする。

        残りの無音（`_wait`、ブロックより短い）のあとに、音源の最初から
        読んだ音を置く。音源から読むのはその残りのぶんだけなので、続きの
        ブロックとの間に隙間も重なりもできない。
        """
        offset = slot._wait
        part = self._read_frames(slot, self.block - offset)
        data = np.zeros((self.block,) + part.shape[1:], dtype=np.float32)
        data[offset:] = part
        slot._wait = 0
        return data

    def _place_spatial(self, listener: Listener, rows: np.ndarray,
                       slots: list[SourceSlot], fades: list[float],
                       jumps: list[int] | None = None) -> None:
        """空間化する音源の方向・音量・フィルタを、まとめて決めて適用する。

        `_blocks[rows]` には音源の音がそのまま入っている。ここで
        距離・前後の音量をブロック内でランプさせて掛け、方向の索引と
        前後ローパスのバケットを決める。計算は音源ごとの版
        （relative_spherical / distance_gain / front_back_gain /
        _bucket_for）と同じ式で、丸め誤差の範囲で一致する。
        """
        occluding = self._occ_active
        if occluding:
            # 回り込んで届く音は、出入口の方向から、道のりの距離で鳴らす
            pos = np.array([s._occ.position if s._occ.position is not None
                            else s.position for s in slots],
                           dtype=np.float64)
        else:
            pos = np.array([s.position for s in slots], dtype=np.float64)
        relative = np.fromiter((s.relative for s in slots), dtype=bool,
                               count=len(slots))
        if relative.any():
            pos[relative] += np.asarray(listener.position, dtype=np.float64)
        az, el, dist = relative_spherical_many(listener, pos)
        depth = self.effective_front_back_depth
        base = np.fromiter((s.gain * f for s, f in zip(slots, fades)),
                           dtype=np.float64, count=len(slots))
        if occluding:
            base *= np.fromiter((10.0 ** (-s._occ_db / 20.0) for s in slots),
                                dtype=np.float64, count=len(slots))
        target = (base
                  * distance_gain_many(dist, self.distance_model,
                                       self.reference_distance,
                                       self.max_distance)
                  * front_back_gain_many(az, depth, el))
        cur = np.fromiter((s._cur_gain for s in slots), dtype=np.float64,
                          count=len(slots))
        if jumps:
            # 鳴り始め: フェードインせず、データのとおりに鳴らす
            cur[jumps] = target[jumps]
        # ramp the gain across the block instead of stepping it
        ramp = (cur[:, None]
                + (target - cur)[:, None] * self._ramp01[None, :])
        self._blocks[rows] *= ramp.astype(np.float32)
        for slot, value in zip(slots, target.tolist()):
            slot._cur_gain = value
        occl = np.fromiter((s._occl_now for s in slots), dtype=np.float64,
                           count=len(slots))
        if occluding:
            # 手で決めたこもり（set_source_lowpass）と、壁のこもりを合わせる
            auto = np.fromiter((s._occ_lp for s in slots), dtype=np.float64,
                               count=len(slots))
            occl = 1.0 - (1.0 - occl) * (1.0 - auto)
        self._lp_bucket[rows] = self._buckets_for(az, occl, el)
        if self.effective_elevation_emphasis > 0.0:
            self._tilt_step[rows] = self._emphasis_steps(el)
        # 方向探索は1回にまとめる。音源ごとに index_for を呼ぶと、方向数の
        # 多い実測HRTF（1513方向）では呼び出し回数がそのままコストになる。
        # 実測で64音源3.6ms→6.5msの差が出た。
        self._dir_idx[rows] = self.hrir.indices_for(az, el)

    def _place_early(self, listener: Listener, rows: np.ndarray) -> np.ndarray:
        """空間の初期反射を、音源の行の後ろの行に置く。畳み込む行を返す。

        反射の入力は空間化した音源の和（前後ローパスと距離の減衰の後）。
        方向は world 座標の面の向きを、いまの聞き手（首振りなどを含む）から
        見た向きに直して HRTF を選ぶ。前後ローパスは掛けない（人工の手がかり
        なので、反射には要らない）。

        切り替えの途中は、新しい空間の反射（音源の行のすぐ後ろ）と、消えて
        いく前の空間の反射（その後ろ）を両方置き、倍率をブロックの中で
        なめらかに動かす。
        """
        dry = (self._blocks[rows].sum(axis=0) if rows.size
               else np.zeros(self.block, dtype=np.float32))
        parts = [rows]
        if self._early is not None:
            g0 = self._early_gain
            g1 = min(1.0, g0 + self._early_step)
            self._early_gain = g1
            parts.append(self._early_rows(self._early, dry, listener,
                                          self.max_sources, g0, g1))
        if self._old_early is not None:
            g0 = self._old_early_gain
            g1 = max(0.0, g0 - self._old_early_step)
            self._old_early_gain = g1
            parts.append(self._early_rows(self._old_early, dry, listener,
                                          self.max_sources + MAX_PATHS,
                                          g0, g1, hold=True))
            if g1 <= 0.0:
                self._old_early = None
        return np.concatenate(parts)

    def _early_rows(self, early: EarlyReflections, dry: np.ndarray,
                    listener: Listener, first_row: int, g0: float,
                    g1: float, hold: bool = False) -> np.ndarray:
        # hold: 消えていく前の空間の反射は、聞き手が壁を抜けて外に出ても、
        # 最後に中にいた位置で続ける（0本にすると、その場で切れてプツッと鳴る）
        signals, dirs, dists = early.process(dry, listener.position,
                                             hold_outside=hold)
        if not len(signals):
            return np.zeros(0, dtype=np.intp)
        if g0 != 1.0 or g1 != 1.0:
            signals = signals * (g0 + (g1 - g0) * self._ramp01)[None, :]
        er_rows = first_row + np.arange(len(signals), dtype=np.intp)
        self._blocks[er_rows] = signals
        points = (np.asarray(listener.position, dtype=np.float64)
                  + dirs * dists[:, None])
        az, el, _ = relative_spherical_many(listener, points)
        self._dir_idx[er_rows] = self.hrir.indices_for(az, el)
        return er_rows

    def _ring_out_tails(self) -> np.ndarray:
        """前の空間の残響を鳴らし終える。入力は transition_s で絞り、その後は
        新しい音を入れずに自然に減衰させる。消えたら外す。"""
        zero = np.zeros(self.block, dtype=np.float32)
        total = np.zeros((self.block, 2), dtype=np.float32)
        keep = []
        for tail in self._tails:
            i0 = tail["in_gain"]
            if i0 > 0.0:
                i1 = max(0.0, i0 - tail["in_step"])
                tail["in_gain"] = i1
                feed = self._send * (i0 + (i1 - i0) * self._ramp01)
            else:
                feed = zero
            y = tail["reverb"].process(feed)
            g0 = tail["gain"]
            if tail["step"] > 0.0:
                g1 = max(0.0, g0 - tail["step"])
                tail["gain"] = g1
                y = y * (g0 + (g1 - g0) * self._ramp01)[:, None]
            else:
                g1 = g0
            total += y
            # -120dB を下回るか、打ち切りのフェードが終わったら外す
            # （入力を絞っている途中は、静かでも外さない）
            if g1 > 0.0 and (tail["in_gain"] > 0.0
                             or float(np.max(np.abs(y))) > 1e-6):
                keep.append(tail)
        self._tails = keep
        return total

    def _apply_limiter(self, out: np.ndarray) -> np.ndarray:
        """Keep the mix under `limit_threshold` without hard clipping.

        Attack is instant: a block whose peak exceeds the threshold is
        scaled down to fit its own peak.  Release recovers over roughly
        half a second.  The gain change is ramped across the block, so the
        correction itself never steps audibly; a final hard clip guards
        the ramp-in edge, where a few samples can still exceed full scale.
        Blocks below the threshold at unity gain pass through untouched.
        """
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        need = (self.limit_threshold / peak
                if peak > self.limit_threshold else 1.0)
        release_step = self._block_seconds / 0.5
        target = min(need, min(1.0, self._limiter_gain + release_step))
        if target >= 1.0 and self._limiter_gain >= 1.0:
            return out
        if target < self._limiter_gain:
            self.limited_blocks += 1
        gains = (self._limiter_gain
                 + (target - self._limiter_gain) * self._ramp01)
        out *= gains[:, None]
        self._limiter_gain = target
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def render_seconds(self, seconds: float) -> np.ndarray:
        """Offline render, for tests and for baking demo files."""
        n = int(np.ceil(seconds * self.sample_rate / self.block))
        return np.concatenate([self.render() for _ in range(n)], axis=0)

    # -- damping (front/back and occlusion) -------------------------------
    #
    # Level alone is a poor front/back cue: it is indistinguishable from
    # distance.  The pinna makes sounds from behind measurably duller, and
    # reproducing that is what actually resolves front/back confusion for
    # players -- especially with a synthetic HRIR set, whose mirrored
    # directions are numerically identical.  Occlusion ("a wall is in the
    # way") is the same cue for the same reason, so both feed one filter.
    #
    # The filter is a one-pole low-pass whose coefficient follows the forward
    # component, so a source circling the listener sweeps smoothly rather
    # than switching at a threshold.  Calling lfilter once per source costs
    # far more in call overhead than in arithmetic, so coefficients are
    # quantised into buckets and every source sharing a bucket is filtered in
    # a single call; 24 buckets keeps the step inaudible while cutting up to
    # 64 calls down to at most 24.
    LP_BUCKETS = 24

    @property
    def effective_front_back_depth(self) -> float:
        """`front_back_depth` capped at 0.9.

        At depth 1.0 the level term of the cue reaches exactly zero for a
        source directly behind, which mutes it outright -- the cue is
        supposed to muffle, never to erase.  The cap keeps the API tolerant
        of out-of-range values instead of failing silently (literally).
        """
        return min(float(self.front_back_depth), 0.9)

    @property
    def effective_elevation_emphasis(self) -> float:
        """`elevation_emphasis` clamped to 0..1."""
        return min(1.0, max(0.0, float(self.elevation_emphasis)))

    def emphasis_db(self, elevation_deg: float) -> float:
        """上下の強調で、その仰角の音の 3kHz より上を何dB変えるか（丸める前）。"""
        return (ELEVATION_EMPHASIS_MAX_DB * self.effective_elevation_emphasis
                * float(elevation_deg) / 90.0)

    def _emphasis_steps(self, elevation_deg: np.ndarray) -> np.ndarray:
        """上下の強調の段（1dB 刻みの整数）。同じ段の音源をまとめてかける。

        後ろのローパスと同じく、段に丸めて同じ段の行を1回のフィルタで
        処理する。1dB 刻みなら、音源が上下に動いても段の変わり目は
        聞こえない（音量の 1dB の段差は聞き分けにくい大きさ）。
        """
        amount = self.effective_elevation_emphasis
        db = ELEVATION_EMPHASIS_MAX_DB * amount * np.asarray(
            elevation_deg, dtype=np.float64) / 90.0
        return np.round(db).astype(np.intp)

    def _emphasis_sos(self, step: int) -> np.ndarray:
        """段 `step` dB の係数（2次の1節）。

        RBJ のハイシェルフ（傾き1、3kHz で増減の半分、その上で全量）に、
        増減の半分を打ち消す全体の倍率を掛ける。アプリが使っていた
        「元の音 + k・高域通過」の形は、3kHz の手前で位相が回って打ち消し
        合い、真上で低域が -7.8dB・高域が +5.7dB と狙いからずれた。
        """
        sos = self._tilt_sos.get(step)
        if sos is None:
            sos = _rbj_shelf("high", ELEVATION_EMPHASIS_CORNER_HZ,
                             10.0 ** (step / 20.0), self.sample_rate)
            sos = sos.copy()
            sos[:3] *= 10.0 ** (-step * ELEVATION_EMPHASIS_LEVEL_SHARE / 20.0)
            sos = sos[None, :]
            self._tilt_sos[step] = sos
        return sos

    def _apply_elevation_emphasis(self) -> None:
        """Filter the staged rows by their elevation-emphasis step.

        強調がオフ（どの行も段 0）なら何もしない。状態を消すのも、前の
        ブロックまでにどこかの行で使っていたときの1回だけ。
        """
        steps = self._tilt_step
        active = steps != 0
        if not active.any():
            if self._tilt_dirty:
                self._tilt_zi[:] = 0.0
                self._tilt_dirty = False
            return
        for step in np.unique(steps[active]):
            rows = np.flatnonzero(steps == step)
            out, zf = signal.sosfilt(
                self._emphasis_sos(int(step)), self._blocks[rows],
                axis=1, zi=self._tilt_zi[:, rows])
            self._blocks[rows] = out.astype(np.float32)
            self._tilt_zi[:, rows] = zf
        # rows not being filtered must not keep a stale filter tail
        self._tilt_zi[:, ~active] = 0.0
        self._tilt_dirty = True

    def _bucket_for(self, azimuth_deg: float, occlusion: float = 0.0,
                    elevation_deg: float = 0.0) -> int:
        """Quantised filter strength for a source; -1 means no filtering.

        The direction term is scaled by `elevation_weight`, as in
        `front_back_gain`: overhead there is no front or back to muffle.

        Direction and occlusion combine as cascaded absorptions,
        `1 - (1-a)(1-b)`, which is the same as multiplying the two
        coefficients: a source that is both behind the listener and behind
        a wall is duller than either alone, and neither cue can cancel the
        other out.
        """
        depth = self.effective_front_back_depth
        strength = 0.0
        if depth > 0.0:
            forward = math.cos(math.radians(azimuth_deg))  # +1 ahead, -1 behind
            openness = 0.5 * (forward + 1.0)               # 1 ahead, 0 behind
            strength = (depth * (1.0 - openness)
                        * elevation_weight(elevation_deg))
        if occlusion > 0.0:
            strength = 1.0 - (1.0 - strength) * (1.0 - occlusion)
        bucket = int(round(strength * (self.LP_BUCKETS - 1)))
        return -1 if bucket <= 0 else bucket

    def _buckets_for(self, azimuth_deg: np.ndarray,
                     occlusion: np.ndarray,
                     elevation_deg: np.ndarray | None = None) -> np.ndarray:
        """`_bucket_for` for many sources at once (same rules)."""
        depth = self.effective_front_back_depth
        if depth > 0.0:
            openness = 0.5 * (np.cos(np.radians(azimuth_deg)) + 1.0)
            strength = depth * (1.0 - openness)
            if elevation_deg is not None:
                strength = strength * elevation_weight_many(elevation_deg)
        else:
            strength = np.zeros(len(azimuth_deg), dtype=np.float64)
        occluded = occlusion > 0.0
        if occluded.any():
            strength = np.where(
                occluded, 1.0 - (1.0 - strength) * (1.0 - occlusion),
                strength)
        # np.round も round() と同じく偶数への丸め
        bucket = np.round(strength * (self.LP_BUCKETS - 1)).astype(np.intp)
        bucket[bucket <= 0] = -1
        return bucket

    def _coeff_of(self, bucket: int) -> float:
        # never let the coefficient reach 0: that would freeze the filter
        # (y[n] = y[n-1] forever) and mute the source outright instead of
        # muffling it, which front_back_depth=1.0 used to do.  And never
        # let it exceed 1: bucket -1 (no filtering wanted) would otherwise
        # yield 1.043 and *boost* the highs by ~9% -- audible as a brief
        # sheen exactly when an occlusion ramp passes near zero.
        return max(0.05, min(1.0, 1.0 - bucket / (self.LP_BUCKETS - 1)))

    def _filter_flat(self, slot: SourceSlot, data: np.ndarray,
                     strength: float) -> np.ndarray:
        """One-pole filter a flat source's samples, keeping its state.

        The flat path is not staged in `_blocks` and so cannot join the
        batched pass.  State is per slot and per channel: a stereo source
        needs one value for each ear, and seeding the right channel from
        the left's state every block put an audible step at every block
        boundary.  `data` may be (n,) or (n, channels).
        """
        coeff = self._coeff_of(self._bucket_for(0.0, strength))
        width = data.shape[1] if data.ndim == 2 else 1
        if slot._flat_zi is None or slot._flat_zi.shape[0] != width:
            slot._flat_zi = np.zeros(width, dtype=np.float64)
        if data.ndim == 2:
            out, zf = signal.lfilter(
                [coeff], [1.0, -(1.0 - coeff)], data, axis=0,
                zi=slot._flat_zi[None, :])
            slot._flat_zi = zf[0]
        else:
            out, zf = signal.lfilter(
                [coeff], [1.0, -(1.0 - coeff)], data, zi=slot._flat_zi)
            slot._flat_zi = zf
        return out.astype(np.float32)

    def _apply_damping(self) -> None:
        """Filter the staged rows in place, grouped by coefficient bucket."""
        buckets = self._lp_bucket
        active = buckets > 0
        if active.any():
            for bucket in np.unique(buckets[active]):
                rows = np.flatnonzero(buckets == bucket)
                coeff = self._coeff_of(int(bucket))
                out, zf = signal.lfilter(
                    [coeff], [1.0, -(1.0 - coeff)], self._blocks[rows],
                    axis=1, zi=self._lp_zi[rows])
                self._blocks[rows] = out.astype(np.float32)
                self._lp_zi[rows] = zf
        # rows not being filtered must not keep a stale filter tail
        self._lp_zi[~active] = 0.0
