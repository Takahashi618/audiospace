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
"""
from __future__ import annotations

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
        self._space_blocks = 0          # いまの空間で鳴らしたブロック数
        self._reverb_in_gain = 1.0      # いまの残響への入力の倍率（入る途中）
        self._reverb_in_step = 1.0
        # 場所ごとの空間（set_spaces）
        self._space_map: list[Space] = []
        self._map_options: dict = {}
        # 精密版の後期残響（案5）。鍵 -> 畳み込みの核（clone して使う）
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
        """
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
        with self._lock:
            self._listener = Listener(position, yaw_deg, pitch_deg, roll_deg)
            self._conv.reset()
            if self.reverb is not None:
                self.reverb.reset()
            if self._early is not None:
                self._early.reset()
            # 瞬間移動では前の場所の響きは残さない
            self._tails.clear()
            self._old_early = None

    # -- sources ---------------------------------------------------------
    def add_source(self, source: Source, position: Vec3 = (0.0, 0.0, 0.0),
                   gain: float = 1.0, reverb_send: float = 0.25,
                   relative: bool = False,
                   spatialise: bool = True, pan: float = 0.0,
                   lowpass: float = 0.0) -> int:
        """Register a source and return its handle.

        `lowpass` starts the source already muffled (see
        set_source_lowpass); a sound that begins behind a wall should not
        be heard opening up over the first few blocks.

        Raises RuntimeError when every slot is taken, rather than silently
        dropping the sound -- a dropped cue is a bug the player cannot see.
        Raises ValueError for a stereo source with spatialise=True: a
        direction can only be given to a mono signal.  This must fail here,
        at registration -- failing later, inside render(), would take every
        other sound down with it on every block.
        """
        if spatialise and getattr(source, "channels", 1) != 1:
            raise ValueError(
                "a spatialised source must be mono; play a stereo asset "
                "with spatialise=False, or downmix it")
        with self._lock:
            try:
                row = self._slot_of_row.index(None)
            except ValueError:
                raise RuntimeError(
                    f"all {self.max_sources} source slots are in use; "
                    "raise max_sources or release finished sources") from None
            handle = self._next_id
            self._next_id += 1
            amount = float(min(1.0, max(0.0, lowpass)))
            slot = SourceSlot(source, position, gain, reverb_send,
                              relative, spatialise, pan, lowpass=amount)
            slot._cur_gain = 0.0
            slot._occl_now = amount          # already muffled at birth
            self._slots[handle] = slot
            self._lp_zi[row] = 0.0           # no tail from the previous tenant
            self._tilt_zi[:, row] = 0.0
            self._conv.forget(row)           # nor its direction to fade from
            self._slot_of_row[row] = handle
            return handle

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
            if fade_ms > 0.0:
                slot._fade_step = max(
                    1e-6, self._block_seconds * 1000.0 / float(fade_ms))
                return
            del self._slots[handle]
            row = self._slot_of_row.index(handle)
            self._slot_of_row[row] = None
            self._blocks[row] = 0.0

    def set_source_position(self, handle: int, position: Vec3) -> None:
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
        with self._lock:
            slot = self._slots[handle]
            target = float(gain)
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
        amount = float(min(1.0, max(0.0, amount)))
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
        with self._lock:
            self._slots[handle].reverb_send = float(np.clip(send, 0.0, 1.0))

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
            if not self._space_map:
                self._switch_space(None, early, late, transition_s,
                                   release_s)
                return
            self._follow_space_map(self._listener.position)

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
        else:
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
                self._precise_kernels[key] = kernel
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
                    found = SpaceIr.load(out).resampled(self.sample_rate)
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
                self._precise_kernels[key] = kernel
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

            for row, handle in enumerate(self._slot_of_row):
                if handle is None:
                    continue
                slot = self._slots[handle]
                if not slot.active:
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
                mono = slot.source.read(self.block)

                if not slot.spatialise:
                    # pan 0 leaves both ears at full gain, which is what
                    # this did before panning existed.
                    left = min(1.0, 1.0 - slot.pan)
                    right = min(1.0, 1.0 + slot.pan)
                    # same per-block gain ramp as spatialised sources, so
                    # starting or stopping a UI sound never clicks
                    target = slot.gain * fade
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
                sp_rows.append(row)
                sp_slots.append(slot)
                sp_fades.append(fade)
                if slot.reverb_send > 0:
                    sends.append((row, slot.reverb_send))

            for handle in finished_fades:
                self.remove_source(handle)

            rows = np.asarray(sp_rows, dtype=np.intp)
            if rows.size:
                self._place_spatial(listener, rows, sp_slots, sp_fades)

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
                if self.reverb is not None and hasattr(self.reverb, "reset"):
                    self.reverb.reset()
                self._tails = []
            if self.space is not None:
                self._space_blocks += 1
            out *= self.master_gain
            if self.limiter_enabled:
                out = self._apply_limiter(out)
            self._elapsed += self._block_seconds

        self.last_peak = float(np.max(np.abs(out))) if out.size else 0.0
        return out

    def _place_spatial(self, listener: Listener, rows: np.ndarray,
                       slots: list[SourceSlot], fades: list[float]) -> None:
        """空間化する音源の方向・音量・フィルタを、まとめて決めて適用する。

        `_blocks[rows]` には音源の音がそのまま入っている。ここで
        距離・前後の音量をブロック内でランプさせて掛け、方向の索引と
        前後ローパスのバケットを決める。計算は音源ごとの版
        （relative_spherical / distance_gain / front_back_gain /
        _bucket_for）と同じ式で、丸め誤差の範囲で一致する。
        """
        pos = np.array([s.position for s in slots], dtype=np.float64)
        relative = np.fromiter((s.relative for s in slots), dtype=bool,
                               count=len(slots))
        if relative.any():
            pos[relative] += np.asarray(listener.position, dtype=np.float64)
        az, el, dist = relative_spherical_many(listener, pos)
        depth = self.effective_front_back_depth
        base = np.fromiter((s.gain * f for s, f in zip(slots, fades)),
                           dtype=np.float64, count=len(slots))
        target = (base
                  * distance_gain_many(dist, self.distance_model,
                                       self.reference_distance,
                                       self.max_distance)
                  * front_back_gain_many(az, depth, el))
        cur = np.fromiter((s._cur_gain for s in slots), dtype=np.float64,
                          count=len(slots))
        # ramp the gain across the block instead of stepping it
        ramp = (cur[:, None]
                + (target - cur)[:, None] * self._ramp01[None, :])
        self._blocks[rows] *= ramp.astype(np.float32)
        for slot, value in zip(slots, target.tolist()):
            slot._cur_gain = value
        occl = np.fromiter((s._occl_now for s in slots), dtype=np.float64,
                           count=len(slots))
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
                                          g0, g1))
            if g1 <= 0.0:
                self._old_early = None
        return np.concatenate(parts)

    def _early_rows(self, early: EarlyReflections, dry: np.ndarray,
                    listener: Listener, first_row: int, g0: float,
                    g1: float) -> np.ndarray:
        signals, dirs, dists = early.process(dry, listener.position)
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
