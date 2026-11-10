"""SpatialPlayer -- the high-level API.

Wraps device selection, the engine and the audio thread so that the common
case is three lines:

    player = SpatialPlayer.open()
    player.play("footstep.wav", position=(2.0, 5.0, 0.0))
    player.set_listener((0, 0, 0), yaw_deg=90)

Everything the low-level API offers stays reachable through `player.engine`.

Silent mode
-----------
With AUDIOSPACE_SILENT=1 no device is opened at all: the player runs the
engine on a timer thread and discards the audio.  Callers therefore do not
need a second code path for tests or for muted verification runs, and the
timing behaviour of the game logic stays the same.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .device import (DeviceChoice, choose_output, is_silent,
                     open_output_stream)
from .defaults import default_hrir
from .engine import SpatialEngine
from .i18n import tr
from .geometry import Vec3
from .hrtf import HrirSet
from .motion import WOBBLE_PRESETS, HeadWobble
from .presets import (BUILTIN_PRESETS, RoomIr, load_room_ir,
                      preset_reverb_spec, render_room_ir)
from .reverb import ConvolutionReverb, Reverb, reverb_from_spec
from .sources import (BufferSource, PushSource, Source, StereoBufferSource,
                      load_flat, load_mono, resampled_by)


def _mark_exclusive(choice: DeviceChoice, exclusive: bool) -> DeviceChoice:
    """排他で開くときは、読み上げ（describe）も排他モードにする。

    実際に排他になるのは WASAPI だけ（device.open_output_stream と同じ条件）。
    """
    if exclusive and "wasapi" in choice.host_api.lower():
        return replace(choice, exclusive=True)
    return choice


class SpatialPlayer:
    """Real-time binaural playback with a simple call surface."""

    def __init__(self, engine: SpatialEngine,
                 choice: DeviceChoice | None = None,
                 exclusive: bool = False) -> None:
        self.engine = engine
        # どの作り方でも、読み上げ（describe）を実際のモードに合わせる
        self.choice = (_mark_exclusive(choice, exclusive)
                       if choice is not None else None)
        self.exclusive = exclusive
        self.silent = choice is None
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.underruns = 0
        self.callbacks = 0
        self.worst_ms = 0.0
        self._sum_ms = 0.0
        self._measured = 0
        # the first callback is always slow (page faults, lazy imports, FFT
        # plan setup), so it is tracked separately rather than skewing both
        # the mean and the worst-case figure
        self.first_ms = 0.0
        self._seen_first = False
        # a render() exception must not kill the stream: for a player who
        # cannot see the screen, audio dying silently is the worst failure
        # mode there is.  The callback swallows it, plays silence for that
        # block and records what happened for status().
        self.render_errors = 0
        self.last_error = ""
        # decoded-file cache: a game replays the same footstep hundreds of
        # times, and decoding plus resampling on every play() is disk I/O
        # in the middle of gameplay
        self._sound_cache: OrderedDict[
            tuple[str, float, bool], tuple[np.ndarray, float]] = OrderedDict()
        self.sound_cache_limit = 32
        # per-entry cap: repeatable effects fit far under this; music does
        # not, and must not accumulate (see _cached_sound)
        self.sound_cache_max_bytes = 16 * 1024 * 1024
        # device-loss watch: the callback stamps this, and check_device()
        # compares it against the wall clock (see `device_lost`)
        self._last_callback = 0.0
        self._device_lost = False

    # -- construction ----------------------------------------------------
    @classmethod
    def open(cls, block: int = 512, max_sources: int = 64,
             hrir: HrirSet | None = None,
             sofa_path: str | Path | None = None,
             reverb: Reverb | str | None = "small_room",
             device: int | str | None = None,
             prefer_host_api: str = "WASAPI",
             exclusive: bool = False,
             sample_rate: int | None = None,
             head_wobble: "HeadWobble | str | None" = None
             ) -> "SpatialPlayer":
        """Pick a device, build a matching engine, and start playback.

        The engine is always built at the *device's* sample rate, because
        WASAPI shared mode refuses any other rate (see device.py).
        """
        choice: DeviceChoice | None = None
        if is_silent():
            rate = int(sample_rate or 48000)
        else:
            choice = choose_output(prefer_host_api, device)
            rate = choice.sample_rate

        if hrir is None:
            if sofa_path:
                hrir = HrirSet.from_sofa(sofa_path, sample_rate=rate)
            else:
                # 既定は聴取テストで選んだ実測HRTF（水平面のみ、約200KB）。
                # アセットが無ければ合成に落ちるが、合成は前後を区別
                # できないので首振りと front_back_depth への依存が増える。
                hrir = default_hrir(rate)
        elif hrir.sample_rate != rate:
            hrir = hrir.resampled(rate)

        engine = SpatialEngine(hrir, block=block, max_sources=max_sources)
        if head_wobble is not None:
            engine.set_head_wobble(
                WOBBLE_PRESETS[head_wobble]
                if isinstance(head_wobble, str) else head_wobble)
        player = cls(engine, choice, exclusive)
        player.set_reverb(reverb)
        player.start()
        return player

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()          # allow start() after stop()
        if self.silent:
            self._thread = threading.Thread(target=self._silent_loop,
                                            name="audiospace-silent",
                                            daemon=True)
            self._thread.start()
            return
        self._stream = open_output_stream(self.choice, self._callback,
                                          block=self.engine.block,
                                          exclusive=self.exclusive)
        self._stream.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "SpatialPlayer":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _callback(self, outdata, frames, time_info, status) -> None:
        started = time.perf_counter()
        self._last_callback = started
        if status:
            self.underruns += 1
        try:
            if frames == self.engine.block:
                outdata[:] = self.engine.render()
            else:
                outdata[:] = 0.0
        except Exception as exc:  # noqa: BLE001 -- a dead stream is worse
            outdata[:] = 0.0
            self.render_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
        self.callbacks += 1
        elapsed = (time.perf_counter() - started) * 1000.0
        if not self._seen_first:
            self._seen_first = True
            self.first_ms = elapsed
        else:
            self._sum_ms += elapsed
            self._measured += 1
            self.worst_ms = max(self.worst_ms, elapsed)

    @property
    def mean_ms(self) -> float:
        """Mean callback time, excluding the cold first callback."""
        return self._sum_ms / max(1, self._measured)

    @property
    def budget_ms(self) -> float:
        return self.engine.block / self.engine.sample_rate * 1000.0

    # -- device loss -----------------------------------------------------
    def check_device(self) -> bool:
        """Poll for a dead output stream; True when the device is gone.

        Unplugging a USB interface, or Windows switching its default
        device, kills the stream without raising anywhere the application
        can see: the callback simply stops being called.  **Silence is the
        one failure a blind player cannot notice**, and a game can sit
        there for hours, so this has to be pollable and reportable rather
        than merely logged.

        Call it from the game loop (once a second is plenty).  Once it
        latches, `status()` says so and `reopen()` attempts recovery.
        """
        if self.silent or self._stream is None or self._device_lost:
            return self._device_lost
        try:
            active = bool(self._stream.active)
        except Exception:                     # noqa: BLE001 -- torn-down stream
            active = False
        stalled = False
        if self._last_callback:
            # PortAudio drives callbacks every `block` samples; ten blocks
            # of nothing (or one second, whichever is longer) is a dead
            # stream, not scheduling jitter
            limit = max(1.0, 10.0 * self.budget_ms / 1000.0)
            stalled = (time.perf_counter() - self._last_callback) > limit
        if not active or stalled:
            self._device_lost = True
            if stalled:
                self.last_error = tr("出力デバイスが応答しません",
                                     "the output device is not responding")
            else:
                self.last_error = tr("出力ストリームが停止しました",
                                     "the output stream has stopped")
        return self._device_lost

    @property
    def device_lost(self) -> bool:
        """Whether a lost device has been detected (see `check_device`)."""
        return self._device_lost

    def reopen(self, device: int | str | None = None,
               prefer_host_api: str = "WASAPI") -> bool:
        """Re-open the output after a device change.  True when playing again.

        Everything that was playing keeps playing: only the stream is
        rebuilt, the engine and its sources are untouched.  The one case
        this cannot absorb is a replacement device running at a different
        sample rate -- the engine builds its HRIR and convolver around one
        rate -- so that raises RuntimeError telling the caller to rebuild
        the player rather than pretending to recover.

        Note: PortAudio is re-initialised here to refresh its device list
        (it enumerates once, at startup, so a replugged device would
        otherwise be invisible and the old, dead index picked again).
        Any *other* sounddevice streams the application has open are
        invalidated by that; this player assumes it owns audio output.
        """
        if self.silent:
            return False
        if self._stream is not None:
            for step in (self._stream.stop, self._stream.close):
                try:
                    step()
                except Exception:             # noqa: BLE001 -- already dead
                    pass
            self._stream = None
        try:
            import sounddevice as sd
            sd._terminate()
            sd._initialize()
        except Exception:                     # noqa: BLE001 -- enumeration
            pass                              # below will tell the truth
        try:
            choice = _mark_exclusive(choose_output(prefer_host_api, device),
                                     self.exclusive)
        except Exception:
            # no usable device at all: stay in the lost state so
            # check_device()/status() keep saying so instead of going quiet
            self._device_lost = True
            self.last_error = tr("利用できる出力デバイスがありません",
                                 "no output device is available")
            raise
        if choice.sample_rate != self.engine.sample_rate:
            self._device_lost = True
            raise RuntimeError(tr(
                f"{choice.name} は {choice.sample_rate}Hz で、"
                f"エンジンの {self.engine.sample_rate}Hz と違います。"
                "この場合はプレイヤーを作り直してください",
                f"{choice.name} runs at {choice.sample_rate} Hz, but the "
                f"engine runs at {self.engine.sample_rate} Hz. "
                "Create the player again in this case"))
        self.choice = choice
        self._last_callback = 0.0
        self.start()
        self._device_lost = False
        self.last_error = ""
        return True

    def reset_stats(self) -> None:
        """Zero the running statistics for a fresh measuring interval.

        Long sessions need per-section dropout checks ("did the boss fight
        stutter?"), which cumulative counters cannot answer.  The cold
        first-callback figure is kept -- it describes startup, not the
        interval.  The engine's limiter counter is reset too.
        """
        self.underruns = 0
        self.callbacks = 0
        self.worst_ms = 0.0
        self._sum_ms = 0.0
        self._measured = 0
        self.render_errors = 0
        self.last_error = ""
        self.engine.limited_blocks = 0
        self.engine.nonfinite_blocks = 0

    def _silent_loop(self) -> None:
        period = self.engine.block / self.engine.sample_rate
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.engine.render()
            except Exception as exc:  # noqa: BLE001 -- keep the loop alive
                self.render_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            self.callbacks += 1
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    # -- convenience -----------------------------------------------------
    def play(self, sound: str | Path | np.ndarray | Source,
             position: Vec3 = (0.0, 0.0, 0.0), gain: float = 1.0,
             loop: bool = False, reverb_send: float = 0.25,
             relative: bool = False, spatialise: bool = True,
             pan: float = 0.0, loop_start: int = 0,
             loop_end: int = 0, rate: float = 1.0,
             lowpass: float = 0.0) -> int:
        """Start a sound and return its handle.

        `pan` places a flat (spatialise=False) source between the ears:
        -1 hard left, 0 both, +1 hard right.  A stereo asset played flat
        stays stereo in a single source; a spatialised one is downmixed,
        because a direction can only be given to a mono signal.

        `rate` changes playback speed, pitch and tempo together, the way a
        sampler does: 1.05 is five percent faster and slightly higher.
        Use it for footstep variation and for pitched cues instead of
        resampling the array yourself -- rated versions are cached too.

        `lowpass` starts the source muffled (see
        `SpatialEngine.set_source_lowpass`), for a sound that begins
        behind a wall.

        Files are decoded once and cached (see `clear_sound_cache`), and
        sources that have played out are reaped here, so calling play()
        for every footstep never exhausts the source slots.
        """
        # release finished one-shots first: without this, 64 played-out
        # sounds would still occupy their slots and the 65th play() of a
        # session would raise, despite every slot being reclaimable
        self.engine.reap_finished()
        stereo = not spatialise
        if isinstance(sound, (str, Path)):
            data, scale = self._cached_sound(sound, rate, stereo)
            # loop points are given in the file's own frames; the data has
            # been converted to the engine rate (scale) and then shortened
            # by the playback rate, so both factors apply
            scale = scale / float(rate)
            factory = StereoBufferSource if data.ndim == 2 else BufferSource
            src: Source = factory(
                data, loop=loop,
                loop_start=int(round(loop_start * scale)),
                loop_end=int(round(loop_end * scale)))
        elif isinstance(sound, np.ndarray):
            data = resampled_by(np.asarray(sound, dtype=np.float32), rate)
            factory = (StereoBufferSource
                       if stereo and data.ndim == 2 else BufferSource)
            src = factory(data, loop=loop,
                          loop_start=int(round(loop_start / float(rate))),
                          loop_end=int(round(loop_end / float(rate))))
        else:
            if abs(float(rate) - 1.0) > 1e-9:
                # silently ignoring it would be a "nothing changed" failure,
                # which is the kind a blind player cannot notice
                raise ValueError(
                    "rate applies to files and arrays only; a Source "
                    "object is played as-is -- resample its data instead")
            src = sound
        return self.engine.add_source(src, position, gain, reverb_send,
                                      relative, spatialise, pan, lowpass)

    def _cached_sound(self, path: str | Path, rate: float,
                      stereo: bool) -> tuple[np.ndarray, float]:
        """Decode via an LRU cache keyed by path, playback rate and layout.

        The rate belongs in the key because a pitched footstep is a
        different buffer, and resampling it on every step is exactly the
        per-play work the cache exists to avoid.  The cached array is
        shared between plays; the buffer sources never write into their
        data, so sharing is safe.

        An entry larger than `sound_cache_max_bytes` is returned but not
        kept.  The cache exists for sounds that repeat -- footsteps, UI
        blips -- and a three-minute stereo music track is ~66 MB that is
        read once per track change; keeping 32 of those would quietly
        grow to gigabytes.
        """
        resolved = Path(path).resolve()
        key = (str(resolved), round(float(rate), 6), bool(stereo))
        hit = self._sound_cache.get(key)
        if hit is None:
            # load_flat decides mono-vs-stereo from the single decode it
            # already does (a mono recording in a stereo container must
            # not spend a second channel of mixing on an identical copy)
            loader = load_flat if stereo else load_mono
            data, scale = loader(resolved, self.engine.sample_rate)
            hit = (resampled_by(data, rate), scale)
            if hit[0].nbytes <= self.sound_cache_max_bytes:
                self._sound_cache[key] = hit
                while len(self._sound_cache) > self.sound_cache_limit:
                    self._sound_cache.popitem(last=False)
        else:
            self._sound_cache.move_to_end(key)
        return hit

    def clear_sound_cache(self) -> None:
        """Drop all cached decodes (e.g. after replacing files on disk)."""
        self._sound_cache.clear()

    def play_ui(self, sound: str | Path | np.ndarray,
                gain: float = 1.0) -> int:
        """Play a non-diegetic cue: centred, dry, unaffected by the listener.

        Menu and status sounds must not move with the player's head, or the
        interface becomes harder to use than the game world.
        """
        return self.play(sound, gain=gain, reverb_send=0.0, spatialise=False)

    def stop_sound(self, handle: int, fade_ms: float = 0.0) -> None:
        """Stop one sound.  `fade_ms` > 0 fades it out instead of cutting;
        30 ms is enough to be click-free on a sustained loop."""
        self.engine.remove_source(handle, fade_ms)

    def pause_sound(self, handle: int) -> None:
        """Pause one sound in place; resume_sound() continues it."""
        self.engine.set_source_active(handle, False)

    def resume_sound(self, handle: int) -> None:
        self.engine.set_source_active(handle, True)

    def pause(self) -> None:
        """Silence everything without losing any state (pause menu)."""
        self.engine.set_paused(True)

    def resume(self) -> None:
        self.engine.set_paused(False)

    def set_sound_gain(self, handle: int, gain: float,
                       ramp_ms: float = 0.0) -> None:
        """Set a sound's gain, ramping over `ramp_ms` when asked.

        The ramp runs on the audio thread, so a music fade finishes on
        time even while the UI thread is blocked in a modal loop or a
        window drag.  Cross-fade by ramping one sound down and another up
        with the same duration.
        """
        self.engine.set_source_gain(handle, gain, ramp_ms)

    def set_sound_gain_db(self, handle: int, db: float,
                          ramp_ms: float = 0.0) -> None:
        """Set a sound's gain in decibels (0 = unchanged, -6 = half)."""
        self.engine.set_source_gain_db(handle, db, ramp_ms)

    def set_sound_lowpass(self, handle: int, amount: float,
                          ramp_ms: float = 60.0) -> None:
        """Muffle one sound: 0 = clear, 1 = maximum (occlusion)."""
        self.engine.set_source_lowpass(handle, amount, ramp_ms)

    def open_stream_source(self, capacity: int | None = None,
                           position: Vec3 = (0.0, 0.0, 0.0),
                           **kwargs) -> tuple[int, PushSource]:
        """Create a source the caller pushes samples into (low-level use)."""
        self.engine.reap_finished()
        cap = capacity or self.engine.sample_rate
        push = PushSource(cap)
        handle = self.engine.add_source(push, position, **kwargs)
        return handle, push

    def set_listener(self, position: Vec3 | None = None,
                     yaw_deg: float | None = None,
                     pitch_deg: float | None = None,
                     roll_deg: float | None = None) -> None:
        self.engine.set_listener(position, yaw_deg, pitch_deg, roll_deg)

    def set_emphasis(self, front_back: float | None = None,
                     elevation: float | None = None) -> None:
        """前後の強調と上下の強調の強さ（0〜1）を変える。None は変えない。

        - front_back: 後ろの音を小さく、こもらせる（`front_back_depth`、
          既定 0.35、0.9 で頭打ち）
        - elevation: 上の音ほど明るく、下の音ほど暗くする
          （`elevation_emphasis`、既定 0）

        どちらも鳴っている音にその場で効く。併用してよい（前後は水平で
        最も強く上下で弱まり、上下は水平で 0 になり上下で強まる）。
        """
        if front_back is not None:
            self.engine.front_back_depth = float(front_back)
        if elevation is not None:
            self.engine.elevation_emphasis = float(elevation)

    def set_reverb(self, reverb: Reverb | str | None) -> None:
        """Accept a Reverb, a built-in preset name, or None.

        A preset name uses the cheap FDN by default; call
        `use_room_ir(name)` to switch that preset to convolution.
        """
        if reverb is None or not isinstance(reverb, str):
            self.engine.set_reverb(reverb)
            return
        # 用意した部屋の値は preset_reverb_spec にまとめてある（自分で作る
        # 残響の元にもするので、ここと二重に持たない）
        self.engine.set_reverb(reverb_from_spec(preset_reverb_spec(reverb),
                                                self.engine.sample_rate))

    def use_room_ir(self, room: str | RoomIr | np.ndarray,
                    max_seconds: float = 3.0) -> None:
        """Switch to convolution reverb.

        A preset name loads the pre-baked IR from assets/reverb when one
        exists (resampling if it was baked at another rate), so runtime
        needs nothing beyond numpy.  Only when nothing is baked does it
        fall back to simulating the room live, which needs pyroomacoustics
        and takes seconds -- run tools/build_reverb_presets.py once to
        avoid that.
        """
        if isinstance(room, str):
            if room not in BUILTIN_PRESETS:
                raise ValueError(f"unknown reverb preset {room!r}; "
                                 f"known: {sorted(BUILTIN_PRESETS)}")
            baked = load_room_ir(room, self.engine.sample_rate)
            room = baked if baked is not None else render_room_ir(
                BUILTIN_PRESETS[room], self.engine.sample_rate, max_seconds)
        ir = room.ir if isinstance(room, RoomIr) else np.asarray(room)
        self.engine.set_reverb(ConvolutionReverb(ir, block=self.engine.block))

    def set_space(self, space, early: bool = True, late: bool = True,
                  transition_s: float = 0.3,
                  release_s: float | None = None) -> None:
        """材質つきの空間の響きにする（初期反射と、材質から決まる後期残響）。

        `space` は `Space`、空間の名前（`space_labels()` の名前。いまの聞き手の
        位置に合わせて置く）、または None（空間の響きを外す）。
        切り替えのつなぎ（transition_s, release_s）を含め、詳しくは
        `SpatialEngine.set_space`。
        """
        if isinstance(space, str):
            from .acoustics import space_preset
            space = space_preset(space, self.engine.listener.position)
        self.engine.set_space(space, early=early, late=late,
                              transition_s=transition_s, release_s=release_s)

    def set_spaces(self, spaces, early: bool = True, late: bool = True,
                   transition_s: float = 0.3,
                   release_s: float | None = None) -> None:
        """場所ごとの空間を登録する（聞き手のいる場所で自動で切り替わる）。

        `spaces` は world 座標に置いた `Space` のリスト。詳しくは
        `SpatialEngine.set_spaces`。
        """
        self.engine.set_spaces(list(spaces), early=early, late=late,
                               transition_s=transition_s,
                               release_s=release_s)

    # -- diagnostics -----------------------------------------------------
    def status(self) -> str:
        """Plain-text status line, safe to read aloud."""
        # status() is what gets read aloud when something has gone wrong,
        # so it must not raise on a half-torn-down player (no device)
        if self.silent or self.choice is None:
            where = tr("無音モード（デバイスを開いていません）",
                       "Silent mode (no device is open)")
        else:
            where = self.choice.describe()
        sep = tr("／", " / ")
        text = sep.join([
            where,
            tr(f"音源{self.engine.active_sources}個",
               f"{self.engine.active_sources} sources"),
            tr(f"コールバック{self.callbacks}回",
               f"{self.callbacks} callbacks"),
            tr(f"途切れ{self.underruns}回", f"{self.underruns} underruns"),
            tr(f"平均処理時間{self.mean_ms:.2f}ミリ秒",
               f"mean {self.mean_ms:.2f} ms"),
            tr(f"最悪処理時間{self.worst_ms:.2f}ミリ秒",
               f"worst {self.worst_ms:.2f} ms"),
            tr(f"予算{self.budget_ms:.2f}ミリ秒",
               f"budget {self.budget_ms:.2f} ms")])
        if self.engine.limited_blocks:
            text += sep + tr(f"音量制限{self.engine.limited_blocks}ブロック",
                             f"limiter {self.engine.limited_blocks} blocks")
        if self.engine.nonfinite_blocks:
            text += sep + tr(
                f"計算の異常で無音{self.engine.nonfinite_blocks}ブロック",
                f"silenced after bad numbers "
                f"{self.engine.nonfinite_blocks} blocks")
        if self.engine.paused:
            text += sep + tr("一時停止中", "paused")
        if self._device_lost:
            # 先頭に出す。ほかの数字より先に伝わるべき事実なので
            text = tr(f"【出力デバイスが失われました（{self.last_error}）。"
                      f"reopen() で再接続できます】",
                      f"[The output device was lost ({self.last_error}). "
                      f"reopen() can reconnect]") + sep + text
        elif self.render_errors:
            text += sep + tr(f"レンダリングエラー{self.render_errors}回"
                             f"（{self.last_error}）",
                             f"{self.render_errors} rendering errors "
                             f"({self.last_error})")
        return text
