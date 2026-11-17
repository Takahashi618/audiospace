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
import weakref
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .device import (DeviceChoice, choose_output, is_silent,
                     open_output_stream)
from .defaults import default_hrir
from .engine import SoundEnd, SpatialEngine
from .occlusion import Occlusion, Opening
from .i18n import tr
from .geometry import Vec3
from .hrtf import HrirSet
from .motion import WOBBLE_PRESETS, HeadWobble
from .presets import (BUILTIN_PRESETS, RoomIr, load_room_ir,
                      preset_reverb_spec, render_room_ir)
from .reverb import ConvolutionReverb, Reverb, reverb_from_spec
from .sources import (BufferSource, FileStreamSource, PushSource, Source,
                      StereoBufferSource, audio_file_info,
                      can_resample_streaming, load_flat, load_mono,
                      resampled_by)


class DeviceRateError(RuntimeError):
    """つなぎ直そうとした出力機器のサンプルレートが、エンジンと違う。

    `SpatialPlayer.reopen` が送出する。エンジンは HRIR と畳み込みを1つの
    レートで組むので、この場合はプレイヤーを作り直す必要がある。
    RuntimeError の一種なので、以前の書き方の except RuntimeError でも捕まる。
    """


#: 出力デバイスの見張りが知らせること（`DeviceEvent.kind`）
DEVICE_LOST = "lost"              # 失われ、つなぎ直せなかった（試し続ける）
DEVICE_RESTORED = "restored"      # つなぎ直した（鳴っていた音はそのまま続く）
DEVICE_RATE_CHANGED = "rate_changed"  # 替わった機器のレートが違う（作り直しが要る）


@dataclass(frozen=True)
class DeviceEvent:
    """出力デバイスの見張り（`poll_device` / `watch_device`）が知らせること。

    kind     DEVICE_LOST、DEVICE_RESTORED、DEVICE_RATE_CHANGED のどれか
    message  読み上げ用の文（いまの言語）。そのまま読めば伝わる
    detail   理由やつないだ機器の説明（例外の文、`DeviceChoice.describe()`）

    同じ知らせは続けて出さない（つなぎ直しに失敗し続けても、DEVICE_LOST は
    1回だけ）。状態が変わったときにだけ出る。
    """

    kind: str
    message: str
    detail: str = ""


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
        # コールバックがブロックと違うフレーム数を求めた回数と、そのフレーム数。
        # ブロックの大きさを指定して開くので通常は 0。起きたときは無音を出す
        # ので、気づけるよう status() に出す
        self.block_mismatches = 0
        self.last_mismatch_frames = 0
        # decoded-file cache: a game replays the same footstep hundreds of
        # times, and decoding plus resampling on every play() is disk I/O
        # in the middle of gameplay
        self._sound_cache: OrderedDict[
            tuple[str, float, bool], tuple[np.ndarray, float]] = OrderedDict()
        self.sound_cache_limit = 32
        # per-entry cap: repeatable effects fit far under this; music does
        # not, and must not accumulate (see _cached_sound)
        self.sound_cache_max_bytes = 16 * 1024 * 1024
        # 長いファイルを少しずつ読む音源（play の stream）。先読みの秒数と、
        # 状態の読み上げのために、作った音源を弱い参照で覚えておく
        self.stream_buffer_seconds = 2.0
        self._streams: weakref.WeakSet = weakref.WeakSet()
        # device-loss watch: the callback stamps this, and check_device()
        # compares it against the wall clock (see `device_lost`)
        self._last_callback = 0.0
        self._device_lost = False
        # ストリームを開く・閉じる・つなぎ直す処理を1つずつにする。見張りの
        # スレッド（watch_device）と呼び出し側が同時に触っても壊れないように
        self._lifecycle = threading.RLock()
        #: つなぎ直しに失敗した後、次に試すまでの秒数（poll_device）
        self.device_retry_s = 3.0
        self._retry_at = 0.0             # 次につなぎ直しを試す時刻
        self._reported: str | None = None   # 最後に知らせた kind
        self._watch_thread: threading.Thread | None = None
        self._watch_stop: threading.Event | None = None
        # 鳴り終わりの知らせ（play の on_end）。ハンドル -> 呼ぶ関数。
        # play と poll_ends が同時に触るので鍵をかける。play は音を足して
        # 関数を覚えるまでを1つにする（間に知らせを取り出されると、
        # 呼ぶ関数が見つからず知らせが消えるため）
        self._end_callbacks: dict[int, Callable[[SoundEnd], None]] = {}
        self._ends_lock = threading.RLock()
        self._ends_thread: threading.Thread | None = None
        self._ends_stop: threading.Event | None = None
        #: on_end の関数が例外を出した回数と、最後の理由（status() に出す）
        self.callback_errors = 0
        self.last_callback_error = ""

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
             head_wobble: "HeadWobble | str | None" = None,
             full_sphere: bool = False
             ) -> "SpatialPlayer":
        """Pick a device, build a matching engine, and start playback.

        The engine is always built at the *device's* sample rate, because
        WASAPI shared mode refuses any other rate (see device.py).

        `full_sphere=True` uses the full-sphere default HRTF (sounds above
        and below are rendered too) instead of the horizontal one.  It
        cannot be combined with `hrir` or `sofa_path` (ValueError): those
        already choose the HRTF.
        """
        if full_sphere and (hrir is not None or sofa_path):
            raise ValueError(
                "full_sphere chooses the default HRTF; it cannot be combined "
                "with hrir or sofa_path")
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
                hrir = default_hrir(rate, full_sphere=full_sphere)
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
        with self._lifecycle:
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

    def close(self) -> None:
        """Finish: stop playback, release every sound and close its file.

        `stop()` alone keeps the sounds so that `start()` can go on; a sound
        read a little at a time (`stream=True`) then keeps its file open.
        Call this (or leave a `with` block) when the player is no longer
        needed.  After it, the files can be deleted or replaced.
        """
        self.stop()
        for source in self.engine.remove_all_sources():
            if isinstance(source, FileStreamSource):
                source.close(wait_s=1.0)

    def stop(self) -> None:
        """Stop playback.  Also ends `watch_device` and `watch_ends`.

        The sounds stay (start() resumes them).  To finish for good and
        release the files of streamed sounds, use close().
        """
        self.unwatch_device()
        self.unwatch_ends()
        with self._lifecycle:
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
        self.close()

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
                self.block_mismatches += 1
                self.last_mismatch_frames = int(frames)
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
        rate -- so that raises DeviceRateError (a RuntimeError) telling the
        caller to rebuild the player rather than pretending to recover.

        Note: PortAudio is re-initialised here to refresh its device list
        (it enumerates once, at startup, so a replugged device would
        otherwise be invisible and the old, dead index picked again).
        Any *other* sounddevice streams the application has open are
        invalidated by that; this player assumes it owns audio output.
        """
        if self.silent:
            return False
        with self._lifecycle:
            return self._reopen(device, prefer_host_api)

    def _reopen(self, device, prefer_host_api: str) -> bool:
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
            self.last_error = tr(
                f"{choice.name} は {choice.sample_rate}Hz で、"
                f"エンジンの {self.engine.sample_rate}Hz と違います。"
                "この場合はプレイヤーを作り直してください",
                f"{choice.name} runs at {choice.sample_rate} Hz, but the "
                f"engine runs at {self.engine.sample_rate} Hz. "
                "Create the player again in this case")
            raise DeviceRateError(self.last_error)
        self.choice = choice
        self._last_callback = 0.0
        self.start()
        self._device_lost = False
        self.last_error = ""
        return True

    # -- 出力デバイスの見張り ---------------------------------------------
    def poll_device(self) -> DeviceEvent | None:
        """出力デバイスを調べ、失われていたらつなぎ直す。知らせることがあれば返す。

        ゲームのループや画面のタイマーから、1秒に1回くらい呼ぶ。
        `check_device` で失われたと分かったら、すぐ `reopen` を試す。
        失敗したら `device_retry_s` 秒ごとに試し続ける（ヘッドホンをつなぎ
        直せば、次に試したときに戻る）。返すのは状態が変わったときだけ:

        - DEVICE_RESTORED: つなぎ直した。鳴っていた音はそのまま続く
        - DEVICE_LOST: 失われ、まだつなぎ直せない（失敗し続けても1回だけ）
        - DEVICE_RATE_CHANGED: 替わった機器のレートが違い、このプレイヤーでは
          鳴らせない。プレイヤーを作り直す（`SpatialPlayer.open` をもう一度）。
          元の機器が戻れば、試し続けたつなぎ直しで DEVICE_RESTORED になる

        何も無ければ None。無音モードでは常に None。
        `reopen` と同じく PortAudio を初期化し直すので、ほかに sounddevice の
        ストリームを開いているアプリでは、それらが無効になる。
        """
        if self.silent:
            return None
        with self._lifecycle:
            if not self.check_device():
                self._reported = None
                self._retry_at = 0.0
                return None
            now = time.monotonic()
            if now < self._retry_at:
                return None
            self._retry_at = now + float(self.device_retry_s)
            try:
                self.reopen()
            except DeviceRateError as exc:
                return self._report(DEVICE_RATE_CHANGED, str(exc), tr(
                    f"出力デバイスが替わりましたが、サンプルレートが違うため"
                    f"鳴らせません（{exc}）",
                    f"The output device changed, but it runs at another "
                    f"sample rate, so it cannot play ({exc})"))
            except Exception as exc:          # noqa: BLE001
                detail = str(exc) or type(exc).__name__
                return self._report(DEVICE_LOST, detail, tr(
                    f"出力デバイスが失われました（{detail}）。"
                    "つなぎ直しを試し続けます",
                    f"The output device was lost ({detail}). "
                    "It keeps trying to reconnect"))
            self._reported = None
            self._retry_at = 0.0
            where = self.choice.describe() if self.choice is not None else ""
            return DeviceEvent(DEVICE_RESTORED, tr(
                f"出力デバイスをつなぎ直しました。{where}",
                f"The output device was reconnected. {where}").strip(),
                where)

    def _report(self, kind: str, detail: str,
                message: str) -> DeviceEvent | None:
        """同じ知らせを続けて出さない。"""
        if self._reported == kind:
            return None
        self._reported = kind
        return DeviceEvent(kind, message, detail)

    def watch_device(self, on_event: Callable[[DeviceEvent], None] | None
                     = None, interval_s: float = 1.0) -> None:
        """別のスレッドで出力デバイスを見張り、失われたら自動でつなぎ直す。

        `interval_s` 秒ごとに `poll_device` を呼び、知らせがあれば
        `on_event(event)` を呼ぶ。**`on_event` は見張りのスレッドで呼ばれる**。
        画面の部品を触るときは、画面のスレッドへ渡す（wxPython なら
        `wx.CallAfter`）。on_event が例外を出しても見張りは止まらない。
        渡さなくても、つなぎ直しは行い、`status()` が状態を伝える。

        画面のあるアプリは、別のスレッドを立てずに、画面のタイマーから
        `poll_device` を呼ぶほうが扱いやすい。`stop` か `unwatch_device` で
        終わる。もう一度呼ぶと、前の見張りを止めてから始め直す。無音モード
        では何もしない（見張るデバイスが無い）。
        """
        self.unwatch_device()
        if self.silent:
            return
        stop = threading.Event()
        period = max(0.05, float(interval_s))

        def loop() -> None:
            while not stop.wait(period):
                try:
                    event = self.poll_device()
                except Exception as exc:      # noqa: BLE001
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    continue
                if event is not None and on_event is not None:
                    try:
                        on_event(event)
                    except Exception:         # noqa: BLE001
                        pass

        self._watch_stop = stop
        self._watch_thread = threading.Thread(
            target=loop, name="audiospace-device-watch", daemon=True)
        self._watch_thread.start()

    def unwatch_device(self) -> None:
        """`watch_device` の見張りを止める（見張っていなければ何もしない）。"""
        stop, thread = self._watch_stop, self._watch_thread
        self._watch_stop = self._watch_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    @property
    def watching_device(self) -> bool:
        """`watch_device` で見張っているか。"""
        return self._watch_thread is not None and self._watch_thread.is_alive()

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
        self.block_mismatches = 0
        self.last_mismatch_frames = 0
        self.engine.limited_blocks = 0
        self.engine.nonfinite_blocks = 0
        self.engine.source_errors = 0
        self.engine.last_source_error = ""

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
             lowpass: float = 0.0, delay_s: float = 0.0,
             group: str | None = None,
             on_end: Callable[[SoundEnd], None] | None = None,
             occlude: bool = True,
             stream: bool | None = None) -> int:
        """Start a sound and return its handle.

        `stream` reads a file a little at a time on a thread instead of
        decoding it whole (FileStreamSource): None (the default) streams
        files too large for the sound cache (`sound_cache_max_bytes`, 16 MB:
        about 44 s of 48 kHz stereo), which were decoded whole on every
        play before; True always streams; False never does.  Streaming
        needs `rate` 1.0, and samplerate when the file's rate differs.

        `occlude=False` keeps the automatic occlusion (set_auto_occlusion)
        off this sound, e.g. for a guide beacon that must stay clear.

        `on_end(event)` is called once when the sound ends: it played out
        (event.reason SOUND_FINISHED), was stopped (SOUND_STOPPED) or was
        taken out after an error (SOUND_ERROR).  It is called by
        poll_ends(), on the thread that calls it, or by the thread that
        watch_ends() starts -- never on the audio thread.

        `group` puts it in a named group ("sfx", "music", "voice" ...) so
        that set_group_gain, pause_group, stop_group and set_ducking act on
        it with the rest of the group.

        `delay_s` starts it that many seconds later, to the sample: a
        footstep pattern, a rhythm cue or an echo without a timer in the
        game.  Several sounds that must keep their offsets exactly go in
        one `with player.together():` block (see `together`).  Stopping a
        sound that has not started yet removes it unheard.

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
        if on_end is not None and not callable(on_end):
            raise ValueError(f"on_end must be callable, got {on_end!r}")
        # release finished one-shots first: without this, 64 played-out
        # sounds would still occupy their slots and the 65th play() of a
        # session would raise, despite every slot being reclaimable
        self.engine.reap_finished()
        stereo = not spatialise
        if stream and not isinstance(sound, (str, Path)):
            # 黙って丸ごと鳴らすと「少しずつ読んだつもり」に気づけない
            raise ValueError(tr(
                "stream=True はファイル（パス）にだけ使えます。配列や音源は"
                "そのまま鳴らします",
                "stream=True works only with a file (a path); an array or "
                "a source object is played as it is"))
        # パスは1回だけ解決する（Windows では1回 0.1ms 近くかかり、足音の
        # ように何百回も鳴らす音では、それが play の時間の大半になる）
        resolved = (Path(sound).resolve() if isinstance(sound, (str, Path))
                    else None)
        streamed = (self._stream_channels(resolved, rate, stereo, stream)
                    if resolved is not None else None)
        if streamed is not None:
            src: Source = FileStreamSource(
                sound, self.engine.sample_rate, channels=streamed,
                loop=loop, loop_start=loop_start, loop_end=loop_end,
                buffer_seconds=self.stream_buffer_seconds)
            self._streams.add(src)
        elif isinstance(sound, (str, Path)):
            data, scale = self._cached_sound(resolved, rate, stereo)
            # loop points are given in the file's own frames; the data has
            # been converted to the engine rate (scale) and then shortened
            # by the playback rate, so both factors apply
            scale = scale / float(rate)
            factory = StereoBufferSource if data.ndim == 2 else BufferSource
            src = factory(
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
        if on_end is None:
            return self.engine.add_source(src, position, gain, reverb_send,
                                          relative, spatialise, pan, lowpass,
                                          delay_s=delay_s, group=group,
                                          occlude=occlude)
        with self._ends_lock:
            handle = self.engine.add_source(
                src, position, gain, reverb_send, relative, spatialise, pan,
                lowpass, delay_s=delay_s, group=group, notify_end=True,
                occlude=occlude)
            self._end_callbacks[handle] = on_end
        return handle

    def together(self):
        """Keep the offsets of several sounds exact.

            with player.together():
                player.play("step.wav", position=feet)
                player.play("step.wav", position=floor, gain=0.6,
                            delay_s=0.0094)

        Every sound started inside waits until the with-block ends and
        then counts its `delay_s` from the same moment.  Separate play()
        calls can otherwise straddle an audio block, and the later sound
        starts one block (about 11 ms) late.  If the block raises, its
        sounds are dropped unplayed.  Use it from the thread that plays
        the sounds; files are decoded as usual inside it.
        """
        return self.engine.together()

    def _stream_channels(self, path, rate: float, stereo: bool,
                         stream: bool | None) -> int | None:
        """少しずつ読むなら鳴らすチャンネル数（1 か 2）、読まないなら None。

        既定（stream=None）では、サウンドキャッシュに入っている音は調べない
        （ファイルを開かない）。キャッシュに入るのは少しずつ読まない大きさの
        音だけなので、入っていれば答えは決まっている。足音のように何百回も
        鳴らす音で、毎回ファイルを開かないため。
        """
        if stream is False:
            return None
        if stream is None and self._cache_key(path, rate, stereo) in \
                self._sound_cache:
            return None
        if abs(float(rate) - 1.0) > 1e-9:
            if stream:
                raise ValueError(tr(
                    "stream=True と rate（1 以外）は一緒に使えません。"
                    "少しずつ読む音は、そのままの速さで鳴らします",
                    "stream=True cannot be used with a rate other than 1; "
                    "a streamed sound plays at its own speed"))
            return None
        try:
            frames, file_rate, file_channels = audio_file_info(path)
        except Exception:
            if stream:
                raise
            return None             # 丸ごと読む道へ（そこで理由が分かる）
        channels = 2 if (stereo and file_channels >= 2) else 1
        if file_rate != self.engine.sample_rate and not can_resample_streaming():
            if stream:
                raise ValueError(tr(
                    f"{Path(path).name} は {file_rate}Hz で、少しずつ読むには"
                    "レートの変換に samplerate が要ります",
                    f"{Path(path).name} is {file_rate} Hz; streaming it "
                    "needs samplerate for the rate conversion"))
            return None
        if stream:
            return channels
        # 既定: キャッシュに入らない大きさ（いままで毎回丸ごと読んでいた音）
        out_frames = frames * self.engine.sample_rate / max(1, file_rate)
        size = out_frames * channels * 4
        return channels if size > self.sound_cache_max_bytes else None

    @staticmethod
    def _cache_key(resolved: Path, rate: float, stereo: bool) -> tuple:
        """サウンドキャッシュの鍵（解決したパス・再生レート・平置きか）。"""
        return (str(resolved), round(float(rate), 6), bool(stereo))

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
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        key = self._cache_key(resolved, rate, stereo)
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
                gain: float = 1.0, delay_s: float = 0.0,
                group: str | None = None,
                on_end: Callable[[SoundEnd], None] | None = None) -> int:
        """Play a non-diegetic cue: centred, dry, unaffected by the listener.

        Menu and status sounds must not move with the player's head, or the
        interface becomes harder to use than the game world.  `delay_s`,
        `group` and `on_end` work as in play(), e.g. for a two-note cue.
        """
        return self.play(sound, gain=gain, reverb_send=0.0, spatialise=False,
                         delay_s=delay_s, group=group, on_end=on_end)

    # -- end notices -----------------------------------------------------
    def poll_ends(self) -> list[SoundEnd]:
        """Call the on_end functions of the sounds that have ended.

        Call it from the game's own loop or timer (about every 20 to 50
        ms); the functions run on that thread, so a GUI can touch its
        widgets from them.  Returns the events in the order the sounds
        ended.  A function that raises does not stop the others; the count
        and the last reason are in status().  Played-out sounds are reaped
        here too, so their handles are released.
        """
        return self._deliver_ends(None)

    def _deliver_ends(self, dispatch) -> list[SoundEnd]:
        self.engine.reap_finished()
        with self._ends_lock:
            events = self.engine.take_ended()
            calls = [(self._end_callbacks.pop(e.handle, None), e)
                     for e in events]
        for fn, event in calls:
            if fn is None:
                continue
            try:
                if dispatch is None:
                    fn(event)
                else:
                    dispatch(fn, event)
            except Exception as exc:          # noqa: BLE001
                self.callback_errors += 1
                self.last_callback_error = f"{type(exc).__name__}: {exc}"
        return events

    def watch_ends(self, interval_s: float = 0.02,
                   dispatch: Callable | None = None) -> None:
        """Deliver on_end on a thread of the player's own.

        Every `interval_s` seconds it does what poll_ends() does.  **The
        functions run on that thread.**  To have them run on a GUI's
        thread, pass that toolkit's hand-over function as `dispatch`; it
        is called as dispatch(fn, event).  With wxPython:

            player.watch_ends(dispatch=wx.CallAfter)

        `stop` or `unwatch_ends` ends it; calling it again restarts it.
        Works in silent mode too.
        """
        self.unwatch_ends()
        stop = threading.Event()
        period = max(0.005, float(interval_s))

        def loop() -> None:
            while not stop.wait(period):
                try:
                    self._deliver_ends(dispatch)
                except Exception as exc:      # noqa: BLE001
                    self.callback_errors += 1
                    self.last_callback_error = f"{type(exc).__name__}: {exc}"

        self._ends_stop = stop
        self._ends_thread = threading.Thread(
            target=loop, name="audiospace-end-notices", daemon=True)
        self._ends_thread.start()

    def unwatch_ends(self) -> None:
        """End watch_ends (does nothing if it is not running)."""
        stop, thread = self._ends_stop, self._ends_thread
        self._ends_stop = self._ends_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    @property
    def watching_ends(self) -> bool:
        """Whether watch_ends is delivering on_end."""
        return self._ends_thread is not None and self._ends_thread.is_alive()

    # -- groups ----------------------------------------------------------
    def set_group_gain(self, name: str, gain: float,
                       ramp_ms: float = 0.0) -> None:
        """Set the volume of a group, e.g. a settings screen's effects
        volume.  It multiplies each sound's own gain; see
        SpatialEngine.set_group_gain."""
        self.engine.set_group_gain(name, gain, ramp_ms)

    def set_group_gain_db(self, name: str, db: float,
                          ramp_ms: float = 0.0) -> None:
        """Set a group's volume in decibels (0 = unchanged, -6 = half)."""
        self.engine.set_group_gain_db(name, db, ramp_ms)

    def group_gain(self, name: str) -> float:
        """The volume set for a group (1.0 if never set)."""
        return self.engine.group_gain(name)

    def pause_group(self, name: str) -> None:
        """Pause every sound in a group in place (e.g. music in a menu)."""
        self.engine.set_group_paused(name, True)

    def resume_group(self, name: str) -> None:
        self.engine.set_group_paused(name, False)

    def stop_group(self, name: str, fade_ms: float = 0.0) -> list[int]:
        """Stop every sound in a group, fading over `fade_ms`."""
        return self.engine.stop_group(name, fade_ms)

    def group_sounds(self, name: str) -> list[int]:
        """Handles of the sounds now in a group."""
        return self.engine.sources_in_group(name)

    def set_sound_group(self, handle: int, name: str | None) -> None:
        """Move one sound to another group (None = no group)."""
        self.engine.set_source_group(handle, name)

    def set_ducking(self, name: str, by: str | tuple[str, ...] = (),
                    db: float = -12.0, attack_ms: float = 80.0,
                    release_ms: float = 400.0) -> None:
        """Lower group `name` by `db` while a sound of group `by` plays.

            player.set_ducking("music", by="voice", db=-12)

        The volume set with set_group_gain is kept; ducking is a factor
        on top of it.  See SpatialEngine.set_ducking.
        """
        self.engine.set_ducking(name, by, db, attack_ms, release_ms)

    def clear_ducking(self, name: str) -> None:
        self.engine.clear_ducking(name)

    def hold_ducking(self, name: str, on: bool = True) -> None:
        """Duck a group by hand, e.g. while a screen reader speaks."""
        self.engine.hold_ducking(name, on)

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

    def set_auto_occlusion(self, enabled: bool = True, *,
                           strength: float = 1.0,
                           openings: list[Opening] | tuple = (),
                           materials: dict | None = None,
                           portals: bool = True) -> None:
        """Muffle and lower sounds heard through walls, automatically.

        Works with the spaces given to set_spaces.  `openings` are the
        doors, windows and stairwells cut into walls (`Opening`).  See
        SpatialEngine.set_auto_occlusion.
        """
        self.engine.set_auto_occlusion(enabled, strength=strength,
                                       openings=openings,
                                       materials=materials, portals=portals)

    def set_opening(self, name: str, material: str | None = None) -> None:
        """Open (None) or close (a material such as "wooden_door") a door."""
        self.engine.set_opening(name, material)

    def sound_occlusion(self, handle: int) -> Occlusion:
        """How one sound is occluded: loss_db, lowpass, walls, via."""
        return self.engine.source_occlusion(handle)

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
        if self.engine.source_errors:
            text += sep + tr(
                f"音源の異常で外した音{self.engine.source_errors}個"
                f"（{self.engine.last_source_error}）",
                f"{self.engine.source_errors} sources removed after errors "
                f"({self.engine.last_source_error})")
        if self.block_mismatches:
            text += sep + tr(
                f"ブロックの大きさの食い違いで無音{self.block_mismatches}回"
                f"（{self.last_mismatch_frames}フレームを求められた。"
                f"ブロックは{self.engine.block}）",
                f"silent {self.block_mismatches} times because the device "
                f"asked for {self.last_mismatch_frames} frames (the block "
                f"is {self.engine.block})")
        if self.callback_errors:
            text += sep + tr(
                f"鳴り終わりの関数の誤り{self.callback_errors}回"
                f"（{self.last_callback_error}）",
                f"{self.callback_errors} errors in on_end functions "
                f"({self.last_callback_error})")
        if not self.watching_ends:
            # on_end を渡したのに poll_ends も watch_ends も呼んでいないと、
            # 関数は呼ばれない。気づけるように、2秒より古い知らせを数える
            now = self.engine.elapsed_s
            stale = sum(1 for e in self.engine.pending_ends()
                        if now - e.at_s > 2.0)
            if stale:
                text += sep + tr(
                    f"届けていない鳴り終わりの知らせ{stale}件"
                    "（poll_ends か watch_ends を呼びます）",
                    f"{stale} end notices not delivered "
                    "(call poll_ends or watch_ends)")
        if self.engine.auto_occlusion and not self.engine.occlusion_spaces:
            # 空間が無いと、自動の遮蔽を入れても何も変わらない。気づけるように
            text += sep + tr(
                "自動の遮蔽は入っていますが、空間がありません（set_spaces か "
                "set_space で空間を決めます）",
                "automatic occlusion is on, but there is no space (set one "
                "with set_spaces or set_space)")
        late = sum(s.underruns for s in list(self._streams))
        if late:
            text += sep + tr(
                f"ファイルの読み込みが間に合わず無音{late}ブロック",
                f"silent {late} blocks because a file was not read in time")
        if self.engine.paused:
            text += sep + tr("一時停止中", "paused")
        if self._device_lost:
            # 先頭に出す。ほかの数字より先に伝わるべき事実なので
            if self.watching_device:
                how = tr("自動でつなぎ直しを試しています",
                         "trying to reconnect automatically")
            else:
                how = tr("reopen() で再接続できます", "reopen() can reconnect")
            text = tr(f"【出力デバイスが失われました（{self.last_error}）。"
                      f"{how}】",
                      f"[The output device was lost ({self.last_error}). "
                      f"{how}]") + sep + text
        elif self.render_errors:
            text += sep + tr(f"レンダリングエラー{self.render_errors}回"
                             f"（{self.last_error}）",
                             f"{self.render_errors} rendering errors "
                             f"({self.last_error})")
        return text
