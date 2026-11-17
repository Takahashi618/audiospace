"""Sound sources.

A source is a sample generator; spatialisation happens in the engine,
never here.  Four shapes cover the useful cases:

    BufferSource        a decoded mono array, optionally looping
    StereoBufferSource  a stereo asset played flat, as one source
    PushSource          a ring buffer the application writes into
    CallbackSource      a synthesiser function called per block
    FileStreamSource    a long file read a little at a time on a thread

Sources are mono by default -- `read(n)` returns `(n,)` and `channels` is
1 -- because a direction can only be given to a mono signal.  A source may
declare `channels = 2` and return `(n, 2)`, which the engine accepts only
on the flat (spatialise=False) path: music and ambience are already
stereo, and folding them to the middle or splitting them across two mixer
slots are both worse than simply playing them.

Sources are pulled by the engine, so a low-level caller that wants to feed
blocks itself uses `PushSource.write()` and never has to match the audio
callback's timing exactly.
"""
from __future__ import annotations

import threading
import weakref
from fractions import Fraction
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .i18n import tr


class DecodeError(ValueError):
    """音声ファイルを読めない（形式が違う、壊れている、読む道具が無い）。"""


class Source(Protocol):
    """Sample generator, mono unless it declares otherwise."""

    #: 1 for mono `(n,)` reads, 2 for stereo `(n, 2)` reads
    channels: int

    def read(self, n: int) -> np.ndarray:
        """Return exactly `n` float32 frames, zero-padding if exhausted."""
        ...

    @property
    def finished(self) -> bool:
        """True once the source will only ever produce silence again."""
        ...

    # 任意: `sounding` を持つ音源は、それで「いま鳴っているか」を伝える
    # （ダッキングのきっかけに使う）。持たない音源は、鳴り終わっていなければ
    # 鳴っているとみなす。


class _LoopingBuffer:
    """Cursor and loop-point handling shared by the buffer sources.

    `_data` is (frames,) for mono or (frames, channels) for multichannel;
    the read logic is identical either way because the cursor indexes the
    first axis.
    """

    channels = 1

    def __init__(self, data: np.ndarray, loop: bool, start: int,
                 loop_start: int, loop_end: int) -> None:
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self.loop = bool(loop)
        self._pos = int(start) % max(1, self._data.shape[0])
        self._done = self._data.shape[0] == 0
        # Loop points, in this buffer's samples: on wrap the cursor goes
        # back to loop_start, and loop_end (0 = the end) is where the
        # wrap happens -- a tracker-style intro-then-loop.
        frames = self._data.shape[0]
        self._end = min(int(loop_end), frames) if loop_end else frames
        if self._end <= 0:
            self._end = frames
        self.loop_start = max(0, min(int(loop_start), max(0, self._end - 1)))

    @property
    def frames(self) -> int:
        return int(self._data.shape[0])

    @property
    def position(self) -> int:
        return self._pos

    @property
    def finished(self) -> bool:
        return self._done

    def rewind(self) -> None:
        self._pos = 0
        self._done = self.frames == 0

    def read(self, n: int) -> np.ndarray:
        out = np.zeros((n,) + self._data.shape[1:], dtype=np.float32)
        if self._done:
            return out
        filled = 0
        while filled < n:
            take = min(n - filled, self._end - self._pos)
            if take > 0:
                out[filled:filled + take] = \
                    self._data[self._pos:self._pos + take]
                filled += take
                self._pos += take
            if self._pos >= self._end:
                if self.loop:
                    self._pos = self.loop_start
                else:
                    self._done = True
                    break
        return out


class BufferSource(_LoopingBuffer):
    """Play a decoded mono array once, or loop it forever."""

    channels = 1

    def __init__(self, samples: np.ndarray, loop: bool = False,
                 start: int = 0, loop_start: int = 0,
                 loop_end: int = 0) -> None:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 2:
            # Decoders produce (frames, channels); average the channels.
            # Guessing at channels-first data would silently destroy it (a
            # 5.1 clip used to fold into six samples), so a shape that
            # cannot be frames-first is rejected instead of guessed at.
            if data.shape[0] < data.shape[1]:
                raise ValueError(
                    f"expected (frames, channels), got {data.shape}; "
                    "pass samples.T for channels-first data")
            data = data.mean(axis=1)
        elif data.ndim > 2:
            raise ValueError(f"samples must be 1-D or 2-D, got {data.shape}")
        super().__init__(data, loop, start, loop_start, loop_end)

    @classmethod
    def from_file(cls, path: str | Path, loop: bool = False,
                  sample_rate: int | None = None, loop_start: int = 0,
                  loop_end: int = 0) -> "BufferSource":
        """Decode an audio file, resampling if the engine rate differs.

        Uses soundfile when available (covers wav/flac/ogg) and falls back
        to miniaudio, which additionally handles mp3.
        """
        data, scale = load_mono(path, sample_rate)
        return cls(data, loop=loop,
                   loop_start=int(round(loop_start * scale)),
                   loop_end=int(round(loop_end * scale)))


class StereoBufferSource(_LoopingBuffer):
    """Play a stereo asset flat, as a single source.

    Only the flat path (`spatialise=False`) can use this: giving a stereo
    signal a direction is meaningless, so the engine rejects it elsewhere.
    Music, ambience and jingles are the cases that want it -- they are
    stereo on disk, and the alternatives are to fold them to the middle
    (losing the image the composer mixed) or to start one hard-panned
    source per channel (two mixer slots and two convolution rows for one
    piece of music).

    `pan` on the slot acts as a balance for a source like this rather than
    as a position.
    """

    channels = 2

    def __init__(self, samples: np.ndarray, loop: bool = False,
                 start: int = 0, loop_start: int = 0,
                 loop_end: int = 0) -> None:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)      # mono in a stereo slot
        elif data.ndim == 2:
            if data.shape[0] < data.shape[1]:
                raise ValueError(
                    f"expected (frames, channels), got {data.shape}; "
                    "pass samples.T for channels-first data")
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]                     # take the front pair
        else:
            raise ValueError(f"samples must be 1-D or 2-D, got {data.shape}")
        super().__init__(data, loop, start, loop_start, loop_end)

    @classmethod
    def from_file(cls, path: str | Path, loop: bool = False,
                  sample_rate: int | None = None, loop_start: int = 0,
                  loop_end: int = 0) -> "StereoBufferSource":
        data, scale = load_stereo(path, sample_rate)
        return cls(data, loop=loop,
                   loop_start=int(round(loop_start * scale)),
                   loop_end=int(round(loop_end * scale)))


class PushSource:
    """Ring buffer written by the app thread, read by the audio thread.

    Single producer / single consumer: the writer only ever advances
    `_written` and the reader only `_consumed`, both monotonic sample
    counts, so no counter is written from two threads and the object is
    safe without a lock.  Writing more than the free space overwrites the
    oldest samples rather than blocking (blocking near an audio callback
    is far worse than a dropout); the reader detects being lapped and
    resumes from the oldest surviving sample, so playback stays in order
    with a gap instead of coming out scrambled.
    """

    channels = 1

    def __init__(self, capacity: int = 48000) -> None:
        self._buf = np.zeros(int(capacity), dtype=np.float32)
        self._written = 0        # writer-owned monotonic total
        self._consumed = 0       # reader-owned monotonic total
        self.closed = False

    @property
    def available(self) -> int:
        return min(self._written - self._consumed, int(self._buf.shape[0]))

    @property
    def sounding(self) -> bool:
        """書いた音が残っているか（ダッキングのきっかけ）。

        閉じるまで鳴り終わらない音源なので、「鳴り終わっていない」で判断
        すると、何も書いていない間も声が鳴っている扱いになり、BGM が下がった
        ままになる。
        """
        return self.available > 0

    @property
    def finished(self) -> bool:
        return self.closed and self._written - self._consumed <= 0

    def close(self) -> None:
        self.closed = True

    def write(self, samples: np.ndarray) -> int:
        x = np.asarray(samples, dtype=np.float32).ravel()
        cap = self._buf.shape[0]
        if x.shape[0] > cap:
            x = x[-cap:]
        n = int(x.shape[0])
        pos = self._written % cap
        end = pos + n
        if end <= cap:
            self._buf[pos:end] = x
        else:
            split = cap - pos
            self._buf[pos:] = x[:split]
            self._buf[:end - cap] = x[split:]
        self._written += n
        return n

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        cap = int(self._buf.shape[0])
        written = self._written          # snapshot: the writer may move on
        start = self._consumed
        if written - start > cap:        # lapped; oldest surviving sample
            start = written - cap
        take = min(n, written - start)
        if take > 0:
            pos = start % cap
            end = pos + take
            if end <= cap:
                out[:take] = self._buf[pos:end]
            else:
                split = cap - pos
                out[:split] = self._buf[pos:]
                out[split:take] = self._buf[:end - cap]
            self._consumed = start + take
        return out


class CallbackSource:
    """Wrap a generator function `fn(n) -> (n,) float32`.

    `sounding` is False while the function returns silence (the last block
    it gave was all zeros), so that a quiet source does not keep another
    group ducked.
    """

    channels = 1
    #: これ以下の大きさは無音とみなす（-120dB）
    SILENCE = 1e-6

    def __init__(self, fn: Callable[[int], np.ndarray]) -> None:
        self._fn = fn
        self._done = False
        self._sounding = True           # 読む前は、これから鳴るものとして扱う

    @property
    def sounding(self) -> bool:
        """直前に関数が返した音が無音でなかったか（ダッキングのきっかけ）。"""
        return self._sounding and not self._done

    @property
    def finished(self) -> bool:
        return self._done

    def stop(self) -> None:
        self._done = True

    def read(self, n: int) -> np.ndarray:
        if self._done:
            return np.zeros(n, dtype=np.float32)
        out = np.asarray(self._fn(n), dtype=np.float32).ravel()
        if out.shape[0] < n:
            out = np.pad(out, (0, n - out.shape[0]))
        out = out[:n]
        self._sounding = bool(out.size) and bool(
            np.max(np.abs(out)) > self.SILENCE)
        return out


# ---------------------------------------------------------------------------
# 長いファイルを少しずつ読む
# ---------------------------------------------------------------------------
class _SoundfileReader:
    """soundfile（libsndfile）で少しずつ読む。wav・flac・ogg・mp3。"""

    def __init__(self, path: Path) -> None:
        import soundfile as sf
        self._file = sf.SoundFile(str(path))
        self.rate = int(self._file.samplerate)
        self.channels = int(self._file.channels)
        self.frames = int(self._file.frames)

    def read(self, n: int) -> np.ndarray:
        return self._file.read(n, dtype="float32", always_2d=True)

    def seek(self, frame: int) -> None:
        self._file.seek(int(frame))

    def close(self) -> None:
        self._file.close()


class _MiniaudioReader:
    """miniaudio で少しずつ読む（soundfile で読めない mp3 など）。"""

    CHUNK = 4096

    def __init__(self, path: Path) -> None:
        import miniaudio
        self._ma = miniaudio
        self._path = str(path)
        info = miniaudio.get_file_info(self._path)
        self.rate = int(info.sample_rate)
        self.channels = int(info.nchannels)
        self.frames = int(info.num_frames)
        self._gen = None
        self._pending = np.zeros((0, self.channels), dtype=np.float32)
        self.seek(0)

    def seek(self, frame: int) -> None:
        self.close()
        self._gen = self._ma.stream_file(
            self._path, output_format=self._ma.SampleFormat.FLOAT32,
            nchannels=self.channels, sample_rate=self.rate,
            frames_to_read=self.CHUNK, seek_frame=int(frame))
        self._pending = np.zeros((0, self.channels), dtype=np.float32)

    def read(self, n: int) -> np.ndarray:
        parts = [self._pending]
        have = self._pending.shape[0]
        while have < n and self._gen is not None:
            try:
                chunk = next(self._gen)
            except StopIteration:
                self._gen = None
                break
            data = np.frombuffer(chunk, dtype=np.float32).reshape(
                -1, self.channels)
            parts.append(data)
            have += data.shape[0]
        joined = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self._pending = joined[n:]
        return np.ascontiguousarray(joined[:n])

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None


def _open_reader(path: str | Path, backend: str | None = None):
    """少しずつ読む道具を開く。soundfile、だめなら miniaudio。

    どちらでも開けないときは、ファイルの名前と両方の理由を添えた
    DecodeError（`_decode` と同じ考え方）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    reasons = []
    for name, cls in (("soundfile", _SoundfileReader),
                      ("miniaudio", _MiniaudioReader)):
        if backend is not None and backend != name:
            continue
        try:
            return cls(path)
        except Exception as exc:                 # noqa: BLE001
            reasons.append(f"{name}: {type(exc).__name__}: {exc}")
    raise DecodeError(tr(
        f"音声ファイルを少しずつ読めません: {path.name}（{'、'.join(reasons)}）",
        f"Cannot stream the audio file {path.name} "
        f"({'; '.join(reasons)})"))


def audio_file_info(path: str | Path) -> tuple[int, int, int]:
    """ファイルの (フレーム数, サンプルレート, チャンネル数) を、中身を
    読まずに返す。読めなければ DecodeError か FileNotFoundError。"""
    reader = _open_reader(path)
    try:
        return reader.frames, reader.rate, reader.channels
    finally:
        reader.close()


def can_resample_streaming() -> bool:
    """少しずつのリサンプリング（samplerate）が使えるか。"""
    try:
        import samplerate
        return hasattr(samplerate, "Resampler")
    except ImportError:
        return False


class _StreamState:
    """読み込みのスレッドと音声のスレッドが分け合う状態。

    書くのは読み込みのスレッドだけ（written）、読むのは音声のスレッド
    だけ（consumed）。どちらも増えていくだけの数なので、PushSource と
    同じく鍵なしで安全。書く側は空きがあるときだけ書く（読まれていない
    ところを上書きしない）。
    """

    def __init__(self, reader, ratio: float, channels: int, capacity: int,
                 loop: bool, loop_start: int, loop_end: int,
                 chunk: int) -> None:
        #: 空きが足りず書ききれなかった残り。次の step で先に書く
        self.pending: np.ndarray | None = None
        self.reader = reader
        self.ratio = ratio
        self.channels = channels
        self.capacity = capacity
        shape = (capacity,) if channels == 1 else (capacity, 2)
        self.buf = np.zeros(shape, dtype=np.float32)
        self.written = 0
        self.consumed = 0
        self.loop = loop
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.chunk = chunk
        self.chunk_out = int(chunk * ratio) + 64
        self.pos = 0
        self.cycle = 0              # いまの一巡で読んだフレーム数
        self.eof = False
        self.error = ""
        self.underruns = 0
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.resampler = None
        if abs(ratio - 1.0) > 1e-12:
            import samplerate
            self.resampler = samplerate.Resampler("sinc_medium",
                                                  channels=channels)

    @property
    def space(self) -> int:
        return self.capacity - (self.written - self.consumed)

    def _convert(self, data: np.ndarray) -> np.ndarray:
        """読んだ (フレーム, ch) を、鳴らすチャンネル数にする。"""
        if self.channels == 1:
            return (data.mean(axis=1) if data.shape[1] > 1
                    else data[:, 0]).astype(np.float32)
        if data.shape[1] == 1:
            return np.repeat(data, 2, axis=1)
        return data[:, :2]

    @property
    def drained(self) -> bool:
        """読み終え、書ききった（あとは鳴らすだけ）。"""
        return self.eof and self.pending is None

    def _write(self, y: np.ndarray) -> None:
        """空きのぶんだけ書き、残りは pending に持ち越す（待たない。待つと、
        作るときの先読みで読む側がいないまま止まる）。"""
        n = int(y.shape[0])
        take = min(max(0, self.space), n)
        if take > 0:
            pos = self.written % self.capacity
            end = pos + take
            if end <= self.capacity:
                self.buf[pos:end] = y[:take]
            else:
                split = self.capacity - pos
                self.buf[pos:] = y[:split]
                self.buf[:end - self.capacity] = y[split:take]
            self.written += take
        self.pending = y[take:] if take < n else None

    def step(self) -> None:
        """1かたまり読んで、変換して、輪のバッファに書く。"""
        if self.pending is not None:
            rest, self.pending = self.pending, None
            self._write(rest)
            return
        end = self.loop_end or self.reader.frames
        want = self.chunk if end <= 0 else min(self.chunk, end - self.pos)
        data = (self.reader.read(want) if want > 0
                else np.zeros((0, self.reader.channels), dtype=np.float32))
        k = int(data.shape[0])
        self.pos += k
        self.cycle += k
        if k:
            y = self._convert(data)
            if self.resampler is not None:
                y = self.resampler.process(y, self.ratio, end_of_input=False)
            self._write(np.asarray(y, dtype=np.float32))
        at_end = k == 0 or (end > 0 and self.pos >= end)
        if not at_end:
            return
        if self.loop and self.cycle > 0:
            self.reader.seek(self.loop_start)
            self.pos = self.loop_start
            self.cycle = 0
            return
        if self.resampler is not None:
            empty = np.zeros((0,) if self.channels == 1 else (0, 2),
                             dtype=np.float32)
            self._write(np.asarray(self.resampler.process(
                empty, self.ratio, end_of_input=True), dtype=np.float32))
        self.eof = True


def _feed(state: _StreamState, owner) -> None:
    """読み込みのスレッド。音源が捨てられたか、止められたか、読み終えたら
    終わり、ファイルを閉じる。音源そのものは弱い参照でしか持たない（持つと
    捨てられた音源のスレッドが残り続ける）。"""
    try:
        while not state.stop.is_set() and not state.drained:
            if owner() is None:
                break
            if state.space < state.chunk_out:
                state.wake.wait(0.1)
                state.wake.clear()
                continue
            state.step()
    except Exception as exc:                      # noqa: BLE001
        state.error = f"{type(exc).__name__}: {exc}"
        state.eof = True
    finally:
        try:
            state.reader.close()
        except Exception:                         # noqa: BLE001
            pass


class FileStreamSource:
    """長い音声ファイルを、別のスレッドで少しずつ読みながら鳴らす。

    1曲を丸ごと読むと、48kHz ステレオの3分の曲で約66MB になる。これは
    読み込みのスレッドが `buffer_seconds` 秒先までを輪のバッファに読み、
    音声のスレッドはそこから取り出すだけ。メモリは2秒のステレオで約0.8MB。

    path          wav・flac・ogg・mp3（soundfile、だめなら miniaudio で読む）
    sample_rate   鳴らすレート（エンジンのレート）。ファイルと違えば
                  samplerate で少しずつ変換する（無ければ ValueError）
    channels      1（空間化する音。チャンネルを平均する）か 2（平置きの
                  ステレオ。モノラルのファイルは両方に入れる）
    loop          最後まで読んだら loop_start へ戻る（つなぎ目は途切れない）
    loop_start, loop_end   ループの区間（ファイルのフレーム。0 は終わり）

    作るときに最初の少し（0.5秒）を読んでおくので、鳴らし始めで途切れない。
    読み込みが間に合わなかったブロックは無音になり、`underruns` に数える。
    読み込み中にファイルが読めなくなったら、残りを鳴らした後の read() が
    例外を出す（エンジンはその音だけを外し、鳴り終わりの知らせは error）。
    エンジンから外されて捨てられると、読み込みのスレッドも自然に終わる。
    """

    PREFILL_SECONDS = 0.5
    CHUNK_FRAMES = 4096

    def __init__(self, path: str | Path, sample_rate: int | None = None, *,
                 channels: int = 1, loop: bool = False, loop_start: int = 0,
                 loop_end: int = 0, buffer_seconds: float = 2.0,
                 _backend: str | None = None) -> None:
        if channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {channels!r}")
        reader = _open_reader(path, _backend)
        try:
            rate = int(sample_rate or reader.rate)
            ratio = rate / reader.rate
            if abs(ratio - 1.0) > 1e-12 and not can_resample_streaming():
                raise ValueError(tr(
                    f"{Path(path).name} は {reader.rate}Hz で、鳴らすレート "
                    f"{rate}Hz と違います。少しずつ変換するには samplerate "
                    "が要ります",
                    f"{Path(path).name} is {reader.rate} Hz, not the "
                    f"playback rate {rate} Hz. Converting it a little at a "
                    "time needs samplerate"))
            frames = max(0, reader.frames)
            end = min(int(loop_end), frames) if loop_end else 0
            # ループの始めは、終わりの手前に収める（BufferSource と同じ。
            # 終わり以降だと一巡が空になり、繰り返さずに鳴り終わっていた）
            stop = end if end > 0 else frames
            start = max(0, min(int(loop_start), max(0, stop - 1)))
            capacity = max(rate // 10, int(float(buffer_seconds) * rate))
            state = _StreamState(reader, ratio, channels, capacity,
                                 bool(loop), start, end, self.CHUNK_FRAMES)
            # 最初の少しを、ここで読んでおく（鳴らし始めで途切れないように）
            prefill = min(capacity - state.chunk_out,
                          int(self.PREFILL_SECONDS * rate))
            while state.written < prefill and not state.drained:
                state.step()
        except Exception:
            reader.close()
            raise
        self.channels = channels
        self.sample_rate = rate
        self.duration_s = frames / reader.rate if reader.rate else 0.0
        self._state = state
        self._thread = None
        if not state.drained:
            self._thread = threading.Thread(
                target=_feed, args=(state, weakref.ref(self)),
                name="audiospace-stream", daemon=True)
            self._thread.start()
        else:
            reader.close()

    @property
    def underruns(self) -> int:
        """読み込みが間に合わず、無音を出したブロックの数。"""
        return self._state.underruns

    @property
    def finished(self) -> bool:
        st = self._state
        # 読めなくなったときは「鳴り終わった」にしない。次の read() が例外を
        # 出して、エンジンがその音を error として外す
        return st.stop.is_set() or (st.drained and not st.error
                                    and st.consumed >= st.written)

    def close(self, wait_s: float = 0.0) -> None:
        """読み込みを止める（その後は無音で、finished は True）。

        ファイルは読み込みのスレッドが閉じる。`wait_s` を渡すと、閉じ終わる
        のをその秒数まで待つ（すぐ後でファイルを消す・置き換えるとき）。
        """
        self._state.stop.set()
        self._state.wake.set()
        thread = self._thread
        if (wait_s > 0.0 and thread is not None
                and thread is not threading.current_thread()):
            thread.join(timeout=float(wait_s))

    def read(self, n: int) -> np.ndarray:
        st = self._state
        shape = (n,) if self.channels == 1 else (n, 2)
        out = np.zeros(shape, dtype=np.float32)
        if st.stop.is_set():
            return out
        cap = st.capacity
        written = st.written            # 書く側が先へ進んでもよいように控える
        start = st.consumed
        take = min(n, written - start)
        if take > 0:
            pos = start % cap
            end = pos + take
            if end <= cap:
                out[:take] = st.buf[pos:end]
            else:
                split = cap - pos
                out[:split] = st.buf[pos:]
                out[split:take] = st.buf[:end - cap]
            st.consumed = start + take
        if take < n:
            if st.error:
                # 鳴らせる分を鳴らし終えた。外してもらう
                raise RuntimeError(st.error)
            if not st.drained:
                st.underruns += 1
        st.wake.set()
        return out


def load_mono(path: str | Path,
              sample_rate: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a file to a mono float32 array, resampling when asked.

    Returns (samples, scale) where `scale` converts frame counts given in
    the file's native rate (loop points, cue markers) into the returned
    rate.  Split out of `BufferSource.from_file` so a player can cache the
    decoded array and reuse it across many plays.
    """
    data, scale = _load(path, sample_rate)
    if data.ndim == 2:                       # decoders are frames-first
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), scale


def load_stereo(path: str | Path,
                sample_rate: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a file to an (frames, 2) float32 array, resampling when asked.

    A mono file is duplicated into both channels, so the caller always
    gets a stereo buffer and never has to branch on the file's layout.
    """
    data, scale = _load(path, sample_rate)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32), scale


def load_flat(path: str | Path,
              sample_rate: int | None = None) -> tuple[np.ndarray, float]:
    """Decode for flat playback with a single read of the file.

    Returns stereo (frames, 2) when the content really is stereo, and
    mono (frames,) otherwise.  A mono recording in a stereo container is
    extremely common in game assets, and playing that as stereo would
    spend a second channel of mixing on an identical copy -- while
    deciding via `is_stereo_file` and then loading again would decode a
    multi-minute music file twice.
    """
    data, scale = _load(path, sample_rate)
    if data.ndim == 2 and data.shape[1] >= 2 and _is_effectively_stereo(data):
        if data.shape[1] > 2:
            data = data[:, :2]                 # take the front pair
    elif data.ndim == 2:
        data = data.mean(axis=1)               # fold the identical copies
    return np.ascontiguousarray(data, dtype=np.float32), scale


def is_stereo_file(path: str | Path) -> bool:
    """True when the file has two or more distinct channels.

    Convenience wrapper; when the data is going to be loaded anyway, use
    `load_flat`, which decides from the one decode it already does.
    """
    data, _rate = _decode(Path(path))
    return data.ndim == 2 and data.shape[1] >= 2 \
        and _is_effectively_stereo(data)


def _is_effectively_stereo(data: np.ndarray) -> bool:
    """Whether (frames, >=2ch) data carries a real stereo image.

    Side against mid, which is what "how stereo is it" means.  Game
    assets fall cleanly into two groups measured this way: genuinely
    stereo files sit far above one percent, mono-in-a-stereo-container
    far below it.
    """
    left, right = data[:, 0], data[:, 1]
    mid = float(np.sqrt(np.mean(((left + right) * 0.5) ** 2)))
    side = float(np.sqrt(np.mean(((left - right) * 0.5) ** 2)))
    return side > mid * 1e-2


def resampled_by(data: np.ndarray, rate: float) -> np.ndarray:
    """Return `data` played at `rate` times speed.

    Pitch and tempo move together, which is what a sampler's playback-rate
    control does (and what the original DirectSound games used for footstep
    variation).  Sinc resampling is used where available: linear
    interpolation audibly dulls and grits speech and effects.
    """
    rate = float(rate)
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}")
    if abs(rate - 1.0) < 1e-9 or data.shape[0] < 2:
        return data
    ratio = 1.0 / rate                       # faster playback = fewer frames
    try:
        import samplerate

        out = np.asarray(samplerate.resample(data, ratio, "sinc_medium"),
                         dtype=np.float32)
    except Exception:                            # noqa: BLE001
        from scipy import signal

        frac = Fraction(ratio).limit_denominator(2000)
        out = signal.resample_poly(data, frac.numerator, frac.denominator,
                                   axis=0).astype(np.float32)
    if data.ndim == 2 and out.ndim == 1:         # samplerate can flatten mono
        out = out[:, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def _load(path: str | Path,
          sample_rate: int | None) -> tuple[np.ndarray, float]:
    """Decode and rate-convert, keeping whatever channel layout the file has."""
    data, file_rate = _decode(Path(path))
    scale = 1.0
    if sample_rate and file_rate != sample_rate:
        data = _resample(data, file_rate, sample_rate)
        scale = sample_rate / float(file_rate)
    return data, scale


def _decode(path: Path) -> tuple[np.ndarray, int]:
    """Decode to float32, keeping the file's channel layout.

    Returns (frames,) for mono and (frames, channels) otherwise; callers
    downmix or keep the channels as they need.  A missing file must raise,
    not fall through to the next backend, or the caller is told about
    miniaudio when the real problem is a typo in a path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(rate)
    except Exception as exc:                     # noqa: BLE001
        first = f"soundfile: {type(exc).__name__}: {exc}"
    # どちらでも読めないときは、ファイルの名前と両方の理由を添える。
    # 理由を捨てると「failed to decode file」としか分からず、miniaudio が
    # 入っていない環境では、ファイルの問題が「miniaudio が無い」に化ける
    try:
        import miniaudio
    except ImportError:
        raise DecodeError(tr(
            f"音声ファイルを読めません: {path.name}（{first}。mp3 を読むには "
            "miniaudio が要ります）",
            f"Cannot decode the audio file {path.name} ({first}; reading "
            "mp3 needs miniaudio)")) from None
    try:
        decoded = miniaudio.decode_file(
            str(path), output_format=miniaudio.SampleFormat.FLOAT32)
    except Exception as exc:                     # noqa: BLE001
        raise DecodeError(tr(
            f"音声ファイルを読めません: {path.name}（{first}、"
            f"miniaudio: {exc}）",
            f"Cannot decode the audio file {path.name} ({first}; "
            f"miniaudio: {exc})")) from exc
    samples = np.frombuffer(decoded.samples, dtype=np.float32)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)
    return np.ascontiguousarray(samples), int(decoded.sample_rate)


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Rate-convert along the frame axis, mono or multichannel."""
    try:
        import samplerate
        out = np.asarray(
            samplerate.resample(data, dst_rate / src_rate, "sinc_medium"),
            dtype=np.float32)
    except Exception:                            # noqa: BLE001
        from scipy import signal
        from math import gcd
        g = gcd(int(src_rate), int(dst_rate))
        out = signal.resample_poly(data, dst_rate // g, src_rate // g,
                                   axis=0).astype(np.float32)
    if data.ndim == 2 and out.ndim == 1:
        out = out[:, None]
    return out
