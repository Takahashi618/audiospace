"""Sound sources.

A source is a sample generator; spatialisation happens in the engine,
never here.  Four shapes cover the useful cases:

    BufferSource        a decoded mono array, optionally looping
    StereoBufferSource  a stereo asset played flat, as one source
    PushSource          a ring buffer the application writes into
    CallbackSource      a synthesiser function called per block

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

from fractions import Fraction
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


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
    """Wrap a generator function `fn(n) -> (n,) float32`."""

    channels = 1

    def __init__(self, fn: Callable[[int], np.ndarray]) -> None:
        self._fn = fn
        self._done = False

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
        return out[:n]


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
    except Exception:                            # noqa: BLE001
        pass
    import miniaudio
    decoded = miniaudio.decode_file(str(path),
                                    output_format=miniaudio.SampleFormat.FLOAT32)
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
