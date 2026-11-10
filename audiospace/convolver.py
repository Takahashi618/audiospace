"""Uniform-partitioned overlap-add convolution for many sources at once.

Measured on this machine (CPython 3.14, block=512 @ 48 kHz, 128-tap HRIR):
256 simultaneous sources cost 1.99 ms against a 10.67 ms budget, and the
result matches direct convolution to within 4.5e-08.  (That figure predates
the 2026-10 changes below; tools/bench_polyphony.py measures the
whole engine.)

Directional crossfade
---------------------
Swapping the HRIR for a moving source mid-stream produces an audible click.
A source whose direction index changed this block is rendered twice -- once
with the previous index and once with the new one -- and crossfaded with a
linear ramp.  Sources whose index did not change are rendered once: for
them the two renders would be identical, so by linearity the blend equals
the single render.  (Until 2026-10 every source was rendered twice whenever
any one of them moved; with head wobble on, that was every block.)

Cost scaling (2026-10)
----------------------
Only the rows that are playing -- plus, for HRIRs longer than one block,
rows whose input is still in the delay line -- are transformed and summed.
The cost follows the number of sounding sources, not `max_sources`.

When the HRIR fits in one block, the FFT is only as long as the linear
convolution needs (block + taps - 1, rounded to a fast size: 640 instead
of 1024 for 512 + 128), which cuts the transform and the summation by a
third.
"""
from __future__ import annotations

import numpy as np

from scipy import fft as sp_fft

from .hrtf import HrirSet


class BinauralConvolver:
    """Convolve N mono source blocks against per-source HRIRs -> stereo.

    Parameters
    ----------
    hrir        the directional HRIR grid to render with
    block       samples per callback; must stay constant for the lifetime
    max_sources allocated capacity (the number of rows in `blocks`)
    crossfade   render twice and blend when a source's direction changes
    """

    def __init__(self, hrir: HrirSet, block: int = 512,
                 max_sources: int = 64, crossfade: bool = True) -> None:
        if block < 32 or (block & (block - 1)) != 0:
            raise ValueError("block must be a power of two >= 32")
        self.hrir = hrir
        self.block = int(block)
        self.max_sources = int(max_sources)
        self.crossfade = bool(crossfade)

        self.nparts = int(np.ceil(hrir.taps / self.block))
        if self.nparts == 1:
            # One partition: plain overlap-add needs only room for the
            # linear convolution, block + taps - 1 samples.
            self.nfft = int(sp_fft.next_fast_len(
                self.block + hrir.taps - 1, real=True))
        else:
            self.nfft = self.block * 2
        padded = np.zeros((hrir.n_directions, 2, self.nparts * self.block),
                          dtype=np.float32)
        padded[..., :hrir.taps] = hrir.data
        parts = padded.reshape(hrir.n_directions, 2, self.nparts, self.block)
        # (direction, ear, partition, bin)
        self._H = np.fft.rfft(parts, self.nfft, axis=-1).astype(np.complex64)
        self.nbins = int(self._H.shape[-1])
        # one-partition view without the partition axis, for the fast path
        self._H1 = (np.ascontiguousarray(self._H[:, :, 0, :])
                    if self.nparts == 1 else None)

        # frequency delay line, one row of partitions per source.  With a
        # single partition nothing outlives the block, so none is kept.
        self._fdl = (np.zeros((self.max_sources, self.nparts, self.nbins),
                              dtype=np.complex64)
                     if self.nparts > 1 else None)
        self._tail = np.zeros((2, self.nfft - self.block), dtype=np.float32)
        self._prev_idx = np.zeros(self.max_sources, dtype=np.intp)
        # rows rendered in the previous block: only these can crossfade.  A
        # row that just started has no previous direction to fade from.
        self._was_rendered = np.zeros(self.max_sources, dtype=bool)
        # blocks for which each row's input is still in the delay line
        self._history = np.zeros(self.max_sources, dtype=np.intp)
        self._ramp = np.linspace(0.0, 1.0, self.block,
                                 endpoint=False, dtype=np.float32)
        self._scratch = np.zeros((self.max_sources, self.block),
                                 dtype=np.float32)

    def reset(self) -> None:
        """Clear all convolution state (use when teleporting the listener)."""
        if self._fdl is not None:
            self._fdl[:] = 0
        self._tail[:] = 0
        self._prev_idx[:] = 0
        self._was_rendered[:] = False
        self._history[:] = 0

    def forget(self, row: int) -> None:
        """A new source now occupies `row`: do not crossfade it from the
        direction the previous occupant was rendered at.

        Without this, a sound that starts in a row freed in the same block
        window (stop-then-play, or a reaped one-shot) spent its first block
        fading in from the old occupant's direction.  The previous
        occupant's overlap tail is untouched: it is already summed into the
        shared output tail and plays out where it was.
        """
        self._was_rendered[row] = False

    def process(self, blocks: np.ndarray, direction_idx: np.ndarray,
                active: np.ndarray | None = None) -> np.ndarray:
        """Render one block.

        Parameters
        ----------
        blocks         (max_sources, block) float32, silent rows allowed
        direction_idx  (max_sources,) index into the HRIR grid
        active         rows that carry sound this block.  Rows left out are
                       treated as silent and cost nothing.  None means the
                       rows whose samples are not all zero, so callers that
                       never pass it still get the saving, and a source
                       starting in a row that was silent is not faded in
                       from that row's stale direction.

        Returns
        -------
        (block, 2) float32 interleaved-by-column stereo
        """
        if blocks.shape != (self.max_sources, self.block):
            raise ValueError(
                f"blocks must be {(self.max_sources, self.block)}, "
                f"got {blocks.shape}")
        idx = np.asarray(direction_idx, dtype=np.intp)
        if active is None:
            active = np.flatnonzero(np.any(blocks != 0.0, axis=1))
        else:
            active = np.asarray(active, dtype=np.intp)

        if self.nparts == 1:
            rows = active
            spectra = np.fft.rfft(blocks[rows], self.nfft, axis=-1)
        else:
            # a row keeps sounding for nparts blocks after its last input
            self._history[active] = self.nparts
            rows = np.flatnonzero(self._history > 0)
            if rows.size:
                self._fdl[rows, 1:] = self._fdl[rows, :-1]
                self._fdl[rows, 0] = 0
            if active.size:
                self._fdl[active, 0] = np.fft.rfft(blocks[active], self.nfft,
                                                   axis=-1)
            spectra = self._fdl[rows]
            self._history[rows] -= 1

        cur = idx[rows]
        new = self._time(self._sum(spectra, cur))
        # The tail is nfft - block samples long: a whole block when nfft is
        # 2*block, but only taps-1 (rounded up) with the short FFT, so it
        # overlaps just the start of this block.
        out = new[:, :self.block].copy()
        out[:, :self._tail.shape[1]] += self._tail
        if self.crossfade and rows.size:
            prev = self._prev_idx[rows]
            moved = (cur != prev) & self._was_rendered[rows]
            if moved.any():
                # the blend old*(1-r) + new*r differs from the plain render
                # only on the moved rows: add (old - new)*(1-r) for those
                diff = (self._sum(spectra[moved], prev[moved])
                        - self._sum(spectra[moved], cur[moved]))
                out += self._time(diff)[:, :self.block] * (1.0 - self._ramp)
        self._tail[:] = new[:, self.block:]

        self._was_rendered[:] = False
        self._was_rendered[rows] = True
        self._prev_idx[rows] = cur
        return np.ascontiguousarray(out.T)

    def _sum(self, spectra: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Sum sources through their HRIRs in the frequency domain -> (2, bins).

        Plain einsum, deliberately not optimize=True: for this two-operand
        contraction the "optimised" route goes through batched small matrix
        products and measured 6-7x slower (128 sources: 0.80 ms against
        0.12 ms; 256: 4.2 ms against 0.69 ms).
        """
        if len(idx) == 0:
            return np.zeros((2, self.nbins), dtype=np.complex64)
        if self._H1 is not None:
            return np.einsum("sb,seb->eb", spectra, self._H1[idx])
        return np.einsum("spb,sepb->eb", spectra, self._H[idx])

    def _time(self, acc: np.ndarray) -> np.ndarray:
        return np.fft.irfft(acc, self.nfft, axis=-1).astype(np.float32)

    @property
    def scratch(self) -> np.ndarray:
        """Reusable (max_sources, block) staging buffer for the engine.

        Exposed so the audio callback can fill it in place and avoid any
        allocation, which is what keeps worst-case callback time flat.
        """
        return self._scratch


class MonoConvolver:
    """Partitioned convolution of one mono signal against a stereo IR.

    Used for the convolution reverb send, where a single summed bus is
    convolved with a long room impulse response.
    """

    def __init__(self, ir: np.ndarray, block: int = 512) -> None:
        ir = np.atleast_2d(np.asarray(ir, dtype=np.float32))
        if ir.shape[0] == 1:
            ir = np.repeat(ir, 2, axis=0)
        if ir.shape[0] != 2:
            raise ValueError(f"ir must be mono or stereo, got {ir.shape}")
        self.block = int(block)
        self.nfft = self.block * 2
        self.taps = int(ir.shape[1])
        self.nparts = int(np.ceil(self.taps / self.block))
        padded = np.zeros((2, self.nparts * self.block), dtype=np.float32)
        padded[:, :self.taps] = ir
        parts = padded.reshape(2, self.nparts, self.block)
        self._H = np.fft.rfft(parts, self.nfft, axis=-1).astype(np.complex64)
        # Frequency delay line kept as a ring: each block's spectrum is
        # written once at `_head`, which moves one row back per block, so
        # newest-to-oldest is _fdl[_head:] then _fdl[:_head] -- two plain
        # slices that line up with _H[:, :P-head] and _H[:, P-head:].
        # Shifting every row down instead copied the whole line (several
        # hundred partitions for a long room) on every block.
        self._fdl = np.zeros((self.nparts, self._H.shape[-1]),
                             dtype=np.complex64)
        self._head = 0
        self._tail = np.zeros((2, self.nfft - self.block), dtype=np.float32)

    def reset(self) -> None:
        self._fdl[:] = 0
        self._head = 0
        self._tail[:] = 0

    def process(self, mono_block: np.ndarray) -> np.ndarray:
        """(block,) mono in -> (block, 2) stereo reverb out.

        Plain einsum, not optimize=True: for this contraction the
        "optimised" route was about 8x slower (a 2.1 s room IR, 201
        partitions: 2.2 ms against 0.28 ms per block).
        """
        parts = self.nparts
        k = self._head = (self._head - 1) % parts
        self._fdl[k] = np.fft.rfft(mono_block, self.nfft)
        acc = np.einsum("pb,epb->eb", self._fdl[k:], self._H[:, :parts - k])
        if k:
            acc += np.einsum("pb,epb->eb", self._fdl[:k],
                             self._H[:, parts - k:])
        y = np.fft.irfft(acc, self.nfft, axis=-1).astype(np.float32)
        out = y[:, :self.block] + self._tail
        self._tail[:] = y[:, self.block:]
        return np.ascontiguousarray(out.T)
