"""Audio device selection.

Measured on this machine, the same output card reports:

    MME           90.0 ms      (sounddevice's default host API)
    DirectSound  120.0 ms
    WASAPI shared 22.0 ms
    WASAPI excl.  10.7 ms

so an audio game should never accept the default.  This module picks
WASAPI when present and falls back cleanly otherwise.

The important trap: PortAudio does not resample in WASAPI shared mode, so
opening a stream at any rate other than the one configured for the device
in the Windows sound settings fails with

    Error opening OutputStream: Invalid sample rate [PaErrorCode -9997]

Therefore we always read the device's own `default_samplerate` and build
the engine around it, rather than assuming 48000.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .i18n import tr

SILENT_ENV = "AUDIOSPACE_SILENT"


def is_silent() -> bool:
    """True when audio output is disabled by environment variable.

    Set AUDIOSPACE_SILENT=1 for headless tests and for verification runs
    where nothing should be audible.
    """
    return os.environ.get(SILENT_ENV, "").strip() not in ("", "0", "false")


@dataclass(frozen=True)
class DeviceChoice:
    """A resolved output device."""

    index: int
    name: str
    host_api: str
    sample_rate: int
    latency_ms: float
    exclusive: bool = False

    def describe(self) -> str:
        """One-line summary intended to be read aloud."""
        if self.exclusive:
            mode = tr("排他", "exclusive")
        else:
            mode = tr("共有", "shared")
        return tr(f"{self.name}（{self.host_api} {mode}モード、"
                  f"{self.sample_rate}Hz、遅延約{self.latency_ms:.0f}ミリ秒）",
                  f"{self.name} ({self.host_api} {mode} mode, "
                  f"{self.sample_rate} Hz, latency about "
                  f"{self.latency_ms:.0f} milliseconds)")


def list_output_devices() -> list[DeviceChoice]:
    """All stereo-capable outputs, best latency first."""
    import sounddevice as sd

    apis = sd.query_hostapis()
    found: list[DeviceChoice] = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] < 2:
            continue
        found.append(DeviceChoice(
            index=index,
            name=str(dev["name"]),
            host_api=str(apis[dev["hostapi"]]["name"]),
            sample_rate=int(dev["default_samplerate"]),
            latency_ms=float(dev["default_low_output_latency"]) * 1000.0))
    found.sort(key=lambda d: d.latency_ms)
    return found


def choose_output(prefer_host_api: str = "WASAPI",
                  device: int | str | None = None) -> DeviceChoice:
    """Resolve the output device to use.

    `device` may be an index or a substring of the device name; when given
    it wins over `prefer_host_api`, so a user preference in a settings file
    is always honoured.
    """
    import sounddevice as sd

    candidates = list_output_devices()
    if not candidates:
        raise RuntimeError("no stereo output device found")

    if device is not None:
        if isinstance(device, int):
            match = [c for c in candidates if c.index == device]
        else:
            needle = device.lower()
            match = [c for c in candidates if needle in c.name.lower()]
        if not match:
            raise ValueError(f"no output device matching {device!r}")
        # among identical names prefer the lowest-latency host API
        return match[0]

    apis = sd.query_hostapis()
    api_index = next((i for i, a in enumerate(apis)
                      if prefer_host_api.lower() in a["name"].lower()), None)
    if api_index is not None:
        default = apis[api_index]["default_output_device"]
        preferred = [c for c in candidates if c.index == default]
        if preferred:
            return preferred[0]
        on_api = [c for c in candidates
                  if prefer_host_api.lower() in c.host_api.lower()]
        if on_api:
            return on_api[0]
    return candidates[0]


def open_output_stream(choice: DeviceChoice, callback, block: int = 512,
                       channels: int = 2, exclusive: bool = False):
    """Open a low-latency stream at the device's own sample rate.

    Returns an un-started `sounddevice.OutputStream`.
    """
    import sounddevice as sd

    extra = None
    if exclusive and "wasapi" in choice.host_api.lower():
        extra = sd.WasapiSettings(exclusive=True)
    return sd.OutputStream(device=choice.index,
                           samplerate=choice.sample_rate,
                           blocksize=block, channels=channels,
                           dtype="float32", latency="low",
                           callback=callback, extra_settings=extra)


def describe_devices() -> str:
    """Numbered plain-text device list for screen-reader friendly menus."""
    lines = []
    for i, dev in enumerate(list_output_devices(), start=1):
        lines.append(f"{i}. {dev.describe()}")
    return "\n".join(lines) if lines else tr(
        "利用可能な出力デバイスがありません", "No output device is available")
