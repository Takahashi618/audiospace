"""audiospace -- binaural 3D audio for accessible games and tools.

MIT License, Copyright (c) 2026 Takahashi (see LICENSE).  The default HRTF
data is CC BY 4.0 (see ATTRIBUTION.md).  All dependencies are permissively
licensed (numpy, scipy, and optionally sofar / pyroomacoustics /
sounddevice, all MIT or BSD), so it can be used in software that is sold
without any source-disclosure obligation.  The code was written with
Claude (an AI model by Anthropic); see "How this was made" in README.md.
It exists to offer binaural audio that does not depend on OpenAL, under the
MIT License, with a choice of measured HRTFs.

Two levels of API:

    # high level
    from audiospace import SpatialPlayer
    player = SpatialPlayer.open()
    player.play("beacon.wav", position=(3.0, 4.0, 0.0), loop=True)

    # low level -- no audio device involved
    from audiospace import SpatialEngine, HrirSet, BufferSource
    engine = SpatialEngine(HrirSet.synthetic(sample_rate=48000))
    engine.add_source(BufferSource(samples), position=(1.0, 2.0, 0.0))
    stereo_block = engine.render()

Set AUDIOSPACE_SILENT=1 to run the whole stack with no audible output,
for tests and for verification passes.

Text meant for people (names, status lines, error messages) is Japanese
when the operating system's display language is Japanese, and English
otherwise.  Set AUDIOSPACE_LANG=ja or AUDIOSPACE_LANG=en before starting
the program to choose one.
"""
from __future__ import annotations

__version__ = "0.7.1"

from .acoustics import (FACE_LABELS, FACE_MATERIALS, FURNISH_LEVELS,
                        FURNISHINGS, SPACE_KINDS, SPACE_PRESETS, Furnishing,
                        contents_text, furnish_counts, furnishing_labels,
                        reverb_text,
                        ReflectionPath, Space, absorption, band_centers,
                        material_choices, material_label, material_names,
                        space_from_spec, space_labels, space_preset,
                        space_spec_problems, space_to_spec)
from .convolver import BinauralConvolver, MonoConvolver
from .defaults import (DEFAULT_PINNA, attribution, default_hrir,
                       default_is_measured)
from .device import (DeviceChoice, choose_output, describe_devices,
                     is_silent, list_output_devices)
from .engine import SourceSlot, SpatialEngine
from .geometry import (Listener, distance_gain, front_back_gain,
                       relative_spherical)
from .hrtf import HrirSet
from .i18n import language
from .motion import (ROLL_PRESETS, WOBBLE_PRESETS, HeadWobble, roll_labels,
                     wobble_labels)
from .player import SpatialPlayer
from .presets import (BUILTIN_PRESETS, RoomIr, RoomPreset, describe_presets,
                      load_all, load_room_ir, preset_names,
                      preset_reverb_spec, render_room_ir)
from .early import EarlyReflections
from .reverb import (BandFdnReverb, ConvolutionReverb, FdnReverb,
                     VectorFdnReverb, db_to_gain, gain_to_db,
                     reverb_from_spec, reverb_spec_problems, rt60_of)
from .sources import (BufferSource, CallbackSource, PushSource,
                      StereoBufferSource, is_stereo_file, load_flat,
                      load_mono, load_stereo, resampled_by)

__all__ = [
    "__version__",
    # 表示の言語（ja か en。環境変数 AUDIOSPACE_LANG か OS の表示言語で決まる）
    "language",
    # high level
    "SpatialPlayer",
    # engine
    "SpatialEngine", "SourceSlot",
    # hrtf / geometry
    "HrirSet", "Listener", "relative_spherical", "distance_gain",
    "front_back_gain",
    # 既定の HRTF（聴取テストで選定したもの）
    "default_hrir", "default_is_measured", "attribution", "DEFAULT_PINNA",
    # 動的な方向手がかり
    "HeadWobble", "WOBBLE_PRESETS", "wobble_labels",
    "ROLL_PRESETS", "roll_labels",
    # convolution
    "BinauralConvolver", "MonoConvolver",
    # reverb
    "FdnReverb", "VectorFdnReverb", "ConvolutionReverb", "BandFdnReverb",
    "rt60_of", "db_to_gain", "gain_to_db",
    # 材質つきの空間（初期反射と材質から決まる後期残響）
    "Space", "ReflectionPath", "SPACE_PRESETS", "space_preset",
    "space_labels", "material_names", "absorption", "band_centers",
    "SPACE_KINDS", "FACE_LABELS", "material_label", "material_choices",
    "space_from_spec", "space_to_spec", "space_spec_problems",
    "reverb_from_spec", "reverb_spec_problems", "preset_reverb_spec",
    "FURNISHINGS", "Furnishing", "FURNISH_LEVELS", "FACE_MATERIALS",
    "furnishing_labels", "furnish_counts", "contents_text", "reverb_text",
    "EarlyReflections",
    # rooms
    "RoomPreset", "RoomIr", "BUILTIN_PRESETS", "preset_names",
    "describe_presets", "render_room_ir", "load_room_ir", "load_all",
    # sources
    "BufferSource", "StereoBufferSource", "PushSource", "CallbackSource",
    "load_mono", "load_stereo", "load_flat", "is_stereo_file",
    "resampled_by",
    # devices
    "DeviceChoice", "choose_output", "list_output_devices",
    "describe_devices", "is_silent",
]
