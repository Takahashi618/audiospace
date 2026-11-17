# audiospace

A binaural 3D audio library for games and tools for blind and visually
impaired people. (日本語の説明は [README.md](README.md) にあります。)

Licensed under MIT (Copyright (c) 2026 Takahashi; full text in `LICENSE`).
You are free to use it in games, sell it, and use it in research. All
dependencies are permissively licensed as well (numpy / scipy, and
optionally sofar, pyroomacoustics and sounddevice, all MIT or BSD), so it
can go into **software that you sell without having to disclose your
source**. Only the default HRTF is CC BY 4.0 and needs a credit
(`ATTRIBUTION.en.md`).

It was made to have a sound component that does not depend on OpenAL, is
under the MIT License, and lets you choose an HRTF that suits your ears
(OpenAL Soft is LGPL and has few HRTF profiles).

## Display language

Text meant for people (names of spaces, materials and furniture, status
lines, error messages, and everything in the Localization Tool) is
**Japanese when the operating system's display language is Japanese, and
English otherwise**. On Windows this is the display language
(`GetUserDefaultUILanguage`); on macOS the first preferred language; elsewhere
`LC_ALL`, `LC_MESSAGES` or `LANG`.

To choose, set the environment variable `AUDIOSPACE_LANG` to `en` or `ja`
before starting the program. Lists of names such as `FURNISH_LEVELS` and
`FACE_LABELS` are built when audiospace is imported, so the language cannot
be switched while the program is running. `audiospace.language()` returns
the language in use (`"ja"` or `"en"`).

## How this was made

The code of audiospace was written by an AI model (Claude, by Anthropic).
The author did not write the code. The author's part was:

- Deciding what to build: 3D audio that blind and visually impaired people
  can use in games and tools, independent of OpenAL, and verifiable with a
  keyboard and a screen reader alone.
- Listening tests on headphones. The author chose the default pinna by
  comparing measured pinnae by ear. Front-back confusions noticed in the
  first listening session led to the three aids: head wobble, training and
  pinna selection.
- Checking the behaviour and deciding what to fix.
- Publishing, answering questions and maintenance.

The code is checked by 304 automated tests and by the author on real
hardware, but like any code it may still contain mistakes. It comes without
warranty.

Under current guidance in Japan and the United States, parts written by an
AI may not be protected by copyright. Either way, audiospace is free to use
under the MIT License.

## Distributions (wheel and zip)

There are two. Neither contains the author's listening records (`logs/`)
or working notes.

- **wheel** (`audiospace-<version>-py3-none-any.whl`, about 1.8 MB): for
  people who only use the library. It contains `audiospace/` with the
  default HRTFs (horizontal and full sphere), `LICENSE` and `ATTRIBUTION.md`. The 21 measured HRTF SOFA
  files and the baked reverb IRs are not included
- **zip** (`audiospace-<version>.zip`, about 70 MB): the full source. The
  library, the listening app (`demo/`), the tests, the tools, the 21 SOFA
  files, 7 reverb IRs and 7 baked space reverbs (`assets/space_ir/`). Get
  this one if you want to choose a different pinna or use it in research.
  It also has the user documents in English (`README (English).txt`,
  `Manual (English).html` and the same content as `Manual (English).txt`)
  and Japanese, and the launcher `Start Localization Tool.bat`

Installing the wheel (use the real file name):

```
python -m pip install audiospace-0.8.2-py3-none-any.whl
python -m pip install "audiospace-0.8.2-py3-none-any.whl[playback]"
```

The second line is for playing on a real device. With the wheel alone there
are no baked reverb IRs, so `use_room_ir()` computes the room at run time
(this needs pyroomacoustics, installed by `[rooms]` or `[full]`). If you keep
the zip's `assets/reverb/` at hand, `load_room_ir(name, rate, directory=...)`
loads them.

With the zip, run this in the extracted folder:

```
python -m pip install -e .[full]
```

The tables and numbers in the manuals (the list of keys, the settings, the
number of questions, the lists of spaces, reverbs and materials, the
version) are rebuilt from the code of the app and the library by `python -X
utf8 tools\build_manual.py`, in Japanese and in English (each language from
the app running in that language). It also makes the txt editions and the
read-me files (from `tools/readme_first.txt` and `tools/readme_first_en.txt`).
With `--check` it only reports whether they are out of date.

## Installing

For development in this folder:

```
python -m pip install -e .
```

The only required dependencies are numpy and scipy. Add `.[playback]`
(sounddevice / soundfile) to play on a device, or `.[full]` for everything.
For the Localization Tool (`demo/`), use `.[app]` (wxPython /
accessible_output2 / sounddevice / sofar). The default HRTF
(`audiospace/data/default_horizontal.npz`, about 205 KB) and the full sphere
of the same pinna (`audiospace/data/default_sphere.npz`, about 1.4 MB) are
package data inside the wheel, so they are bundled when you freeze the
package with PyInstaller. **A CC BY 4.0 credit is required when you distribute it** (see
the text of `attribution()` and `ATTRIBUTION.en.md`).

## Usage

### High-level API

```python
from audiospace import SpatialPlayer

player = SpatialPlayer.open(head_wobble="clear")  # picks WASAPI automatically
h = player.play("beacon.wav", position=(3.0, 4.0, 0.0), loop=True)
player.play_ui("menu_move.wav")                   # UI sounds stay centred
player.set_listener((0.0, 0.0, 0.0), yaw_deg=90)  # face left
player.set_sound_gain_db(h, -6.0)                 # gain in dB
player.stop_sound(h, fade_ms=30)                  # stop with a fade-out
player.pause(); player.resume()                   # pause and resume everything
print(player.status())                            # status line to read aloud
player.close()                                    # finish (or use `with`)
```

Files passed to `play()` are decoded once and cached, so the same sound
effect never reads the disk again. Finished one-shot sounds are collected by
the next `play()`, so you do not need to worry about running out of slots.
To pause a single sound, use `pause_sound(h)` / `resume_sound(h)`.

**Long sounds are read a little at a time** (`stream`). Decoding a whole
three-minute 48 kHz stereo track takes about 66 MB. Files too large for the
cache (`sound_cache_max_bytes`, 16 MB by default: about 44 s of 48 kHz
stereo) are turned into a `FileStreamSource` by `play()` automatically,
which plays while a thread reads two seconds ahead. With a one-minute track,
`play()` went from 488 ms to 7 ms and memory from 46 MB to 0.8 MB (measured).
A short sound already in the cache is played as it is, without opening the
file to check it (a footstep played hundreds of times never reads the
disk). `stream=True` works only with a file (on an array or a source object
it raises ValueError).
While a file is streamed it stays open. When you are done with the player,
call `close()` (leaving a `with` block does the same): it releases every
sound and closes the files, so they can then be deleted or replaced.
`stop()` only stops the output and keeps the sounds (`start()` goes on from
there).

```python
player.play("bgm.ogg", spatialise=False, loop=True)   # streamed if it is long
player.play("bgm.ogg", spatialise=False, stream=True)  # streamed even if short
player.play("voice.wav", stream=False)                 # decoded whole even if long
```

- It reads wav, flac, ogg and mp3 (soundfile, or miniaudio if that fails).
  If the file's rate differs, samplerate converts it a little at a time (the
  values match converting it whole). Without samplerate, a file at another
  rate is decoded whole (with `stream=True`, ValueError)
- `loop_start` / `loop_end` (in the file's frames) loop without a gap. It
  cannot be combined with a `rate` other than 1 (that is decoded whole)
- The first 0.5 s is read inside `play()`, so the start never drops out. If
  reading falls behind, that block is silent and `status()` says "silent N
  blocks because a file was not read in time". The read-ahead is
  `player.stream_buffer_seconds` (2 s by default)
- The file stays open while it plays. If it cannot be read any more, only
  that sound is taken out (its end notice is `SOUND_ERROR`)
- In the low-level API, pass `FileStreamSource(path, engine rate,
  channels=1 or 2)` to `add_source`. `audio_file_info(path)` gives the
  frames, rate and channels without decoding

```python
player.set_sound_lowpass(h, 0.7)              # behind a wall (muffled)
player.set_sound_gain(h, 0.0, ramp_ms=500)    # fade out over 500 ms
player.play("step.wav", rate=1.05)            # 5% faster and higher
player.play("bgm.wav", spatialise=False, loop=True)   # keep it stereo
```

**Occlusion (`set_sound_lowpass`)** signals "on the other side of a wall".
Lowering the volume alone sounds like distance, so it cuts the highs. It
combines with the darkening of sounds behind (a sound behind and through a
wall is darker than either alone). Values ramp over 60 ms by default, so the
tone does not jump when it changes at every step.

**Gain ramps** (`ramp_ms`) run in the audio thread. Even if the UI thread
stalls in a modal loop or while dragging a window, the fade still ends on
time. Write a crossfade as two ramps.

**Playback rate** (`rate`) changes pitch and tempo together. Use it for
varied footsteps or pitched cues. Different rates share the same cache.

**Start time** (`delay_s`) starts a sound that many seconds later, to the
sample. A footstep pattern, a rhythm cue or an echo needs no timer in the
game. Sounds whose offsets must stay exact go inside
`with player.together():`. Sounds played inside wait until the with-block
ends, and all count from the same moment.

```python
for i in range(4):                                # four steps, 0.4 s apart
    player.play("step.wav", position=(0.0, 2.0 + i, 0.0), delay_s=0.4 * i)

with player.together():                           # direct sound and floor reflection
    player.play("bell.wav", position=(0.0, 4.0, 0.0))
    # with the floor 3 m below the ears, the reflection comes from the mirrored
    # position, late by the 3.2 m longer path
    player.play("bell.wav", position=(0.0, 4.0, -6.0), gain=0.6,
                delay_s=0.0094)
```

- Without `together()`, an audio block may come between two `play()` calls,
  and only the later sound starts one block late (about 11 ms at 512
  frames). Reflections and echoes work on offsets shorter than a block, so
  the error is easy to hear
- If an exception is raised inside `together()`, the sounds played inside
  are dropped unplayed
- A waiting sound holds a source slot. Stopping it with `stop_sound()`
  removes it at once, even with a fade
- While a sound waits, its gain and muffling ramps do not run.
  `set_sound_gain(h, 1.0, ramp_ms=3000)` called while it waits makes it
  grow over 3 seconds from when it starts. While it is paused with
  `pause_sound()`, or everything is paused with `pause()`, the remaining
  wait does not shrink either
- Delayed or not, a sound starts exactly as its data starts (see "A sound
  starts as its data" below)

**Sound groups** (`group`) give a name to a set of sounds, such as effects,
music or voice, and set the volume, pause, stop and ducking for the whole
set. A group's volume multiplies each sound's own gain, distance and fades.
An "effects volume" on a settings screen is one call, without changing the
playing sounds one by one.

```python
player.set_group_gain("sfx", 0.7)                 # effects volume (also before playing)
player.play("step.wav", position=(1.0, 2.0, 0.0), group="sfx")
player.play("bgm.ogg", spatialise=False, loop=True, group="music")
player.set_ducking("music", by="voice", db=-12)   # lower the music while a voice plays
player.play("line01.wav", spatialise=False, group="voice")
player.pause_group("music"); player.resume_group("music")
player.stop_group("sfx", fade_ms=30)              # stop all effects
```

- `set_group_gain(name, gain, ramp_ms=)` and `set_group_gain_db`. The ramp
  runs on the audio thread, as with `set_sound_gain`. `group_gain(name)`
  returns the volume set (where a ramp is heading). A name not used yet has
  volume 1
- `pause_group` / `resume_group`: stops the sounds in place, and on resume
  they continue (the first block after resuming fades in). A sound paused
  on its own with `pause_sound()` stays paused when the group resumes. While
  the group is paused, the wait of a delayed sound does not shrink either
- `stop_group(name, fade_ms=)` returns the handles it stopped.
  `group_sounds(name)` lists the sounds now in the group, and
  `set_sound_group(h, name)` moves a sound to another group (None for no
  group)
- **Ducking** (`set_ducking(name, by=, db=-12, attack_ms=80, release_ms=400)`):
  while any sound of the `by` group plays, the group is lowered by `db` (a
  push source from `open_stream_source` counts as playing only while
  samples written to it are left; while nothing is written, it does not
  duck. A `CallbackSource` does not duck while its function returns
  silence). It
  takes `attack_ms` to go down and `release_ms` to come back. It is a factor
  separate from the volume set with `set_group_gain`, so the player's
  volume setting is kept. `by` may also be several names. `clear_ducking`
  removes it
- For a voice that does not go through the engine, such as a screen
  reader, lower the group by hand with `hold_ducking("music", True)` and
  `hold_ducking("music", False)` (set the depth and speed first with
  `set_ducking("music", db=-12)`)
- Sounds in no group sound exactly as before, and cost the same

**End notices** (`on_end`) call a function once when a sound has ended. Use
them to play the next line when a line of speech ends, or to move on when
an effect is over. The function gets a `SoundEnd` (`handle`, `reason`,
`detail`, `at_s`).

```python
from audiospace import SOUND_FINISHED

def next_line(event):
    if event.reason == SOUND_FINISHED:           # it played to the end
        player.play("line02.wav", spatialise=False, group="voice")

player.play("line01.wav", spatialise=False, group="voice", on_end=next_line)

# call from the game's loop or timer (every 20 to 50 ms or so)
player.poll_ends()
```

- `reason` is `SOUND_FINISHED` (the data played to the end),
  `SOUND_STOPPED` (stopped with `stop_sound`, `stop_group` and so on; with a
  fade, once it has faded out) or `SOUND_ERROR` (taken out after an error in
  the source; the reason is in `detail`)
- The function is never called on the audio thread. It runs on the thread
  that calls `poll_ends()`, so when you call that from a GUI timer, the
  function may touch the widgets
- Without a loop of your own, use `player.watch_ends()`. It delivers on a
  thread of the player's. To hand over to a GUI thread, use
  `watch_ends(dispatch=wx.CallAfter)` (called as `dispatch(function,
  notice)`). `stop()` ends it
- If a function raises, the other functions are still called. The count
  and the last reason are in `status()`
- If you pass `on_end` but call neither `poll_ends` nor `watch_ends`, the
  functions are never called. When notices older than 2 seconds are left,
  `status()` says so ("end notices not delivered")
- In the low-level API: `add_source(..., notify_end=True)` and
  `take_ended()`

The end of the engine has a **limiter** (on by default). It lowers only
the blocks that would go over full scale when many sources overlap, so there
is no harsh clipping on headphones. How often it acted appears in
`status()` as the limiter count. To turn it off:
`SpatialEngine(..., limiter=False)`.

If you do not give an HRTF, the **measured HRTF chosen in the listening
tests** (Viking pinna E, horizontal plane only, 205 KB) loads automatically.
Without the asset it falls back to the synthetic one.

To render sounds above and below too, use `full_sphere=True` for the **full
sphere** of the same pinna (1513 directions, from 45 degrees below to straight
up, about 1.4 MB). It is the SOFA measurement saved as npz unchanged, and it
loads without sofar. If the full-sphere asset is missing, it raises
`FileNotFoundError` (silently using the horizontal or synthetic one instead
would hide that up and down cannot be heard). Pinna E was chosen in the
front/back tests, and the pinna that suits up and down differs from person
to person (`CUES.en.md`). If it does not suit you, choose another pinna with
`sofa_path`.

```python
from audiospace import default_hrir, default_is_measured, attribution

hrir = default_hrir(48000)      # already converted to the output rate
default_is_measured()           # measured, or fell back to synthetic?
print(attribution())            # credit to show when distributing (CC BY 4.0)

player = SpatialPlayer.open(full_sphere=True)     # open with the full sphere
sphere = default_hrir(48000, full_sphere=True)    # load only the full sphere
print(attribution(full_sphere=True))              # its credit text
```

### Low-level API (no device)

```python
from audiospace import SpatialEngine, HrirSet, BufferSource

engine = SpatialEngine(HrirSet.synthetic(sample_rate=48000), block=512)
handle = engine.add_source(BufferSource(samples, loop=True),
                           position=(1.0, 2.0, 0.0))
stereo = engine.render()          # returns (512, 2) float32; use it as you like
```

To feed your own blocks, use `PushSource`.

```python
handle, push = player.open_stream_source(position=(0.0, 5.0, 0.0))
push.write(my_generated_samples)  # ring buffer; no need to match timing
```

## Coordinates

Right-handed, in metres. x = east, y = north, z = up. The listener's
`yaw_deg` is counterclockwise from north. `pitch_deg` turns the face up and
down, positive is up (applied after the body's `yaw_deg`, around the
left-right axis).

`roll_deg` tilts the head, positive lowers the left ear toward the left
shoulder (applied after the pitch, around the front-back axis).

```python
player.set_listener(pitch_deg=30)    # face 30 degrees up; position and yaw unchanged
player.set_listener(roll_deg=10)     # lower the left ear by 10 degrees
```

Like head wobble, head tilt can sway automatically. Sounds above and below
move left and right in opposite directions while level sounds in front do
not move, which gives an up/down cue that does not depend on the listener's
ears (with Viking pinna E, the left-right movements of sounds 45 degrees up
and 45 degrees down have a correlation of -0.997).

```python
from audiospace import ROLL_PRESETS
player.engine.set_roll_wobble(ROLL_PRESETS["clear"])   # 12 degrees
```

With `pitch_deg=0` and `roll_deg=0` (the default) the computation is
bit-for-bit the same as before these arguments existed. A sound outside the
measured range of a measured HRTF (below 45 degrees down for Viking) plays
with the nearest measured direction, so it will not sound exact.

Azimuth and elevation follow the **SOFA convention** (azimuth 0 = front,
counterclockwise, 90 = left), so measured HRTFs read from .sofa files need
no conversion.

## Silent mode

Setting the environment variable `AUDIOSPACE_SILENT=1` means no device is
ever opened. `SpatialPlayer` runs the engine on a timer thread and discards
the output, so callers need no separate code path for tests.

```
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_tests.py
```

## Reverb

Two methods share one interface.

```python
player.set_reverb("hall")            # light FDN; no assets needed
player.use_room_ir("hall")           # convolve with a baked IR
player.set_reverb(None)              # no reverb
```

`use_room_ir(name)` reads the baked IR from `assets/reverb/` (converting the
rate if needed), so pyroomacoustics is not needed at run time. Only when no
baked IR exists does it fall back to computing one (a few seconds). Bake the
IRs in advance:

```
python -X utf8 tools\build_reverb_presets.py --rate 48000
python -X utf8 tools\build_reverb_presets.py --list
```

There are 7 presets: bathroom / small_room / corridor / hall / cave /
warehouse / cathedral. Their measured RT60 ranges from 0.18 seconds
(bathroom) to 5.37 seconds (cathedral).

## Spaces with materials (early reflections and material reverb)

The same footstep sounds different in a wooden room, a concrete room, an
empty road and a street between buildings. `Space` creates that difference:
a box-shaped space with a material for the floor, the ceiling and each of
the four walls.

```python
player.set_space("wood_floor_concrete_walls")   # by the name of a provided space
player.set_space(Space.room((5, 4, 2.7), floor="wood_16mm",
                            walls="rough_concrete", ceiling="plasterboard"))
player.set_space(Space.street(width=12, facades="brickwork"))   # between buildings
player.set_space(None)                                          # remove it
```

A space adds two kinds of sound.

- **Early reflections with direction**: the distance from the listener to
  each surface sets the time, direction and level of each reflection, and
  the surface's material colours it. Standing by a wall, you hear its
  reflection from that side; between two facing surfaces, reflections
  bounce back and forth (the flutter echo of a street between buildings).
  Open surfaces (`"open"`, the sky or the road running on) return nothing
- **Late reverb from the materials**: the reverb time of each band comes
  from the materials' absorption by Eyring's formula (including air
  absorption), and an FDN that decays at a different speed in each band
  (`BandFdnReverb`) plays it. It is computed in an instant, so it switches at
  once when the place changes during a game. Open outdoor places get no
  diffuse reverb

There are 91 materials from the pyroomacoustics material database
(`material_names()`, 7 bands from 125 Hz to 8 kHz) plus `"open"`. You can
also pass 7 absorption values directly. The provided spaces are in
`space_labels()`: a room with wooden floor and walls, a room with wooden
floor and concrete walls, a concrete room, a room with carpet and an
absorbing ceiling, a tiled bathroom, a concrete corridor, a stone hall, an
empty road, a street between buildings and a narrow alley.

Computed reverb times at 1 kHz (a 5 m × 4 m × 2.7 m room): carpet 0.27
seconds, wooden walls 0.81 seconds, concrete walls 1.99 seconds, all
concrete 3.21 seconds. They agree with pyroomacoustics' precise computation
(image method and ray tracing) to about 20 percent.

Notes:

- Reflections are computed from the listener's position (not for each
  source). This approximation is widely used in game audio engines; the
  cost does not depend on the number of sources (at most 24 reflections).
  In exchange, fine differences in the reflections of sources far from the
  listener are not represented
- The level of the early reflections does not depend on the reverb send
  (`reverb_send`), to keep their ratio to the direct sound physical (about
  -6 to -10 dB at 1 m from a wall)
- `set_space` replaces the reverb (the setting of `set_reverb` /
  `use_room_ir` is removed)
- Processing time (performance core, 128 sources): 2.2 ms with the hall IR
  reverb, 2.9 ms with a space. On an efficiency core, the 99th percentile of
  128 sources plus a space is right at the budget (10.2 to 10.5 ms).
- The decay of each band is approximated with two shelves (low and high), so
  the reverb time may differ from the target by 20 to 30 percent (1.8
  seconds against a target of 1.35 seconds at 4 kHz in the concrete room)

### Precise reverb (`late="precise"`)

```python
player.set_space("concrete_room", late="precise")
print(player.engine.late_status)     # e.g. "precise reverb", to read aloud
```

Replaces the late reverb of a closed room with a tail computed precisely by
pyroomacoustics (7 bands of material for each surface, scattering, air
absorption, image method and ray tracing). The early reflections with
direction stay in the engine (so they follow the listener's position and
facing); the tail is used from after the reflections have mixed (50 to 150
ms onward), with smooth joins at both ends. The tail is scaled so that its
energy in the band where footsteps and voices sit (100 to 4000 Hz) equals
the fast version's, so switching hardly changes how loud the reverb is (in
the built-in rooms the difference is -2.0 to +1.4 dB at 100 to 1000 Hz and
-0.6 to +0.4 dB at 1 to 4 kHz, measured). A tail saved by an earlier version
is brought to the same scaling when it is loaded.

It differs from the fast version (Eyring's formula) in rooms where the
absorption is uneven. In a room with a carpet floor and an absorbing ceiling
but reflective walls, the fast version gives 0.27 seconds and the precise
version 0.66 seconds (only the precise computation can show the sound
bouncing on between the walls).

- The 7 provided closed rooms are baked already (`assets/space_ir/`, 8.4 MB
  in all) and ready at once (`tools/build_space_irs.py` rebuilds them)
- Other rooms are computed in a **separate process** the first time they are
  used (a few seconds to 20 seconds); meanwhile the fast version plays, and
  it switches when ready. The result is saved in `engine.space_ir_cache` (by
  default under the temporary folder) and used at once next time. Computing
  in a thread of the same process stalled the sound for up to 12.8 seconds,
  because pyroomacoustics holds Python's GIL; in a separate process, the
  slowest block while computing was 3.7 ms
- **Spaces with open sides (a street between buildings, an alley, an empty
  road) keep the fast version.** Rays escape through the open sides, the rest
  keep bouncing between the walls, and the tail stops abruptly where the
  computation is cut off (in the street between buildings, it dropped 30 dB
  within 10 ms at 0.9 seconds)
- Without pyroomacoustics, or in a frozen exe, only the baked rooms get the
  precise version. Others play the fast version, and `late_status` says so
- Processing time: about 0.6 ms per block with a 6-second tail (concrete
  room); the fast version takes 0.09 ms

### Designing spaces and reverbs (designs)

Spaces and reverbs can also be made from a dictionary (a design) that can be
written as JSON as it is. You can keep the sound of each map of a game in a
file, or let users design and save their own.

```python
from audiospace import space_from_spec, reverb_from_spec, space_to_spec

room = space_from_spec({
    "kind": "room",                       # room / street / open
    "size": [6.0, 5.0, 2.8],              # width, depth, height (m)
    "materials": {"floor": "wood_16mm", "ceiling": "plasterboard",
                  "north": "glass_window", "south": "wooden_door",
                  "west": "brickwork", "east": "curtains_velvet"},
    "at": [0.4, 0.4],                     # listener position (fraction of width and depth)
    "early_order": 4, "early_gain": 1.0, "late_gain": 1.0})
player.set_space(room)

player.set_reverb(reverb_from_spec(
    {"rt60": 2.5, "room_scale": 1.2, "damping": 0.3, "width": 1.0,
     "level": 1.2}, player.engine.sample_rate))
```

- The listener is taken to face north (front): north is the front wall,
  south the back, west the left and east the right. In `street` the front,
  back and sky are open; in `open` only the floor (ground) reflects
- Bad values raise `ValueError`. `space_spec_problems` /
  `reverb_spec_problems` return the problems as sentences in the display
  language (ready to be read aloud)
- To start from a provided space or reverb, use
  `space_to_spec(space_preset(name))` and `preset_reverb_spec(name)`. Both
  round-trip to the same space and the same sound
- The 91 materials have readable names in Japanese and English.
  `material_choices()` returns (name, "Category: readable name") in the order
  of the categories, ready for a choice list; `material_label(name)` gives
  one
- `FdnReverb` has a `level` (reverb level)

### Adding furniture and people (`contents`)

Even in a room of the same size, furniture and people change the sound.
Give a space "contents" as a count of each kind, and the sound changes in two
ways.

- Absorption: the absorption of each thing (m², 7 bands) is added to the
  reverb time computation (Eyring). The more things, the shorter and quieter
  the reverb
- Clutter: the area the things block divided by the floor area. It weakens
  reflections. Furniture stands near the floor, so it weakens floor
  reflections most, walls moderately and the ceiling only a little, and high
  sounds more than low ones

```python
from audiospace import space_preset, furnish_counts
room = space_preset("wood_room")
furnished = room.with_contents({"sofa": 1, "table": 1, "person": 4})
player.set_space(furnished)                 # switches smoothly
# counts to suit the floor area, from an amount (empty / sparse / normal / full)
counts = furnish_counts(room.size[0] * room.size[1], "normal")
```

- Things you can place (`FURNISHINGS`): standing person, seated person,
  sofa, upholstered armchair, wooden chair, table, full bookshelf, bed,
  wooden cabinet, cardboard box. The values are estimates based on typical
  figures from the literature, meant to change the sound broadly (where
  things stand is not modelled)
- Reverb time at 1 kHz in the wooden room (5 × 4 × 2.7 m): empty 0.81
  seconds, normal furniture 0.64 seconds, plus 10 people 0.45 seconds. Floor
  reflections weaken by about 1.6 dB, walls by about 0.9 dB and the ceiling
  by about 0.4 dB
- In a design: `"contents": {"sofa": 1, "person": 3}`. When empty, the space
  is exactly the same as before (neither the output nor the key of the
  precise version changes)
- Precise reverb (`late="precise"`) handles only the materials of the
  surfaces, so a space with things in it uses the fast version
  (`late_status` gives the reason)
- `material_choices(face)` puts the materials that suit the surface (floor /
  ceiling / walls; `FACE_MATERIALS`) first, and the rest after them with
  "Other" in front

### Moving between places (crossfading the sound of spaces)

When you switch spaces, the sound changes smoothly.

- The reverb of the place you left stops taking new sound and keeps ringing
  until it dies away naturally (after leaving a room, you hear its reverb
  fade). Pass `release_s` to cut it after that many seconds
- The input to the reflections and reverb crossfades from the old place to
  the new one over `transition_s` (0.3 seconds by default). 0 switches at
  once and clicks (the sample jump is about 7 times the steady state; with the
  crossfade it is the same as the steady state)

Register a space for each place, and it switches automatically by the
listener's position.

```python
room = Space.room((5, 4, 2.7), floor="concrete_floor", walls="rough_concrete",
                  ceiling="rough_concrete", listener=(0, 0, 0), at=(0.5, 0.5))
street = Space.street(width=12, length=80)
road = Space.open_ground()
player.set_spaces([road, street, room])     # then just move the listener
```

Of the spaces whose range (box) contains the listener, the smallest one is
chosen (entering a room placed inside a road gives the room's sound). It
does not move to another space until the listener leaves the current one
(except when entering a smaller space), so it does not flip back and forth at
a boundary. Calling `set_space` clears the registration.

To hear it, run `python -X utf8 demo\space_walk.py`. It walks from a
concrete room through a street between buildings to an empty road, clapping
in each place, and writes the result to `demo/out/space_walk.wav` (no device
is opened).

### Sounds through walls (automatic occlusion)

Using the spaces registered with `set_spaces` (or, with no registration, the
space given to `set_space`), a sound behind a wall is muffled and lowered
automatically. With neither, nothing changes, so `status()` says "there is no
space". When the line from a source to the
listener passes through a closed face (wall, floor, ceiling), that face's
material decides how much.

```python
from audiospace import Opening

kitchen_door = Opening((4.9, 2.0, 0.0), (5.1, 2.9, 2.0), name="kitchen_door")
player.set_spaces([living, kitchen, hall], early=True, late=True)
player.set_auto_occlusion(openings=[kitchen_door])
player.set_opening("kitchen_door", "wooden_door")   # close the door
player.set_opening("kitchen_door")                  # open it
print(player.sound_occlusion(h).walls)              # materials passed through
```

- How much it drops (with the muffling included, measured with broadband
  noise): about 22 dB through concrete, brick or stone, 14 dB through a
  wooden door or wall, 11 dB through plasterboard, 10 dB through glass and 3
  dB through a curtain. That is far less than real sound insulation (about 50
  dB for concrete). Hearing a muffled sound through a wall is itself a cue,
  so it stays audible. The values are per material category
  (`audiospace.OCCLUSION_BY_CATEGORY`, `material_occlusion(name)`).
  `materials={"wooden_door": (6.0, 0.4)}` overrides one material, and
  `strength=` scales the whole effect
- Faces in nearly one plane (within 0.5 m), such as the walls of two
  neighbouring rooms, count as one wall (the stronger value). With one room
  in between, it drops by two walls
- Doors, windows and stairwells are holes cut with `Opening` (a box around
  the hole). Place the box so that it covers the wall's thickness (so the
  faces of the rooms on both sides are inside it). A line through an open
  hole is not blocked. With a `material`, it is a closed door or window;
  `set_opening` opens and closes it. A whole face set to "open" passes sound
  like a hole
- **Going round**: when the straight line is blocked by walls, it looks for
  a way through open faces and holes. If that is louder, the sound comes from
  the direction of the listener's doorway, at the length of the way round.
  The sharper the turn, the lower and more muffled. A sound round the corner
  of a corridor is heard from the room's doorway. When there is both a short
  way through a closed door and a way round through open doorways, the
  louder one is used
- It is occluded from its very start (a sound starting behind a wall does
  not seem to open up and then close). When a source or the listener moves,
  it changes smoothly, at 30 dB per 100 ms. After `teleport_listener` it takes
  the new value at once (nothing of the old place is left)
- It combines with the muffling set with `set_sound_lowpass` (it does not
  replace it). Sounds played with `play(..., occlude=False)` are never
  occluded (a guide beacon, say), nor are sounds that move with the listener
  (`relative=True`) and flat sounds
- Off by default; when off, everything sounds as before. With 64 sources
  and a listener walking all the time, a block takes 1.8 ms instead of 0.98
  ms (up to 16 sources are rechecked per block; the rest in the next one)

## Telling front from back (important)

A list of the processing applied to sources as front/back and up/down cues
(with numbers and where in the code) is in [CUES.en.md](CUES.en.md).

### What happens

With someone else's HRTF on headphones, **front-back confusions almost
always happen**. This was confirmed by listening: people learn to hear the
difference with practice, but you cannot expect it from a first-time player.
Pinna shapes differ greatly between people, and the brain judges front and
back by the spectral features of its own pinnae, so someone else's pinnae
give mismatched cues.

### Aid 1: head wobble (the most reliable)

When the head moves slightly, sounds in front and sounds behind always move
in opposite directions. With a sound in front, turning the head left moves
the sound to the right relative to the head; with a sound behind, it moves
the other way. Measurement showed a **correlation of -0.999** (almost exactly
opposite). This was measured on the synthetic HRIR, where front and back are
numerically identical, so it **works regardless of individual pinnae**.

```python
from audiospace import WOBBLE_PRESETS

player.engine.set_head_wobble(WOBBLE_PRESETS["clear"])   # 8 degrees, recommended
player.engine.set_head_wobble(amplitude_deg=6.0)          # custom
```

4 to 10 degrees is enough. Larger amplitudes make the sound image sway
unpleasantly. If the game lets players turn freely, that is best and head
wobble is not needed.

### Aid 2: training

People learn to hear the difference, so give them the chance.
`TrainingSession` in `audiospace/quiz.py` uses a weighted draw that brings back
directions answered wrongly more often, and weighs front-back confusions
especially heavily. Right after a wrong answer it plays the correct direction
again so the player can compare.

### Aid 3: choosing a pinna

Viking has 20 pinnae. Which one suits a listener depends on the person, so
`CalibrationSession` lets them compare by ear. Switching does not stop the
sound.

```python
player.engine.set_hrir(other_hrir)     # swap without stopping playback
```

#### Choosing a pinna inside a game (`PinnaSelection`)

The questions and scoring (`audiospace.quiz`) and the running totals per
pinna (`audiospace.records`) are part of the library and depend on neither
a GUI nor audio output. `PinnaSelection` wraps the same flow as the
listening app's "choose a pinna" (F8) in a form a game screen can use.

```python
from audiospace import (Difficulty, PinnaSelection, RecordStore,
                        load_pinna, viking_candidates)

candidates = viking_candidates("assets/hrtf/viking")    # [(tag, name)]
store = RecordStore(settings_dir / "pinna_records.json")
selection = PinnaSelection(player, candidates, store,
                           difficulty=Difficulty.EIGHT)
speak(selection.intro())                    # how many, the reference, progress
while not selection.finished:
    if selection.prepare():                 # switched to the next pinna
        speak(selection.round_text())       # "Pinna 2 of 3."
    selection.play(sound)                   # the question sound (also replays)
    selection.answer(wait_for_answer())     # direction as seen from you now
speak(selection.finish())                   # add to the totals, save, restore the HRTF

best = selection.best()                     # the best overall (tag, name)
if best is not None:
    player.engine.set_hrir(load_pinna(best[0], player.engine.sample_rate))
```

- One round compares a few pinnae (about 48 questions in all). **The current
  best is always included as the reference**, and the rest are pinnae not yet
  tested. Listeners improve over time, so only the difference from the
  reference within one round counts for the ranking. All of them can be
  tested over several rounds
- The question sound plays in the direction seen from the listener, at a
  position that moves with the listener. If the listener has turned, use
  `answer(direction, listener_yaw_deg=current_heading)`. For a replay, call
  `play` again (the previous sound stops); `note_replay()` counts it
- To choose a pinna for up and down, use `mode=MODE_ELEVATION` (answer the
  height at the front with `answer_elevation(elevation)`; it needs full-sphere
  SOFA data). Keep its totals in a separate file from the front/back ones
- To stop early, call `cancel()` (nothing is added to the totals, and the
  HRTF is restored). With `preload=True`, all pinnae of the round are loaded
  at the start (no waiting in the middle)
- The pinna data (SOFA) is not in the wheel. Ship `assets/hrtf/viking` from
  the zip with your game (CC BY 4.0; the credit text is "using the SOFA files
  unchanged" in ATTRIBUTION.en.md). Reading SOFA needs sofar. An npz baked
  with `tools/bake_hrtf_asset.py --pinna B` can also be a candidate

### Volume alone is not a cue

The engine has `front_back_depth` (0.35 by default), which low-passes
sounds behind. At first it only lowered the volume, but measurement showed
the frequency balance of front and back did not change (front 13.12 against
back 13.12), so it was no cue. A difference in volume cannot be told from a
difference in distance. With the low-pass, there is a 0.57 times difference
in clarity.

The volume and low-pass of sounds behind weaken away from the horizontal:
above, by cos(elevation) (about 0.71 times at 45 degrees up); below, they
keep the horizontal strength down to 45 degrees and weaken from there
(`min(1, √2·cos(elevation))`). Straight above and below there is no front or
back, so they have no effect (previously only the azimuth was used, and a
sound straight up given with azimuth 180 degrees came out quiet and
muffled). The lower side keeps its strength down to 45 degrees so that,
combined with the up/down emphasis below, a low sound behind does not end up
as bright as a level sound behind.

### Up/down emphasis (`elevation_emphasis`)

An artificial cue that makes higher sounds brighter and lower sounds darker.
The up/down differences of measured HRTFs vary greatly between pinnae and
from one azimuth to another, so this adds the rule "higher is brighter" at
every azimuth. Off (0) by default.

```python
engine = SpatialEngine(hrir, front_back_depth=0.35, elevation_emphasis=0.35)
player.set_emphasis(front_back=0.35, elevation=0.35)   # high-level API
```

- At strength 1, it changes the range above 3 kHz by +12 dB straight up,
  +6 dB at 45 degrees up and −6 dB at 45 degrees down (proportional to
  elevation; an RBJ high shelf). The elevation is as seen from the face, so it
  follows the pitch and roll of the head
- **Half of the change is offset by the overall volume.** Straight up, the
  highs get +6 dB and the lows −6 dB: the brightness changes but the loudness
  hardly does, because a difference in volume gets mixed up with a difference
  in distance
- It acts at once on sounds that are playing or moving. It is rounded to
  1 dB steps, and sources on the same step are processed together. There is
  no jump at the change of step (checked by tests)
- It can be combined with the front/back emphasis. Front/back is strongest
  at the horizontal and weakens up and down, while up/down is 0 at the
  horizontal and grows up and down, so they rarely cancel each other
- Cost (performance core, full sphere, all sources moving up and down too,
  strength 0.5): 64 sources +0.50 ms, 128 sources +0.56 ms, 256 sources
  +0.78 ms (`tools/bench_tilt.py`). At strength 0 there is no cost
- Listening records (up/down with four choices, mixed azimuths): without
  emphasis 42 to 46 percent, with 35 percent emphasis 88 percent (values from
  when the app baked it into the sound)

## Restricting to the horizontal plane for 2D games

A measured HRTF has 1513 directions over the full sphere, and its spectrum
table is 11.8 MB. Gathering rows from it every block misses the cache, which
was the main reason measured HRTFs were slow.

```python
hrir = HrirSet.from_sofa(path).horizontal()      # drop the elevations
hrir = hrir.decimated(15.0)                      # thin out the azimuths too
```

| | Directions | Table | 64 sources |
|---|---|---|---|
| Measured, full sphere | 1513 | 11.8 MB | 4.91 ms |
| Measured, horizontal only | 216 | 1.7 MB | **2.07 ms** |

The high-frequency left/right ratio of 284 times at 90 degrees is fully kept
in the horizontal-only set.

**After the speed-ups of 2026-10-16, the full sphere and the horizontal plane
cost almost the same** (3.4 ms and 3.2 ms with 128 sources; see "Measured
performance" below). The table above was measured before that. There is no
longer a need to restrict for speed; if you do not use elevation, the only
benefit left is smaller loading and memory. Measured on 2026-11-17 with 64
sources and head wobble, the bundled horizontal set (216 directions) and the
full sphere (1513 directions) both took about 0.9 ms per block. A 3D game
that uses up and down can simply open the player with
`SpatialPlayer.open(full_sphere=True)`.

## Measured performance (on this machine)

Windows / CPython 3.14.5 / Steinberg UR44C / WASAPI shared 44100 Hz / block
of 512 samples (budget 11.61 ms). Measured with every source moving every
block and the listener rotating continuously.

| Setup | Mean | Worst | Underruns |
|---|---|---|---|
| 8 sources + FDN | 1.56 ms (13.4%) | 2.66 ms | 0 |
| 32 sources + FDN | 2.27 ms (19.6%) | 3.30 ms | 0 |
| 64 sources + FDN | 3.24 ms (27.9%) | 4.04 ms | 0 |
| 8 sources + IR convolution (68073 taps) | 3.00 ms (25.8%) | 4.18 ms | 0 |
| 32 sources + IR convolution | 3.68 ms (31.7%) | 4.88 ms | 0 |
| 64 sources + IR convolution | 4.68 ms (40.3%) | 5.75 ms | 0 |

Convolution alone is faster still: 256 sources fit in 18.7 percent of the
budget. The difference from the numbers above is Python work in the mixer,
not the convolution.

### After the speed-ups (2026-10-16)

Engine only (no device), 48000 Hz, block of 512 (budget 10.67 ms), full
sphere measured HRTF, every source moving every block, head wobble on.
Measured with `tools/bench_polyphony.py`, alternating with the library from
before the change.

| Sources | Before, mean | After, mean | After review, mean | After review, 99% |
|---|---|---|---|---|
| 64 | 2.81 ms | 1.75 ms | 0.91 ms | 1.61 ms |
| 128 | 6.60 ms | 3.38 ms | 2.06 ms | 2.86 ms |
| 192 | — | 5.4 to 5.7 ms | 2.7 to 3.0 ms | 3.6 to 5.6 ms |
| 256 | 13.11 ms (over budget) | 8.0 ms | 4.5 to 5.3 ms | 5.3 to 7.0 ms (once 11.5 ms) |
| 384 | — | — | 7.2 to 8.2 ms | 9.0 to 11.1 ms (right at the budget) |

"After review" is after a code review the same day, which replaced the
optimised `einsum` path for the HRIR sum with plain `einsum` (for this shape,
the optimised path was 6 to 7 times slower).

- **Polyphony guide (with reverb): 128 sources is safe. Up to 256 sources if
  it runs on a performance core.** This PC (Core i7-12700K) has fast
  performance cores and slow efficiency cores, and when Windows moves the
  work to an efficiency core the time roughly doubles. Values with the
  process pinned to one kind of core (full sphere, all sources moving, head
  wobble on, sent to the hall IR reverb):

  | Setup | Performance core mean / 99% | Efficiency core mean / 99% |
  |---|---|---|
  | 128 sources + IR reverb | 2.32 / 3.25 ms | 5.12 / 8.67 ms |
  | 256 sources + IR reverb | 4.98 / 6.32 ms | 10.02 / 12.29 ms (over budget) |

  The values in the table above (cores not pinned) include this variation
- The work is proportional to the number of sources playing. A large
  capacity (`max_sources`) costs almost nothing for the slots that are idle
  (16 sounds playing with a capacity of 256: 9.2 ms → 0.74 ms)
- Four improvements went in: direction and volume are computed for all
  sources at once as arrays / only playing sources are convolved / only
  sources whose direction changed are computed twice for the crossfade / the
  FFT shrinks from 1024 to 640 when the HRIR fits in one block
- The output matches the old one within rounding error (peak ratio 3e-7,
  about -130 dB). However, **the first block (about 11 ms) of a newly started
  sound changed.** Before, that row crossfaded in from the direction it had
  last used (index 0 for a new row), so the start of a sound briefly came
  from the wrong direction. Now it starts in the right direction

### Faster reverb (2026-10-16)

`tools/bench_reverb.py` (no device, pinned to a performance core). Per block.

| Room (IR length) | IR reverb before | IR reverb after | FDN before | FDN after |
|---|---|---|---|---|
| hall (1.54 s) | 1.50 ms | 0.19 ms | 0.31 ms | 0.06 ms |
| cathedral (3.00 s) | 3.47 ms | 0.33 ms | 0.13 ms | 0.06 ms |

- The IR convolution reverb became about 10 times faster (on an efficiency
  core, cathedral went from 10.05 to 0.69 ms; before, its 99th percentile was
  20 ms, twice the budget on its own). Two changes: the sum uses plain
  `einsum` instead of the optimised path (which was about 8 times slower for
  this shape) / the spectra of past blocks are no longer shifted every time;
  the write position rotates instead (a ring buffer)
- The FDN reverb became 2 to 5 times faster. Reading and writing the delay
  lines now uses slices that wrap once at the end, instead of index arrays
  built every time
- **The sound did not change.** The FDN output is bit-for-bit the same. The
  IR convolution matches within rounding error (peak ratio 4e-7 or less,
  about -128 dB), and its response to a single click matches the room IR
  itself within 1.5e-8, checked for all 7 rooms

## Design notes

### Do not hard-code the sample rate

WASAPI shared mode does not resample in PortAudio, so a rate that differs
from the one set for the device in the Windows sound settings fails to open
with `Invalid sample rate [PaErrorCode -9997]`.

`SpatialPlayer.open()` reads the device's `default_samplerate` and builds the
HRIR at that rate, so normally you need not think about it. If you load an
IR yourself, pass it through `RoomIr.resampled(rate)`.

### The synthetic HRIR cannot tell front from back

`HrirSet.synthetic()` is a spherical head model: both the ITD and the head
shadow depend only on sin(azimuth). So **the HRIRs at 30 degrees and 150
degrees are numerically identical**. Front and back are conveyed by the aids
above.

For true front/back, elevation and externalisation, load measured data with
`HrirSet.from_sofa()`. With the included Viking data, pairs of positions
mirrored front to back differ by 86.7 percent on average in waveform (0.0
percent for the synthetic one).

### Stereo sources play as stereo only when not spatialised

With `spatialise=False`, a stereo asset plays as one source with 2 channels
(BGM, ambience, jingles). "Fake stereo" whose channels are identical is
folded to 1 channel automatically, so it does not mix two channels for
nothing.

On the other hand, passing a stereo source with `spatialise=True` raises
`ValueError`. Directions can only be given to mono, and silently folding to
mono would hide that it is not playing in stereo as intended, so it fails
explicitly.

### When the output device disappears

Unplugging USB audio or changing the default device kills the stream
silently. **Silence is the one failure a blind user cannot notice**, so let
the player watch the device and reconnect by itself.

```python
from audiospace import DEVICE_LOST, DEVICE_RATE_CHANGED, DEVICE_RESTORED

# about once a second, from the game loop or a UI timer
event = player.poll_device()
if event is not None:
    speak(event.message)                  # text to read aloud (current language)
    if event.kind == DEVICE_RATE_CHANGED:
        player = SpatialPlayer.open()     # a device at another rate: rebuild
```

When the device is lost, `poll_device()` reconnects with `reopen()` (playing
sounds are kept); if that fails, it keeps trying every
`player.device_retry_s` (3) seconds. It returns an event (`DeviceEvent`)
only when the state changes: reconnected (`DEVICE_RESTORED`), lost and not
back yet (`DEVICE_LOST`, only once however often it fails), or the new device
runs at another rate (`DEVICE_RATE_CHANGED`). If the original device comes
back, a later attempt reconnects and gives `DEVICE_RESTORED`.

To avoid writing the loop, `player.watch_device(on_event)` does the same on
another thread (`stop()` or `unwatch_device()` ends it). `on_event` is called
on the watching thread, so hand UI work over to the UI thread (`wx.CallAfter`
in wxPython). An app with a window is easier to write with `poll_device()`
from a UI timer (the Localization Tool works this way).

To build it yourself, check with `player.check_device()` and reconnect with
`player.reopen()`. When the replacement device runs at a different rate, it
raises `DeviceRateError` (a kind of `RuntimeError`).

### Broken sources and bad values

When the `read()` of a source you wrote raises an exception or returns an
array of the wrong shape, only that source is removed and counted
(`status()` reports "sources removed after errors" with the reason). The
other sounds keep playing.

Passing NaN or infinity as a position, gain, muffling, reverb send, or
listener position or angle raises `ValueError` at once. Accepting it would
turn the whole output into bad numbers while that source exists, and every
sound would go silent.

An audio file that cannot be read raises `DecodeError` (a kind of
`ValueError`), with the file name and the reason in the message.

### A sound starts as its data

A newly played sound plays from its first sample at the gain you set (no
fade-in). The attack of a footstep or a click reaches the ear as it is.

- Until 2026-11-18, the first block (about 11 ms) faded in from gain 0. A
  click that was a single first sample was silent, and the attack of other
  sounds was weakened
- Data cut in the middle of a waveform clicks at the start, as in any other
  player. Give the data a short fade, or play it with `gain=0.0` and raise it
  with `set_sound_gain(h, 1.0, ramp_ms=10)` (to leave no gap, make both
  calls inside `together()`)
- Gain changes, `stop_sound(h, fade_ms=)` and resuming after
  `pause_sound()` still change smoothly within a block, as before

### Source slots have a fixed cost

Every block processes `max_sources` rows. The processing time stays the same
whatever the number of playing sounds, so the worst case is easy to predict.
When the slots run out, it raises `RuntimeError` instead of silently dropping
the sound (a sound that never played is something a blind player cannot
notice).

## Tests

Run this in the audiospace folder:

```
python -X utf8 tests\run_all.py
```

It runs 5 suites, 304 tests in all (the silent setting is applied
internally). The test output is in Japanese, and the last line also says
"all passed" in English. To run them one by one:

```
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_tests.py         (domain, 171)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_motion_tests.py  (head wobble, 11)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_quiz_tests.py    (quiz, 39)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_records_tests.py (records, 23)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_app_tests.py     (GUI structure, 60)
```

They check numerically that the sound really carries spatial information:
whether the ITD is within the human range, whether the highs drop at the
shadowed ear, whether the convolution matches direct computation, whether
left-right mirroring is symmetric, whether front and back move in opposite
directions under head wobble, and so on.

The GUI tests check widget names, the order in which labels are created,
that no letter keys are assigned to actions, and that the numeric keypad
layout matches the azimuths geometrically. One of them starts the app in
English in a separate process (`tests/en_probe.py`) and checks that no
Japanese is left in what the user sees.

Without wxPython, `python -X utf8 tests\run_all.py --skip-gui` runs the rest
and leaves out the GUI suite (the last line says so).

### Automatic tests (GitHub Actions)

On the GitHub repository, `.github/workflows/tests.yml` runs the tests
without sound on every push to main and every pull request (this file is
not in the zip).

- Windows, macOS and Linux, with Python 3.10 (the oldest supported) and
  3.13: six runs
- The GUI suite and the English text check run on Windows only; macOS and
  Linux use `--skip-gui`
- The detection of the OS display language (`tests/check_language.py`) is
  checked on each OS. On macOS it is also checked after switching the
  display language to Japanese
- The per-test result files (`tests/*_result.txt`) are kept as Artifacts,
  whether the run passed or failed

In a folder where the repository is checked out, GitHub CLI (gh) shows the
results as text.

```
gh run list --workflow tests.yml        the list of runs
gh run view NUMBER --log-failed         the output where it failed
gh run download NUMBER                  fetch the result files
```

## The Localization Tool (listening app)

```
python -X utf8 demo\localization_app.py
```

Headphones are required. It works with the keyboard alone and supports
screen readers. Install what it needs with `python -m pip install -e
".[app]"`. How to use it is in `Manual (English).html` (the same content as
`Manual (English).txt`); in the app, F1 opens the keyboard help. In the zip
you can start it with `Start Localization Tool.bat`.

The choices in the settings area (HRTF, sound type, head wobble, reverb,
space, furniture amount, number of people, precise reverb, difficulty and
front/back emphasis) are saved in `settings/app_settings.json` and carry over
to the next start, kept by name rather than by position. Head tilt, up/down
emphasis and floor reflection always start off, and repeat playback and the
reference sound always start on (their state at start is fixed). To get the
first state back, close the tool and delete that file.

- F2 preview (the sound circles around you) / F3 quiz / F7 training / F8
  choose a pinna
- F11 head wobble measurement (alternates with and without, question by
  question, and totals each condition; the effect of improvement cancels out
  within one session)
- Shift+F3 up/down test (four choices: straight up, 45 degrees up, level,
  45 degrees down; answer with numpad / for straight up, 8 up, 5 level, 2
  down). Azimuths are mixed from the difficulty. It works only with a
  measured HRTF that covers elevation (Viking); with the synthetic HRIR and
  the horizontal-only asset, up and down sound the same, so it does not start
  and says why. Down goes only to 45 degrees (the range of Viking). F12 is not
  used because it clashes with PC-Talker
- Ctrl+F3 up/down test from the front (the same four choices). The sound
  always comes from the front and only the height changes. The same height
  may come twice in a row (if it never did, choosing "not the last one" would
  raise the chance level)
- Numpad * turns the reference sound on and off (on at start). In the up/down
  tests, a level sound from the same direction plays before the question
  sound, so instead of naming the step you can compare "higher or lower than
  the reference". Its state is kept in the result log
- Shift+F2 compare heights (not scored). Numpad / 8 5 2 just play that height
  in front. Pressing a key does not speak (speech would cover the sound)
- Ctrl+F8 choose a pinna for up and down. The up/down version of F8 (front
  and back), trying pinnae with the up/down test from the front. Its totals
  build up in `logs/pinna_records_elev.json`, apart from the front/back ones,
  and F10 reads them after the front/back totals
- Ctrl+Shift+F2 compare 8 directions (not scored). Numpad 1 to 9 (except 5)
  play a level sound from that direction. The keys are laid out like the
  directions, as seen from where you face now
- The Design a space and Design a reverb buttons in the settings let you set
  the size, the materials of the six surfaces (with readable names and
  categories), the listener's position and the levels of reflections and
  reverb, or the reverb time and more, and save them with a name. Saved ones
  appear in the choices as "Yours". The Current settings field in the window
  describes the reverb length in words, and Play with these settings lets you
  hear it before saving. They are saved in `settings/user_presets.json` (not
  part of the distribution). In the material lists, materials that suit the
  surface come first, and the rest follow with "Other" in front. The window
  also sets the numbers of furniture and people
- Furniture amount (none, a little, normal, a lot) and Number of people in the
  settings add furniture and people to the chosen space. Furniture is added
  only to closed rooms (numbers to suit the floor area)
- Ctrl+F2 rotation test (24 questions). A sound travels half a circle around
  you in 3 seconds; answer counterclockwise or clockwise for a level circle
  (numpad 4 and 6), or back to front or front to back for a sound passing
  overhead (8 and 2). The starting azimuth is drawn from 8 directions.
  Confusing front and back makes the direction sound reversed, so this tests
  front and back through movement. Overhead sounds are asked only with
  measured Viking data (only level ones with an HRTF that has no up/down
  measurements)
- Three up/down cues in the settings (all off at start): head tilt (an
  automatic sway of tilting the head toward each shoulder; off, subtle 6
  degrees, standard 12 degrees, strong 20 degrees), up/down emphasis (the
  engine's `elevation_emphasis`; above 3 kHz brighter for higher sounds and
  darker for lower ones, +12 dB straight up at 100 percent, also applied to
  the circling sound of the F2 preview) and floor reflection (the floor is 3 m
  below the ears; the higher the sound, the later the reflection: 17.5 ms
  straight up, 9.4 ms level, 0.7 ms at 45 degrees down). The settings are kept
  in the log
- Page Up turns the face up 15 degrees, Page Down down (stopping straight up
  and straight down). In the up/down tests, answer with the height as seen
  from the body, whichever way the face points
- F4 replay / F5 results / F6 stop / F9 read the settings / F10 totals / F1
  keyboard help (a window of its own opens with the focus in the text; read it
  line by line with the up and down arrows; Escape closes it and returns to
  where you were)
- Answers use the numeric keypad (the layout 7 8 9 / 4 6 / 1 2 3 maps
  directly to directions). Answers are not accepted until the question sound
  plays (to prevent wrong answers from a double press)
- Numpad 0 turns repeat on and off (on at start). When on, the question sound
  keeps playing with pauses until you answer (every 2.2 seconds for noise
  bursts). If the screen reader's speech covers it, just wait to hear it
  again. F6 stops it and F4 starts it again. Its state is kept in the result
  log
- The left and right arrows turn you 15 degrees at a time (helps with front
  and back)

Results are saved automatically in `logs/` with the settings, because
comparisons are impossible once you no longer know the conditions.

## Checking on real hardware

```
python -X utf8 tools\verify_realtime.py            # measure without sound
python -X utf8 tools\verify_realtime.py --audible  # only if you want sound
python -X utf8 tools\bench_wobble.py               # cost of head wobble
python -X utf8 tools\bench_sofa.py                 # cost of measured HRTFs
python -X utf8 tools\verify_sofa.py                # check SOFA loading
```

By default every source plays at gain=0, so nothing is heard. What is
measured is the callback processing time and the number of underruns.

## License notes

The library itself is under the **MIT License** (Copyright (c) 2026
Takahashi; full text in `LICENSE`). When you include it in software you
distribute or sell, include the copyright notice and the license text in
your product. The default HRTF is not covered by MIT; it is CC BY 4.0 (see
"Included data" below).

The required dependencies (numpy / scipy) are permissive too. The optional
ones, sofar, pyroomacoustics, sounddevice, miniaudio and samplerate, are all
MIT or BSD.

On the other hand, **pedalboard (GPLv3) and sound_lib (a wrapper around the
commercially licensed BASS) cannot be used in software you sell**. The
former requires publishing the source of the software that includes it, and
the latter needs a paid licence for commercial use.

### Included data

`audiospace/data/default_horizontal.npz` (the default HRTF),
`audiospace/data/default_sphere.npz` (the full-sphere default HRTF) and the
SOFA files in `assets/hrtf/viking/` are from the **Viking HRTF Dataset v2 (CC
BY 4.0)**. Commercial use is allowed, but **a credit is required**. When you
distribute them in your product, show the text in `ATTRIBUTION.en.md` there
too (the text is also returned by `attribution()`).

The IRs in `assets/reverb/` were generated with pyroomacoustics and carry no
outside restrictions (treated as MIT, like the library).

## Support

If audiospace is useful to you, please consider supporting its development.
Support is entirely optional and does not change what you can do with it.
It goes toward the cost of development tools, including AI, and the time
spent on listening tests and maintenance.

A way to support the project is in preparation.

Paid help is available for integrating audiospace, advice on audio design
for blind and visually impaired players, listening evaluations and feature
requests. Contact details are in preparation.
