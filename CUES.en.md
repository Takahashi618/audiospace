# Front/back and up/down cues (the processing applied to sources)

As of 2026-10-16. This lists the processing that the library
(`audiospace/`) and the listening test app (`demo/localization_app.py`)
apply to sources to make front and back, and up and down, easier to tell
apart. (日本語版は [CUES.md](CUES.md)。)

The counts are:

- Front/back cues: 4 (measured HRTF, quieter sounds behind, more muffled
  sounds behind, head wobble) and 1 user action (turning with the left and
  right arrows)
- Up/down cues: 6 (full-sphere measured HRTF, reference sound, head tilt,
  up/down emphasis, floor reflection, directional reflections of a space) and
  1 user action (turning the face up and down with Page Up and Page Down)
- Common to both: distance attenuation and the reverb send

For each one, this gives what it does, where it is, its setting at start and
its numbers, in that order. At the end it describes one problem found and
fixed while writing this list.

## Front/back cues

### 1. Measured HRTF (differences in tone caused by the pinna)

The shape of the pinna changes the tone of sounds from the front and from
behind. audiospace uses the Viking HRTF (measured on a KEMAR dummy head
fitted with 20 different silicone pinnae) and convolves with the HRTF of the
measured direction nearest to the source. There is no interpolation.

- Where: `HrirSet.indices_for` in `audiospace/hrtf.py` (chooses the nearest
  direction); the convolution is in `audiospace/convolver.py`
- At start: pinna E (chosen in the listening tests)
- With the synthetic HRIR there is no front/back difference in tone (front
  and back are numerically identical). Only cues 2 onward remain

### 2. Quieter sounds behind (an artificial cue)

The further behind a sound is, the lower its volume. The strength is
Front/back emphasis, percent in the settings.

- Formula: volume = 1 − strength × (1 − cos azimuth) ÷ 2 × elevation weight.
  Azimuth 0 is the front. The elevation weight above the horizontal is cos
  elevation (about 0.71 at 45 degrees up); below, it is 1 down to 45 degrees
  and weakens from there toward straight down (√2 × cos elevation, about 0.71
  at 60 degrees down). It has no effect straight up or straight down (there is
  no front or back there)
- The lower side keeps its strength down to 45 degrees so that, combined with
  the up/down emphasis, a low sound behind does not end up as bright as a
  level sound behind
- At the starting value of 35 percent: 0.825 times (about −1.7 dB) at the
  side, 0.65 times (about −3.7 dB) straight behind. The front does not change
- The strength is capped at 90 percent (so that a sound straight behind does
  not disappear)
- Where: `front_back_gain_many` in `audiospace/geometry.py`, called from
  `_place_spatial` in `audiospace/engine.py`. It changes smoothly within a
  block

### 3. More muffled sounds behind (an artificial cue)

The further behind a sound is, the more of its highs are cut. The strength
is the same Front/back emphasis as in 2.

- A first-order low-pass. The strength is rounded to 24 steps, and sources on
  the same step are processed together
- As in 2, the strength is multiplied by the elevation weight. It is not
  applied straight up or straight down
- Straight behind at the starting 35 percent, the highs are −3 dB at about
  8.9 kHz (about −2.6 dB at 8 kHz). A mild muffling
- Straight behind at 90 percent, it falls from about 690 Hz and is about −21
  dB at 8 kHz
- Combined with a sound behind a wall (occlusion), it is muffled by both
  multiplied together. Neither cancels the other
- Where: `_buckets_for` and `_coeff_of` in `audiospace/engine.py`

### 4. Head wobble (a small side-to-side movement of the head)

Turns the listener automatically a little to the left and right. When the
head turns, sounds in front and sounds behind move in opposite directions, so
front and back can be told apart regardless of the listener's pinnae.

- Head wobble in the settings. At start: Standard (8 degrees, 0.45 Hz, a
  period of about 2.2 seconds)
- Subtle is 4 degrees at 0.6 Hz; Strong is 14 degrees at 0.35 Hz, as a
  triangle wave
- Where: `HeadWobble` and `WOBBLE_PRESETS` in `audiospace/motion.py`. The
  engine computes directions with the swayed facing every block

### User action: turning with the left and right arrows

The left and right arrows turn you 15 degrees on the spot. The sounds do not
move, so which way a sound moves when you turn tells front from back.
Answers are taken as "the direction as seen from where you face now".

### Tools for training the cues (not processing)

- F7 training: asks the directions you got wrong more often
- F8 choose a pinna: tries several pinnae to find one that suits you
- F11 head wobble measurement: alternates questions with and without head
  wobble and compares them
- Ctrl+F2 rotation test: answer which way a sound travels half a circle
  (counterclockwise or clockwise for a level circle, front to back or back to
  front overhead). Confusing front and back makes the direction sound
  reversed. With pinna E and 35% front/back emphasis, the brightness halfway
  through a circle starting from the left is −8.9 dB for counterclockwise
  (passing behind) and +2.2 dB for clockwise (passing in front)

## Up/down cues

### 1. Full-sphere measured HRTF (differences in tone up and down caused by the pinna)

Uses the 1513 directions of the Viking HRTF (elevations from 45 degrees down
to 90 degrees up). As for front and back, it convolves with the HRTF of the
nearest measured direction.

- The up/down tests work only when a measured Viking pinna is chosen. The
  default HRTF asset and the synthetic HRIR are horizontal only, so the app
  refuses to start them
- Down goes only to 45 degrees, because the measurements stop there
- This is the cue where individual pinnae differ most. Choose a pinna for up
  and down (Ctrl+F8) finds one that suits you. In the records so far, pinna E
  (chosen for front and back) was not among the best for up and down

### 2. Reference sound (a level sound from the same azimuth first)

Before the question sound, a level sound from the same azimuth plays. You can
answer whether the second is higher or lower than the first.

- Numpad * turns it on and off. On at start. Used only in the up/down tests
- 0.5 seconds between the reference and the question sound, 1.5 seconds
  between pairs when repeating (the other way round, you could not tell which
  is the reference)
- Where: `play_at` (`with_reference`) in `demo/localization_app.py`

### 3. Head tilt (tilting the head toward each shoulder)

Tilts the listener's head toward each shoulder in turn. Sounds above and
below move left and right in opposite directions, and only level sounds in
front do not move. It is heard through the difference between the ears, so
it does not depend on individual pinnae. The up/down version of head wobble.

- Head tilt in the settings. Off at start
- Subtle 6 degrees at 0.4 Hz, Standard 12 degrees at 0.35 Hz, Strong 20
  degrees at 0.3 Hz
- The amplitudes are larger than for head wobble. How far a sound 45 degrees
  up moves left and right is roughly sin(amplitude) × cos(45 degrees), which is
  less than for head wobble
- Where: `ROLL_PRESETS` in `audiospace/motion.py`; the engine adds it to the
  listener's roll angle

### 4. Up/down emphasis (an artificial cue)

Lifts the highs of higher sounds and cuts those of lower sounds. It also
follows the tendency of hearing that high-pitched sounds seem to come from
above. It is not natural, but it is easy to learn.

- Up/down emphasis, percent in the settings. 0 (none) at start
- A feature of the library (`SpatialEngine.elevation_emphasis`,
  `SpatialPlayer.set_emphasis`). The app passes the value of the setting as
  it is
- Changes the range above 3 kHz (an RBJ high shelf). At 100 percent: +12 dB
  straight up, +6 dB at 45 degrees up, 0 dB level, −6 dB at 45 degrees down.
  Proportional to elevation
- Half of the change is offset by the overall volume. Straight up at 100
  percent, the highs get +6 dB and the lows −6 dB. The brightness changes, but
  the loudness hardly does
- The elevation is as seen from the face. Facing up with Page Up makes level
  sounds darker
- It acts at once on playing sounds, including the circling sound of the F2
  preview. The floor reflection is darkened by its own direction (down)
- Rounded to 1 dB steps, and sources on the same step are processed together
- Where: `_emphasis_steps`, `_emphasis_sos` and `_apply_elevation_emphasis`
  in `audiospace/engine.py`
- Until 2026-10-16 the app baked it into the sound (adding or subtracting a
  high-passed part of the original). That form cancelled itself just below 3
  kHz and missed its target: −7.8 dB in the lows and +5.7 dB in the highs
  straight up

### 5. Floor reflection (app)

Adds the sound bouncing off the floor, a little later than the direct sound.
The higher the sound, the longer the gap between the direct sound and the
reflection.

- Floor reflection in the settings. Off at start
- The floor is 3 m below the ears (deep enough that a sound 4 m away at 45
  degrees down does not go under the floor). The reflection is 0.6 times as
  loud
- The delays for a sound 4 m away: 17.5 milliseconds straight up, 15.4
  milliseconds at 45 degrees up, 9.4 milliseconds level, 0.7 milliseconds at
  45 degrees down
- The reflection would really come from 56 to 90 degrees down, but its
  direction is moved to 45 degrees down, the edge of the measured range
  (straight down, all azimuths at 45 degrees down would be equally near, and
  the sound could lean to one side). The time gap keeps its true value
- The reflection is not sent to the reverb (it would be doubled). The
  reference sound gets one too
- Where: `floor_image` and `_play_one` in `demo/localization_app.py`

### 6. Directional reflections of a space (library)

When you choose a room or a street in Space in the settings, sound bouncing
off the floor, walls and ceiling arrives from each surface's direction,
coloured by its material. The floor and ceiling reflections are up/down cues
as well.

- None at start. While a space is chosen, it takes the place of the Reverb
  setting
- A space includes the floor or ground reflection, so do not combine it with
  the floor reflection of 5 (the floor reflection would be doubled)
- Where: `EarlyReflections` in `audiospace/early.py`, `Space` in
  `audiospace/acoustics.py`

### User action: turning the face up and down with Page Up and Page Down

Turns the face 15 degrees up or down at a time (up to 90 degrees either
way). The sounds do not move, so facing up brings sounds above toward the
front. In the up/down tests, answer with the height of the sound as seen from
your body, whichever way your face points. Listening to a low sound with the
face turned far up goes beyond the measured range (down to 45 degrees), and
it will not sound right.

### Tools for training the cues (not processing)

- Shift+F2 compare heights: numpad / 8 5 2 play straight up, up, level and
  down in front of you. Not scored
- Ctrl+F3 up/down test from the front: the azimuth is fixed at the front and
  only the height changes
- Ctrl+F8 choose a pinna for up and down

## Common to both

- Distance attenuation: unchanged up to the reference distance, then falling
  in inverse proportion to distance (inverse_clamped, as in OpenAL). Questions
  play 4 m away
- Reverb send: 0.3 for question sounds, 0 for the floor reflection
- The question repeats until you answer with the keypad (numpad 0, on at
  start). Not processing, but if speech covers the sound, you can just wait to
  hear it again

## A problem fixed: sounds straight up got front/back processing from their nominal azimuth

(Found while writing this list on 2026-10-16 and fixed the same day.)

Front/back cues 2 and 3 (quieter and more muffled sounds behind) used to look
**only at the azimuth**, not at the elevation. So:

- A sound straight up should be the same point whatever azimuth it is given
  with, but the computation kept the azimuth. A sound straight up given with
  azimuth 0 stayed at full volume, while one given with azimuth 180 dropped to
  0.65 times and was muffled (at the starting 35 percent)
- The up/down test of Shift+F3 (which mixes azimuths) treated straight-up
  questions as "the same sound whatever the azimuth", but their volume and
  tone actually changed with the azimuth. It did not happen with Ctrl+F3
  (fixed in front), where the azimuth is always 0
- Sounds behind at 45 degrees up were as quiet and muffled as sounds straight
  behind at ear level

The fix: the strength of the front/back processing is multiplied by
cos(elevation) (`elevation_weight` in `audiospace/geometry.py`). It has no
effect straight up and down, about 0.71 times at 45 degrees up and down, and
is unchanged at the horizontal. (The same day, to work with the up/down
emphasis, the weight was changed so that the lower side keeps its strength
down to 45 degrees; see front/back cue 2.) A test checks that a sound
straight up played with azimuth 0 and with azimuth 180 gives exactly the same
output (before the fix the largest difference was 0.037).
