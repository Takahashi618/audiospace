# Third-party data credits (ATTRIBUTION)

This lists the data in this repository that comes from others. (日本語版は
[ATTRIBUTION.md](ATTRIBUTION.md)。) **If you distribute or sell software that
includes this data, you must repeat these credits in your product (an About
screen, a README, the documentation and so on).**

---

## Table of material absorption (pyroomacoustics)

(The baked reverb tails in `assets/space_ir/` are audiospace's own output,
computed with pyroomacoustics using the materials of this table, and, like
the IRs in `assets/reverb/`, carry no outside restrictions.)

The absorption of each material in `audiospace/data/materials.json` (7
bands, 125 Hz to 8 kHz) was exported from the pyroomacoustics material
database (`tools/build_materials.py`). Materials with only 6 bands were
extended to 7 by repeating the last value (marked in `padded`). "open"
(absorption 1) was added by audiospace and is not in the source table. The
short readable names in `materials_ja.json` and `materials_en.json` were
written for audiospace.

The pyroomacoustics license (MIT):

```
Copyright (c) 2014-2017 EPFL-LCAV

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Viking HRTF Dataset v2

- **Where**: `assets/hrtf/viking/subj_*.sofa` (unchanged) and the default
  HRTF made from it, `audiospace/data/default_horizontal.npz` (modified)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
  https://creativecommons.org/licenses/by/4.0/
- **Authors of v2**: Simone Spagnol, Riccardo Miccini, Rúnar Unnthórsson
  — The Viking HRTF dataset v2 (2020), https://doi.org/10.5281/zenodo.4160401
- **Authors of the first edition**: Simone Spagnol, Kristján Bjarki Purkhús,
  Sverrir Karl Björnsson, Rúnar Unnthórsson — The Viking HRTF dataset,
  Proceedings of the 16th Sound & Music Computing Conference (SMC 2019),
  pp. 55-60, Málaga, Spain
- **Source**: https://sofacoustics.org/data/database/viking/

The authors of v2 and of the first edition differ. The `documentation.pdf`
that comes with the dataset asks users to cite both, so the credit includes
both.

### Credit text (to show in your product)

The text for using the default HRTF. `audiospace.attribution()` returns the
same text.

```
This software includes HRTF data from the Viking HRTF Dataset v2
by Spagnol, Miccini and Unnthorsson (2020,
https://doi.org/10.5281/zenodo.4160401), based on The Viking HRTF
dataset by Spagnol, Purkhus, Bjornsson and Unnthorsson (SMC 2019),
used under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Modified: resampled, truncated to two receivers, restricted to the
horizontal plane.
```

When you use the SOFA files unchanged, leave out the last sentence about
modifications (from "Modified" onward). If you bake the asset with
`tools/bake_hrtf_asset.py` under other conditions, the text stored in the npz
changes to match them, so use the output of `attribution()`.

### Citation for research

- Simone Spagnol, Kristján Bjarki Purkhús, Sverrir Karl Björnsson, and
  Rúnar Unnthórsson (2019). The Viking HRTF dataset. In: Proceedings of the
  16th Sound & Music Computing Conference (SMC 2019), pages 55-60,
  Málaga, Spain.
- Simone Spagnol, Riccardo Miccini and Rúnar Unnthórsson (2020).
  The Viking HRTF dataset v2. DOI: 10.5281/zenodo.4160401

### What CC BY 4.0 allows and requires

- **Commercial use is allowed.** You can include it in paid software and
  sell it
- **Modification is allowed.** You may resample, thin out and convert the
  format
- **Required credit**: the authors, the source, the name of the license and
  a link to it, and a note if you modified it
- It is not copyleft, so **there is no obligation to disclose the source
  code of the software that includes it**

### What the data is

Measured on a KEMAR dummy head fitted in turn with 20 artificial silicone
pinnae.

- `subj_A.sofa` to `subj_T.sofa`: measurements for each of the 20 pinnae
- `subj_Z.sofa`: the condition without pinnae (pinna-less)
- Each file has 1513 directions (azimuth 0 to 355 degrees, elevation -45 to
  +90 degrees), 128 taps, 48000 Hz
- It comes from a dummy head, so it contains no personal biometric data

The shape of the pinna differs greatly between people and directly affects
the perception of front and back and of elevation. With 20 pinnae, you can
compare several by ear and choose the one that suits you.

---

## Record of modifications

`audiospace` may apply the following conversions when loading. They count as
modifications under CC BY ("indicate if changes were made"), so they are
recorded here.

- Sample rate conversion (`HrirSet.resampled()`, which by default follows the
  output device)
- Truncating the receivers to the first two channels
- Converting Cartesian coordinates to spherical coordinates

The original data itself is kept unchanged in `assets/hrtf/viking/`.
