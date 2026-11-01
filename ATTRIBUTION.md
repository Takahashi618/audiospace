# サードパーティデータの表示（ATTRIBUTION）

このリポジトリに含まれるデータのうち、外部提供のものを以下に示す。
**これらを組み込んだソフトウェアを配布・販売する場合、この表示を
製品側（About画面、README、ドキュメント等）に再掲する必要がある。**

---

## 材質の吸音率の表（pyroomacoustics）

（`assets/space_ir/` の焼き込み済みの残響の尾は、この表の材質を使って
pyroomacoustics で計算した自作の出力で、`assets/reverb/` の IR と同じく外部の
制約は無い。）

`audiospace/data/materials.json` の材質ごとの吸音率（7帯域、125Hz〜8kHz）は、
pyroomacoustics の材質データベースから書き出したもの
（`tools/build_materials.py`）。帯域が6つしか無い材質は最後の値を延ばして
7つにそろえた（`padded` に印がある）。「open」（吸音率1）は audiospace で
足したもので、出典の表には無い。

pyroomacoustics のライセンス（MIT）:

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

- **場所**: `assets/hrtf/viking/subj_*.sofa`（無改変）と、
  そこから作った既定 HRTF `audiospace/data/default_horizontal.npz`（改変あり）
- **ライセンス**: Creative Commons Attribution 4.0 International (CC BY 4.0)
  https://creativecommons.org/licenses/by/4.0/
- **v2 の作者**: Simone Spagnol, Riccardo Miccini, Rúnar Unnthórsson
  — The Viking HRTF dataset v2 (2020), https://doi.org/10.5281/zenodo.4160401
- **初版の作者**: Simone Spagnol, Kristján Bjarki Purkhús,
  Sverrir Karl Björnsson, Rúnar Unnthórsson — The Viking HRTF dataset,
  Proceedings of the 16th Sound & Music Computing Conference (SMC 2019),
  pp. 55-60, Málaga, Spain
- **入手元**: https://sofacoustics.org/data/database/viking/

v2 と初版で著者が異なる。データセット付属の `documentation.pdf` は、
利用時に両方を引用するよう求めている。表示文にも両方を入れる。

### 表示文（製品に記載する文言）

既定 HRTF を使う場合の文面。`audiospace.attribution()` も同じ英文を返す。

```
This software includes HRTF data from the Viking HRTF Dataset v2
by Spagnol, Miccini and Unnthorsson (2020,
https://doi.org/10.5281/zenodo.4160401), based on The Viking HRTF
dataset by Spagnol, Purkhus, Bjornsson and Unnthorsson (SMC 2019),
used under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Modified: resampled, truncated to two receivers, restricted to the
horizontal plane.
```

日本語版:

```
本ソフトウェアは Viking HRTF Dataset v2（Spagnol, Miccini,
Unnthorsson 作、2020年、https://doi.org/10.5281/zenodo.4160401）の
HRTF データを含みます。元は The Viking HRTF dataset（Spagnol, Purkhus,
Bjornsson, Unnthorsson 作、SMC 2019）です。CC BY 4.0
（https://creativecommons.org/licenses/by/4.0/）のもとで使用し、
リサンプリング・受信点を2つに削減・水平面のみに限定して改変しています。
```

SOFA を無改変のまま使う場合は、最後の改変の文（Modified 以降）を除く。
`tools/bake_hrtf_asset.py` を別の条件で焼いた場合は、npz に入る文面が
その条件に合わせて変わるので、`attribution()` の出力を使う。

### 研究で使う場合の引用

- Simone Spagnol, Kristján Bjarki Purkhús, Sverrir Karl Björnsson, and
  Rúnar Unnthórsson (2019). The Viking HRTF dataset. In: Proceedings of the
  16th Sound & Music Computing Conference (SMC 2019), pages 55-60,
  Málaga, Spain.
- Simone Spagnol, Riccardo Miccini and Rúnar Unnthórsson (2020).
  The Viking HRTF dataset v2. DOI: 10.5281/zenodo.4160401

### CC BY 4.0 で許可されること・必要なこと

- **商用利用可**。有償ソフトウェアへの組み込み・販売ができる
- **改変可**。リサンプリング、間引き、フォーマット変換をしてよい
- **必要な表示**: 作者名、出典、ライセンス名とそのリンク、改変した場合は
  その旨
- コピーレフトではないため、**組み込んだ側のソースコード開示義務はない**

### データの内容

KEMAR ダミーヘッドに20種類の人工シリコン耳介を付け替えて測定したもの。

- `subj_A.sofa` 〜 `subj_T.sofa`: 20種類の耳介ごとの測定
- `subj_Z.sofa`: 耳介なし（pinna-less）の条件
- 各ファイル 1513方向（方位0〜355度、仰角 -45〜+90度）、128タップ、48000Hz
- ダミーヘッド由来なので個人の生体データは含まれない

耳介の形は人によって大きく違い、前後判別や仰角の知覚に直結する。
20種類あるので、複数の耳介で聞き比べて自分に合うものを選べる。

---

## 改変の記録

`audiospace` は読み込み時に以下の変換を行うことがある。CC BY の
「改変した場合はその旨を示す」に該当するため記録しておく。

- サンプルレート変換（`HrirSet.resampled()`、既定では出力デバイスに追従）
- 受信機を先頭2チャンネルに切り詰め
- 座標が Cartesian の場合は球面座標へ変換

元データそのものは `assets/hrtf/viking/` に無改変で保存してある。
