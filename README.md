# audiospace

視覚障害者向けゲーム・ツールのための、バイノーラル3D音響ライブラリ。
（English: [README.en.md](README.en.md)）

ライセンスは MIT（Copyright (c) 2026 Takahashi、全文は `LICENSE`）。
ゲームへの組み込みや販売、研究での利用は自由。依存もすべて許容型
（numpy / scipy、任意で sofar・pyroomacoustics・sounddevice、いずれも
MIT か BSD）なので、**ソース開示義務なしに販売できる**ソフトウェアに
組み込める。既定 HRTF だけは CC BY 4.0 で、表示が必要（`ATTRIBUTION.md`）。

OpenAL に頼らず、ライセンスが MIT で、HRTF を耳に合わせて選べる音響の部品が
ほしくて作った（OpenAL Soft は LGPL で、HRTF の種類も少ない）。

## 表示の言語

利用者に見せる文（空間・材質・家具の名前、状態の文、誤りの説明、定位確認
ツールのすべて）は、**OS の表示言語が日本語なら日本語、それ以外なら英語**に
なる。Windows は表示言語（`GetUserDefaultUILanguage`）、macOS は「優先する
言語」の先頭、それ以外は `LC_ALL`・`LC_MESSAGES`・`LANG` を見る。

決めたいときは、プログラムを起動する前に環境変数 `AUDIOSPACE_LANG` を `ja` か
`en` にする。`FURNISH_LEVELS` や `FACE_LABELS` などの名前の一覧は audiospace を
読み込んだときに作るので、動いている途中では切り替えられない。いまの言語は
`audiospace.language()`（`"ja"` か `"en"`）。文は `audiospace/i18n.py` の
`tr("日本語", "English")` で並べて書いてある。材質の英語名は
`audiospace/data/materials_en.json`。

## この作り方について

audiospace のコードは、AI（Anthropic の Claude）が書きました。作者は
コードを書いていません。作者が担ったのは次のことです。

- 何を作るかを決めること。視覚障害者がゲームやツールで使える3D音響であること、
  OpenAL に頼らないこと、キーボードと読み上げだけで確かめられること
- ヘッドホンでの聴き取りテスト。既定の耳介は、作者が実測の耳介を聴き比べて
  選びました。最初の聴き取りで前後の取り違えに気づいたことから、首振り・
  訓練・耳介の選定の3つの補助機能が生まれました
- 動作の確認と、直すか直さないかの判断
- 公開、問い合わせへの対応、保守

コードは自動テスト（241件）と作者の実機での確認で検証しています。それでも、
人が書いたコードと同じく、誤りが残っている可能性があります。無保証です。

日本の文化庁や米国の著作権局の考え方では、AI が書いた部分には著作権が
生じない可能性があります。いずれにしても、audiospace は MIT ライセンスの
もとで自由に使えます。

### How this was made

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

The code is checked by 241 automated tests and by the author on real
hardware, but like any code it may still contain mistakes. It comes without
warranty.

Under current guidance in Japan and the United States, parts written by an
AI may not be protected by copyright. Either way, audiospace is free to use
under the MIT License.

## 配布物（wheel と zip）

配布物は2種類ある。どちらも作者の聴取記録（`logs/`）や作業記録は含まない。

- **wheel**（`audiospace-<版>-py3-none-any.whl`、約0.3MB）: ライブラリだけを
  使う人向け。入るのは `audiospace/` と既定 HRTF、`LICENSE`、`ATTRIBUTION.md`。
  実測 HRTF の SOFA 21種と焼き込み済みの残響 IR は入らない
- **zip**（`audiospace-<版>.zip`、約70MB）: ソース一式。ライブラリ、
  聴感確認アプリ（`demo/`）、テスト、ツール、SOFA 21種、残響 IR 7種、焼き込み済みの
  空間の残響7種（`assets/space_ir/`）が入る。
  耳介を選び直したい人や研究に使う人はこちら。利用者向けの
  `はじめにお読みください.txt`、`マニュアル.html`（同じ内容の
  `マニュアル.txt`）、起動用の `定位確認ツールを起動.bat` も入る。
  英語版の `README (English).txt`、`Manual (English).html`（`Manual (English).txt`）、
  `Start Localization Tool.bat`、`README.en.md`、`CUES.en.md`、
  `ATTRIBUTION.en.md` も入る

wheel を入れる場合（ファイル名は実物に合わせる）:

```
python -m pip install audiospace-0.7.1-py3-none-any.whl
python -m pip install "audiospace-0.7.1-py3-none-any.whl[playback]"
```

2行目は実機再生も使う場合。wheel だけの環境では焼き込み済みの残響 IR が
無いので、`use_room_ir()` は実行時に部屋を計算する（pyroomacoustics が要る。
`[rooms]` か `[full]` で入る）。zip の `assets/reverb/` を手元に置けば、
`load_room_ir(名前, レート, directory=...)` で読み込める。

zip の場合は、展開したフォルダで次を実行する:

```
python -m pip install -e .[full]
```

マニュアルの表や数（キーの一覧、設定欄の項目、問題数、空間・残響・材質の
一覧、版の番号）は、`python -X utf8 tools\build_manual.py` がアプリと
ライブラリのコードから作り直す。日本語と英語の両方を作る（それぞれ、その
言語で起動したアプリから作る）。`マニュアル.txt` と
`はじめにお読みください.txt` もここで作る（元は `マニュアル.html` と
`tools/readme_first.txt`。英語は `Manual (English).html` と
`tools/readme_first_en.txt`）。`--check` を付けると書き換えずに古いかだけを
調べる。

## インストール

このフォルダで開発中に使う場合:

```
python -m pip install -e .
```

必須依存は numpy / scipy のみ。実機再生には `.[playback]`
（sounddevice / soundfile）、全部入りは `.[full]`。
定位確認ツール（`demo/`）を使うなら `.[app]`
（wxPython / accessible_output2 / sounddevice / sofar）。
既定 HRTF（`audiospace/data/default_horizontal.npz`、約205KB）は
パッケージデータとして wheel に入るので、PyInstaller でも
パッケージごと固めれば同梱される。**配布時は CC BY 4.0 の表示が必要**
（`attribution()` の文面と `ATTRIBUTION.md` を参照）。

## 使い方

### 高レベル API

```python
from audiospace import SpatialPlayer

player = SpatialPlayer.open(head_wobble="clear")  # WASAPI を自動選択
h = player.play("beacon.wav", position=(3.0, 4.0, 0.0), loop=True)
player.play_ui("menu_move.wav")                   # UI音は定位させず中央固定
player.set_listener((0.0, 0.0, 0.0), yaw_deg=90)  # 左を向く
player.set_sound_gain_db(h, -6.0)                 # dB でも指定できる
player.stop_sound(h, fade_ms=30)                  # フェードアウトで停止
player.pause(); player.resume()                   # 全体の一時停止と再開
print(player.status())                            # 読み上げ用の状態文字列
player.stop()
```

`play()` に渡したファイルはデコード結果がキャッシュされるので、同じ
効果音を何度鳴らしてもディスクを読み直さない。鳴り終わった使い捨ての
音は次の `play()` が自動で回収するため、スロットあふれを気にしなくて
よい。個別の音の一時停止は `pause_sound(h)` / `resume_sound(h)`。

```python
player.set_sound_lowpass(h, 0.7)              # 壁越し（こもらせる）
player.set_sound_gain(h, 0.0, ramp_ms=500)    # 500ms かけて消す
player.play("step.wav", rate=1.05)            # 5% 速く・高く鳴らす
player.play("bgm.wav", spatialise=False, loop=True)   # ステレオのまま
```

**遮蔽（`set_sound_lowpass`）** は「壁の向こう」の合図。音量を下げるだけ
では距離と区別がつかないので、高域を落とす。方位由来の前後の暗さとは
合成される（後ろかつ壁越しは、どちらか一方より暗い）。歩くたびに値が
変わっても、既定 60ms でランプするので音色が飛ばない。

**ゲインのランプ**（`ramp_ms`）は音声スレッドで進む。UI スレッドが
モーダルループやウィンドウのドラッグで止まっても、フェードは指定時間で
必ず終わる。クロスフェードは 2 本のランプで書く。

**再生レート**（`rate`）はピッチとテンポが一緒に動く。足音の揺らぎや
ピッチ付きの合図に使う。レート違いも同じキャッシュに乗る。

エンジンの終段には**リミッター**が入っている（既定で有効）。多数の
音源が重なってフルスケールを超えそうなブロックだけ全体を下げるので、
ヘッドホンで耳障りなクリップが出ない。効いた回数は `status()` に
「音量制限」として出る。無効にするには `SpatialEngine(..., limiter=False)`。

HRTF を指定しなければ、**聴取テストで選定した実測HRTF**（Viking 耳介E、
水平面のみ、205KB）が自動で読み込まれる。アセットが無ければ合成に落ちる。

```python
from audiospace import default_hrir, default_is_measured, attribution

hrir = default_hrir(48000)      # 出力レートに合わせて変換済み
default_is_measured()           # 実測か、合成に落ちたか
print(attribution())            # 配布時に表示する文面（CC BY 4.0）
```

### 低レベル API（デバイスを開かない）

```python
from audiospace import SpatialEngine, HrirSet, BufferSource

engine = SpatialEngine(HrirSet.synthetic(sample_rate=48000), block=512)
handle = engine.add_source(BufferSource(samples, loop=True),
                           position=(1.0, 2.0, 0.0))
stereo = engine.render()          # (512, 2) float32 が返る。使い道は自由
```

自分でブロックを供給したい場合は `PushSource` を使う。

```python
handle, push = player.open_stream_source(position=(0.0, 5.0, 0.0))
push.write(my_generated_samples)  # リングバッファ。タイミングを合わせる必要なし
```

## 座標系

右手系、単位はメートル。x = 東、y = 北、z = 上。
リスナーの `yaw_deg` は北から反時計回り。`pitch_deg` は顔の上下で、
正で上を向く（体の向き `yaw_deg` を決めたあと、左右の軸まわりに傾ける）。

`roll_deg` は首かしげで、正で左耳を左肩へ下げる（顔の上下のあと、
前後の軸まわりに傾ける）。

```python
player.set_listener(pitch_deg=30)    # 30度上を向く。位置と yaw はそのまま
player.set_listener(roll_deg=10)     # 左耳を10度下げる
```

首かしげは首振りと同じく自動で揺らせる。上の音と下の音が左右へ逆向きに
動き、正面・水平の音は動かないので、耳介の個人差と無関係に上下の手がかりに
なる（Viking 耳介E で、上45度と下45度の左右差の動きの相関は -0.997）。

```python
from audiospace import ROLL_PRESETS
player.engine.set_roll_wobble(ROLL_PRESETS["clear"])   # 振り幅12度
```

`pitch_deg=0` かつ `roll_deg=0`（既定）のときの計算は、これらの引数を
足す前とビット単位で同じ。
実測 HRTF の測定範囲の外（Viking なら下45度より下）に音が来ると、最も
近い測定方向で鳴るので正確な音にはならない。

方位・仰角は **SOFA 規約**に合わせてある（方位0度=正面、反時計回り、
90度=左）。実測 HRTF を .sofa から読み込んでも変換が要らない。

## 無音モード

環境変数 `AUDIOSPACE_SILENT=1` を設定すると、デバイスを一切開かない。
`SpatialPlayer` はタイマースレッドでエンジンを回して出力を捨てるので、
呼び出し側はテスト用の別経路を書かなくてよい。

```
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_tests.py
```

## 残響

2方式を同じインターフェースで切り替えられる。

```python
player.set_reverb("hall")            # 軽量な FDN。アセット不要
player.use_room_ir("hall")           # 焼き込み済み IR を畳み込む
player.set_reverb(None)              # 残響なし
```

`use_room_ir(名前)` は `assets/reverb/` の焼き込み済み IR を読む
（レートが違えば変換する）ので、実行時に pyroomacoustics は要らない。
焼き込みが無いときだけライブ生成（数秒かかる）に落ちる。IR は
事前に焼いておく。

```
python -X utf8 tools\build_reverb_presets.py --rate 48000
python -X utf8 tools\build_reverb_presets.py --list
```

プリセットは bathroom / small_room / corridor / hall / cave / warehouse /
cathedral の7種類。実測 RT60 は 0.18秒（バスルーム）から 5.37秒（大聖堂）。

## 材質つきの空間（初期反射と材質の残響）

同じ足音でも、木の部屋・コンクリートの部屋・何もない道・建物の間では聞こえ方が
違う。その違いを出すのが `Space`（箱形の空間と、床・天井・4つの壁それぞれの
材質）。

```python
player.set_space("wood_floor_concrete_walls")   # 用意した空間の名前で
player.set_space(Space.room((5, 4, 2.7), floor="wood_16mm",
                            walls="rough_concrete", ceiling="plasterboard"))
player.set_space(Space.street(width=12, facades="brickwork"))   # 建物の間
player.set_space(None)                                          # 外す
```

空間を設定すると、2つの響きが入る。

- **方向つきの初期反射**: 聞き手から各面までの距離で、反射の時刻・方向・強さと、
  その面の材質による音色が決まる。壁際に立てばその壁の側から反射が聞こえ、
  向かい合う2面の間では反射が往復する（建物の間の、はためくような反響）。
  開いた面（`"open"`、空や道の先）からは何も返らない
- **材質から決まる後期残響**: 材質の吸音率から帯域ごとの残響時間を Eyring の
  式（空気吸収を含む）で求め、帯域ごとに違う速さで減衰する FDN
  （`BandFdnReverb`）で鳴らす。計算は一瞬なので、ゲーム中に場所が変わっても
  すぐ切り替えられる。開けた屋外には拡散した残響は付かない

材質は pyroomacoustics の材質データベース由来の91種類（`material_names()`、
7帯域 125Hz〜8kHz）と `"open"`。7つの吸音率を直接渡してもよい。
用意した空間は `space_labels()`: 木の床と木の壁の部屋、木の床とコンクリートの
壁の部屋、コンクリートの部屋、じゅうたんと吸音天井の部屋、タイル張りの浴室、
コンクリートの廊下、石造りの広間、何もない道、建物の間の道、狭い路地。

計算した 1kHz の残響時間（5m×4m×2.7m の部屋）: じゅうたん 0.27秒、木の壁
0.81秒、コンクリートの壁 1.99秒、全面コンクリート 3.21秒。pyroomacoustics の
精密な計算（像の方法と光線追跡）と2割程度で一致する。

注意:

- 反射は聞き手の位置から計算する（音源ごとには計算しない）。ゲーム向けの音響
  エンジンで広く使われる近似で、費用は音源の数によらず一定（反射は最大24本）。
  代わりに、聞き手から遠い音源の反射の細かい違いは表さない
- 初期反射の大きさは残響の送り量（`reverb_send`）に左右されない。直接音に対する
  比を物理的な値（壁から1mでおよそ -6〜-10dB）に保つため
- `set_space` は残響を置き換える（`set_reverb` / `use_room_ir` の設定は外れる）
- 処理時間（性能コア、128音源）: hall の IR 残響 2.2ms に対し、空間 2.9ms。
  効率コアでは 128音源＋空間の99%値が予算すれすれ（10.2〜10.5ms）。
- 帯域ごとの減衰は低域・高域の2つのシェルフで近似するので、残響時間は目標と
  2〜3割ずれることがある（コンクリートの部屋の 4kHz で 1.35秒の目標に 1.8秒）

### 精密版の残響（`late="precise"`）

```python
player.set_space("concrete_room", late="precise")
print(player.engine.late_status)     # 「精密版の残響」など、読み上げ用
```

閉じた部屋の後期残響を、pyroomacoustics の精密な計算（面ごとの材質の7帯域、
拡散、空気吸収、像の方法と光線追跡）で作った尾に替える。方向つきの初期反射は
エンジンのまま（聞き手の位置と向きに合わせて鳴らせるため）で、尾は反射が
混ざり合った後（50〜150ms 以降）から使い、頭と終わりを滑らかにつなぐ。尾の
大きさは速い版と同じエネルギーにそろえてあるので、切り替えても音量は変わらない。

速い版（Eyring の式）との違いが出るのは、吸音が偏っている部屋。じゅうたんの
床と吸音天井で壁は反射する部屋では、速い版の 0.27秒に対し精密版は 0.66秒
（壁の間を音が往復し続けるのを、精密な計算だけが表せる）。

- 用意した閉じた部屋7種は焼き込み済み（`assets/space_ir/`、計8.4MB）ですぐ使える
  （`tools/build_space_irs.py` で作り直せる）
- それ以外の部屋は初めて使うときに**別のプロセス**で計算し（数秒〜20秒）、
  その間は速い版で鳴らして、できたらつなぎ替える。結果は
  `engine.space_ir_cache`（既定は一時フォルダの下）に保存し、次からすぐ使う。
  同じプロセスのスレッドで計算すると、pyroomacoustics が Python の GIL を
  握り続けて音が最大12.8秒止まった。別のプロセスなら、計算中の描画の
  最も遅いブロックは 3.7ms
- **開いた面のある空間（建物の間の道・路地・何もない道）は速い版のまま。**
  開いた面から光線が抜け、残りが壁の間を往復し続けて、計算の打ち切りで尾が
  急に止まる（建物の間の道では 0.9秒で 10ms のうちに 30dB 落ちた）
- pyroomacoustics が無い環境や、固めた exe では、焼き込み済みの部屋だけ精密版。
  それ以外は速い版で鳴らし、`late_status` にその旨が入る
- 処理時間: 6秒の尾（コンクリートの部屋）で 1ブロック約0.6ms（速い版は 0.09ms）

### 空間と残響を自分で作る（設計図）

空間と残響は、JSON にそのまま書ける辞書（設計図）からも作れる。ゲームの
マップごとの響きをファイルに置いたり、利用者に作らせて保存したりできる。

```python
from audiospace import space_from_spec, reverb_from_spec, space_to_spec

room = space_from_spec({
    "kind": "room",                       # room / street / open
    "size": [6.0, 5.0, 2.8],              # 幅・奥行き・高さ（m）
    "materials": {"floor": "wood_16mm", "ceiling": "plasterboard",
                  "north": "glass_window", "south": "wooden_door",
                  "west": "brickwork", "east": "curtains_velvet"},
    "at": [0.4, 0.4],                     # 聞き手の位置（幅・奥行きの割合）
    "early_order": 4, "early_gain": 1.0, "late_gain": 1.0})
player.set_space(room)

player.set_reverb(reverb_from_spec(
    {"rt60": 2.5, "room_scale": 1.2, "damping": 0.3, "width": 1.0,
     "level": 1.2}, player.engine.sample_rate))
```

- 聞き手は北（前）を向いている前提で、north が前の壁、south が後ろ、west が
  左、east が右。`street` では前後と空が開き、`open` では床（地面）だけ返る
- 値がおかしいと `ValueError`。問題点は `space_spec_problems` /
  `reverb_spec_problems` が表示の言語の文で返す（読み上げにそのまま使える）
- 用意した空間・残響を元にするには `space_to_spec(space_preset(名前))` と
  `preset_reverb_spec(名前)`。どちらも往復して同じ空間・同じ音になる
- 材質91種類には日本語名と英語名がある。`material_choices()` は（名前, 「分類: 読み上げ名」）
  を分類順に返し、選ぶ一覧にそのまま使える。`material_label(名前)` で1つ
- `FdnReverb` に `level`（残響の大きさ）を足した

### 家具や人を置く（`contents`）

部屋の大きさは同じでも、家具や人がいると響きは変わる。空間に「置いてある
もの」を種類ごとの数で持たせると、2つの効き方で響きが変わる。

- 吸音: それぞれの吸音の量（m²、7帯域）を残響時間の計算（Eyring）に足す。
  物が多いほど残響が短く、小さくなる
- 混み具合: 置いたものがふさぐ広さ ÷ 床の広さ。反射を弱める。家具は床の近く
  にあるので床の反射を最も、壁を中くらい、天井を少しだけ弱め、高い音ほど
  強く弱める

```python
from audiospace import space_preset, furnish_counts
room = space_preset("wood_room")
furnished = room.with_contents({"sofa": 1, "table": 1, "person": 4})
player.set_space(furnished)                 # 切り替えはなめらかにつながる
# 家具の量（empty / sparse / normal / full）から床の広さに合わせた数
counts = furnish_counts(room.size[0] * room.size[1], "normal")
```

- 置けるもの（`FURNISHINGS`）: 立っている人、座っている人、ソファ、布張りの
  ひじ掛け椅子、木の椅子、テーブル、本の詰まった本棚、ベッド、木の戸棚、
  段ボール箱。値は文献の典型値をもとにした目安で、大まかに響きを変えるための
  もの（置いた位置は扱わない）
- 木の部屋（5×4×2.7m）の 1kHz の残響時間: 空 0.81秒、普通の家具 0.64秒、
  さらに10人 0.45秒。床の反射は約 1.6dB、壁は約 0.9dB、天井は約 0.4dB 弱まる
- 設計図では `"contents": {"sofa": 1, "person": 3}`。空なら今までとまったく
  同じ空間（出力も精密版の鍵も変わらない）
- 精密な残響（`late="precise"`）は面の材質しか扱えないので、物があるときは
  速い版になる（`late_status` に理由が出る）
- 材質の一覧 `material_choices(face)` は、面（floor / ceiling / 壁）に合う材質
  （`FACE_MATERIALS`）を上に、その他を「その他」付きで下に並べる

### 場所の移動（響きのつなぎ替え）

空間を切り替えると、響きはなめらかにつながる。

- 出てきた場所の残響は、新しい音を受け取らなくなり、自然に消えるまで鳴り続ける
  （部屋を出たあと、部屋の響きが消えていくのが聞こえる）。`release_s` を渡すと
  その秒数で打ち切る
- 反射と残響への入力は `transition_s`（既定 0.3秒）かけて、前の場所から
  新しい場所へクロスフェードする。0 にすると即座に切り替わり、プツッと鳴る
  （標本の跳びが定常時の約7倍。つなぐと定常時と同じ）

場所ごとに空間を登録しておけば、聞き手の位置で自動で切り替わる。

```python
room = Space.room((5, 4, 2.7), floor="concrete_floor", walls="rough_concrete",
                  ceiling="rough_concrete", listener=(0, 0, 0), at=(0.5, 0.5))
street = Space.street(width=12, length=80)
road = Space.open_ground()
player.set_spaces([road, street, room])     # あとは聞き手を動かすだけ
```

聞き手を範囲（箱）に含む空間のうち、いちばん小さいものが選ばれる（道の中に
置いた部屋に入れば部屋の響き）。いまの空間を出るまで、ほかへは移らない
（より小さい空間に入ったときは移る）ので、境目で行ったり来たりしない。
`set_space` を呼ぶと登録は外れる。

聞いて確かめるには `python -X utf8 demo\space_walk.py`。コンクリートの部屋から
建物の間の道を抜けて何もない道へ歩き、各場所で手をたたく音を
`demo/out/space_walk.wav` に書き出す（デバイスは開かない）。

## 前後の判別について（重要）

前後と上下の手がかりとして音源にかけている処理の一覧（数値と場所つき）は
[CUES.md](CUES.md) にまとめてある。

### 何が起きるか

ヘッドホンで他人の HRTF を使うと、**前後の取り違えがほぼ必ず起きる**。
実際に聴いて確認済みで、慣れると分かるようになるが、初見のプレイヤーに
それは期待できない。耳介の形は個人差が大きく、脳は自分の耳介のスペクトル
特性で前後を判断しているため、他人の耳介では手がかりが合わない。

### 対策1: 首振り（最も確実）

頭をわずかに動かしたときの音の動き方は、前後で必ず逆になる。音源が前に
あれば頭を左に振ると音は相対的に右へ動き、後ろにあれば逆に動く。実測で
**相関 -0.999**（ほぼ完全な逆位相）を確認した。しかもこれは前後が
数値的に同一な合成 HRIR 上での測定なので、**耳介の個人差と無関係に効く**。

```python
from audiospace import WOBBLE_PRESETS

player.engine.set_head_wobble(WOBBLE_PRESETS["clear"])   # 振り幅8度、推奨
player.engine.set_head_wobble(amplitude_deg=6.0)          # 個別指定
```

振り幅は 4〜10 度で十分。大きくすると音像が揺れて不快になる。
ゲームがプレイヤーの向きを自由に変えられるなら、それが最良で首振りは不要。

### 対策2: 訓練

慣れれば分かるようになるのだから、慣れる機会を作る。`demo/quiz.py` の
`TrainingSession` は、間違えた方向ほど出やすくなる重み付き抽選で、
前後の取り違えは特に重く扱う。誤答直後に正解方向をもう一度鳴らして
聞き比べさせる。

### 対策3: 耳介を選ぶ

Viking には耳介が20種類ある。合うかどうかは組み合わせ次第なので、
`CalibrationSession` で聞き比べて選べる。切り替えは音を止めずにできる。

```python
player.engine.set_hrir(other_hrir)     # 再生を止めずに差し替え
```

### 音量だけの手がかりは効かない

エンジンには後方の音をローパスする `front_back_depth`（既定0.35）がある。
当初は音量を下げるだけだったが、実測で前後の周波数特性が変わっておらず
（前13.12 対 後13.12）、手がかりになっていなかった。音量差は距離と
区別がつかない。ローパス導入後は明瞭度比 0.57 倍の差がついた。

後方の音量とローパスの強さは、水平から離れると弱める。上側は cos(仰角)
（上45度で約 0.71 倍）、下側は下45度までは水平と同じ強さで、そこから弱める
（`min(1, √2·cos(仰角))`）。真上と真下には前も後ろも無いので効かない（以前は
方位だけを見ていて、方位 180 度で指定した真上の音が小さくこもっていた）。
下側を45度まで弱めないのは、次の上下の強調と併用したとき、後ろの下の音が
後ろの水平の音と同じ明るさにならないようにするため。

### 上下の強調（`elevation_emphasis`）

上の音ほど高い音域を明るく、下の音ほど暗くする人工の手がかり。実測 HRTF の
上下の違いは耳介の個人差が大きく、方位ごとにばらばらなので、どの方位でも
「上ほど明るい」という規則を足す。既定はオフ（0）。

```python
engine = SpatialEngine(hrir, front_back_depth=0.35, elevation_emphasis=0.35)
player.set_emphasis(front_back=0.35, elevation=0.35)   # 高レベル API
```

- 強さ 1 で、3kHz より上を真上 +12dB、上45度 +6dB、下45度 −6dB 変える
  （仰角に比例。RBJ のハイシェルフ）。仰角は顔から見た値なので、顔の上下や
  首かしげにも追従する
- **増減の半分を全体の音量で打ち消す。** 真上なら高域 +6dB・低域 −6dB に
  なり、明るさは変わるが大きさはあまり変わらない。音量の差は距離の差と
  紛れるため
- 鳴っている音・動いている音にもその場で効く。1dB 刻みの段に丸め、同じ段の
  音源をまとめてかける。段の変わり目に段差は出ない（テストで確かめている）
- 前後の強調と併用してよい。前後は水平で最も強く上下で弱まり、上下は水平で
  0 になり上下で強まるので、ほとんどの方向で打ち消し合わない
- 重さ（性能コア、全天球、全音源が上下にも動く、強さ 0.5）: 64音源 +0.50ms、
  128音源 +0.56ms、256音源 +0.78ms（`tools/bench_tilt.py`）。強さ 0 なら費用は無い
- 聴き取りの記録（方位を混ぜた上下の4択）: 強調なし 42〜46%、35% で 88%
  （アプリが音に焼き込んでいたころの値）

## 2Dゲームでは水平面に絞る

実測 HRTF は全天球1513方向あり、スペクトル表が 11.8MB になる。毎ブロック
そこから行を集めるとキャッシュに乗らず、これが実測 HRTF が重い主因である。

```python
hrir = HrirSet.from_sofa(path).horizontal()      # 仰角を捨てる
hrir = hrir.decimated(15.0)                      # さらに方位を間引く
```

| | 方向数 | 表 | 64音源 |
|---|---|---|---|
| 実測 全天球 | 1513 | 11.8 MB | 4.91 ms |
| 実測 水平面のみ | 216 | 1.7 MB | **2.07 ms** |

90度での高域 L/R 比 284 倍は水平面のみでも完全に維持される。

**2026-10-16 の軽量化の後は、全天球と水平面のみの差はほぼ無くなった**
（128音源で 3.4ms と 3.2ms、下の「実測性能」）。上の表はそれ以前の測定。
速さのために絞る必要はもう無い。仰角を使わないなら、読み込みと記憶域が
小さくなる利点だけが残る。

## 実測性能（このマシン）

Windows / CPython 3.14.5 / Steinberg UR44C / WASAPI 共有 44100Hz /
ブロック512サンプル（予算11.61ミリ秒）。全音源を毎ブロック移動させ、
リスナーも回転させ続けた状態での測定。

| 構成 | 平均 | 最悪 | 途切れ |
|---|---|---|---|
| 8音源 + FDN | 1.56 ms (13.4%) | 2.66 ms | 0 |
| 32音源 + FDN | 2.27 ms (19.6%) | 3.30 ms | 0 |
| 64音源 + FDN | 3.24 ms (27.9%) | 4.04 ms | 0 |
| 8音源 + IR畳み込み(68073タップ) | 3.00 ms (25.8%) | 4.18 ms | 0 |
| 32音源 + IR畳み込み | 3.68 ms (31.7%) | 4.88 ms | 0 |
| 64音源 + IR畳み込み | 4.68 ms (40.3%) | 5.75 ms | 0 |

畳み込み単体の性能はさらに高く、256音源で予算の18.7%に収まる。
上の数値との差はミキサー側の Python 処理であって、畳み込みではない。

### 軽量化の後（2026-10-16）

エンジンのみ（デバイスなし）、48000Hz・ブロック512（予算10.67ms）、
全天球の実測 HRTF、全音源が毎ブロック動き、首振りあり。
`tools/bench_polyphony.py` で測り、改善前のライブラリと交互に実行して
比べた。

| 音源数 | 改善前 平均 | 改善後 平均 | レビュー後 平均 | レビュー後 99% |
|---|---|---|---|---|
| 64 | 2.81 ms | 1.75 ms | 0.91 ms | 1.61 ms |
| 128 | 6.60 ms | 3.38 ms | 2.06 ms | 2.86 ms |
| 192 | — | 5.4〜5.7 ms | 2.7〜3.0 ms | 3.6〜5.6 ms |
| 256 | 13.11 ms（予算超え） | 8.0 ms | 4.5〜5.3 ms | 5.3〜7.0 ms（1回だけ11.5ms） |
| 384 | — | — | 7.2〜8.2 ms | 9.0〜11.1 ms（予算すれすれ） |

「レビュー後」は同日のコードレビューで、HRIR の和の計算を `einsum` の
最適化経路から素の `einsum` に替えた後（この形の計算では最適化経路の
ほうが6〜7倍遅かった）。

- **同時発音数の目安（残響込み）: 確実なのは 128 音源。性能コアで動くなら
  256 音源まで。** この PC（Core i7-12700K）は速い性能コアと遅い効率コアを
  持ち、Windows が処理を効率コアへ回すと時間がおよそ2倍になる。プロセスを
  コアに固定して測った値（全天球、全音源移動、首振りあり、hall の IR 残響に
  送る）:

  | 構成 | 性能コア 平均 / 99% | 効率コア 平均 / 99% |
  |---|---|---|
  | 128音源＋IR残響 | 2.32 / 3.25 ms | 5.12 / 8.67 ms |
  | 256音源＋IR残響 | 4.98 / 6.32 ms | 10.02 / 12.29 ms（予算超え） |

  上の表（コアを固定していない計測）の値も、この揺れを含む
- 処理は鳴っている音源の数に比例する。容量（`max_sources`）を大きく
  取っても、鳴っていない分はほぼ費用がかからない（容量256で16音を
  鳴らしたとき 9.2ms → 0.74ms）
- 入れた改善は4つ: 方向・音量の計算を全音源まとめて配列で行う／鳴っている
  音源だけ畳み込む／方向が変わった音源だけ二重に計算してクロスフェード
  する／HRIR が1ブロックに収まるときは FFT を 1024 から 640 に縮める
- 出力は改善前と丸め誤差（ピーク比 3e-7、約 -130dB）の範囲で一致する。
  ただし**新しく鳴り始めた音の最初の1ブロック（約11ms）だけは変わった。**
  改善前は、その行が以前使っていた方向（新しい行なら索引0）からクロス
  フェードで入ってきており、鳴り始めが一瞬ずれた方向から聞こえていた。
  改善後は最初から正しい方向で鳴る

### 残響の軽量化（2026-10-16）

`tools/bench_reverb.py`（デバイスなし、性能コアに固定）。1ブロックあたり。

| 部屋（IR の長さ） | IR 残響 改善前 | IR 残響 改善後 | FDN 改善前 | FDN 改善後 |
|---|---|---|---|---|
| hall（1.54秒） | 1.50 ms | 0.19 ms | 0.31 ms | 0.06 ms |
| cathedral（3.00秒） | 3.47 ms | 0.33 ms | 0.13 ms | 0.06 ms |

- IR 畳み込みの残響が約10倍速くなった（効率コアでは cathedral が
  10.05 → 0.69ms。改善前の99%値は20msで、それだけで予算の2倍だった）。
  変えたのは2つ: 和の計算を `einsum` の最適化経路から素の `einsum` に
  した（この形では最適化経路が約8倍遅かった）／過去のブロックの
  スペクトルを毎回ずらして詰め直さず、書く位置を回す（リングバッファ）
- FDN の残響は2〜5倍速くなった。遅延線の読み書きを、毎回作っていた
  添字の配列から、端で1回だけ折り返す切り出しに替えた
- **音質は変わらない。** FDN はビット単位で同じ出力。IR 畳み込みは丸め
  誤差（ピーク比 4e-7 以下、約 -128dB）の範囲で一致し、クリック1つへの
  応答が部屋の IR そのものと 1.5e-8 以内で一致することを、7部屋すべてで
  確かめた

## 設計上の注意

### サンプルレートを固定値で書かない

WASAPI 共有モードは PortAudio 側でリサンプリングしないため、Windows の
サウンド設定でそのデバイスに設定されている値と一致しないと
`Invalid sample rate [PaErrorCode -9997]` で開けない。

`SpatialPlayer.open()` はデバイスの `default_samplerate` を読んでから
HRIR をそのレートで構築するので、通常は意識しなくてよい。IR を自前で
読み込む場合は `RoomIr.resampled(rate)` を通すこと。

### 合成 HRIR は前後を区別できない

`HrirSet.synthetic()` は球体頭部モデルで、ITD も頭部遮蔽も sin(方位) にしか
依存しない。したがって **30度と150度の HRIR は数値的に完全に同一**になる。
前後は上記の対策で伝える。

真の前後判別・仰角・頭外定位が必要なら `HrirSet.from_sofa()` で実測データを
読み込む。同梱の Viking では、前後対称な位置ペアの波形差が平均 86.7%
（合成は 0.0%）。

### ステレオ音源は平置きでだけそのまま鳴る

`spatialise=False` ならステレオ資産は 1 音源のまま 2ch で鳴る（BGM・
環境音・ジングル）。中身が左右同一の「偽ステレオ」は自動で 1ch に畳む
ので、無駄に 2ch 分を混ぜない。

一方 `spatialise=True` にステレオ音源を渡すと `ValueError` になる。
方向は mono にしか与えられないためで、黙ってモノラルに畳むと
「ステレオで鳴らしたつもり」に気づけないので、明示的に失敗させている。

### 出力デバイスが消えたとき

USB オーディオの抜き差しや既定デバイスの切り替えでストリームは黙って
死ぬ。**沈黙は目の見えない利用者が唯一気づけない失敗**なので、
ゲームループから `player.check_device()` を毎秒程度呼ぶ。真になったら
`status()` の先頭にその旨が入るので読み上げ、`player.reopen()` で
つなぎ直す（鳴っている音は保たれる）。代わりのデバイスがレート違いの
ときだけ `RuntimeError` になるので、その場合はプレイヤーを作り直す。

### 音源スロットは固定コスト

`max_sources` の行数ぶんを毎ブロック処理する。鳴っている音の数に関係なく
処理時間が一定になるので、最悪ケースが読みやすい。スロットが尽きたときは
黙って捨てずに `RuntimeError` を送出する（鳴らなかった音は目の見えない
プレイヤーには気づけないため）。

## テスト

audiospace のフォルダで次を実行する:

```
python -X utf8 tests\run_all.py
```

これで5スイート計241件がまとめて走る（無音設定は内部で行う。表示の言語は
日本語に固定する）。
個別に走らせる場合:

```
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_tests.py         (ドメイン109件)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_motion_tests.py  (首振り11件)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_quiz_tests.py    (クイズ39件)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_records_tests.py (記録20件)
set AUDIOSPACE_SILENT=1 && python -X utf8 tests\run_app_tests.py     (GUI構成49件)
```

音が実際に空間情報を持っているかを数値で検証する。ITD が人間の
範囲内か、遮蔽側の耳で高域が落ちているか、畳み込みが直接計算と一致するか、
左右反転で対称か、前後で首振りに対する動きが逆になるか、など。

GUI テストはウィジェットの名前、ラベルの生成順序、文字キーが動作に
割り当てられていないこと、テンキー配置と方位の幾何的一致まで検証する。
英語で起動したアプリ（`tests/en_probe.py`、別のプロセス）で、利用者に見える
文に日本語が残っていないことも調べる。

## 聴感確認アプリ

```
python -X utf8 demo\localization_app.py
```

ヘッドホン必須。キーボードのみで操作でき、スクリーンリーダーに対応する。
必要な部品は `python -m pip install -e ".[app]"` で入る。使い方は
`マニュアル.html`（同じ内容の `マニュアル.txt`）にあり、アプリの中では
F1 でキーの説明が開く。zip では `定位確認ツールを起動.bat` から起動できる。

- F2 試聴（音が周囲を回る） / F3 クイズ / F7 訓練 / F8 耳介の選定
- F11 首振りの効果測定（ありなしを1問ごとに交互に出して条件別に集計。
  上達の影響を同一セッション内の差で打ち消す）
- Shift+F3 上下の聴き取り（真上・上45度・水平・下45度の4択。テンキーの
  / で真上、8で上、5で水平、2で下と答える）。方位は難易度の方向から混ぜる。
  仰角を測っている実測 HRTF（Viking）を選んだときだけ使える。合成 HRIR と
  水平面だけのアセットでは上下が同じ音になるため、始めずに理由を読み上げる。
  下は45度まで（Viking の測定範囲）。F12 は PC-Talker と衝突するので使わない
- Ctrl+F3 正面固定の上下の聴き取り（同じ4択）。音は毎回正面から鳴り、
  変わるのは高さだけ。同じ高さが続くこともある（続かないと分かっていると
  「直前と違う方」を選ぶだけで当て推量の水準が上がるため）
- テンキーの * で基準音のオンとオフ（起動時はオン）。上下の聴き取りで、
  問題の音の前に同じ方位・水平の音を鳴らす。「どの段か」を当てる代わりに
  「基準より上か下か」を比べて答えられる。状態は結果のログに残る
- Shift+F2 上下の聞き比べ（採点なし）。テンキーの / 8 5 2 でその高さの音が
  正面で鳴るだけ。押したときは読み上げない（声が音にかぶるため）
- Ctrl+F8 上下用の耳介の選定。F8（前後用）の上下版で、正面固定の上下4択で
  耳介を順に試す。通算成績は `logs/pinna_records_elev.json` に前後用とは
  別に積み上がり、F10 で前後用に続けて読み上げる
- Ctrl+Shift+F2 8方向の聞き比べ（採点なし）。テンキーの1〜9（5を除く）で、その
  方向の音が水平に鳴る。キーの並びがそのまま方向の配置で、いまの自分から見た向き
- 設定欄の「空間を自分で作る」「残響を自分で作る」のボタンで、大きさ・6つの面の
  材質（日本語名、分類つき）・聞き手の位置・反射と残響の強さ、または残響時間などを
  決め、名前を付けて保存する。保存したものは「自作」として選択肢に並ぶ。窓の中の
  「いまの設定」に残響の長さを文で出し、「この設定で鳴らす」で保存前に聞ける。
  保存先は `settings/user_presets.json`（配布物には入れない）。材質の一覧は
  面ごとに合う材質が上、その他は「その他」付きで下。窓の中で家具や人の数も決める
- 設定欄の「家具の量」（何も置かない・少し・普通・多い）と「人の数」で、選んで
  いる空間に家具と人を足す。家具は閉じた部屋のときだけ（床の広さに合わせた数）
- Ctrl+F2 回る向きの聴き取り（24問）。音が自分の周りを3秒で半周し、
  水平なら左回りか右回りか（テンキーの4と6）、頭上を越える音なら後ろから
  前へか前から後ろへか（8と2）を答える。回り始めの方位は8方向からばらばら。
  前後を取り違えると回る向きが逆に聞こえるので、前後の聞き分けを動きで試す。
  頭上は実測 Viking のときだけ出題する（上下の測定がない HRTF では水平だけ）
- 設定欄の上下の手がかり3つ（起動時はすべてオフ）:
  首かしげ（首を左右の肩へ交互に傾ける自動の揺れ。なし・弱め6度・標準12度・
  強め20度）、上下の強調（エンジンの `elevation_emphasis`。上の音ほど3kHz より上を
  明るく、下ほど暗くする。100パーセントで真上 +12dB。F2 の試聴の回る音にも
  かかる）、床の反射（床は耳の3m下。高い音ほど反射音の
  遅れが長い。真上17.5ms、水平9.4ms、下45度0.7ms）。設定はログに残る
- Page Up で15度ずつ上を向き、Page Down で下を向く（真上・真下で止まる）。
  上下の聴き取りでは、顔の向きに関係なく体から見た高さで答える
- F4 再生 / F5 結果 / F6 停止 / F9 設定の読み上げ / F10 通算成績 / F1 キー操作の説明
  （専用の窓が開き、本文にフォーカスが移る。上下の矢印で1行ずつ読み、Escape で閉じると
  元の場所に戻る）
- 回答はテンキー（7 8 9 / 4 6 / 1 2 3 の並びがそのまま方向に対応）。
  出題音が鳴るまで回答は受け付けない（二度押しの誤回答を防ぐ）
- テンキーの0で繰り返し再生のオンとオフ（起動時はオン）。オンのときは、
  答えるまで出題音を間を空けて鳴らし続ける（ノイズバーストなら2.2秒ごと）。
  スクリーンリーダーの読み上げに紛れて聞き逃しても、待てばまた聞ける。
  F6 で止まり、F4 で再開する。状態は結果のログに残る
- 左右の矢印キーで自分の向きを15度ずつ変えられる（前後判別に有効）

結果は設定とともに `logs/` へ自動保存される。どの条件で試したか
分からなくなると比較ができないため。

## 実機確認

```
python -X utf8 tools\verify_realtime.py            # 無音のまま計測
python -X utf8 tools\verify_realtime.py --audible  # 音を出す場合のみ
python -X utf8 tools\bench_wobble.py               # 首振りのコスト
python -X utf8 tools\bench_sofa.py                 # 実測HRTFのコスト
python -X utf8 tools\verify_sofa.py                # SOFA読み込みの検証
```

既定では全音源を gain=0 で鳴らすので何も聞こえない。測っているのは
コールバックの処理時間と途切れの回数。

## ライセンス上の注意

このライブラリ自体は **MIT ライセンス**（Copyright (c) 2026 Takahashi、
全文は `LICENSE`）。組み込んで配布・販売するときは、著作権表示と
ライセンス文を製品側に載せること。既定 HRTF は MIT の対象外で
CC BY 4.0（下の「同梱データ」）。

必須依存（numpy / scipy）も許容型。任意依存も
sofar・pyroomacoustics・sounddevice・miniaudio・samplerate はすべて MIT か BSD。

一方で **pedalboard（GPLv3）や sound_lib（中身が商用有償の BASS）は
販売するソフトには使えない**。前者は組み込んだソフトのソース公開が
必要になり、後者は商用利用に有償のライセンスが要る。

### 同梱データ

`audiospace/data/default_horizontal.npz`（既定 HRTF）と
`assets/hrtf/viking/` の SOFA は **Viking HRTF Dataset v2（CC BY 4.0）**。
商用利用は可能だが**表示が必須**。組み込んで配布する場合は
`ATTRIBUTION.md` の文面を製品側にも再掲すること（文面は
`attribution()` でも取れる）。

`assets/reverb/` の IR は pyroomacoustics で生成したもので、外部の制約は無い
（本体と同じ MIT で扱う）。

## 支援のお願い

audiospace が役に立ったら、開発の支援をお願いします。支援はまったくの任意で、
してもしなくても、使える機能や条件は変わりません。いただいた支援は、
開発に使う AI などの道具の費用と、聴き取りテストや保守を続ける時間に充てます。

支援の窓口は準備中です。

組み込みの相談、視覚障害者向けの音響設計の助言、聴き取りによる評価、
機能追加のご依頼は、有償でお受けします。連絡先も準備中です。

## Support

If audiospace is useful to you, please consider supporting its development.
Support is entirely optional and does not change what you can do with it.
It goes toward the cost of development tools, including AI, and the time
spent on listening tests and maintenance.

A way to support the project is in preparation.

Paid help is available for integrating audiospace, advice on audio design
for blind and visually impaired players, listening evaluations and feature
requests. Contact details are in preparation.
