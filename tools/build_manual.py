# -*- coding: utf-8 -*-
"""配布する文書（マニュアルと「はじめにお読みください」）を作り直す。

    python -X utf8 tools/build_manual.py          今のコードに合わせて作り直す
    python -X utf8 tools/build_manual.py --check  作り直しが要るかだけ調べる
                                                  （要れば 1 で終わる。書き換えない）
    python -X utf8 tools/build_manual.py --lang en  英語の文書だけ（ja なら日本語だけ）

作るもの（日本語と英語の2組）:

- マニュアル.html / Manual (English).html  元の文書。コードから作れる部分
  （キーの一覧、答えのキー、設定欄の項目と起動時の値、モードごとの問題数、
  空間・残響・置けるもの・材質の一覧、表示文、版の番号）は、印の間を作り
  直す。手で書き写すと、コードを変えたときに文書だけ古いまま残る
- マニュアル.txt / Manual (English).txt  html から作る（tools/html_to_txt.py）。
  上から順に読む人向け。BOM 付き UTF-8・CRLF（古い日本語のエディタでの
  文字化けを避ける）
- はじめにお読みください.txt / README (English).txt  tools/readme_first.txt と
  tools/readme_first_en.txt の {VERSION} を埋めて作る

印の書き方（HTML のコメント。読み上げにも txt にも出ない）:

    <!-- gen:keys -->
    …ここが作り直される…
    <!-- /gen:keys -->

本物の MainFrame を作って設定欄を読み、各モードを始めて問題数を数える。
画面の言語は起動したときに決まる（audiospace.i18n）ので、言語ごとに別の
プロセスで作る（AUDIOSPACE_LANG を渡す）。英語の文書は英語の画面から作る。
音は出さない（AUDIOSPACE_SILENT=1、デバイスを開かない）。自作の空間と残響の
保存先は一時フォルダにする（利用者の保存ファイルを読み書きしない）。
作った html は tools/check_html.py の検査（見出しの階層など）を通す。
"""
from __future__ import annotations

import html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# 音を出さない。読み上げとデバイスは import の時点で決まるものがあるので先に
os.environ["AUDIOSPACE_SILENT"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

#: 言語ごとの (マニュアルの html, txt, はじめにお読みください, その元)
DOCS = {
    "ja": (ROOT / "マニュアル.html", ROOT / "マニュアル.txt",
           ROOT / "はじめにお読みください.txt",
           ROOT / "tools" / "readme_first.txt"),
    "en": (ROOT / "Manual (English).html", ROOT / "Manual (English).txt",
           ROOT / "README (English).txt",
           ROOT / "tools" / "readme_first_en.txt"),
}

MARK_RE = re.compile(
    r"(<!-- gen:(?P<name>[a-z_]+) -->)(?P<body>.*?)(<!-- /gen:(?P=name) -->)",
    re.S)
VERSION_SPAN_RE = re.compile(r'(<span class="version">)[^<]*(</span>)')
TITLE_RE = re.compile(r"(<title>[^<]*?)\d+\.\d+\.\d+([^<]*</title>)")
NL = chr(10)


def tr(ja: str, en: str) -> str:
    """いまの言語のほうの文（audiospace.i18n と同じ決め方）。"""
    from audiospace.i18n import tr as _tr
    return _tr(ja, en)


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def sep() -> str:
    return tr("、", ", ")


def table(intro: str, header: tuple[str, ...], rows: list[tuple]) -> str:
    """表の前に「何行で、どの列か」の段落を付ける（caption は使わない。
    txt では表の後に出てしまうため）。"""
    cols = sep().join(header)
    count = tr(f"表は{len(rows)}行で、列は{cols}です。",
               f" The table has {len(rows)} rows. The columns are {cols}.")
    lines = [f"<p>{esc(intro)}{esc(count)}</p>",
             "<table>", "<thead><tr>"
             + "".join(f'<th scope="col">{esc(h)}</th>' for h in header)
             + "</tr></thead>", "<tbody>"]
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row)
                     + "</tr>")
    lines += ["</tbody>", "</table>"]
    return NL.join(lines)


def no_value() -> str:
    """値を持たない部品（ボタンや一覧）の「起動時の値」。選択肢の「なし」と
    取り違えないよう、別の言葉にする。"""
    return tr("値なし", "no value")


def mnemonic(label: str) -> str:
    """「HRTF (&H)」を「HRTF（Alt+H）」に、「&HRTF」を「HRTF (Alt+H)」にする
    （Alt と文字で移動できる）。英語は文中の & の文字を使うので、最初の文の
    後ろにキーを書く。"""
    m = re.search(r"\s*\(&(\w)\)", label)
    if m:
        key = m.group(1).upper()
        return (label[:m.start()] + tr(f"（Alt+{key}）", f" (Alt+{key})")
                + label[m.end():])
    m = re.search(r"&(\w)", label)
    if not m:
        return label
    plain = label[:m.start()] + label[m.start() + 1:]
    hint = f" (Alt+{m.group(1).upper()})"
    head, dot, tail = plain.partition(". ")
    return f"{head}{hint}{dot}{tail}" if dot else plain + hint


# ---------------------------------------------------------------------------
# アプリの F1 の説明（HELP_TEXT）から作るキーの一覧
# ---------------------------------------------------------------------------
KEY_LINE = {
    "ja": re.compile(
        r"^(?P<key>(?:(?:Ctrl|Shift)\+)*(?:F\d{1,2}|Escape)|テンキーの[0-9*/])"
        r"\s+(?P<desc>\S.*)$"),
    "en": re.compile(
        r"^(?P<key>(?:(?:Ctrl|Shift)\+)*(?:F\d{1,2}|Escape)|Numpad [0-9*/])"
        r"\s+(?P<desc>\S.*)$"),
}

#: 向きを変えるキーの行（説明の文章の行と混ざっているので、この形の行だけ拾う）
TURN_LINE = {
    "ja": re.compile(
        r"^(?P<key>左右の矢印キー|Page Up|Page Down)\s{2,}(?P<desc>\S.*)$"),
    "en": re.compile(
        r"^(?P<key>Left/Right arrows|Page Up|Page Down)\s{2,}(?P<desc>\S.*)$"),
}

#: F1 の説明の節の名前（モード、操作、向きを変える、顔を上下に向ける）
HELP_SECTIONS = {
    "ja": ("モード", "操作", "向きを変える", "顔を上下に向ける"),
    "en": ("Modes", "Controls", "Turning", "Facing up and down"),
}


def lang() -> str:
    from audiospace.i18n import language
    return language()


def help_sections(help_text: str) -> dict[str, list[str]]:
    """節に分ける。名前の括弧書きは除く。

    日本語は「■ 名前」の行、英語は字下げの無い「名前:」の行で区切る。
    """
    sections: dict[str, list[str]] = {}
    current = None
    for line in help_text.split(NL):
        name = None
        if line.startswith("■ "):
            name = line[2:]
        elif (lang() == "en" and line.endswith(":") and line[:1].strip()
              and not KEY_LINE["en"].match(line)):
            name = line[:-1]
        if name is not None:
            current = re.sub(r"\s*\(.*\)$", "", name).strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def parse_key_lines(lines: list[str], section: str) -> list[tuple[str, str]]:
    """「キー  説明」の行と、4字下げの続きの行を組にする。形が違えば止める
    （説明文の書き方が変わったのに黙って一覧から落とさないため）。"""
    items: list[list] = []
    stop = tr("。", ".")
    for line in lines:
        if not line.strip():
            continue
        m = KEY_LINE[lang()].match(line)
        if m:
            items.append([m.group("key"), m.group("desc").strip(), False])
        elif line.startswith("    ") and items:
            # 1行目（「上下の聞き比べ (採点なし)」のような見出し）の後に
            # 続きの文をつなぐときは句点を補う（そのままつなぐと
            # 「(採点なし)テンキーの…」と一続きに読まれる）。2行目以降は
            # 文の途中で折り返しただけなので、そのままつなぐ（英語は空白で）
            if not items[-1][2] and not items[-1][1].endswith(stop):
                items[-1][1] += stop
            joint = "" if lang() == "ja" else " "
            items[-1][1] += joint + line.strip()
            items[-1][2] = True
        else:
            raise ValueError(f"F1 の説明の「{section}」の節に、キーの行として"
                             f"読めない行があります: {line!r}")
    return [(k, d) for k, d, _ in items]


def part_keys(help_text: str) -> str:
    sections = help_sections(help_text)
    modes, controls, turning, facing = HELP_SECTIONS[lang()]
    out = []
    for name, title, intro in (
            (modes, tr("モードのキー", "Mode keys"),
             tr("モードを始めるキーです。", "Keys that start a mode.")),
            (controls, tr("操作のキー", "Control keys"),
             tr("どのモードでも使えるキーです。",
                "Keys that work in any mode."))):
        if name not in sections:
            raise ValueError(f"F1 の説明に「{name}」の節がありません")
        items = parse_key_lines(sections[name], name)
        out.append(f"<h3>{esc(title)}</h3>")
        out.append(f"<p>{esc(intro)}"
                   + esc(tr(f"{len(items)}項目あります。",
                            f" There are {len(items)} of them."))
                   + "</p>")
        out.append("<dl>")
        for key, desc in items:
            out.append(f"<dt>{esc(key)}</dt><dd>{esc(desc)}</dd>")
        out.append("</dl>")
    turns = []
    for name in (turning, facing):
        for line in sections.get(name, []):
            m = TURN_LINE[lang()].match(line)
            if m:
                turns.append((m.group("key"), m.group("desc").strip()))
    if len(turns) != 3:
        raise ValueError(f"F1 の説明から向きを変えるキーを3つ読めません: {turns}")
    out.append("<h3>" + esc(tr("向きを変えるキー", "Turning keys")) + "</h3>")
    out.append("<p>" + esc(tr(f"自分の向きと顔の向きを変えるキーです。"
                              f"{len(turns)}項目あります。音源は動きません。",
                              f"Keys that turn you and your face. There are "
                              f"{len(turns)} of them. The sounds do not "
                              "move.")) + "</p>")
    out.append("<dl>")
    for key, desc in turns:
        out.append(f"<dt>{esc(key)}</dt><dd>{esc(desc)}</dd>")
    out.append("</dl>")
    return NL.join(out)


# ---------------------------------------------------------------------------
# 本物の画面から読むもの
# ---------------------------------------------------------------------------
def numpad_name(code) -> str:
    import wx
    names = {wx.WXK_NUMPAD_DIVIDE: tr("テンキーのスラッシュ", "Numpad slash"),
             wx.WXK_NUMPAD_MULTIPLY: tr("テンキーのアスタリスク",
                                        "Numpad asterisk")}
    for digit in range(10):
        names[getattr(wx, f"WXK_NUMPAD{digit}")] = tr(f"テンキーの{digit}",
                                                      f"Numpad {digit}")
    return names[code]


def part_answers() -> str:
    """答えのキー。アプリが実際に使う対応表（MainFrame の3つの関数）から作る。"""
    import wx

    from demo.localization_app import MainFrame
    from demo.quiz import direction_name, elevation_name, rotation_name
    codes = [wx.WXK_NUMPAD_DIVIDE, wx.WXK_NUMPAD_MULTIPLY] + [
        getattr(wx, f"WXK_NUMPAD{d}") for d in (7, 8, 9, 4, 5, 6, 1, 2, 3, 0)]
    key = tr("キー", "Key")
    out = []
    rows = [(numpad_name(c), direction_name(MainFrame._azimuth_for_key(c)))
            for c in codes if MainFrame._azimuth_for_key(c) is not None]
    out.append("<h3>" + esc(tr("方向の答え", "Direction answers")) + "</h3>")
    out.append(table(tr("クイズ、訓練、耳介の選定、首振りの効果測定で方向を答える"
                        "キーと、8方向の聞き比べで鳴らす方向です。キーの並びが"
                        "そのまま方向の配置です。",
                        "Keys for answering a direction in the quiz, "
                        "training, choosing a pinna and the head wobble "
                        "measurement, and the directions they play when "
                        "comparing 8 directions. The keys are laid out like "
                        "the directions."),
                     (key, tr("方向", "Direction")), rows))
    rows = [(numpad_name(c), elevation_name(MainFrame._elevation_for_key(c)))
            for c in codes if MainFrame._elevation_for_key(c) is not None]
    out.append("<h3>" + esc(tr("高さの答え", "Height answers")) + "</h3>")
    out.append(table(tr("上下の聴き取りと上下用の耳介の選定で高さを答えるキーと、"
                        "上下の聞き比べで鳴らす高さです。上から縦に並んでいます。",
                        "Keys for answering a height in the up/down tests "
                        "and when choosing a pinna for up and down, and the "
                        "heights they play when comparing heights. They run "
                        "down the keypad from top to bottom."),
                     (key, tr("高さ", "Height")), rows))
    rows = [(numpad_name(c), rotation_name(MainFrame._rotation_for_key(c)))
            for c in codes if MainFrame._rotation_for_key(c) is not None]
    out.append("<h3>" + esc(tr("回る向きの答え", "Rotation answers")) + "</h3>")
    out.append(table(tr("回る向きの聴き取りで答えるキーです。音が進む向きを"
                        "キーの位置で表しています。",
                        "Keys for answering in the rotation test. The "
                        "position of the key shows which way the sound "
                        "moves."),
                     (key, tr("回る向き", "Rotation")), rows))
    return NL.join(out)


def make_frame():
    """自作の保存先を一時フォルダにした MainFrame。wx.App は使い回す。"""
    import wx

    from demo.localization_app import MainFrame
    app = wx.App.Get() or wx.App(False)
    tmp = tempfile.TemporaryDirectory(prefix="audiospace_manual_")
    frame = MainFrame(preset_path=Path(tmp.name) / "user_presets.json")
    return app, frame, tmp


def kinds_text(values: list[str]) -> str:
    # 英語は名前の中に「,」があるので、名前と名前の間は「;」で区切る
    return tr(f"{len(values)}種類: ", f"{len(values)} choices: ") + \
        tr("、", "; ").join(values)


def part_settings(frame) -> str:
    """設定欄の項目。画面の部品を作られた順に読む（Tab の順と同じ）。"""
    import wx
    rows = []
    label = None
    choice, number = tr("選択肢", "Choice"), tr("数値", "Number")
    for child in frame.panel.GetChildren():
        if isinstance(child, wx.StaticText):
            label = mnemonic(child.GetLabel())
            continue
        if isinstance(child, wx.Choice):
            values = [child.GetString(i) for i in range(child.GetCount())]
            rows.append((label, choice, child.GetStringSelection(),
                         kinds_text(values)))
        elif isinstance(child, wx.SpinCtrl):
            rows.append((label, number, str(child.GetValue()),
                         tr(f"{child.GetMin()}から{child.GetMax()}まで",
                            f"{child.GetMin()} to {child.GetMax()}")))
        elif isinstance(child, wx.CheckBox):
            rows.append((label, tr(f"チェックボックス（{child.GetLabel()}）",
                                   f"Check box ({child.GetLabel()})"),
                         tr("オン", "On") if child.GetValue()
                         else tr("オフ", "Off"),
                         tr("オンかオフ", "On or off")))
        elif isinstance(child, wx.Button):
            rows.append((mnemonic(child.GetLabel()), tr("ボタン", "Button"),
                         no_value(), tr("押すと窓が開く",
                                        "Opens a window when pressed")))
            continue
        elif isinstance(child, wx.ListBox):
            rows.append((label, tr("一覧", "List"), no_value(),
                         tr("回答の結果が1問ずつ並ぶ",
                            "The result of each answer, one per line")))
        elif isinstance(child, wx.TextCtrl):
            rows.append((label, tr("読み取り専用の文章欄", "Read-only text"),
                         no_value(),
                         tr("起動時はキー操作の説明。モードの案内や結果が入る",
                            "The keyboard help at start; then mode "
                            "guidance and results")))
        label = None
    return table(tr("画面の上から順（Tab で移動する順）に並べました。"
                    "項目の名前の括弧の中は、その項目へ移るキーです。"
                    f"起動時の値が「{no_value()}」の項目は、値を持たない部品です。",
                    "Listed from the top of the screen (the order of Tab). "
                    "The key in brackets after a name moves straight to that "
                    f"item. Items whose value at start is \"{no_value()}\" "
                    "are controls that have no value."),
                 (tr("項目", "Item"), tr("種類", "Type"),
                  tr("起動時の値", "Value at start"),
                  tr("選べる値", "Possible values")), rows)


def part_editor(frame, editor_cls, intro: str) -> str:
    """自分で作る窓の欄。本物の窓を作って、欄を作られた順（Tab の順）に読む。"""
    import wx
    dlg = editor_cls(frame)
    choice, number = tr("選択肢", "Choice"), tr("数値", "Number")
    try:
        rows = []
        label = None
        for child in dlg.body.GetChildren():
            if isinstance(child, wx.StaticText):
                label = child.GetLabel()
                continue
            if isinstance(child, wx.Choice):
                values = [child.GetString(i) for i in range(child.GetCount())]
                if len(values) > 30:
                    shown = tr(f"{len(values)}種類（付録の材質の一覧と同じ。"
                               "その面に合う材質が上）",
                               f"{len(values)} choices (the same as the list "
                               "of materials in the appendix; materials that "
                               "suit the surface come first)")
                else:
                    shown = kinds_text(values)
                rows.append((label, choice, shown))
            elif isinstance(child, wx.SpinCtrl):
                rows.append((label, number,
                             tr(f"{child.GetMin()}から{child.GetMax()}まで",
                                f"{child.GetMin()} to {child.GetMax()}")))
            elif isinstance(child, wx.TextCtrl):
                if child.IsEditable():
                    from demo.user_presets import NAME_MAX
                    rows.append((label, tr("文字", "Text"),
                                 tr(f"{NAME_MAX}文字まで",
                                    f"Up to {NAME_MAX} characters")))
                else:
                    rows.append((label, tr("読み取り専用の文章欄",
                                           "Read-only text"),
                                 tr("いまの値での響きの説明",
                                    "Describes the sound with the current "
                                    "values")))
            label = None
        for child in dlg.GetChildren():
            if isinstance(child, wx.Button):
                rows.append((mnemonic(child.GetLabel()), tr("ボタン", "Button"),
                             tr("押すと実行する", "Acts when pressed")))
    finally:
        dlg.Destroy()
    return table(intro + tr("Tab で移動する順に並べました。",
                            " Listed in the order of Tab."),
                 (tr("欄", "Field"), tr("種類", "Type"),
                  tr("範囲や選択肢", "Range or choices")), rows)


def reset_mode(frame) -> None:
    frame._stop_repeat()
    frame.stop_sound()
    frame.session = None
    frame.mode = "idle"
    frame._accepting = False
    frame._begin_gen += 1           # 予約済みの遅延出題を無効にする


def part_counts(frame) -> str:
    """モードごとの問題数。実際にモードを始めて数える（式を書き写さない）。"""
    from demo.quiz import Difficulty
    diffs = [Difficulty.LEFT_RIGHT, Difficulty.FOUR, Difficulty.EIGHT,
             Difficulty.TWELVE]
    labels = [d.label for d in diffs]

    def count(start, attr="n_trials", per_difficulty=True) -> list[str]:
        values = []
        for d in (diffs if per_difficulty else diffs[:1]):
            frame.cmb_diff.SetSelection(frame.difficulties.index(d))
            start()
            session = frame.session
            if session is None:
                raise RuntimeError(f"{start.__name__} でモードが始まりません")
            values.append(getattr(session, attr))
            reset_mode(frame)
        frame.cmb_diff.SetSelection(1)
        return values

    def per_diff(values) -> str:
        return sep().join(tr(f"{lab}は{v}問", f"{lab}: {v}")
                          for lab, v in zip(labels, values))

    def fixed(start) -> str:
        n = count(start, per_difficulty=False)[0]
        return tr(f"{n}問（難易度によらない）",
                  f"{n} (whatever the difficulty)")

    rows = [
        (tr("クイズ", "Quiz"), "F3", per_diff(count(frame.start_quiz))),
        (tr("訓練", "Training"), "F7",
         tr("続けて正解すると終わる数: ", "Correct in a row to finish: ")
         + per_diff(count(frame.start_training, "target_streak"))),
        (tr("耳介の選定", "Choose a pinna"), "F8",
         per_diff(count(frame.start_calibration))),
        (tr("首振りの効果測定", "Head wobble measurement"), "F11",
         per_diff(count(frame.start_ab))),
        (tr("上下の聴き取り", "Up/down test"), "Shift+F3",
         per_diff(count(frame.start_elevation))),
        (tr("正面固定の上下の聴き取り", "Up/down test from the front"),
         "Ctrl+F3", fixed(frame.start_elevation_front)),
        (tr("上下用の耳介の選定", "Choose a pinna for up and down"),
         "Ctrl+F8", fixed(frame.start_calibration_elev)),
        (tr("回る向きの聴き取り", "Rotation test"), "Ctrl+F2",
         fixed(frame.start_rotation)),
    ]
    return table(tr("採点するモードの問題数です。難易度（方向の数）で変わるものは、"
                    "難易度ごとに書きました。",
                    "The number of questions in the scored modes. Where it "
                    "depends on the difficulty (the number of directions), "
                    "it is given for each difficulty."),
                 (tr("モード", "Mode"), tr("キー", "Key"),
                  tr("問題数", "Questions")), rows)


# ---------------------------------------------------------------------------
# ライブラリの定義から作るもの
# ---------------------------------------------------------------------------
def size_text(w, d, h) -> str:
    return tr(f"幅{w:g}、奥行き{d:g}、高さ{h:g}メートル",
              f"width {w:g}, depth {d:g}, height {h:g} metres")


def seconds(value: float) -> str:
    return tr(f"{value:.1f}秒", f"{value:.1f} seconds")


def part_reverbs() -> str:
    from audiospace import BUILTIN_PRESETS, preset_reverb_spec
    rows = []
    for name, preset in BUILTIN_PRESETS.items():
        rt = preset_reverb_spec(name)["rt60"]
        rows.append((name, preset.description, size_text(*preset.dimensions),
                     seconds(rt)))
    return table(tr("設定欄の「残響」で選べる部屋です。名前は英語のままです。",
                    "Rooms you can choose in Reverb in the settings."),
                 (tr("名前", "Name"), tr("説明", "Description"),
                  tr("大きさ", "Size"), tr("残響時間", "Reverb time")), rows)


def _materials_text(space) -> str:
    from audiospace import FACE_LABELS, material_label
    mats = space.materials

    def say(face_label: str, mat: str) -> str:
        return tr(f"{face_label}は{material_label(mat)}",
                  f"{face_label}: {material_label(mat)}")

    walls = [mats[f] for f in ("north", "south", "west", "east")]
    parts = [say(FACE_LABELS["floor"], mats["floor"])]
    if all(w == walls[0] for w in walls):
        if walls[0] != "open":
            parts.append(say(tr("壁", "Walls"), walls[0]))
    else:
        for face in ("west", "east", "north", "south"):
            if mats[face] != "open":
                parts.append(say(FACE_LABELS[face], mats[face]))
    if mats["ceiling"] != "open":
        parts.append(say(FACE_LABELS["ceiling"], mats["ceiling"]))
    return tr("、", "; ").join(parts)


def part_spaces() -> str:
    from audiospace import SPACE_KINDS, space_labels, space_preset, space_to_spec
    from audiospace.acoustics import band_centers
    mid = band_centers().index(1000)
    rows = []
    kinds = dict(SPACE_KINDS)
    for key, label in space_labels():
        space = space_preset(key)
        kind = space_to_spec(space)["kind"]
        w, d, h = space.size
        if kind == "room":
            size = size_text(w, d, h)
        elif kind == "street":
            size = tr(f"道の幅{w:g}、長さ{d:g}、建物の高さ{h:g}メートル",
                      f"street width {w:g}, length {d:g}, building height "
                      f"{h:g} metres")
        else:
            size = tr("広さの限り無し", "no limit")
        rt = space.rt60_bands()[mid]
        rt_text = (seconds(rt) if rt > 0
                   else tr("無し（開けた場所）", "none (open place)"))
        rows.append((label, kinds[kind], size, _materials_text(space), rt_text))
    return table(tr("設定欄の「空間」で選べる場所です。残響時間は中ほどの高さの音"
                    "（1キロヘルツ）のものです。",
                    "Places you can choose in Space in the settings. The "
                    "reverb time is for middle sounds (1 kilohertz)."),
                 (tr("名前", "Name"), tr("種類", "Kind"), tr("大きさ", "Size"),
                  tr("材質", "Materials"), tr("残響時間", "Reverb time")), rows)


def part_furnishings() -> str:
    from audiospace import (FURNISH_LEVELS, FURNISHINGS, contents_text,
                            furnish_counts)
    from audiospace.acoustics import band_centers
    mid = band_centers().index(1000)

    def area(value: float) -> str:
        return tr(f"{value:g}平方メートル", f"{value:g} square metres")

    rows = [(f.label, area(f.absorption_area[mid]), area(f.blocking_area))
            for f in FURNISHINGS.values()]
    out = [table(tr("空間に置けるものです。吸音の量は、中ほどの高さの音"
                    "（1キロヘルツ）で、完全に音を吸う面の広さに直した値です。"
                    "ふさぐ広さが大きいほど、床や壁からの反射を散らします。",
                    "Things you can put in a space. The absorption is for "
                    "middle sounds (1 kilohertz), given as the area of a "
                    "surface that absorbs completely. The larger the "
                    "blocking area, the more it scatters reflections from "
                    "the floor and walls."),
                 (tr("置けるもの", "Thing"), tr("吸音の量", "Absorption"),
                  tr("ふさぐ広さ", "Blocking area")), rows)]
    rows = [(label, contents_text(furnish_counts(20.0, key)))
            for key, label in FURNISH_LEVELS]
    out.append(table(tr("「家具の量」で選ぶと置かれるものです。床の広さに比例させ、"
                        "ここでは床20平方メートル（例えば5メートル×4メートル）の"
                        "部屋の数を書きました。人は含みません。",
                        "What each Furniture amount puts in the room. It "
                        "grows with the floor area; the numbers here are for "
                        "a floor of 20 square metres (for example 5 by 4 "
                        "metres). People are not included."),
                     (tr("家具の量", "Furniture amount"),
                      tr("床20平方メートルあたり", "Per 20 square metres of "
                         "floor")), rows))
    return NL.join(out)


def part_materials() -> str:
    from audiospace import material_choices
    rows = []
    for key, label in material_choices():
        category, name = label.split(": ", 1)
        rows.append((category, name, key))
    return table(tr("材質を分類の順に並べました。名前（プログラム）は、設計図や"
                    "プログラムで材質を指定するときに書く名前です。",
                    "Materials in the order of their categories. The name "
                    "(program) is what you write in a design or a program "
                    "to choose the material."),
                 (tr("分類", "Category"), tr("日本語の名前", "Name"),
                  tr("名前（プログラム）", "Name (program)")), rows)


def part_attribution() -> str:
    from audiospace import attribution
    # 日本語の文書の中の英文には lang を付ける（読み上げの声を切り替える）
    attr = ' lang="en"' if lang() == "ja" else ""
    return f"<p{attr}>{esc(attribution())}</p>"


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------
def build_parts() -> dict[str, str]:
    from demo.localization_app import HELP_TEXT
    parts = {
        "keys": part_keys(HELP_TEXT),
        "answers": part_answers(),
        "reverbs": part_reverbs(),
        "spaces": part_spaces(),
        "furnishings": part_furnishings(),
        "materials": part_materials(),
        "attribution": part_attribution(),
    }
    app, frame, tmp = make_frame()
    try:
        parts["settings"] = part_settings(frame)
        from demo.preset_editor import ReverbEditor, SpaceEditor
        parts["space_editor"] = part_editor(
            frame, SpaceEditor, tr("「空間を自分で作る」の窓の欄です。",
                                   "The fields of the Design a space "
                                   "window."))
        parts["reverb_editor"] = part_editor(
            frame, ReverbEditor, tr("「残響を自分で作る」の窓の欄です。",
                                    "The fields of the Design a reverb "
                                    "window."))
        if not frame.ensure_player():
            raise RuntimeError("音声エンジンを用意できません")
        parts["counts"] = part_counts(frame)
    finally:
        reset_mode(frame)
        if frame.player is not None:
            frame.player.stop()
        frame.Destroy()
        tmp.cleanup()
    return parts


def apply_parts(text: str, parts: dict[str, str], version: str) -> str:
    """印の間を作り直し、版の番号を合わせる。知らない印があれば止める
    （綴り違いの印は、黙って古い中身のまま残るので）。"""
    names = [m.group("name") for m in MARK_RE.finditer(text)]
    unknown = [n for n in names if n not in parts]
    if unknown:
        raise ValueError(f"作り方を知らない印があります: {unknown}")
    unused = [n for n in parts if n not in names]
    if unused:
        raise ValueError(f"マニュアルに印がありません（作った節が使われない）: "
                         f"{unused}")

    def repl(m: re.Match) -> str:
        return f"{m.group(1)}{NL}{parts[m.group('name')]}{NL}{m.group(4)}"

    text = MARK_RE.sub(repl, text)
    text = VERSION_SPAN_RE.sub(lambda m: f"{m.group(1)}{version}{m.group(2)}",
                               text)
    text = TITLE_RE.sub(lambda m: f"{m.group(1)}{version}{m.group(2)}", text)
    return text


def to_txt(html_text: str) -> str:
    from html_to_txt import DocumentToText
    parser = DocumentToText()
    parser.feed(html_text)
    text = parser.render()
    if len(text) < 2000:
        # 変換器の深度が狂うと、例外なしで中身がほとんど消える
        raise RuntimeError(f"txt が短すぎます（{len(text)} 文字）")
    return text


def encode_txt(text: str) -> bytes:
    """BOM 付き UTF-8・CRLF。"""
    body = text.replace("\r\n", NL).replace(NL, "\r\n")
    return "﻿".encode("utf-8") + body.encode("utf-8")


def check_html_text(html_text: str) -> None:
    import check_html
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manual_check.html"
        path.write_text(html_text, encoding="utf-8")
        failures = check_html.check_file(path)
    if failures:
        raise RuntimeError("マニュアルの構造の検査に通りません: "
                           + "、".join(failures))


def build_outputs() -> dict[Path, bytes]:
    """いまの言語の、作るべき3つのファイルの中身。"""
    import audiospace
    version = audiospace.__version__
    manual_html, manual_txt, readme_first, template_path = DOCS[lang()]
    raw = manual_html.read_bytes().decode("utf-8")
    source = raw.replace("\r\n", NL)
    html_text = apply_parts(source, build_parts(), version)
    check_html_text(html_text)
    template = template_path.read_text(encoding="utf-8").replace("\r\n", NL)
    readme = template.replace("{VERSION}", version)
    if "{" in readme or "}" in readme:
        raise ValueError(f"{readme_first.name} に埋めていない印が残っています")
    # html は元の改行（作業ツリーの改行）を保つ。git の autocrlf で CRLF に
    # なっていても、中身が同じなら変更ありに見えないように
    newline = "\r\n" if b"\r\n" in manual_html.read_bytes() else NL
    return {
        manual_html: html_text.replace(NL, newline).encode("utf-8"),
        manual_txt: encode_txt(to_txt(html_text)),
        readme_first: encode_txt(readme),
    }


def update(check_only: bool = False) -> list[Path]:
    """いまの言語の文書を作り直す（check_only なら調べるだけ）。中身が変わる
    （変わるべき）ファイルを返す。"""
    changed = []
    for path, data in build_outputs().items():
        before = path.read_bytes() if path.exists() else b""
        if before.replace(b"\r\n", b"\n") != data.replace(b"\r\n", b"\n") or (
                path.suffix == ".txt" and before != data):
            changed.append(path)
            if not check_only:
                path.write_bytes(data)
    return changed


def run_one(language: str, check_only: bool) -> int:
    """1つの言語の文書を、このプロセスで作る（言語は import の前に決める）。"""
    os.environ["AUDIOSPACE_LANG"] = language
    from audiospace.i18n import language as current
    if current() != language:
        raise RuntimeError(f"表示の言語が {current()} のままです"
                           f"（{language} で作るには別のプロセスが要ります）")
    changed = update(check_only=check_only)
    for path in changed:
        verb = "作り直しが必要" if check_only else "作り直しました"
        print(f"{verb}: {path.name}")
    if not changed:
        print(f"文書は今のコードと合っています（{language}）")
    return 1 if (check_only and changed) else 0


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    if "--lang" in argv:
        language = argv[argv.index("--lang") + 1]
        if language not in DOCS:
            print(f"--lang は {' か '.join(DOCS)} です")
            return 2
        return run_one(language, check_only)
    # 言語ごとに別のプロセスで作る（画面の言語は起動したときに決まるため）
    code = 0
    for language in DOCS:
        args = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                "--lang", language] + (["--check"] if check_only else [])
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             env=dict(os.environ, AUDIOSPACE_LANG=language,
                                      AUDIOSPACE_SILENT="1"))
        # 検査の経過（check_html の OK の行）は省き、結果の行だけを出す
        for line in out.stdout.splitlines():
            if line.startswith(("作り直し", "文書は")):
                print(line)
        if out.returncode != 0:
            if out.returncode != 1 or not check_only:
                print(out.stdout[-1500:])
                print(out.stderr[-2000:])
            code = max(code, out.returncode)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
