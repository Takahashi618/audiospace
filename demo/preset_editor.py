"""自分で空間と残響を作る窓（設定欄のボタンから開く）。

キーボードだけで操作でき、スクリーンリーダーで読めるように作る:

- 欄は Tab の順に、ラベルを先に作ってからコントロールを作る（直前の
  StaticText が読み上げ名になるので、単位まで含めた文にする）
- 数値は整数の wx.SpinCtrl（SpinCtrlDouble は PC-Talker で名前が読めない。
  小数は「センチメートル」「パーセント」の整数に直す）
- 「いまの設定」の欄（読み取り専用）に、残響の長さなどを文で出す。値を
  変えるたびに書き換えるので、Tab で戻れば確かめられる
- 「この設定で鳴らす」で、保存する前に正面の音で聞ける
- 削除は2回押す（確認の窓を開かない。モーダルの窓はテストから開けず、
  読み上げの流れも切れる）
- モーダルにしない。Escape で閉じると、設定欄で選んでいる響きに戻す

窓が使うのは、親（MainFrame）の次のもの: `user_presets`（保存先）、
`speech`、`preview_space(spec)`、`preview_reverb(spec)`、`end_preview()`、
`on_user_presets_changed(kind, name)`、`current_space_spec()`、
`current_reverb_spec()`。
"""
from __future__ import annotations

import wx

from audiospace import (FACE_LABELS, FURNISH_LEVELS, SPACE_KINDS,
                        contents_text, furnish_counts, furnishing_labels,
                        material_choices, preset_names, preset_reverb_spec,
                        reverb_text, space_from_spec, space_labels,
                        space_preset, space_to_spec)
from audiospace.i18n import tr

#: 自分で作ったものの選択肢の頭
USER_PREFIX = tr("自作: ", "Yours: ")

#: 空間の面の並び（Tab の順）
FACE_ORDER = ("floor", "ceiling", "north", "south", "west", "east")


class _EditorBase(wx.Dialog):
    """2つの窓に共通の部分（元にする選択肢、名前、いまの設定、ボタン）。"""

    #: "space" か "reverb"
    kind = ""
    #: 窓のタイトル
    title_text = ""

    def __init__(self, parent) -> None:
        super().__init__(parent, title=self.title_text, size=(720, 640),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.frame = parent
        self.SetName(self.title_text)
        self._delete_armed: str | None = None
        self._loading = False
        outer = wx.BoxSizer(wx.VERTICAL)
        # 欄が多いので、文字が大きいときに潰れないよう ScrolledWindow に置く
        self.body = wx.ScrolledWindow(self, style=wx.TAB_TRAVERSAL | wx.VSCROLL)
        self.body.SetScrollRate(0, 20)
        self.grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=10)
        self.grid.AddGrowableCol(1, 1)

        self.sources = self._source_items()
        lbl = self.label(tr("元にする響き。選ぶと、その値を下の欄に読み込みます",
                            "Start from. Choosing one loads its values into "
                            "the fields below"))
        self.cmb_source = wx.Choice(self.body,
                                    choices=[v for _, v in self.sources])
        self.cmb_source.SetName(tr("元にする響き", "Start from"))
        self.add(lbl, self.cmb_source)

        lbl = self.label(tr("名前。保存するときに付けます。同じ名前は上書きします",
                            "Name. Used when saving. Saving with the same "
                            "name replaces it"))
        self.txt_name = wx.TextCtrl(self.body)
        self.txt_name.SetName(tr("名前", "Name"))
        self.add(lbl, self.txt_name)

        self.build_fields()

        lbl = self.label(tr("いまの設定", "Current settings"))
        self.txt_info = wx.TextCtrl(
            self.body, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        self.txt_info.SetName(tr("いまの設定の説明",
                                 "Description of the current settings"))
        self.txt_info.SetMinSize(wx.Size(-1, 90))
        self.add(lbl, self.txt_info)

        self.body.SetSizer(self.grid)
        self.body.FitInside()
        outer.Add(self.body, 1, wx.EXPAND | wx.ALL, 8)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_play = wx.Button(self, label=tr("この設定で鳴らす (&P)",
                                                 "&Play with these settings"))
        self.btn_play.SetName(tr("この設定で鳴らす", "Play with these settings"))
        self.btn_save = wx.Button(self, label=tr("保存 (&S)", "&Save"))
        self.btn_save.SetName(tr("保存", "Save"))
        self.btn_delete = wx.Button(self, label=tr("削除 (&D)", "&Delete"))
        self.btn_delete.SetName(tr("削除", "Delete"))
        self.btn_close = wx.Button(self, wx.ID_CLOSE, label=tr("閉じる", "Close"))
        self.btn_close.SetName(tr("閉じる", "Close"))
        for b in (self.btn_play, self.btn_save, self.btn_delete,
                  self.btn_close):
            buttons.Add(b, 0, wx.ALL, 4)
        outer.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALL, 4)
        self.SetSizer(outer)

        self.cmb_source.Bind(wx.EVT_CHOICE, self.on_source)
        self.btn_play.Bind(wx.EVT_BUTTON, lambda e: self.preview())
        self.btn_save.Bind(wx.EVT_BUTTON, lambda e: self.save())
        self.btn_delete.Bind(wx.EVT_BUTTON, lambda e: self.delete())
        self.btn_close.Bind(wx.EVT_BUTTON, lambda e: self.Close())
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key)
        # 窓の中の F1 で親が説明の窓を開き直さないように止める
        self.Bind(wx.EVT_HELP, lambda e: None)

    # -- 組み立ての道具 -------------------------------------------------
    def label(self, text: str) -> wx.StaticText:
        """ラベルを先に作る（読み上げ名の関連付けは生成順で決まる）。"""
        static = wx.StaticText(self.body, label=text)
        static.SetName(text)
        return static

    def add(self, static: wx.StaticText, ctrl: wx.Window) -> None:
        self.grid.Add(static, 0, wx.ALIGN_CENTER_VERTICAL)
        self.grid.Add(ctrl, 0, wx.EXPAND)

    def spin(self, text: str, lo: int, hi: int, value: int) -> wx.SpinCtrl:
        lbl = self.label(text)
        ctrl = wx.SpinCtrl(self.body, min=lo, max=hi, initial=value)
        ctrl.SetName(text)
        self.add(lbl, ctrl)
        ctrl.Bind(wx.EVT_SPINCTRL, self.on_field)
        ctrl.Bind(wx.EVT_TEXT, self.on_field)
        return ctrl

    # -- 共通の動き -----------------------------------------------------
    def on_key(self, event: wx.KeyEvent) -> None:
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.Close()
            return
        event.Skip()

    def on_field(self, event=None) -> None:
        if self._loading:
            return
        self._delete_armed = None
        self.update_info()

    def on_source(self, event=None) -> None:
        key = self.sources[max(0, self.cmb_source.GetSelection())][0]
        spec = self.spec_of_source(key)
        self._loading = True
        try:
            self.load_spec(spec)
            # 自分で作ったものなら名前も入れる（直して上書きしやすく）。
            # 用意したものは空にして、新しい名前を付けてもらう
            self.txt_name.SetValue(key[1] if key[0] == "user" else "")
        finally:
            self._loading = False
        self._delete_armed = None
        self.update_info()

    def select_source(self, key) -> None:
        for i, (k, _) in enumerate(self.sources):
            if k == key:
                self.cmb_source.SetSelection(i)
                break
        else:
            self.cmb_source.SetSelection(0)
        self.on_source()

    def preview(self) -> None:
        try:
            spec = self.collect_spec()
            self.preview_spec(spec)
        except ValueError as exc:
            self.frame.speech.say(tr(f"鳴らせません。{exc}",
                                     f"Cannot play. {exc}"))

    def save(self) -> None:
        name = self.txt_name.GetValue().strip()
        try:
            spec = self.collect_spec()
            replaced = self.put(name, spec)
        except ValueError as exc:
            self.frame.speech.say(tr(f"保存できません。{exc}",
                                     f"Cannot save. {exc}"))
            return
        except OSError as exc:
            self.frame.speech.say(tr(f"ファイルに書けませんでした。{exc}",
                                     f"Could not write the file. {exc}"))
            return
        self._delete_armed = None
        self.refresh_sources(("user", name))
        self.frame.on_user_presets_changed(self.kind, name)
        if replaced:
            verb = tr("上書きしました", "Replaced")
        else:
            verb = tr("保存しました", "Saved")
        self.frame.speech.say(tr(f"「{name}」を{verb}。設定欄の{self.choice_name}で"
                                 "選んであります。",
                                 f'{verb} "{name}". It is selected in '
                                 f"{self.choice_name} in the settings."))

    def delete(self) -> None:
        name = self.txt_name.GetValue().strip()
        if name not in self.store_items():
            self.frame.speech.say(tr("消せるのは自分で作って保存したものだけです。"
                                     "名前の欄に、消したいものの名前を入れて"
                                     "ください。",
                                     "Only things you made and saved can be "
                                     "deleted. Type the name of the one to "
                                     "delete in the Name field."))
            return
        if self._delete_armed != name:
            self._delete_armed = name
            self.frame.speech.say(tr(f"「{name}」を消すには、もう一度削除を"
                                     "押してください。",
                                     f'To delete "{name}", press Delete '
                                     "again."))
            return
        try:
            self.remove(name)
        except OSError as exc:
            self.frame.speech.say(tr(f"ファイルに書けませんでした。{exc}",
                                     f"Could not write the file. {exc}"))
            return
        self._delete_armed = None
        self.refresh_sources(None)
        self.frame.on_user_presets_changed(self.kind, None)
        self.frame.speech.say(tr(f"「{name}」を消しました。",
                                 f'Deleted "{name}".'))

    def refresh_sources(self, select) -> None:
        self.sources = self._source_items()
        self.cmb_source.Set([v for _, v in self.sources])
        for i, (k, _) in enumerate(self.sources):
            if k == select:
                self.cmb_source.SetSelection(i)
                return
        self.cmb_source.SetSelection(0)

    def _source_items(self) -> list[tuple[tuple, str]]:
        items = [(("preset", k), v) for k, v in self.preset_items()]
        items += [(("user", n), f"{USER_PREFIX}{n}") for n in self.store_items()]
        return items

    # -- それぞれの窓で決めるもの ------------------------------------------
    choice_name = ""

    def preset_items(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def store_items(self) -> dict:
        raise NotImplementedError

    def build_fields(self) -> None:
        raise NotImplementedError

    def spec_of_source(self, key) -> dict:
        raise NotImplementedError

    def load_spec(self, spec: dict) -> None:
        raise NotImplementedError

    def collect_spec(self) -> dict:
        raise NotImplementedError

    def update_info(self) -> None:
        raise NotImplementedError

    def preview_spec(self, spec: dict) -> None:
        raise NotImplementedError

    def put(self, name: str, spec: dict) -> bool:
        raise NotImplementedError

    def remove(self, name: str) -> None:
        raise NotImplementedError


class SpaceEditor(_EditorBase):
    """材質つきの空間を自分で作る窓。"""

    kind = "space"
    title_text = tr("空間を自分で作る。Escape で閉じます",
                    "Design a space. Escape closes")
    choice_name = tr("空間", "Space")

    def preset_items(self) -> list[tuple[str, str]]:
        return space_labels()

    def store_items(self) -> dict:
        return self.frame.user_presets.spaces

    def build_fields(self) -> None:
        lbl = self.label(tr("種類", "Kind"))
        self.kind_keys = [k for k, _ in SPACE_KINDS]
        self.cmb_kind = wx.Choice(self.body,
                                  choices=[v for _, v in SPACE_KINDS])
        self.cmb_kind.SetName(tr("空間の種類", "Kind of space"))
        self.add(lbl, self.cmb_kind)
        self.cmb_kind.Bind(wx.EVT_CHOICE, self.on_kind)

        self.spn_w = self.spin(tr("幅（左右）、センチメートル。道なら道の幅",
                                  "Width (left to right), centimetres. For a "
                                  "street, the street width"),
                               50, 50000, 500)
        self.spn_d = self.spin(tr("奥行き（前後）、センチメートル。道なら道の長さ",
                                  "Depth (front to back), centimetres. For a "
                                  "street, the street length"),
                               50, 50000, 400)
        self.spn_h = self.spin(tr("高さ、センチメートル。道なら建物の高さ",
                                  "Height, centimetres. For a street, the "
                                  "building height"),
                               50, 50000, 270)

        # 材質の一覧は面ごとに、その面に合うものを上に出す（床なら床材、
        # 天井なら天井材）。その他は「その他」を頭に付けて下に並べる
        self.material_keys: dict[str, list[str]] = {}
        self.cmb_mat: dict[str, wx.Choice] = {}
        for face in FACE_ORDER:
            items = material_choices(face)
            self.material_keys[face] = [k for k, _ in items]
            text = tr(f"{FACE_LABELS[face]}の材質",
                      f"{FACE_LABELS[face]} material")
            lbl = self.label(text)
            ctrl = wx.Choice(self.body, choices=[v for _, v in items])
            ctrl.SetName(text)
            self.add(lbl, ctrl)
            ctrl.Bind(wx.EVT_CHOICE,
                      lambda e, f=face: self.on_material(f))
            self.cmb_mat[face] = ctrl
        # 吸音率を数値で直接書いた材質（手で編集した保存ファイルなど）。
        # 一覧に無いので、選び直さない限り元の値を保つ
        self._custom_mats: dict[str, list] = {}

        self.spn_ax = self.spin(tr("聞き手の位置、左の壁からのパーセント",
                                   "Listener position, percent from the left "
                                   "wall"),
                                5, 95, 42)
        self.spn_ay = self.spin(tr("聞き手の位置、後ろの壁からのパーセント",
                                   "Listener position, percent from the back "
                                   "wall"),
                                5, 95, 38)
        # 置いてあるもの（家具や人）。部屋の大きさは同じでも、物があれば
        # 残響が短くなり、反射が散らされる
        lbl = self.label(tr("家具の量（部屋のときだけ）。選ぶと、床の広さに"
                            "合わせて下の家具の数を入れます",
                            "Amount of furniture (rooms only). Choosing one "
                            "fills in the furniture numbers below to suit "
                            "the floor area"))
        self.furnish_keys = [None] + [k for k, _ in FURNISH_LEVELS]
        self.cmb_furnish = wx.Choice(
            self.body, choices=[tr("下の数をそのまま使う",
                                   "Use the numbers below as they are")]
            + [v for _, v in FURNISH_LEVELS])
        self.cmb_furnish.SetName(tr("家具の量", "Amount of furniture"))
        self.add(lbl, self.cmb_furnish)
        self.cmb_furnish.Bind(wx.EVT_CHOICE, self.on_furnish)
        self.spn_contents: dict[str, wx.SpinCtrl] = {}
        for key, name in furnishing_labels():
            self.spn_contents[key] = self.spin(
                tr(f"{name}の数", f"{name[:1].upper()}{name[1:]} count"),
                0, 1000, 0)

        self.spn_order = self.spin(tr("反射を追う回数（向かい合う面の間）",
                                      "Reflection order (between facing "
                                      "surfaces)"),
                                   1, 8, 4)
        self.spn_early = self.spin(tr("反射の強さ、パーセント",
                                      "Reflection level, percent"),
                                   0, 300, 100)
        self.spn_late = self.spin(tr("残響の強さ、パーセント",
                                     "Reverb level, percent"),
                                  0, 300, 100)

    def spec_of_source(self, key) -> dict:
        if key[0] == "user":
            return dict(self.frame.user_presets.spaces[key[1]])
        return space_to_spec(space_preset(key[1]))

    def load_spec(self, spec: dict) -> None:
        kind = spec.get("kind", "room")
        self.cmb_kind.SetSelection(self.kind_keys.index(kind))
        size = spec.get("size", [5.0, 4.0, 2.7])
        for ctrl, v in zip((self.spn_w, self.spn_d, self.spn_h), size):
            ctrl.SetValue(int(round(float(v) * 100)))
        mats = spec.get("materials", {})
        self._custom_mats = {}
        for face in FACE_ORDER:
            name = mats.get(face, "open")
            keys = self.material_keys[face]
            if isinstance(name, str) and name in keys:
                self.cmb_mat[face].SetSelection(keys.index(name))
            else:
                # 一覧に無い材質は、近い名前にも開いた面にもせず元の値を保つ
                self.cmb_mat[face].SetSelection(0)
                self._custom_mats[face] = name
        at = spec.get("at", [0.42, 0.38])
        self.spn_ax.SetValue(int(round(float(at[0]) * 100)))
        self.spn_ay.SetValue(int(round(float(at[1]) * 100)))
        contents = dict(spec.get("contents", {}))
        for key, ctrl in self.spn_contents.items():
            ctrl.SetValue(int(contents.get(key, 0)))
        self.cmb_furnish.SetSelection(0)
        self.spn_order.SetValue(int(spec.get("early_order", 4)))
        self.spn_early.SetValue(int(round(float(spec.get("early_gain", 1.0))
                                          * 100)))
        self.spn_late.SetValue(int(round(float(spec.get("late_gain", 1.0))
                                         * 100)))
        self.update_enabled()

    def collect_spec(self) -> dict:
        kind = self.kind_keys[max(0, self.cmb_kind.GetSelection())]
        mats = {face: self._custom_mats.get(
                    face, self.material_keys[face][max(0, ctrl.GetSelection())])
                for face, ctrl in self.cmb_mat.items()}
        spec = {"kind": kind, "name": self.txt_name.GetValue().strip(),
                "materials": mats,
                "early_order": int(self.spn_order.GetValue()),
                "early_gain": self.spn_early.GetValue() / 100.0,
                "late_gain": self.spn_late.GetValue() / 100.0}
        contents = {k: int(c.GetValue()) for k, c in self.spn_contents.items()
                    if c.GetValue() > 0}
        if contents:
            spec["contents"] = contents
        if kind != "open":
            spec["size"] = [self.spn_w.GetValue() / 100.0,
                            self.spn_d.GetValue() / 100.0,
                            self.spn_h.GetValue() / 100.0]
            spec["at"] = [self.spn_ax.GetValue() / 100.0,
                          self.spn_ay.GetValue() / 100.0]
        if kind == "street":
            for face in ("south", "north", "ceiling"):
                mats[face] = "open"
        if kind == "open":
            for face in ("west", "east", "south", "north", "ceiling"):
                mats[face] = "open"
        return spec

    def on_furnish(self, event=None) -> None:
        """家具の量を選んだら、床の広さに合わせた数を家具の欄に入れる。

        人の数は変えない（人は家具の量とは別に決める）。
        """
        level = self.furnish_keys[max(0, self.cmb_furnish.GetSelection())]
        kind = self.kind_keys[max(0, self.cmb_kind.GetSelection())]
        # 道や開けた場所は床がとても広く、広さに合わせると何十台もの
        # ソファになる。設定欄の「家具の量」と同じく部屋のときだけ
        if level is None or kind != "room":
            return
        floor = (self.spn_w.GetValue() / 100.0) * (self.spn_d.GetValue() / 100.0)
        counts = furnish_counts(floor, level)
        self._loading = True
        try:
            for key, ctrl in self.spn_contents.items():
                if key in ("person", "seated_person"):
                    continue
                ctrl.SetValue(counts.get(key, 0))
        finally:
            self._loading = False
        self.on_field()

    def on_material(self, face: str) -> None:
        self._custom_mats.pop(face, None)     # 選び直したら一覧の材質を使う
        self.on_field()

    def on_kind(self, event=None) -> None:
        self.update_enabled()
        self.on_field()

    def update_enabled(self) -> None:
        """種類に合わない欄は使えなくする（Tab でも止まらない）。"""
        kind = self.kind_keys[max(0, self.cmb_kind.GetSelection())]
        used = {"room": set(FACE_ORDER),
                "street": {"floor", "west", "east"},
                "open": {"floor"}}[kind]
        for face, ctrl in self.cmb_mat.items():
            ctrl.Enable(face in used)
        for ctrl in (self.spn_w, self.spn_d, self.spn_h, self.spn_ax,
                     self.spn_ay):
            ctrl.Enable(kind != "open")
        self.spn_ay.Enable(kind == "room")
        self.cmb_furnish.Enable(kind == "room")

    def update_info(self) -> None:
        try:
            spec = self.collect_spec()
            space = space_from_spec(spec)
        except ValueError as exc:
            self.txt_info.SetValue(tr(f"この値では作れません。{exc}",
                                      f"These values cannot make a space. "
                                      f"{exc}"))
            return
        kind = spec["kind"]
        lines = []
        if kind == "street":
            lines.append(tr("建物の間の道です。前後と空は開いていて、"
                            "使う材質は地面（床）と左右の建物の壁だけです。",
                            "A street between buildings. The front, the back "
                            "and the sky are open; only the ground (floor) "
                            "and the building walls on the left and right "
                            "use materials."))
        elif kind == "open":
            lines.append(tr("周りに何もない場所です。地面（床）の反射だけが"
                            "あります。",
                            "An open place. Only the ground (floor) "
                            "reflects."))
        lines.append(reverb_text(space, sep="\n"))
        if space.contents:
            lines.append(tr(f"置いてあるもの: {contents_text(space.contents)}。"
                            f"混み具合 {space.clutter() * 100:.0f}パーセント"
                            "（床の反射ほど弱まります）。",
                            f"Contents: {contents_text(space.contents)}. "
                            f"Clutter {space.clutter() * 100:.0f} percent "
                            "(floor reflections are weakened most)."))
        self.txt_info.SetValue("\n".join(lines))

    def preview_spec(self, spec: dict) -> None:
        self.frame.preview_space(spec)

    def put(self, name: str, spec: dict) -> bool:
        return self.frame.user_presets.put_space(name, spec)

    def remove(self, name: str) -> None:
        self.frame.user_presets.delete_space(name)


class ReverbEditor(_EditorBase):
    """方向の無い残響（FDN）を自分で作る窓。"""

    kind = "reverb"
    title_text = tr("残響を自分で作る。Escape で閉じます",
                    "Design a reverb. Escape closes")
    choice_name = tr("残響", "Reverb")

    def preset_items(self) -> list[tuple[str, str]]:
        return [(n, n) for n in preset_names()]

    def store_items(self) -> dict:
        return self.frame.user_presets.reverbs

    def build_fields(self) -> None:
        self.spn_rt = self.spin(tr("残響時間、ミリ秒（音が60デシベル下がるまで）",
                                   "Reverb time, milliseconds (until the "
                                   "sound drops by 60 decibels)"),
                                100, 8000, 1400)
        self.spn_scale = self.spin(tr("部屋の大きさ、パーセント（響きの粒の間隔）",
                                      "Room size, percent (spacing of the "
                                      "echoes)"),
                                   30, 250, 100)
        self.spn_damp = self.spin(tr("高い音の減衰、パーセント（大きいほどこもる）",
                                     "High damping, percent (more is more "
                                     "muffled)"),
                                  0, 95, 35)
        self.spn_width = self.spin(tr("左右の広がり、パーセント",
                                      "Stereo width, percent"), 0, 100, 100)
        self.spn_level = self.spin(tr("残響の大きさ、パーセント",
                                      "Reverb level, percent"), 0, 300, 100)

    def spec_of_source(self, key) -> dict:
        if key[0] == "user":
            return dict(self.frame.user_presets.reverbs[key[1]])
        return preset_reverb_spec(key[1])

    def load_spec(self, spec: dict) -> None:
        self.spn_rt.SetValue(int(round(float(spec.get("rt60", 1.4)) * 1000)))
        self.spn_scale.SetValue(int(round(float(spec.get("room_scale", 1.0))
                                          * 100)))
        self.spn_damp.SetValue(int(round(float(spec.get("damping", 0.35))
                                         * 100)))
        self.spn_width.SetValue(int(round(float(spec.get("width", 1.0)) * 100)))
        self.spn_level.SetValue(int(round(float(spec.get("level", 1.0)) * 100)))

    def collect_spec(self) -> dict:
        return {"name": self.txt_name.GetValue().strip(),
                "rt60": self.spn_rt.GetValue() / 1000.0,
                "room_scale": self.spn_scale.GetValue() / 100.0,
                "damping": self.spn_damp.GetValue() / 100.0,
                "width": self.spn_width.GetValue() / 100.0,
                "level": self.spn_level.GetValue() / 100.0}

    def update_info(self) -> None:
        spec = self.collect_spec()
        lines = [tr(f"残響時間 {spec['rt60']:.1f}秒、部屋の大きさ "
                    f"{spec['room_scale'] * 100:.0f}パーセント、高い音の減衰 "
                    f"{spec['damping'] * 100:.0f}パーセント。",
                    f"Reverb time {spec['rt60']:.1f} seconds, room size "
                    f"{spec['room_scale'] * 100:.0f} percent, high damping "
                    f"{spec['damping'] * 100:.0f} percent."),
                 tr("方向の無い残響です。壁や床からの方向つきの反射も欲しいときは、"
                    "空間を自分で作ってください。設定欄で空間を選んでいる間は、"
                    "残響の設定は使われません。",
                    "This reverb has no direction. If you also want "
                    "reflections from walls and floor with a direction, "
                    "design a space instead. While a space is selected in "
                    "the settings, the reverb setting is not used.")]
        self.txt_info.SetValue("\n".join(lines))

    def preview_spec(self, spec: dict) -> None:
        self.frame.preview_reverb(spec)

    def put(self, name: str, spec: dict) -> bool:
        return self.frame.user_presets.put_reverb(name, spec)

    def remove(self, name: str) -> None:
        self.frame.user_presets.delete_reverb(name)
