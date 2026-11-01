"""既定の HRTF。聴取テストで選ばれたものを、依存の少ない形で提供する。

なぜ既定を決めるのか
--------------------
HRTF の良し悪しは聞き手との相性で決まり、21種類を毎回試させるのは
現実的でない。実際に聴取テストを行って選んだものを既定にしておけば、
ゲーム側は何も考えずに使える。

選定の経緯（2026-08-10、Viking 21種のうち9種を検証）
- 12方向・首振りあり・のべ数百問の比較
- 上位3種（E・F・B）は統計的に同着（基準との差 +3〜+4 ポイント）
- そのうち **耳介E だけが、判別可能な難易度72問すべてで前後の
  取り違えゼロ**。他は1問ずつ取り違えがあった
- 前後の取り違えは視覚障害者向けゲームで最も避けたい失敗なので、
  正答率が僅差ならこれを基準に選ぶ

同着だったので、E が万人にとって最良という主張ではない。合わない場合は
`demo/localization_app.py` の耳介選定（F8）で選び直せる。
"""
from __future__ import annotations

from pathlib import Path

from .hrtf import HrirSet
from .i18n import tr

#: 実測 SOFA（Viking 21種）の置き場。リポジトリ運用でのみ存在し、
#: wheel には同梱しない（61MB あるため）。
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "hrtf"
#: 聴取テストで選ばれた既定の耳介
DEFAULT_PINNA = "E"
#: 既定アセットはパッケージ内に置く。wheel / PyInstaller にそのまま乗る。
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ASSET = DATA_DIR / "default_horizontal.npz"
#: 2026-08-26 より前は assets/hrtf/ 直下に置いていた。焼き直し済みの
#: 古い作業コピーでも動くように、旧位置も探す。
_LEGACY_ASSET = ASSET_DIR / "default_horizontal.npz"


def _default_asset_path() -> Path | None:
    if DEFAULT_ASSET.exists():
        return DEFAULT_ASSET
    if _LEGACY_ASSET.exists():
        return _LEGACY_ASSET
    return None


def default_hrir(sample_rate: int = 48000,
                 fallback_synthetic: bool = True) -> HrirSet:
    """既定の HRTF を返す。

    焼き込み済みアセット（水平面のみ、約200KB、numpy だけで読める）が
    あればそれを使う。無ければ合成HRIRに落ちる。合成は前後を区別できない
    ので、その場合はエンジンの `front_back_depth` と首振りに頼ることになる。

    Args:
        sample_rate: 出力先に合わせたレート。必要なら変換する。
        fallback_synthetic: アセットが無いとき合成で代替するか。
            False なら FileNotFoundError を送出する。
    """
    path = _default_asset_path()
    if path is not None:
        return HrirSet.load_npz(path).resampled(sample_rate)
    if not fallback_synthetic:
        raise FileNotFoundError(
            tr(f"{DEFAULT_ASSET} がありません。"
               "tools/bake_hrtf_asset.py で作成してください",
               f"{DEFAULT_ASSET} is missing. "
               "Create it with tools/bake_hrtf_asset.py"))
    return HrirSet.synthetic(sample_rate=sample_rate)


def default_is_measured() -> bool:
    """既定が実測HRTFかどうか（合成にフォールバックしていないか）。"""
    return _default_asset_path() is not None


def attribution() -> str:
    """既定アセットを配布物に含める際に表示する文面。

    Viking は CC BY 4.0 なので商用利用できるが、表示が必須。
    """
    path = _default_asset_path()
    if path is None:
        return ""
    meta = HrirSet.read_npz_meta(path)
    return meta.get("attribution", "")
