# MoMask を動かすために必要なデータセット形式・アノテーション

本ドキュメントは、MoMask の学習・評価・生成に必要な **データセットのディレクトリ構成、
ファイル形式、アノテーション仕様** を、本リポジトリの実装
（`data/t2m_dataset.py`, `utils/get_opt.py`, `utils/motion_process.py`）に基づいて整理したものである。

> 用途別の必要物の早見は [§1](#1-用途別必要なものの早見表)、形式の詳細は §3 以降。

---

## 1. 用途別・必要なものの早見表

| 用途 | モーション特徴 `.npy` | テキスト `.txt` | split ファイル | Mean/Std | GloVe + 評価器 |
|------|:---:|:---:|:---:|:---:|:---:|
| **生成のみ**（自前テキストから生成） | 不要 | 不要 | 不要 | 学習済みのもの（同梱） | 不要 |
| **編集**（`edit_t2m.py`） | 元モーション 1 本（dim-263）が必要 | 不要 | 不要 | 同梱 | 不要 |
| **RVQ 学習**（`train_vq.py`） | ✅ 必要 | （未使用） | ✅ train/val | ✅ 必要 | 不要 |
| **Transformer 学習**（M/R） | ✅ 必要 | ✅ 必要 | ✅ train/val | ✅ 必要 | 不要 |
| **評価**（`eval_*.py`） | ✅ 必要 | ✅ 必要 | ✅ test | ✅ 必要 | ✅ 必要 |

- **生成だけ**なら新規データは一切不要。`bash prepare/download_models.sh` で重みと統計量が入る。
- **再学習・評価**には HumanML3D もしくは KIT-ML の完全データが必要（README §3「Get Data」）。

---

## 2. 対応データセットと基本パラメータ

`utils/get_opt.py:59-73` で `--dataset_name` ごとに固定される値。

| 項目 | `t2m`（HumanML3D） | `kit`（KIT-ML） |
|------|:---:|:---:|
| `data_root` | `./dataset/HumanML3D/` | `./dataset/KIT-ML/` |
| 関節数 `joints_num` | 22 | 21 |
| 特徴次元 `dim_pose` | **263** | **251** |
| fps | 20 | 12.5 |
| `max_motion_length` | 196 | 196 |
| `unit_length` | 4（RVQ で時間 1/4 ダウンサンプル） | 4 |

独自データを使う場合も、この **どちらかの形式（22関節263次元 or 21関節251次元）に変換**する必要がある（§6）。

---

## 3. ディレクトリ構成

`data_root`（例 `./dataset/HumanML3D/`）の直下に以下を配置する。

```
dataset/HumanML3D/
├── new_joint_vecs/      # モーション特徴ベクトル  <id>.npy        (opt.motion_dir)
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── texts/               # テキストアノテーション  <id>.txt        (opt.text_dir)
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...
├── Mean.npy             # 特徴量の平均 (dim_pose,)               (train_vq.py:80)
├── Std.npy              # 特徴量の標準偏差 (dim_pose,)
├── train.txt            # 学習用 id 一覧（拡張子なし、1行1id）
├── val.txt              # 検証用 id 一覧
├── test.txt             # 評価用 id 一覧
└── (new_joints/)        # 関節座標 (T,joints,3) ※可視化用・学習には不要
```

- `<id>.npy` と `<id>.txt` は **同じファイル名（id）で対応付け**られる（`data/t2m_dataset.py:110, 115`）。
- パスは `motion_dir = data_root/new_joint_vecs`、`text_dir = data_root/texts` 固定（`get_opt.py:60-61`）。

---

## 4. モーション特徴ファイル（`new_joint_vecs/<id>.npy`）

### 4.1 形状・型
- NumPy 配列、形状 **`(T, dim_pose)`**（`T` = フレーム数）。例: `example_data/000612.npy` は `(199, 263)`。
- dtype は float（float64/float32 いずれも可、内部で float 化される）。
- HumanML3D は `dim_pose=263`、KIT は `251`。

### 4.2 dim-263 特徴ベクトルの内訳（HumanML3D）
`data/t2m_dataset.py:42-62` の正規化処理と `utils/motion_process.py` の `extract_features` から、
1 フレームの 263 次元は次の連結で構成される（`J = joints_num = 22`）。

| 区間 | 内容 | 次元 | HumanML3D |
|------|------|------|:---:|
| `[0:1]` | root の回転角速度（Y軸まわり） | 1 | 1 |
| `[1:3]` | root の水平方向 線速度（X,Z） | 2 | 2 |
| `[3:4]` | root の高さ `root_y` | 1 | 1 |
| `[4 : 4+(J-1)*3]` | **RIC**: root 相対の各関節位置 | (J-1)·3 | 63 |
| `[… : …+(J-1)*6]` | **回転**: 各関節の 6D 連続回転表現 | (J-1)·6 | 126 |
| `[… : …+J*3]` | 各関節の局所速度 | J·3 | 66 |
| 末尾 `[-4:]` | **足接地**フラグ（両足×2点） | 4 | 4 |
| | **合計** | | **263** |

検算: `1+2+1+63+126+66+4 = 263`。KIT（J=21）は `1+2+1+60+120+63+4 = 251`。
この内訳は `t2m_dataset.py:62` の `assert 4 + (joints_num-1)*9 + joints_num*3 + 4 == mean.shape[-1]` で保証される。

### 4.3 制約
- 学習・評価では **フレーム長が範囲内**である必要がある（`t2m_dataset.py:111, 250`）:
  `min_motion_len`（t2m=40, kit=24）以上、200 未満。
- RVQ 学習では `window_size`（既定 64）未満のモーションは捨てられる（`t2m_dataset.py:30`）。

---

## 5. テキストアノテーションファイル（`texts/<id>.txt`）

1 つのモーションに **複数キャプション**を付けられる（1 行 = 1 キャプション）。
各行は `#` 区切りの **4 フィールド**（`data/t2m_dataset.py:118-124, 257-264`）:

```
<caption>#<tokens>#<from_tag>#<to_tag>
```

| フィールド | 意味 | 例 |
|------------|------|----|
| `caption` | 自然文キャプション（モデルへの実入力） | `a person walks forward and waves` |
| `tokens` | **`単語/品詞` を半角空白区切り**にしたもの（POS タグ付き） | `a/DET person/NOUN walk/VERB forward/ADV wave/VERB` |
| `from_tag` | キャプションが対応する区間の**開始秒**。全体記述なら `0.0` | `0.0` |
| `to_tag` | 同 **終了秒**。全体記述なら `0.0` | `0.0` |

具体例（`texts/000000.txt` のイメージ）:
```
a person walks forward.#a/DET person/NOUN walk/VERB forward/ADV#0.0#0.0
someone steps then waves their hand.#someone/NOUN step/VERB then/ADV wave/VERB hand/NOUN#2.0#4.5
```

ポイント:
- `from_tag == to_tag == 0.0` の行は **モーション全体**の記述として扱われる（`t2m_dataset.py:128`）。
- それ以外は `motion[int(f_tag*20) : int(to_tag*20)]`（fps=20）で**部分区間を切り出して**別サンプル化する
  （`t2m_dataset.py:133`）。区間長が範囲外なら破棄。
- `tokens` の品詞タグは **GloVe 評価器が単語埋め込みを引く時のキー**（`sos/OTHER`, `eos/OTHER`, `unk/OTHER`
  を付与、`t2m_dataset.py:189-200`）。**学習・生成の本体（CLIP）は `caption` のみ使用**し、tokens は使わない。
  → 学習・生成だけなら `tokens` は厳密でなくてよいが、**評価**には正しい POS タグ付き tokens が必要。

---

## 6. 統計量・分割・補助ファイル

### 6.1 Mean.npy / Std.npy
- `data_root` 直下に置く `(dim_pose,)` の配列（`train_vq.py:80-81` でロード）。
- 全データの各次元平均・標準偏差。**Z 正規化** `(motion - mean) / std` に使う（`t2m_dataset.py:85`）。
- 学習時は train 集合から再計算され `meta_dir` に保存される（`t2m_dataset.py:63-64`）。
- 生成専用の統計量はチェックポイントに同梱（`checkpoints/<dataset>/.../meta/` 等）。

### 6.2 split ファイル（train.txt / val.txt / test.txt）
- **拡張子なしの id を 1 行 1 つ**列挙したテキスト（`t2m_dataset.py:23-25, 101-103`）。
- ここに書かれた id に対応する `new_joint_vecs/<id>.npy` と `texts/<id>.txt` が読み込まれる。

### 6.3 GloVe・評価器（評価時のみ）
- `bash prepare/download_glove.sh` … 単語ベクトル化器（`utils/word_vectorizer.py`、tokens を埋め込み化）。
- `bash prepare/download_evaluator.sh` … FID/R-Precision 等を測る評価用モーション・テキストエンコーダ。
- 生成・学習自体には不要。**評価指標を出す時だけ**必要。

---

## 7. 独自データを使う手順

### 7.1 既存形式に変換する（推奨）
独自のモーションを使うには、まず **HumanML3D dim-263（または KIT dim-251）特徴に変換**する。

1. 3D 関節座標を用意: 形状 `(T, joints_num, 3)`（22 関節、SMPL 系の関節順）。
2. `utils/motion_process.py` の **`process_file`** に通して dim-263 特徴へ変換
   （README §「Generate from customized descriptions」/`edit_t2m.py` 注記、本リポジトリ `motion_process.py`）。
   - 内部で骨格正規化（`uniform_skeleton`）→ RIC/6D 回転/速度/足接地の抽出（`extract_features`）を行う。
3. 出力 `(T, 263)` を `new_joint_vecs/<id>.npy` として保存。
4. 同じ id で `texts/<id>.txt`（§5 の形式）を作成。
5. `train/val/test.txt` に id を追記。
6. **Mean/Std を再計算**（学習を回せば自動保存される）。

> 逆変換（特徴→関節座標）は `recover_from_ric(features, joints_num)`（`utils/motion_process.py`）。
> 生成結果や可視化はこれで `(T, joints, 3)` に戻す。

### 7.2 最小要件のまとめ
- **学習を回す最低条件**: `new_joint_vecs/*.npy`（正しい次元・フレーム長）, `texts/*.txt`,
  `train.txt`/`val.txt`, `Mean.npy`/`Std.npy`。
- **評価まで**やるなら: 上記 + `test.txt` + GloVe + 評価器、かつ tokens の POS タグが正確であること。

---

## 8. よくある不整合・チェックリスト

- [ ] `.npy` の次元が dataset の `dim_pose`（263 or 251）と一致しているか（§4.2 の `assert` で検出）。
- [ ] `new_joint_vecs/<id>.npy` と `texts/<id>.txt` の **id が一致**しているか。
- [ ] フレーム長が範囲内か（< 200、t2m は ≥ 40 / kit は ≥ 24、RVQ は ≥ window_size）。
- [ ] テキスト各行が **4 フィールド `#` 区切り**になっているか（フィールド欠落で parse 失敗）。
- [ ] 部分区間アノテーションの `from_tag/to_tag` が秒単位（fps を掛けてフレーム index 化される）になっているか。
- [ ] `Mean.npy`/`Std.npy` が当該データから算出した `(dim_pose,)` であるか（他データの統計を流用しない）。

---

## 参考
- アルゴリズム全体: [`momask_algorithm.md`](momask_algorithm.md)
- 元データの作り方: [HumanML3D リポジトリ](https://github.com/EricGuo5513/HumanML3D)（モーション→dim-263 変換スクリプト一式）
- 実装: `data/t2m_dataset.py`, `utils/get_opt.py`, `utils/motion_process.py`, `train_vq.py`
</content>
