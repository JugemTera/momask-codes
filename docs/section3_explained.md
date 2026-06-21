# MoMask 論文 Section 3「Approach」 数式・実装対応つき解説（初学者向け）

本ドキュメントは MoMask（*Generative Masked Modeling of 3D Human Motions*, CVPR 2024 / arXiv:2312.00063）の
**Section 3「Approach」** を、論文の数式 1 つ 1 つについて

- 数式の意味
- 数式に出てくる文字（記号）が何を指すか
- 実装ではどんなデータ構造・どの関数になっているか（**ファイル名:行番号** つき）

をセットで解説したものである。前提知識（深層学習・PyTorch の基本）以上は仮定しない。
記述はすべて本リポジトリの実装（`models/vq/`, `models/mask_transformer/`, `gen_t2m.py`）に基づく。

> 関連ドキュメント: アルゴリズム全体のフローは [`momask_algorithm.md`](momask_algorithm.md)、
> 入力モーションの 263 次元特徴の中身は [`dataset_format.md`](dataset_format.md) を参照。

---

## 0. Section 3 全体の地図

Section 3 は「3 つのモデルを順に学習し、推論時に直列につなぐ」という構成になっている。

| 節 | 内容 | 主な数式 | 実装の中心 |
|----|------|----------|------------|
| 3.（冒頭） | ゴールと記号の定義 | — | — |
| 3.1 | Motion Residual VQ-VAE の学習 | (1) RQ, (2) 損失 | `models/vq/` |
| 3.2 | Masked Transformer の学習 | (3) マスク損失 | `MaskTransformer` |
| 3.3 | Residual Transformer の学習 | (4) 残差損失 | `ResidualTransformer` |
| 3.4 | 推論（生成） | (5) Classifier-Free Guidance | `*.generate()`, `gen_t2m.py` |

論文の Figure 2（学習）と Figure 3（推論）が対応する図である。

---

## 3.（冒頭）ゴールと記号

> 論文: *Our goal is to generate a 3D human pose sequence $\mathbf{m}_{1:N}$ of length $N$ guided by a textual description $c$, where $\mathbf{m}_i \in \mathbb{R}^D$ with $D$ denoting the dimension of pose features.*

### 記号の意味

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $\mathbf{m}_{1:N}$ | 長さ $N$ フレームのモーション系列 | データセットが返す `motion`（形状 `(b, T, D)`） |
| $N$ | フレーム数（時間長） | `T`（テンソルの時間軸） |
| $\mathbf{m}_i \in \mathbb{R}^D$ | $i$ フレーム目のポーズ特徴ベクトル | 1 フレーム分の特徴 |
| $D$ | ポーズ特徴の次元数 | HumanML3D では **263**（KIT-ML は 251） |
| $c$ | テキスト記述（条件） | `conds` / `prompt`（生文字列） |

`D = 263` の内訳（root 角速度・関節位置・速度・回転・接地フラグ）は
[`dataset_format.md`](dataset_format.md) に詳しい。実装では `RVQVAE(input_width=263, ...)`
（`models/vq/model.py:8`）として現れる。

---

## 3.1 Motion Residual VQ-VAE の学習

### 3.1.1 まず通常の VQ（量子化）

論文はまず普通の motion VQ-VAE を説明する。流れは

$$
\mathbf{m}_{1:N} \;\xrightarrow{\;E\;}\; \tilde{\mathbf{b}}_{1:n} \in \mathbb{R}^{n\times d}
\;\xrightarrow{\;Q(\cdot)\;}\; \mathbf{b}_{1:n} \in \mathbb{R}^{n\times d}
\;\xrightarrow{\;D\;}\; \hat{\mathbf{m}}
$$

#### 記号の意味

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $E$ | 1D 畳み込みエンコーダ | `Encoder`（`models/vq/encdec.py:5`） |
| $D$ | デコーダ | `Decoder`（`models/vq/encdec.py:37`） |
| $\tilde{\mathbf{b}}_{1:n}$ | エンコード後の連続な潜在ベクトル列 | `x_encoder`（`models/vq/model.py:55`） |
| $n$ | 潜在系列の長さ（ダウンサンプル後） | `T // 4`。下記参照 |
| $n/N$ | ダウンサンプリング率 | `down_t=2` → $2^2 = 4$ 倍ダウンサンプル（`options/vq_option.py:31`） |
| $d$ | 潜在ベクトルの次元 = コード次元 | `code_dim = 512`（`options/vq_option.py:28`） |
| $Q(\cdot)$ | 量子化（最近傍コードに置換） | `QuantizeEMAReset.quantize`（`models/vq/quantizer.py:67`） |
| $\mathcal{C}=\{\mathbf{c}_k\}_{k=1}^{K}$ | コードブック（コード辞書） | `self.codebook`（`models/vq/quantizer.py:47`） |
| $K$ | コードブックのコード数 | `nb_code = 512`（`options/vq_option.py:29`） |
| $\hat{\mathbf{m}}$ | 再構成モーション | `x_out`（`models/vq/model.py:76`） |

> **ダウンサンプル率が 4 である根拠**: 学習・推論コードが一貫して
> `m_lens = m_lens // 4`（`models/mask_transformer/transformer_trainer.py:46`,
> `gen_t2m.py` の `// 4`）としている。`down_t=2` の畳み込みが stride 2 を 2 回かけるため $2^2=4$。

#### $Q(\cdot)$ の中身（最近傍探索）

量子化 $\mathbf{b}_i = Q(\tilde{\mathbf{b}}_i)$ は「潜在ベクトルを、コードブック中で最も近いコードに置き換える」操作。
実装は二乗距離 $\lVert x - c_k\rVert^2 = \lVert x\rVert^2 - 2 x^\top c_k + \lVert c_k\rVert^2$ を計算して最小のものを選ぶ:

```python
# models/vq/quantizer.py:72-78
distance = torch.sum(x ** 2, dim=-1, keepdim=True) \
           - 2 * torch.matmul(x, k_w) \
           + torch.sum(k_w ** 2, dim=0, keepdim=True)
code_idx = gumbel_sample(-distance, dim=-1, temperature=..., ...)  # 実質 argmin
```

選ばれたコードのインデックス（整数）が **motion token** であり、これがモーションの離散表現になる。

### 3.1.2 Residual Quantization（残差量子化）— 式 (1)

1 回の量子化では誤差が大きい。そこで「誤差（残差）を次の層がさらに量子化する」ことを $V+1$ 層繰り返す。
これが MoMask の肝。論文の定義は

$$
\mathrm{RQ}(\tilde{\mathbf{b}}_{1:n}) = [\mathbf{b}^{v}_{1:n}]_{v=0}^{V}
$$

各層の漸化式が **式 (1)**:

$$
\boxed{\;\mathbf{b}^{v} = Q(\mathbf{r}^{v}), \qquad \mathbf{r}^{v+1} = \mathbf{r}^{v} - \mathbf{b}^{v}\;}
\tag{1}
$$

初期残差は $\mathbf{r}^0 = \tilde{\mathbf{b}}$。最終的に潜在の近似は全層の和

$$
\tilde{\mathbf{b}} \approx \sum_{v=0}^{V} \mathbf{b}^{v}
$$

#### 記号の意味

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $v$ | 量子化層のインデックス（$0$ がベース層） | ループ変数 `quantizer_index` |
| $V$ | 残差層の最大インデックス（全 $V{+}1$ 層） | `num_quantizers`。論文は $V{+}1=6$、本リポジトリ既定は 3（`options/vq_option.py:41`） |
| $\mathbf{r}^v$ | 第 $v$ 層に入る残差 | `residual` |
| $\mathbf{b}^v$ | 第 $v$ 層の量子化結果（コード） | `quantized` |
| $\sum_v \mathbf{b}^v$ | 全層を足した最終近似 | `quantized_out` |

#### 実装での式 (1)

`ResidualVQ.quantize`（`models/vq/residual_vq.py:171`）が式 (1) そのもの:

```python
# models/vq/residual_vq.py:176-181
for quantizer_index, layer in enumerate(self.layers):
    quantized, *rest = layer(residual, return_idx=True)   # b^v = Q(r^v)
    residual = residual - quantized.detach()              # r^{v+1} = r^v - b^v   ← 式(1)
    quantized_out = quantized_out + quantized             # Σ b^v
```

- `self.layers` が $V+1$ 個の `QuantizeEMAReset`（各層が独立コードブックを持つ。`models/vq/residual_vq.py:43-47`）。
- 残差から `.detach()` で引くのは、勾配が下位層に流れないようにするため。
- 各層のトークン列を `torch.stack` して `code_idx`（形状 `(b, n, q)`）として返す（`residual_vq.py:190`）。
  この $q$ 軸が「層 $v$」に対応する。

復号時は全層のコードを足してデコーダに入れる:

```python
# models/vq/model.py:80-87 forward_decoder
x_d = self.quantizer.get_codes_from_indices(x)  # 各層のコード (q, b, n, d)
x = x_d.sum(dim=0)...                            # Σ_v b^v
x_out = self.decoder(x)
```

### 3.1.3 学習損失 — 式 (2)

$$
\boxed{\;\mathcal{L}_{rvq} = \lVert \mathbf{m} - \hat{\mathbf{m}} \rVert_1
\;+\; \beta \sum_{v=1}^{V} \lVert \mathbf{r}^{v} - \mathrm{sg}[\mathbf{b}^{v}] \rVert_2^2 \;}
\tag{2}
$$

#### 記号の意味

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $\lVert \mathbf{m} - \hat{\mathbf{m}}\rVert_1$ | 再構成誤差（L1） | `loss_rec`（`vq_trainer.py:45`） |
| $\mathrm{sg}[\cdot]$ | stop-gradient（勾配を止める） | `.detach()`（`quantizer.py:147`） |
| $\beta$ | embedding/commitment 損失の重み | `commit = 0.02`（`options/vq_option.py:23`） |
| 第 2 項 | 残差とコードを近づける commitment 損失 | `loss_commit`（`quantizer.py:147`） |

#### 実装での式 (2)

**第 1 項（再構成）と全体合成**は `RVQTokenizerTrainer.forward`:

```python
# models/vq/vq_trainer.py:45-50
loss_rec = self.l1_criterion(pred_motion, motions)            # ||m - m̂||_1
...
loss = loss_rec + self.opt.loss_vel * loss_explicit \
       + self.opt.commit * loss_commit                        # β は self.opt.commit
```

> 実装には論文式 (2) に明示されない `loss_explicit`（局所関節位置の追加 L1, 重み `loss_vel=0.5`）が足されている。
> これは再構成の補助項で、式 (2) の精神（再構成誤差の最小化）の拡張と読める（`vq_trainer.py:46-48`）。

**第 2 項（commitment）**は各量子化層の中:

```python
# models/vq/quantizer.py:147
commit_loss = F.mse_loss(x, x_d.detach())   # ||r^v - sg[b^v]||_2^2
```

ここで `x` が層入力の残差 $\mathbf{r}^v$、`x_d` が量子化結果 $\mathbf{b}^v$、`.detach()` が $\mathrm{sg}[\cdot]$。
全層の平均が `all_losses`（`residual_vq.py:157`）として戻る。

#### Straight-Through 推定 とコードブック更新（EMA）

論文の *straight-through gradient estimator* と *EMA + codebook reset* は次の箇所:

```python
# models/vq/quantizer.py:150  Straight-Through（順伝播は x_d、逆伝播は x の勾配）
x_d = x + (x_d - x).detach()
```

- コードブックは勾配ではなく **EMA（指数移動平均）** で更新: `update_codebook`（`quantizer.py:100-123`）。
  係数 $\mu$ は `args.mu`。
- 使われないコードはランダムに張り替える（**codebook reset**）: `usage * code_update + (1-usage) * code_rand`（`quantizer.py:117`）。

### 3.1.4 Quantization Dropout

> 論文: 早い層に情報を集中させるため、確率 $q$ で末尾の層をランダムに無効化する。

実装は `ResidualVQ.forward`:

```python
# models/vq/residual_vq.py:112-136
should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob
if should_quantize_dropout:
    start_drop_quantize_index = randrange(...)   # ここから後ろの層を無効化
...
if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
    all_indices.append(null_indices)             # ドロップした層は -1（無効トークン）
    continue
```

- $q$ = `quantize_dropout_prob = 0.2`（`options/vq_option.py:43`、論文の最適値と一致）。
- 無効化した層のトークンは `-1` で埋め、復号時にゼロベクトル扱い（`residual_vq.py:81-89`）。

### 3.1.5 トークン表現 $T$

学習後、モーションは $V+1$ 本の離散トークン列で表せる:

$$
T = [\,t^{v}_{1:n}\,]_{v=0}^{V}, \qquad t^{v}_i \in \{1, \dots, |\mathcal{C}^v|\}^n
$$

- $t^0$ が**ベース層トークン**（最も支配的な情報）→ 3.2 の M-Transformer が担当。
- $t^1, \dots, t^V$ が**残差層トークン**（細部）→ 3.3 の R-Transformer が担当。
- 実装上は `code_idx` の形状 `(b, n, q)`。`code_idx[..., 0]` がベース層、`code_idx[..., 1:]` が残差層
  （`transformer_trainer.py:54` で `code_idx[..., 0]` を取り出している）。

---

## 3.2 Masked Transformer（M-Transformer）の学習

ベース層トークン $t^0_{1:n}$ を、BERT 風のマスク穴埋めで生成できるように学習する。
実装は `MaskTransformer`（`models/mask_transformer/transformer.py:84`）。

### 3.2.1 マスク率スケジュール

学習時、マスクする割合を毎回ランダムに変える。割合は余弦関数で決める:

$$
\gamma(\tau) = \cos\!\Big(\frac{\pi \tau}{2}\Big) \in [0,1], \qquad \tau \sim \mathcal{U}(0,1)
$$

そしてマスクするトークン数を $m = \lceil \gamma(\tau)\cdot n \rceil$ とする。

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $\tau$ | $[0,1]$ の乱数。$\tau=0$ で完全破壊（全マスク） | `rand_time = uniform(...)`（`transformer.py:273`） |
| $\gamma(\cdot)$ | 余弦マスクスケジュール | `cosine_schedule`（`tools.py:120`） |
| $m$ | マスクするトークン数 | `num_token_masked`（`transformer.py:275`） |
| $n$ | 系列長 | `ntokens` |

```python
# models/mask_transformer/transformer.py:273-282
rand_time = uniform((bs,), device=device)
rand_mask_probs = self.noise_schedule(rand_time)            # γ(τ)
num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)  # m
batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
mask = batch_randperm < num_token_masked.unsqueeze(-1)      # マスク位置を無作為選択
mask &= non_pad_mask                                        # パディングは除外
```

### 3.2.2 マスク損失 — 式 (3)

$$
\boxed{\;\mathcal{L}_{mask} = \sum_{\tilde{t}^0_k = [\mathrm{MASK}]} -\log p_\theta\big(t^0_k \mid \tilde{t}^0,\, c\big)\;}
\tag{3}
$$

「マスクされた位置 $k$ について、正解トークン $t^0_k$ の対数尤度を最大化（=負の対数尤度を最小化）する」。

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $p_\theta$ | M-Transformer（パラメータ $\theta$） | `MaskTransformer` |
| $\tilde{t}^0$ | マスク後のベース層トークン列（入力） | `x_ids`（`transformer.py:287-299`） |
| $t^0_k$ | 位置 $k$ の正解トークン（教師） | `labels`（`transformer.py:285`） |
| $[\mathrm{MASK}]$ | 特殊なマスクトークン | `self.mask_id = opt.num_tokens`（`transformer.py:133`） |
| $c$ | テキスト条件（CLIP 埋め込み） | `cond_vector`（`transformer.py:260`） |
| $\sum_{\tilde{t}^0_k=[\mathrm{MASK}]}$ | マスクされた位置のみ和を取る | `ignore_index=mask_id` で非マスクを無視 |

```python
# models/mask_transformer/transformer.py:285, 301-302
labels = torch.where(mask, ids, self.mask_id)              # マスク位置=正解、他=mask_id
...
logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask)
ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)
```

`cal_performance`→`cal_loss`（`tools.py:132-164`）は `F.cross_entropy(..., ignore_index=mask_id)`。
`labels` の非マスク位置は `mask_id` なので **損失計算から除外され、マスク位置だけが式 (3) の和に入る**。

> テキスト条件 $c$ は CLIP（`ViT-B/32`, frozen）で抽出（`encode_text`, `transformer.py:194`）し、
> `cond_emb` で潜在次元に写像、系列の先頭トークンとして連結する（`trans_forward`, `transformer.py:228-231`）。

### 3.2.3 Replacing & Remasking（BERT 流の置換）

マスク対象に選ばれたトークンを、そのまま全部 `[MASK]` にするのではなく BERT と同じく分散させる:

- 80%: `[MASK]` トークンに置換
- 10%: ランダムな別トークンに置換
- 10%: そのまま（正解のまま残す）

```python
# models/mask_transformer/transformer.py:290-299
mask_rid = get_mask_subset_prob(mask, 0.1)                 # 10% をランダムトークン化
rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
x_ids = torch.where(mask_rid, rand_id, x_ids)
mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)    # 残り90%のうち88%→[MASK]
x_ids = torch.where(mask_mid, self.mask_id, x_ids)
```

残った $0.9 \times 0.12 \approx 10\%$ が「正解のまま」になり、合計 80%/10%/10% を実現する。

### 3.2.4 Classifier-Free Guidance のための条件ドロップ

学習中、確率 0.1 でテキスト条件をゼロにして「無条件」も学べるようにする
（推論時の式 (5) で使う）:

```python
# models/mask_transformer/transformer.py:200-208 mask_cond
mask = torch.bernoulli(torch.ones(bs, ...) * self.cond_drop_prob)  # cond_drop_prob=0.1
return cond * (1. - mask)
```

---

## 3.3 Residual Transformer（R-Transformer）の学習

残差層トークン $t^1,\dots,t^V$ を、**前の層までの結果から**予測できるように学習する。
1 つの Transformer が層番号 $j$ を入力にもらって全層を兼任する。
実装は `ResidualTransformer`（`models/mask_transformer/transformer.py:611`）。

### 3.3.1 残差損失 — 式 (4)

$$
\boxed{\;\mathcal{L}_{res} = \sum_{j=1}^{V}\sum_{i=1}^{n} -\log p_\phi\big(t^{j}_i \mid t^{1:j-1}_i,\, c,\, j\big)\;}
\tag{4}
$$

「層 $j$ のトークン $t^j_i$ を、それより前の層 $t^{1:j-1}$・テキスト $c$・層番号 $j$ から予測する」。

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $p_\phi$ | R-Transformer（パラメータ $\phi$） | `ResidualTransformer` |
| $j$ | 予測対象の量子化層番号（$1\dots V$） | `active_q_layers`（`transformer.py:860`） |
| $t^{1:j-1}$ | $j$ より前の層のトークン（条件） | `history_sum`（`transformer.py:871`） |
| $t^j_i$ | 層 $j$・位置 $i$ の正解トークン | `active_indices`（`transformer.py:870`） |
| $c$ | テキスト条件 | `cond_vector` |

#### 学習時の流れ（`forward`, `transformer.py:840-889`）

```python
# 1) 各サンプルで予測する層 j を 1 つランダムに選ぶ（low=1 なのでベース層0は対象外）
active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)  # :860

# 2) 全層の埋め込みを累積和して「層 j 未満の合計」を作る = t^{1:j-1} の表現
all_codes = token_embed.gather(1, gather_indices)        # 各層の埋め込み :866
cumsum_codes = torch.cumsum(all_codes, dim=-1)           # 層方向の累積和 :868
history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]  # :871

# 3) j 層の正解トークン
active_indices = all_indices[torch.arange(bs), :, active_q_layers]       # :870

# 4) history_sum・テキスト・層番号 j から logits を出して交差エントロピー
logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, ...)
logits = self.output_project(logits, active_q_layers - 1)
ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)
```

- **層番号 $j$ の入力**: `encode_quant`（one-hot）→`quant_emb` で埋め込み、系列に連結
  （`trans_forward`, `transformer.py:800-807`）。M-Transformer がテキスト 1 個を前置するのに対し、
  R-Transformer は **テキスト + 層番号の 2 個**を前置する（`transformer.py:807`）。
- **層ごとに別の埋め込み/出力射影**: `token_embed_weight`, `output_proj_weight`
  （形状 `(V, ntoken, code_dim)`, `transformer.py:694-696`）。

### 3.3.2 埋め込みと出力射影の重み共有

> 論文: *We also share the parameters of the $j$-th prediction layer and the $(j{+}1)$-th motion token embedding layer.*

`share_weight=True` のとき、$j$ 層の出力射影重みと $(j{+}1)$ 層の入力埋め込み重みを共有する:

```python
# models/mask_transformer/transformer.py:759-763 process_embed_proj_weight
self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)
```

`embed_proj_shared_weight` が両者にまたがって現れるのが「共有」の実体。

---

## 3.4 推論（生成）

学習済み 3 モデルを直列につなぐ。論文 Figure 3 がこの流れ。
全体は `gen_t2m.py`（`mids = t2m_transformer.generate(...)` → `res_model.generate(...)` →
`vq_model.forward_decoder(...)`）。

### 3.4.1 M-Transformer による反復生成（ベース層）

> 論文: 空（全マスク）の系列 $t^0(0)$ から始め、$L$ 回の反復でベース層 $t^0$ を埋める。
> 各反復 $l$ で、信頼度の低い $\lceil \gamma(\tfrac{l}{L})\cdot n\rceil$ 個を再マスクして predict し直す。

実装は `MaskTransformer.generate`（`transformer.py:326-430`）:

```python
# 開始: 全トークンを [MASK]、スコア（信頼度）を 0 に初期化
ids = torch.where(padding_mask, self.pad_id, self.mask_id)       # :359
scores = torch.where(padding_mask, 1e5, 0.)                      # :360

for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, ...), reversed(range(timesteps))):
    rand_mask_prob = self.noise_schedule(timestep)               # γ(l/L)  :365
    num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # :371
    # スコアが低い順に num_token_masked 個を再マスク
    ranks = scores.argsort(dim=1).argsort(dim=1)                 # :374-376
    is_mask = (ranks < num_token_masked.unsqueeze(-1))
    ids = torch.where(is_mask, self.mask_id, ids)                # :378
    # 予測 → サンプリング → マスク位置だけ更新
    logits = self.forward_with_cond_scale(ids, ...)              # :384（式(5)を内部で適用）
    pred_ids = Categorical(probs).sample()                       # :412
    ids = torch.where(is_mask, pred_ids, ids)                    # :416
    # 信頼度を更新（高信頼トークンは次回マスクされにくくする）
    scores = probs_without_temperature.gather(2, pred_ids...)    # :421-426
```

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $L$ | 反復回数 | `timesteps`（既定 10。論文も $L=10$） |
| $l$ | 反復ステップ $0\dots L$ | `torch.linspace(0,1,timesteps)` |
| $t^0(l)$ | $l$ 反復目のベース層系列 | `ids` |
| $\gamma(l/L)$ | 各反復のマスク率 | `rand_mask_prob`（`transformer.py:365`） |
| confidence | 各位置の予測確信度 | `scores`（`transformer.py:421-423`） |

ポイント: **確信度の高いトークンは残し、低いものだけ再予測**を繰り返すことで、
自己回帰なしに全位置を並列に詰めていく（拡散モデル的な反復精緻化）。

### 3.4.2 R-Transformer による段階生成（残差層）

ベース層 $t^0$ が決まったら、$v=1,\dots,V$ を**順番に**予測する。
実装は `ResidualTransformer.generate`（`transformer.py:891-966`）:

```python
# models/mask_transformer/transformer.py:933-959
history_sum = 0
for i in range(1, num_quant_layers):                # 層を 1 つずつ進める
    token_embed = self.token_embed_weight[i-1]
    history_sum += token_embed.gather(1, gathered_ids)   # t^{0:i-1} の累積埋め込み
    logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=...)
    pred_ids = gumbel_sample(filtered_logits, temperature=..., dim=-1)
    motion_ids = torch.where(padding_mask, self.pad_id, pred_ids)
    all_indices.append(motion_ids)
all_indices = torch.stack(all_indices, dim=-1)      # (b, n, q) 全層トークン完成
```

各反復で前層までの埋め込み和 `history_sum`（= 式 (4) の $t^{1:j-1}$）を更新しながら次層を予測する。
M-Transformer の「反復（同一層を $L$ 回）」と異なり、R-Transformer は「層を 1 回ずつ進む（$V$ ステップ）」点に注意。

### 3.4.3 Classifier-Free Guidance — 式 (5)

両 Transformer の最終 logits は、条件あり・条件なしの logits を混ぜて作る:

$$
\boxed{\;\omega_g = (1+s)\cdot \omega_c \;-\; s\cdot \omega_u\;}
\tag{5}
$$

| 記号 | 意味 | 実装での対応 |
|------|------|--------------|
| $\omega_c$ | テキスト条件ありの logits | `logits`（`transformer.py:317`） |
| $\omega_u$ | 無条件（条件マスク）の logits | `aux_logits`（`transformer.py:321`, `force_mask=True`） |
| $\omega_g$ | guidance 適用後の logits | `scaled_logits`（`transformer.py:323`） |
| $s$ | ガイダンス強度 | 下記の通り `cond_scale = 1 + s` |

```python
# models/mask_transformer/transformer.py:321-323
aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True)  # ω_u
scaled_logits = aux_logits + (logits - aux_logits) * cond_scale                          # ω_g
```

#### 式 (5) とコードの対応（重要）

コードの式は $\omega_g = \omega_u + (\omega_c - \omega_u)\cdot\texttt{cond\_scale}$ で、展開すると

$$
\omega_g = \texttt{cond\_scale}\cdot \omega_c + (1 - \texttt{cond\_scale})\cdot \omega_u
$$

これを式 (5) $(1+s)\omega_c - s\,\omega_u$ と見比べると

$$
\texttt{cond\_scale} = 1 + s
$$

つまり**コードの `cond_scale` は論文の $s$ そのものではなく $1+s$ に相当**する。
論文の最適 $s\approx 4$ 付近の議論（Sec 4.2, Fig 7）と、コードの `cond_scale`（HumanML3D で M:4, R:5 程度）を
比較する際はこのズレに注意。

### 3.4.4 モーション長の推定と復号

- テキストしか与えない場合、`LengthEstimator`（`models/vq/model.py:90`）が
  トークン長を分類予測する（`gen_t2m.py` の `length_estimator(text_embedding)`）。
- 全層トークン `mids`（`(b, n, q)`）を `vq_model.forward_decoder`（`model.py:80`）に渡し、
  全層コードを足してデコーダで $\hat{\mathbf{m}}$（263 次元特徴）に戻す。
- 263 次元特徴 → 3D 関節位置は `recover_from_ric`、可視化は `plot_3d_motion`（`gen_t2m.py`）。

---

## まとめ：数式 ⇔ 実装 早見表

| 論文 | 内容 | 実装の中心 |
|------|------|------------|
| 式 (1) | 残差量子化 $\mathbf{b}^v=Q(\mathbf{r}^v),\ \mathbf{r}^{v+1}=\mathbf{r}^v-\mathbf{b}^v$ | `models/vq/residual_vq.py:176-181` |
| 式 (2) | RVQ 損失（再構成 + commitment） | `vq_trainer.py:45-50` / `quantizer.py:147` |
| — | Straight-Through / EMA / codebook reset | `quantizer.py:150` / `:100-123` / `:117` |
| — | Quantization Dropout | `residual_vq.py:112-136` |
| 式 (3) | マスク損失（ベース層） | `transformer.py:285,301-302` |
| — | マスクスケジュール $\gamma(\tau)=\cos(\pi\tau/2)$ | `tools.py:120` / `transformer.py:273-275` |
| — | Replacing & Remasking (80/10/10) | `transformer.py:290-299` |
| 式 (4) | 残差損失 $t^j \mid t^{1:j-1},c,j$ | `transformer.py:860-887` |
| — | 重み共有 $j$ 出力 ⇔ $(j{+}1)$ 埋め込み | `transformer.py:759-763` |
| 式 (5) | Classifier-Free Guidance $\omega_g=(1+s)\omega_c-s\omega_u$ | `transformer.py:321-323`（`cond_scale = 1+s`） |
| 推論 | M: 反復生成 / R: 段階生成 | `transformer.py:326-430` / `:891-966` |
