# PolyDiT

PolyDiT is a Diffusion Transformer project for discrete structured generation and representation learning. It asks a simple modeling question: how much do tokenization and representation choices change diffusion generation, learned embeddings, and downstream prediction?

## Motivation

Discrete scientific data has validity constraints, long-range dependencies, and representation-dependent inductive bias. A character tokenizer is simple, BPE shortens sequences, constrained tokens improve validity, fragment tokens expose higher-level units, and graph tokens preserve relational structure. PolyDiT compares these choices under one masked diffusion framework.

## Method

Each representation \(r\) maps an input object \(s\) to a clean discrete state:

$$
x_0^{(r)}=\phi_r(s).
$$

Masked diffusion corrupts mutable positions over time:

$$
\beta_t=\beta_{\min}+(\beta_{\max}-\beta_{\min})\frac{t}{T},
\qquad
x_{t,i}=
\begin{cases}
\texttt{[MASK]}, & u_i < \beta_t,\\
x_{0,i}, & \text{otherwise}.
\end{cases}
$$

The denoiser learns:

$$
p_\theta(x_0\mid x_t,t).
$$

For sequences, the training objective is masked categorical reconstruction:

$$
\mathcal{L}_{\mathrm{seq}}
=
\mathbb{E}
\left[
\frac{1}{|\Omega_t|}
\sum_{i\in\Omega_t}
\operatorname{CE}
\left(
x_{0,i},
p_\theta(x_{0,i}\mid x_t,t)
\right)
\right].
$$

For graphs, node and edge reconstruction are optimized together:

$$
\mathcal{L}_{\mathrm{graph}}
=
\lambda_X\mathcal{L}_X+\lambda_E\mathcal{L}_E.
$$

## Architecture

The sequence model is a timestep-conditioned bidirectional Transformer:

$$
h_i^{(0)}
=
e_{\mathrm{tok}}(x_{t,i})
+e_{\mathrm{pos}}(i)
+e_{\mathrm{time}}(t).
$$

Each block applies pre-normalized self-attention and feed-forward updates:

$$
H^{(\ell+1)}
=
H^{(\ell)}
+\operatorname{MHA}(\operatorname{LN}(H^{(\ell)}))
+\operatorname{FFN}(\cdot).
$$

The graph variant uses edge-aware attention:

$$
a_{ij}^{(k)}
=
\frac{(Q^{(k)}h_i)^\top(K^{(k)}h_j)}{\sqrt{d_k}}
+b^{(k)}(E_{t,ij}).
$$

Model sizes range from 10.9M to 681.1M parameters. Training supports mixed precision, distributed data parallelism, gradient accumulation, dynamic padding, length bucketing, warmup, cosine decay, and multi-node runs such as 2 nodes with 8 GPUs per node.

| Size | Layers | Width | Heads | Params |
|---|---:|---:|---:|---:|
| S | 6 | 384 | 3 | 10.9M |
| M | 12 | 768 | 6 | 85.6M |
| L | 18 | 1,152 | 9 | 288.5M |
| XL | 24 | 1,536 | 12 | 681.1M |

## Data

Pretraining uses 19,345,999 training examples and 1,018,937 validation examples. Representation quality is evaluated on eight regression targets with frozen backbone features and lightweight supervised heads.

## Results

PolyDiT evaluates five state spaces, four model sizes, 200,000 generated samples, and 80 held-out downstream test cells. The best representation depends on the objective.

| Objective | Best representation | Value |
|---|---|---:|
| Validity | constrained tokens | 1.0000 |
| Constraint correctness | graph tokens | 1.0000 |
| Novelty | fragment-level tokens | 1.0000 |
| Diversity | BPE tokens | 0.9033 |
| Best downstream \(R^2\) | character tokens | 0.943 |

The main empirical finding is:

$$
\arg\max_r \mathrm{Validity}(r)
\neq
\arg\max_r \mathrm{Novelty}(r)
\neq
\arg\max_r R_k^2(r).
$$

## How To Use

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and sample one representation:

```bash
cd Bi_Diffusion_SMILES
bash scripts/run_pipeline.sh configs/config.yaml medium 10000
```

Train and sample separately:

```bash
cd Bi_Diffusion_SMILES
bash scripts/run_step1.sh medium
bash scripts/run_step2.sh 10000 medium
```

Run the representation-learning benchmark:

```bash
cd Multi_View_Foundation
bash scripts/run_property_regression.sh medium
```

Run smoke tests:

```bash
pytest tests
```

## Conclusion

PolyDiT shows that representation is part of the model, not just preprocessing. The same Diffusion Transformer objective can favor different state spaces for validity, novelty, diversity, and downstream representation learning, so the best tokenizer or representation should be selected according to the target objective.
