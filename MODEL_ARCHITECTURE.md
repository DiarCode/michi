## Dual-Timescale Graph State-Space Forecasting with Online Residual Correction and Hierarchical Reconciliation (Draft Paper)

### Title

**DTS-GSSF: Dual-Timescale Graph State-Space Forecasting with Online Residual Correction, Drift-Adaptive Low-Rank Updates, and Hierarchical Reconciliation for Real-Time Passenger Flow**

---

### Abstract

Real-time passenger flow forecasting is hard for three reasons that like to team up: (i) **long temporal dependencies** (rush hours, weekly cycles, disruptions), (ii) **spatial coupling** across a transit graph (stations/lines/transfer hubs), and (iii) **non-stationarity** (concept drift): holidays, policy changes, events, weather, construction, sensor recalibration. We propose **DTS-GSSF**, a **dual-timescale** architecture that combines (A) a **graph-structured state-space** forecasting backbone for long-horizon spatio-temporal modeling and (B) an **online residual corrector** that performs fast adaptation via filtering and **drift-triggered low-rank updates**, while (C) enforcing **hierarchical consistency** (station → line → network totals) through optimal reconciliation. We provide a mathematically explicit architecture, training objectives for count data, an online inference/adaptation algorithm, and theoretical justification: (1) residual correction reduces MSE whenever predicted residuals correlate with true residuals; (2) stable residual state dynamics yield bounded correction error; (3) reconciliation is an optimal projection under a weighted error metric; and (4) low-rank adaptation approximates a regularized Bayesian update in a local linear regime. This document is written as a “drop-in” Methods + Theory section suitable for a Q1-style manuscript, with experiments/results left as fill-in.

---

## 1. Problem setup

Let the transit system be a directed/undirected graph (G=(V,E)) with (|V|=N) nodes (stations/segments) and edges encoding connectivity/transfer relations. At discrete time (t\in\mathbb{Z}_{\ge 0}), each node (i) has input features
[
x_{i,t}\in\mathbb{R}^{F}
]
(e.g., recent counts, calendar encodings, weather, events, service disruptions, sensor flags), and target passenger flow count
[
y_{i,t}\in\mathbb{R}_{\ge 0}.
]

**Goal (multi-horizon forecasting).** Given a lookback window length (L), forecast horizon (H), and information set (\mathcal{I}_t = {x_{\cdot,t-L+1:t},, y*{\cdot,t-L+1:t},, G}), output forecasts
[
\hat{y}*{\cdot,t+h} \quad \text{for } h=1,\dots,H.
]

We also assume **hierarchical aggregation**: bottom-level series (y^b\_{t}\in\mathbb{R}^{m}) (e.g., stations) aggregate into higher levels (lines, zones, total) via a known summing matrix (S\in\mathbb{R}^{n\times m}):
[
y_t = S y_t^b,\quad n\ge m.
]
A forecast is **coherent** iff it respects these linear constraints.

---

## 2. Method overview (Idea-3 backbone + Idea-4 online correction)

DTS-GSSF is three coupled modules:

1. **Backbone (slow timescale):** a **Graph-Structured State-Space Forecaster** producing base forecasts (\hat{y}^{(0)}).
   We use state-space sequence modeling ideas (structured SSMs) for long memory and efficiency ([arxiv.org][1]).

2. **Online residual corrector (fast timescale):** maintains a low-dimensional residual state and updates it online (Kalman-style filtering) ([UNC Computer Science][2]). When drift is detected (Page-Hinkley / change detection), trigger a short burst of **low-rank parameter adaptation** (LoRA-style) ([openreview.net][3]).

3. **Hierarchical reconciliation:** projects forecasts onto the constraint subspace using a minimum-trace (MinT) / weighted least squares reconciliation operator ([unitedthc.com][4]).

The point forecast pipeline is:

[
\boxed{
\hat{y}*{t+h} ;=; \mathrm{Reconcile}\Big(\hat{y}^{(0)}*{t+h} + \hat{r}_{t+h}\Big)
}
]

where (\hat{r}\_{t+h}) is the predicted residual correction.

---

## 3. Backbone: Graph-Structured State-Space Forecaster (GSSF)

### 3.1 Input encoding

Per node:
[
u_{i,t} = \phi!\left(W_{\text{in}} x_{i,t} + b_{\text{in}}\right)\in\mathbb{R}^{d}
]
with (\phi) = GELU/ReLU and (d) the model width.

Stack node features into (U_t\in\mathbb{R}^{N\times d}).

### 3.2 Temporal modeling via (selective) state space

For each node (i), maintain a latent state (s\_{i,t}\in\mathbb{R}^{d_s}). A generic discrete state-space block is

[
s_{i,t+1} = A_{i,t} s_{i,t} + B_{i,t} u_{i,t},\qquad
z_{i,t} = C_{i,t} s_{i,t} + D u_{i,t}.
]

- In **structured SSMs (e.g., S4)**, (A) has special structure enabling long memory and fast convolutional evaluation ([arxiv.org][1]).
- In **selective SSMs (e.g., Mamba)**, parameters can depend on input (gating/selectivity), improving expressivity for sequences ([arxiv.org][5]).

We denote this temporal operator as:
[
Z_t = \mathrm{SSM}*\theta(U*{t-L+1:t}) \in \mathbb{R}^{N\times d}.
]

### 3.3 Spatial mixing with adaptive graph propagation

We combine (a) physical adjacency (A^{\text{phys}}) and (b) learned adaptive adjacency (A^{\text{adp}}) (Graph WaveNet-style) ([Google Scholar][6]).

A common adaptive adjacency parameterization:
[
A^{\text{adp}} = \mathrm{softmax}!\big(\mathrm{ReLU}(E_1 E_2^\top)\big),
]
where (E_1,E_2\in\mathbb{R}^{N\times d_e}) are learned node embeddings.

Mix:
[
A^{\text{mix}} = \alpha A^{\text{phys}} + (1-\alpha)A^{\text{adp}},\quad \alpha\in[0,1].
]

Graph propagation (one hop; can stack (K) hops):
[
M_t = \sigma!\left(A^{\text{mix}} Z_t W_g\right)\in\mathbb{R}^{N\times d}.
]

### 3.4 Multi-horizon decoder (counts-aware)

For each horizon (h):
[
\eta_{t+h} = M_t W_h + b_h \in \mathbb{R}^{N}.
]

For count data, two standard choices:

**(i) Poisson head**
[
\hat{\lambda}*{t+h} = \exp(\eta*{t+h}),\qquad
y_{t+h}\sim \mathrm{Poisson}(\hat{\lambda}_{t+h}).
]

**(ii) Negative Binomial head** (handles over-dispersion)
[
\mu_{t+h} = \exp(\eta_{t+h}),\quad
y_{t+h}\sim \mathrm{NB}(\mu_{t+h},,\kappa),
]
with dispersion (\kappa>0) (learned/global/per-node).

We’ll keep notation general: the backbone outputs either point forecasts (\hat{y}^{(0)}) or a distribution (p\_{\theta}(y\mid \mathcal{I}\_t)).

---

## 4. Online residual corrector (fast timescale)

The backbone is powerful but (in the real world) inevitably wrong in _structured_ ways during regime shifts. So we explicitly model the **forecast residual**.

### 4.1 Residual definition

For one-step ahead (streaming update easiest), define
[
r_t = y_t - \hat{y}^{(0)}_{t|t-1}\in\mathbb{R}^{N}.
]

(For multi-horizon, keep a residual head per (h) or correct only (h=1) and propagate via rolling.)

### 4.2 Low-dimensional residual state

We compress residual dynamics into a latent (e_t\in\mathbb{R}^{d_r}) with (d_r\ll N) via an encoder (P\in\mathbb{R}^{d_r\times N}):
[
\tilde{r}_t = P r_t.
]

Model residual evolution as a linear-Gaussian state space:
[
e_{t+1} = F e_t + w_t,\quad w_t\sim\mathcal{N}(0,Q),
]
[
\tilde{r}_t = H e_t + v_t,\quad v_t\sim\mathcal{N}(0,R).
]

Then the predicted residual in original space:
[
\hat{r}*{t+1|t} = P^\top H \hat{e}*{t+1|t}.
]

### 4.3 Kalman filtering update (online)

Given standard Kalman equations ([UNC Computer Science][2]):

**Predict**
[
\hat{e}*{t|t-1}=F\hat{e}*{t-1|t-1},\quad
\Sigma_{t|t-1}=F\Sigma_{t-1|t-1}F^\top + Q.
]

**Update**
[
K_t=\Sigma_{t|t-1}H^\top(H\Sigma_{t|t-1}H^\top + R)^{-1},
]
[
\hat{e}*{t|t}=\hat{e}*{t|t-1}+K_t(\tilde{r}*t - H\hat{e}*{t|t-1}),
]
[
\Sigma_{t|t}=(I-K_tH)\Sigma_{t|t-1}.
]

This yields a **fast**, stable correction signal that reacts to short-term anomalies without retraining the backbone.

---

## 5. Drift detection + drift-triggered low-rank adaptation

Filtering fixes “small but consistent” residual patterns. For **true concept drift**, we also adapt parameters—carefully, to avoid catastrophic forgetting.

### 5.1 Drift test on standardized residuals

Let (z*t) be a scalar drift score. A practical choice:
[
z_t = \frac{1}{N}\sum*{i=1}^{N}\frac{|r*{i,t}|}{\hat{\sigma}*{i,t}+\epsilon}
]
where (\hat{\sigma}) comes from a rolling MAD/variance estimate.

Use a change-detection method such as Page-Hinkley / CUSUM family (classic sequential change detection) ([unitedthc.com][4]). One Page-Hinkley form:

[
m_t = m_{t-1} + (z_t - \bar{z}*t - \delta),\quad
M_t = \min(M*{t-1}, m_t),
]
Trigger drift if
[
m_t - M_t > \lambda,
]
with sensitivity (\delta) and threshold (\lambda).

### 5.2 Low-rank adaptation (LoRA-style) on a short window

When drift triggers, we adapt only a small set of matrices using low-rank updates ([openreview.net][3]).

For a weight matrix (W\in\mathbb{R}^{p\times q}), replace:
[
W \leftarrow W + \Delta W,\quad \Delta W = B A,\quad
B\in\mathbb{R}^{p\times r},; A\in\mathbb{R}^{r\times q},; r\ll \min(p,q).
]

We minimize a **recent-window** negative log likelihood (or MSE) with regularization:
[
\min_{\Delta\theta}
\sum_{\tau=t-W+1}^{t} \mathcal{L}!\left(y_\tau,; f_{\theta+\Delta\theta}(\mathcal{I}_{\tau-1})\right)
;+; \rho |\Delta\theta|_2^2,
]
subject to (\Delta\theta) living only in low-rank adapter parameters.

This is fast (few steps), bounded (regularized), and reversible (store adapters per regime/day).

For drift background and adaptation framing, concept drift reviews are well-established ([arxiv.org][7]).

---

## 6. Hierarchical reconciliation (coherence guarantee)

Raw forecasts for stations/lines/totals often violate sums (classic “incoherent forecast” problem). We enforce:
[
\tilde{y}_{t+h} \in {Sy^b: y^b\in\mathbb{R}^m}.
]

Let (\hat{y}*{t+h}\in\mathbb{R}^{n}) be forecasts for *all* series levels (bottom + aggregated). Reconciliation finds
[
\tilde{y}*{t+h} = P \hat{y}\_{t+h}
]
where (P) is a projection matrix. In MinT reconciliation, (P) is chosen to minimize trace of the reconciled forecast error covariance, yielding a weighted least squares projection ([Rob J Hyndman][8]).

A common expression (one of the standard forms) is:
[
P = S (S^\top W^{-1} S)^{-1} S^\top W^{-1},
]
where (W) approximates the covariance of base forecast errors (estimated from residuals). If (W=I), this becomes ordinary least squares projection.

---

## 7. Training objective

### 7.1 Forecast loss (probabilistic counts)

For Poisson:
[
\mathcal{L}*{\text{pois}}(\theta)=
-\sum*{t}\sum_{h=1}^{H}\sum_{i=1}^{N}
\log \mathrm{Poisson}(y_{i,t+h};\lambda_{i,t+h}(\theta)).
]

For Negative Binomial (one parameterization):
[
\mathcal{L}*{\text{nb}}(\theta)=
-\sum*{t,h,i}\log \mathrm{NB}(y_{i,t+h}; \mu_{i,t+h}(\theta), \kappa).
]

### 7.2 Optional coherence-aware training (soft)

Even though reconciliation enforces hard coherence at the end, you can encourage coherence during training:
[
\mathcal{L} = \mathcal{L}*{\text{forecast}} + \gamma \sum*{t,h}\left| (I-P)\hat{y}_{t+h} \right|_2^2.
]

---

## 8. Online inference algorithm (streaming)

**At each time (t):**

1. Backbone computes (\hat{y}^{(0)}\_{t+1|t}) (and optionally horizons (1..H)).
2. Observe (y*t), compute residual (r_t=y_t-\hat{y}^{(0)}*{t|t-1}).
3. Update residual filter (Kalman) → (\hat{r}\_{t+1|t}).
4. Compute corrected forecast: (\hat{y}_{t+1|t}=\hat{y}^{(0)}_{t+1|t}+\hat{r}\_{t+1|t}).
5. Reconcile: (\tilde{y}_{t+1|t}=P\hat{y}_{t+1|t}).
6. Update drift statistic; if drift triggers:

   - run (k) gradient steps on low-rank adapters using last (W) points.

---

## 9. Mathematical justification (core theorems/lemmas)

### 9.1 Why residual correction reduces MSE (when it should)

Let the true one-step target be (y), base forecast (\hat{y}^{(0)}), residual (r=y-\hat{y}^{(0)}), and corrected forecast (\hat{y}=\hat{y}^{(0)}+\hat{r}). Then the corrected error is:
[
y-\hat{y} = r-\hat{r}.
]

So the mean squared error is:
[
\mathrm{MSE}(\hat{y})=\mathbb{E}|r-\hat{r}|^2
= \mathbb{E}|r|^2 -2\mathbb{E}[r^\top \hat{r}] + \mathbb{E}|\hat{r}|^2.
]

**Optimality statement.** Among all predictors measurable w.r.t. some information (\mathcal{F}) (e.g., last residuals, exogenous signals), the minimizer is:
[
\hat{r}^\star = \mathbb{E}[r\mid \mathcal{F}],
]
and the irreducible error is:
[
\min_{\hat{r}}\mathbb{E}|r-\hat{r}|^2 = \mathbb{E},\mathrm{Var}(r\mid \mathcal{F}).
]

**MSE reduction amount.** Compared to no correction ((\hat{r}=0)):
[
\mathbb{E}|r|^2 - \mathbb{E}|r-\hat{r}^\star|^2 = \mathrm{Var}\big(\mathbb{E}[r\mid \mathcal{F}]\big),
]
i.e., you gain exactly the predictable part of residual variance.

Interpretation: the residual corrector is mathematically “allowed” to help precisely when residuals are not pure noise.

### 9.2 Stability and boundedness of the residual filter

If the residual state dynamics are stable:
[
\rho(F) < 1 \quad (\text{spectral radius}),
]
and noise covariances (Q,R) are bounded, then the Kalman filter error covariance (\Sigma\_{t|t}) remains bounded under standard detectability/stabilizability conditions (classic filtering theory) ([UNC Computer Science][2]). Practically: the correction won’t explode unless the residual model is mis-specified _and_ unregularized.

### 9.3 Reconciliation is an optimal projection

Let the coherent subspace be (\mathcal{C}={Sy^b}). Reconciliation solves:
[
\tilde{y} = \arg\min_{y'\in\mathcal{C}} (y'-\hat{y})^\top W^{-1} (y'-\hat{y}).
]
The solution is the weighted projection (P\hat{y}) with
[
P = S (S^\top W^{-1} S)^{-1} S^\top W^{-1},
]
which is the minimum-trace (MinT) family under standard choices for (W) ([Rob J Hyndman][8]).

### 9.4 Low-rank adaptation as regularized (approx.) Bayesian update

Consider a local linearization around parameters (\theta):
[
f_{\theta+\Delta\theta}(x) \approx f_\theta(x) + J_\theta(x)\Delta\theta.
]
With squared loss and Gaussian noise, minimizing
[
\sum_{\tau}|y_\tau - f_{\theta+\Delta\theta}(x_\tau)|^2 + \rho|\Delta\theta|^2
]
is ridge regression in (\Delta\theta), which corresponds to a Gaussian prior (\Delta\theta\sim\mathcal{N}(0,\rho^{-1}I)). Restricting (\Delta\theta) to a low-rank subspace (LoRA) limits degrees of freedom while preserving rapid adaptation ([openreview.net][3]).

---

## 10. Complexity

Let (T) be number of timesteps processed, (K) graph propagation depth, width (d).

- Backbone temporal SSM blocks can be (O(TNd)) to (O(TNd\log T)) depending on implementation/structure (structured SSM literature emphasizes efficiency for long contexts) ([arxiv.org][1]).
- Graph mixing is roughly (O(T|E|dK)) for sparse adjacency (or (O(TN^2d)) if dense adaptive adjacency is used without sparsification).
- Residual filter is (O(T d_r^3)) if fully dense Kalman; typically (d_r) is small and can be diagonal/low-rank for (O(T d_r)).
- Drift adaptation: (k) gradient steps on only adapter parameters; cost is a tiny fraction of full fine-tuning.

---

## 11. Experimental protocol (what to report)

_(No made-up results here—this is the exact checklist to fill.)_

- **Datasets:** (your passenger flow feeds) + baselines: persistence, SARIMAX, Prophet, DCRNN/GraphWaveNet-style STGNN, Transformer.
- **Splits:** rolling-origin evaluation; also event-focused slices (holidays, disruptions).
- **Metrics:** MAE/RMSE; Poisson deviance; MAPE/SMAPE (careful with zeros); calibration (PIT, coverage) if probabilistic.
- **Ablations:**
  (i) no residual corrector, (ii) filter only, (iii) filter+drift LoRA, (iv) reconciliation on/off, (v) adaptive adjacency on/off.
- **Online setting:** prequential (“predict then observe then update”), report latency and compute.

---

## 12. Novelty (what is actually new here)

To keep this Q1-credible, the novelty should be framed as **a specific integration + theory + real-time constraints**, not “we invented graphs.”

**Core contributions (as written):**

1. **Dual-timescale design**: a powerful offline spatio-temporal backbone plus a mathematically principled online residual state corrector.
2. **Drift-triggered low-rank adaptation**: update only small adapter subspaces when a sequential change detector fires, controlling instability and compute.
3. **Coherence-by-construction**: reconciliation guarantees hierarchical sums without forcing the backbone to learn hard constraints implicitly.
4. **Transparent theory**: explicit MSE decomposition for residual correction, stability conditions for the corrector, and optimality of reconciliation projection.

The underlying building blocks exist in literature (SSMs ([arxiv.org][1]), adaptive graph forecasting ([Google Scholar][6]), LoRA ([openreview.net][3]), concept drift detection/adaptation ([arxiv.org][7]), reconciliation ([Rob J Hyndman][8])); what you’re claiming is the **tight, real-time, mathematically justified system** that makes them play nicely together for passenger flow.
