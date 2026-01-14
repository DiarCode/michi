### Normalized facts (what you _actually_ gave me)

- **Dissertation title:** _Real-Time Adaptive Passenger Flow Prediction: A Hybrid Model Approach._
- **Hard requirements:** real-time streaming + fast inference; uses **historical + very recent** signals; scalable from **one city → whole country**; **storage-efficient**; **secure/privacy-aware**; integrates into **LLM-facing** user answers.
- **Not provided (so I will not fabricate):** which country/cities, sensors (AFC/turnstiles/AVL/CCTV/etc.), prediction horizon (5–60 min? hours?), update frequency (1–15 min?), whether you need station inflow/outflow vs OD matrices, and any experimental results.

I’ll proceed with explicit assumptions and give you a **thesis-ready research framing + method blueprint + core math + evaluation plan** (with “TBD” where experiments would go).

---

## 1) What the field does today (and where it breaks in “real-time + national scale”)

### A. “Classic” spatio-temporal baselines (mostly offline)

Traffic/flow forecasting matured around **spatio-temporal graph neural networks (STGNNs)**:

- **DCRNN** models flow as diffusion on a directed graph + seq2seq temporal modeling. ([arxiv.org][1])
- **STGCN** uses purely convolutional temporal blocks + graph convolutions for speed/parameter efficiency. ([ijcai.org][2])
- **Graph WaveNet** learns an **adaptive adjacency** (hidden spatial dependencies) + dilated temporal conv for long receptive fields. ([arxiv.org][3])
- **GMAN** uses multi-attention blocks in encoder–decoder form to capture long-range spatio-temporal interactions. ([arxiv.org][4])

These are excellent backbones, but most papers assume **centralized training**, **fixed cities**, and evaluation that doesn’t fully stress **streaming drift** and **operational latency**.

### B. Passenger-flow-specific STGNNs (metro/stations)

Passenger flow prediction has its own quirks: commuting periodicity, line structure, sharp event spikes, and inflow/outflow coupling.

- A representative station-level metro model explicitly frames **stations as graph nodes** and learns multi-scale temporal patterns (recent/daily/weekly) for inflow/outflow. ([mdpi.com][5])
- Dynamic graph relation learning + transformers appear in more recent metro-flow models to handle time-varying dependencies (spatial correlations change with time).
- Mixture-of-experts + transformer variants target **long-range + heterogeneous station behaviors**. ([sciencedirect.com][6])

### C. Real-time reality: concept drift + data silos + privacy

Two “adult problems” dominate deployments:

1. **Concept drift:** demand patterns shift (weather, school schedules, disruptions, special events). Drift detection is a whole field (e.g., EDDM monitors error patterns in streams). ([ResearchGate][7])
2. **National scale & privacy:** city/line operators often cannot centralize raw AFC/smart-card data. Federated learning is increasingly used:

   - **FedMetro (KDD’25)**: federated dynamic graph learning + cross-attention; tackles **time-evolving spatial correlations**, heterogeneous temporal correlations, and even **communication compression** for real-time needs. ([hufudb.com][8])

### Core gap your dissertation can own

Most work optimizes **predictive accuracy** in offline settings. Your “Q1-worthy” angle is the **full-stack scientific framing**:

> A _real-time_ passenger-flow forecaster must be **(i)** fast at inference, **(ii)** adaptive under drift, **(iii)** scalable across heterogeneous networks, **(iv)** storage- and privacy-aware, and **(v)** explainable/usable through a constrained LLM interface.

That’s not “marketing”; it’s an engineering-scientific claim about **requirements that change the model design**.

---

## 2) Thesis problem statement + research questions (ready to paste)

### Problem statement

Existing passenger-flow prediction methods often assume static training distributions, centralized data access, and city-specific deployment. In practice, passenger demand is non-stationary, data are distributed across agencies/lines, and decisions require low-latency forecasts with uncertainty and interpretable outputs. This dissertation studies how to design a **real-time adaptive** passenger-flow prediction framework that remains accurate under drift, scalable across networks, and efficient in both computation and storage.

### Research questions (RQs)

- **RQ1 (Modeling):** How can we combine long-term periodic patterns with short-term streaming signals to improve near-future prediction without retraining from scratch?
- **RQ2 (Adaptation):** How can we detect and respond to concept drift in passenger-flow streams with bounded compute and stability guarantees (avoid catastrophic forgetting)?
- **RQ3 (Scalability):** How can a single framework generalize from one city to many, while minimizing data movement and storage?
- **RQ4 (Operationalization):** How can forecasts be served in real time with measurable latency/SLOs and exposed to users via an LLM interface that is faithful to the underlying numbers and uncertainty?

### Contributions (target 3–5 bullets)

1. A **hybrid** forecasting model that couples a strong STGNN/Transformer backbone with a **state-space online correction** layer for immediate updates.
2. A **drift-aware adaptation strategy** using lightweight parameter adapters (small memory footprint per city/line) triggered by streaming drift detectors.
3. A **multi-resolution (hierarchical) spatial formulation** enabling national-scale deployment with storage-efficient aggregation.
4. A **real-time serving + governance design** (privacy/federation option, data minimization, latency/throughput targets).
5. An **LLM-facing forecast interface** that is tool-driven and uncertainty-aware (no hallucinated numbers).

---

## 3) Proposed novel approach: a Hybrid, Real-Time, National-Scale framework

I’ll name it neutrally so you can brand it later:

### **HRA-Flow**: _Hierarchical Real-time Adaptive Flow Forecasting_

**Key idea:** separate the problem into (A) a powerful **offline-trained prior model** and (B) a **fast online corrector** + (C) **drift-triggered micro-updates**.

#### (A) Offline prior (learns structure & long-term patterns)

Backbone: a spatio-temporal graph model with:

- **Spatial module:** learned dynamic adjacency (à la adaptive graphs) + sparse message passing.
- **Temporal module:** multi-scale: short-term causal conv + long-term attention (transformer-like blocks).
- Output is **probabilistic** (mean + variance or quantiles), so your LLM layer can communicate uncertainty.

This borrows the strengths of STGCN/DCRNN/GraphWaveNet/GMAN style families while being passenger-flow specific. ([arxiv.org][1])

#### (B) Online state-space correction (fast, per-timestep)

When a new observation arrives at time _t_ (latest turnstile/AFC counts), do **O(N)** to **O(Nk)** work to adjust predictions immediately—without running full backprop.

We treat the backbone forecast as a **prior**, then correct via a lightweight **state-space filter** on the residuals.

#### (C) Drift-aware micro-adaptation (rare, controlled updates)

A drift detector monitors residual behavior. When drift is detected, update only **small adapters** (e.g., low-rank updates) on a short recent window, keeping the base model fixed to preserve stability.

#### (D) Hierarchical national scaling

Represent the country as:

- **Level 0:** stations/stops (fine graph).
- **Level 1:** zones/districts/cities (coarse graph).
- **Level 2:** inter-city corridors/regions (macro graph).

Train with **consistency constraints** so fine-level forecasts aggregate correctly to coarse-level forecasts. This enables:

- cheaper long-horizon reasoning at coarse levels,
- storage minimization via multi-resolution retention,
- easier cross-city transfer (city embeddings + adapters).

#### (E) Privacy/security option: Federated training across agencies

If data can’t be centralized, you can adopt a federated variant inspired by recent metro/traffic FL work. FedMetro demonstrates that federated dynamic graph learning + compression can work for metro passenger flow in distributed AFC settings. ([hufudb.com][8])

---

## 4) Mathematical formulation (thesis-grade, reproducible, no fake results)

### 4.1 Data and notation

Let the transit network be a graph (G=(V,E)) with (|V|=N) stations.

At discrete time (t) (e.g., every 5–15 minutes), define:

- Observed passenger flow vector (y_t \in \mathbb{R}^{N \times 2}) (inflow/outflow).
- Exogenous features (u_t) (calendar, weather, events, disruptions).
- Optional OD flows (o_t \in \mathbb{R}^{N \times N}) if available (often too heavy for real-time; can be modeled at coarse level).

Goal: predict (y\_{t+h}) for horizons (h \in {1,\dots,H}).

### 4.2 Prior forecasting model (offline-trained)

Define a prior model (f*\theta) producing a distribution:
[
(\mu*{t+1:t+H}, \Sigma*{t+1:t+H}) = f*\theta\big(y*{t-L+1:t}, u*{t-L+1:t}, A_t\big)
]
where (A_t) is a (possibly learned) adjacency / attention sparsity pattern.

Training objective (example, Gaussian NLL):
[
\mathcal{L}_{\text{NLL}}(\theta) = \sum_{t}\sum*{h=1}^{H}
\Big[(y*{t+h}-\mu*{t+h})^\top \Sigma*{t+h}^{-1}(y*{t+h}-\mu*{t+h}) + \log|\Sigma\_{t+h}|\Big]
]
If you prefer robustness, replace with quantile loss (pinball) or Huber.

### 4.3 Online residual state-space corrector (real-time)

Define residual at time (t) for one-step prediction:
[
e_t = y_t - \mu_t
]
Model residual dynamics as:
[
r_{t} = \Phi r_{t-1} + w_t,\quad e_t = r_t + v_t
]
with (w_t \sim \mathcal{N}(0,Q)), (v_t \sim \mathcal{N}(0,R)).

Kalman filter update (per node, or block-diagonal by regions to keep it fast):

- Predict:
  [
  \hat r_t^{-}=\Phi \hat r_{t-1},\quad P_t^{-}=\Phi P_{t-1}\Phi^\top + Q
  ]
- Gain:
  [
  K_t = P_t^{-}(P_t^{-}+R)^{-1}
  ]
- Update:
  [
  \hat r_t=\hat r_t^{-}+K_t(e_t-\hat r_t^{-}),\quad
  P_t=(I-K_t)P_t^{-}
  ]

Corrected forecast for horizons:
[
\mu'*{t+h} = \mu*{t+h} + g(h),\hat r_t
]
where (g(h)) is a decay (e.g., (g(h)=\exp(-\lambda h))) or learned small vector.

**Justification:** Under linear-Gaussian assumptions, the Kalman update yields the minimum mean-square-error state estimate. Under nonlinearity, (\Phi) can be learned and the filter acts as a stable fast corrector (you can frame this as “approximate Bayesian filtering”).

### 4.4 Drift detection trigger (streaming)

Use a drift detector on residual stream (e_t). EDDM-style methods track changes in error patterns over time. ([ResearchGate][7])

When drift is signaled, enable short-window adaptation.

### 4.5 Lightweight online adaptation via low-rank adapters

Let the base model be frozen ((\theta) fixed). Insert adapters with parameters (\phi) (small):

- For a linear transform (W \in \mathbb{R}^{d\times d}), use low-rank update:
  [
  W' = W + \Delta W,\quad \Delta W = AB^\top,\ A,B \in \mathbb{R}^{d\times r},\ r \ll d
  ]
  Parameter cost: (2dr) instead of (d^2).
  Example: (d=512, r=8 \Rightarrow 2dr = 8192) params per adapted matrix (tiny).

Online objective on a rolling window (\mathcal{W}) (recent minutes/hours):
[
\min_{\phi} \sum_{t\in \mathcal{W}} \ell\big(y_{t+1:t+H}, f_{\theta,\phi}(\cdot)\big) + \beta|\phi|^2
]
Stop/rollback rules are part of “safety”: if validation-on-stream worsens, revert adapters.

### 4.6 Hierarchical consistency constraint (national scaling)

Let (S \in {0,1}^{M \times N}) aggregate stations into (M) zones (each station belongs to one zone).

- Zone flows: (\tilde y_t = Sy_t)
- Enforce consistency:
  [
  \mathcal{L}*{\text{cons}} = \sum*{t,h}|S\mu_{t+h}^{\text{station}} - \mu_{t+h}^{\text{zone}}|_2^2
  ]
  This gives you a principled way to “scale up” while keeping forecasts coherent across resolutions.

---

## 5) Storage and speed: concrete calculations (so you can argue feasibility)

### Storage baseline (raw per-minute inflow/outflow)

Assume:

- (N) stations, 2 values (in/out), 4 bytes each (int32), per minute
- Minutes/year = 525,600

Bytes/year:
[
\text{Storage} = N \cdot 2 \cdot 4 \cdot 525{,}600
]

- If (N=5{,}000): (5{,}000 \cdot 2 \cdot 4 \cdot 525{,}600 = 21{,}024{,}000{,}000) bytes ≈ **21.0 GB/year** (decimal).
- If (N=50{,}000): ≈ **210 GB/year**.

That’s manageable, but **OD matrices are not** (they’re (N^2)). So your thesis can justify:

- store **station flows** at fine resolution in “hot” storage,
- store **OD** only at coarse resolution or sampled, or store low-rank factors.

### Multi-resolution retention (minimize “forever storage”)

A defensible policy:

- Hot: last 7–30 days at full resolution.
- Warm: 6–12 months aggregated to 10–15 min.
- Cold: historical baselines (hour-of-week means/variances, holiday profiles, event fingerprints).

This aligns with streaming constraints stated explicitly in drift-detection literature: you can’t keep everything in memory for high-rate streams. ([ResearchGate][7])

### Inference complexity (why your design is real-time)

- Graph message passing with k-neighborhood sparsity: (O(Nkd)) per layer.
- Dense attention is (O(N^2 d)) (too slow at national scale), so you argue for **sparse attention / learned top-k edges**.

---

## 6) Real-time system architecture (fast, secure, scalable, LLM-ready)

**Figure you can draw (pipeline):**

```
Sensors/AFC/AVL → Stream Ingest → Feature Builder → (Prior Model) → (Online Corrector)
                         ↓                 ↓                 ↓
                    Hot Store         Model Cache       Forecast Store
                         ↓                                   ↓
                Drift Monitor → Adapter Update (rare)     LLM Tool API
```

### Components (implementation-agnostic)

1. **Ingest:** event-time aligned stream (handles late arrivals).
2. **Feature builder:** sliding windows + calendar + weather/events; imputes missing sensors.
3. **Forecast service:** serves μ/intervals with strict latency SLOs.
4. **Corrector:** runs per update; constant-time-ish per station/zone.
5. **Drift monitor:** watches residuals; triggers controlled adaptation.
6. **Governance/security:**

   - encryption in transit + at rest,
   - strict retention and aggregation,
   - anonymization/pseudonymization (no user IDs needed for station counts),
   - optional **federated training** when agencies can’t pool data (FedMetro shows feasibility for metro AFC silos). ([hufudb.com][8])

### LLM integration (without “numbers hallucination”)

Treat the LLM as a **verbalization layer**:

- LLM can only answer using a **tool call** that returns:

  - prediction ( \mu\_{t+h} ),
  - uncertainty (intervals),
  - context flags (drift active, data missing, disruption tag).

- The LLM produces explanations _conditioned on those facts_, and must quote numbers from the tool output.

This is scientifically defensible: it separates **numerical forecasting** from **natural-language generation**.

---

## 7) Evaluation plan (what makes it Q1-grade)

### Datasets you can use (real + citable)

- **Shanghai metro flow dataset (May–Aug 2017, 302 stations, 10-min inflow/outflow + OD + weather/topology)** released as open data in _Scientific Data_. ([Nature][9])
- **NYC MTA subway hourly ridership** is referenced as the modern replacement for older turnstile datasets in the NY open-data catalog. ([catalog.data.gov][10])
- Benchmarks like **SHMetro/HZMetro** are discussed in the Scientific Data paper (even if access differs). ([Nature][9])

### Splits and protocol (avoid leakage)

- Time-based split: train on earliest 70%, validate next 10–15%, test last 15–20%.
- Streaming evaluation: **prequential** (predict → observe → update corrector; adapters update only on drift triggers).

### Metrics

- Point: MAE, RMSE, sMAPE (avoid MAPE explosions near zero).
- Probabilistic: coverage of prediction intervals, CRPS (if you output distributions).
- Real-time: p50/p95 latency, throughput, memory, missed-deadline rate.
- Robustness: missingness, disruptions, event spikes, cross-city transfer.

### Baselines (fair + canonical)

- Statistical: seasonal naïve, SARIMA / ETS (citation needed if you include).
- Deep temporal: LSTM/GRU.
- STGNN family: STGCN, DCRNN, Graph WaveNet, GMAN. ([ijcai.org][2])
- Passenger-flow STGNN: station-level metro STGNN variants. ([mdpi.com][5])
- If you include federated: compare with FedMetro-style setup. ([hufudb.com][8])

### Ablations (the “convince reviewers” list)

1. Prior model only (no corrector).
2. - corrector (no drift adaptation).
3. - drift adaptation (no hierarchy).
4. Full HRA-Flow.
5. Sparse vs dense adjacency; top-k sensitivity.
6. With/without exogenous features (weather/events).

---

## 8) Draft text you can paste: Intro + Related Work scaffolding (no fake results)

### Introduction (draft scaffold)

Real-time passenger flow prediction is central to modern public transport operations, supporting congestion warnings, timetable control, and disruption management. In contrast to offline planning, operational decision-making requires forecasts that update continuously as new observations arrive and that remain reliable under non-stationary demand patterns. Passenger flows exhibit strong periodic structure driven by work and school schedules, yet they also show abrupt deviations due to weather, special events, and network disruptions. These properties motivate forecasting methods that combine long-range historical regularities with rapid adaptation to short-term changes.

Recent progress in spatio-temporal learning on graphs has produced strong models for traffic and flow forecasting. Representative architectures include diffusion-based recurrent models that encode directed spatial propagation and temporal dynamics, convolutional graph-temporal frameworks designed for efficient learning, adaptive graph approaches that learn latent spatial dependencies, and attention-based encoder–decoder models that capture long-range interactions. ([arxiv.org][1]) However, many of these methods are evaluated primarily in batch settings, and their assumptions—centralized training data, stationary distributions, and single-network deployment—often do not align with the operational constraints of passenger-flow systems.

Two practical constraints are increasingly important. First, real-time deployments face concept drift: demand distributions change over time, and naive continual retraining can be computationally expensive and unstable. Drift detection in data streams provides principled mechanisms for identifying distributional changes using online error statistics, enabling controlled adaptation. ([ResearchGate][7]) Second, metro and transit data are frequently siloed across operators or organizational units, limiting centralized modeling. Federated learning has emerged as a privacy-preserving alternative in spatio-temporal prediction, and recent work demonstrates its applicability to metro passenger flow with dynamic graph learning and communication-efficient inference. ([hufudb.com][8])

This dissertation addresses these gaps by proposing a hybrid framework that (i) learns a strong spatio-temporal prior from historical data, (ii) performs fast online correction through state-space residual filtering, (iii) adapts selectively under drift via lightweight parameter adapters, and (iv) scales across cities using a hierarchical spatial formulation with storage-efficient retention. The resulting system is designed to support real-time decision-making and to expose uncertainty-aware forecasts through a constrained LLM interface for user-facing explanations.

### Related Work (draft scaffold)

**Spatio-temporal graph forecasting.** Early and widely adopted approaches incorporate graph structure into sequence modeling. DCRNN models traffic as a diffusion process on directed graphs and uses a seq2seq architecture for temporal dependence. ([arxiv.org][1]) STGCN replaces recurrent units with convolutional components to improve efficiency while capturing spatio-temporal dependencies. ([ijcai.org][2]) Graph WaveNet introduces an adaptive dependency matrix learned from node embeddings, addressing the mismatch between physical topology and true statistical dependence, and uses dilated temporal convolutions for long receptive fields. ([arxiv.org][3]) Attention-based architectures such as GMAN apply multi-attention blocks to capture complex spatio-temporal interactions and reduce error propagation in multi-step forecasting. ([arxiv.org][4])

**Passenger-flow prediction in metro networks.** Station-level metro forecasting frames stations as nodes and learns inflow/outflow dynamics with explicit modeling of multi-scale temporal patterns (recent/daily/weekly) and deep graph convolution to capture distant dependencies in citywide networks. ([mdpi.com][5]) More recent metro-flow approaches emphasize dynamic spatial relationships and heterogeneous temporal behaviors, reflecting the time-varying nature of passenger correlations across stations.

**Streaming drift and adaptation.** Real-world forecasting streams violate stationarity assumptions. Drift detection methods such as EDDM monitor changes in online error behavior and can trigger adaptation in a controlled manner. ([ResearchGate][7]) In forecasting contexts, drift-aware mechanisms are especially relevant for handling disruptions and seasonal regime changes without full retraining (citation needed for forecasting-specific drift surveys).

**Privacy-preserving and federated spatio-temporal learning.** Distributed data ownership and privacy constraints motivate federated approaches that avoid raw data sharing. FedMetro demonstrates a federated dynamic graph learning system for metro passenger flow prediction, addressing time-evolving spatial correlations and inference-time communication bottlenecks. ([hufudb.com][8]) This line of work suggests that national-scale deployment can be feasible when combined with communication-efficient model updates and personalization mechanisms (citation needed for broader FL-STGNN surveys).

---

## Minimum experiments checklist (the “do these and you have a defensible dissertation” list)

- **E1:** Offline accuracy on at least 2 datasets (e.g., Shanghai + NYC).
- **E2:** Real-time simulation (stream replay) with measured latency + prequential metrics.
- **E3:** Drift scenario tests (synthetic drift + real disruptions): compare no-adapt vs drift-aware adapters.
- **E4:** Cross-city generalization: train on City A, test on City B; then adapter-only personalization.
- **E5:** Hierarchy benefit: coherence error (aggregation consistency) + accuracy vs single-level model.
- **E6:** Storage strategy evaluation: show forecast quality vs retained resolution (hot/warm/cold).
- **E7:** Ablations: prior vs +corrector vs +adapters vs full model.
- **E8:** LLM interface audit: faithfulness tests (LLM outputs must match tool-returned numbers/intervals).

---
