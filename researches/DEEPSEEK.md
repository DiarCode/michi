## **Real‑Time Adaptive Passenger Flow Prediction: A Scalable Hybrid Framework Integrating Incremental Spatiotemporal Graph Networks and Large Language Models**

**Running Title:** Adaptive Hybrid Passenger‑Flow Prediction

### **Structured Abstract**

- **Background:** Accurate real‑time passenger‑flow forecasting is critical for efficient public‑transport management. Existing methods often struggle to simultaneously capture dynamic spatiotemporal dependencies, adapt to evolving patterns, scale to country‑wide networks, and provide user‑friendly interfaces.
- **Methods:** We propose a scalable hybrid framework that integrates (i) an **Incremental Spatiotemporal Graph Neural Network (ISTGNN)** for adaptive prediction, (ii) a **distributed edge‑cloud data pipeline** for scalable real‑time processing, and (iii) a **fine‑tuned Large Language Model (LLM)** for natural‑language querying and explanation. The ISTGNN combines graph convolutional layers (spatial) with recurrent units (temporal) and employs incremental learning with memory‑aware synapses to continuously adapt to new data without catastrophic forgetting.
- **Expected Results:** The framework is designed to be evaluated on large‑scale metro datasets (e.g., Delhi Metro, Nanjing Metro). We anticipate that it will outperform static hybrid models (e.g., GCMN [reference:0]) and non‑incremental baselines by a significant margin (e.g., >30 % reduction in RMSE) while maintaining low latency (<1 s) for real‑time queries. Ablation studies will validate the contribution of each component.
- **Conclusions:** The proposed framework offers a novel, scalable, and adaptive solution for country‑wide passenger‑flow prediction. By bridging incremental spatiotemporal learning with LLM‑based natural‑language interaction, it aims to set a new standard for real‑time transport‑management systems.

**Keywords:** passenger flow prediction, hybrid deep learning, incremental learning, graph neural networks, large language models, scalable real‑time systems.

---

### **1. Introduction**

Urban public‑transport systems face ever‑growing demand, making real‑time passenger‑flow prediction a cornerstone of efficient operation. Accurate forecasts enable dynamic scheduling, congestion mitigation, and improved passenger experience. However, building a prediction system that is simultaneously **accurate, adaptive, scalable, and user‑friendly** remains an open challenge.

**Current limitations:** Traditional time‑series models (ARIMA, exponential smoothing) fail to capture complex non‑linear spatiotemporal patterns. Recent deep‑learning approaches, especially hybrid models that combine Long Short‑Term Memory (LSTM) networks with Graph Convolutional Networks (GCN), have shown promise[reference:1]. For example, the Graph Convolutional Memory Network (GCMN) achieves an R² of 0.920 on Delhi Metro data[reference:2]. Nevertheless, these models are typically **static** – they require periodic retraining and cannot adapt to gradual pattern shifts (e.g., new stations, changing travel habits). Incremental learning frameworks, such as the Incremental Multi‑Graph Seq2Seq Network (IMGSN), address this by continuously integrating new data while preserving historical knowledge, reporting a 45.69 % average improvement over non‑incremental models[reference:3]. Meanwhile, the integration of Large Language Models (LLMs) for transportation prediction is emerging, with GPT‑2‑based models demonstrating the ability to fuse multi‑source spatiotemporal data[reference:4]. However, no existing work unifies **adaptive spatiotemporal learning, scalable distributed processing, and natural‑language interaction** into a single framework suitable for country‑wide deployment.

**Contributions:** This dissertation presents a novel hybrid framework that fills the above gap. The main contributions are:

1.  **A scalable, real‑time data pipeline** that integrates distributed edge sensors with a cloud‑based processing stack, enabling country‑wide data ingestion with optimized storage (using columnar formats and streaming compression).
2.  **An Incremental Spatiotemporal Graph Neural Network (ISTGNN)** that dynamically captures both spatial (graph‑based) and temporal (recurrent) dependencies, equipped with a memory‑aware synaptic regularization to prevent catastrophic forgetting during continuous learning.
3.  **Seamless integration of a fine‑tuned LLM** that translates numerical predictions into natural‑language summaries and answers user queries (e.g., “When will the next peak occur at Station X?”).
4.  **A comprehensive evaluation plan** using large‑scale metro datasets (Delhi, Nanjing) and a set of strong baselines (GCMN, IMGSN, pure LSTM/GCN) to validate the framework’s accuracy, adaptability, scalability, and latency.
5.  **Full mathematical formulation** of the hybrid model, including the graph‑convolutional operations, incremental‑learning objective, and LLM fine‑tuning procedure.

**Roadmap:** Section 2 reviews related work. Section 3 details the proposed framework. Section 4 outlines the evaluation plan and expected results. Section 5 discusses implications and limitations. Section 6 concludes.

---

### **2. Related Work**

**2.1 Statistical & Machine‑Learning Approaches**
Early passenger‑flow prediction relied on statistical time‑series models (ARIMA, exponential smoothing) and machine‑learning methods (SVR, Bayesian networks). These models often ignore spatial correlations and complex non‑linearities.

**2.2 Deep Spatiotemporal Models**
Deep learning has revolutionized the field. Recurrent networks (LSTM, GRU) capture temporal patterns, while graph convolutional networks (GCN) model spatial dependencies across stations. Hybrid architectures that combine both have become the state‑of‑the‑art for metro‑flow prediction[reference:5].

**2.3 Incremental/Lifelong Learning for Transportation**
Static models degrade as travel patterns evolve. Incremental learning frameworks, such as IMGSN[reference:6], address this by continuously updating the model with new data while protecting previously learned knowledge via techniques like memory‑aware synapses.

**2.4 LLMs for Transportation Prediction**
Recent studies explore LLMs for traffic and passenger‑flow forecasting. GPT‑2 has been fine‑tuned to fuse multi‑source data (weather, events) and produce predictions[reference:7]. However, these approaches are not yet integrated with adaptive spatiotemporal models.

**2.5 Scalable Distributed Systems**
For country‑wide deployment, scalable architectures are essential. Distributed edge‑AI systems[reference:8] demonstrate how to process high‑volume sensor data in real time, reducing network load and latency.

**Positioning:** Our framework synthesizes the strengths of the above lines: the spatiotemporal modeling of hybrid GCN‑LSTM, the adaptability of incremental learning, the user‑friendly interface of LLMs, and the scalability of distributed edge‑cloud pipelines.

---

### **3. Methods**

**3.1 Overall Framework**
The framework comprises three layers:

1.  **Data Layer:** A distributed pipeline of edge sensors (cameras, fare‑gate logs) that stream data to a cloud‑based time‑series database (e.g., InfluxDB). Historical data is stored in compressed columnar format (Parquet).
2.  **Prediction Layer:** The core ISTGNN model, which ingests streaming graphs and produces rolling forecasts.
3.  **Interface Layer:** A fine‑tuned LLM (e.g., GPT‑2 or a smaller transformer) that converts numerical predictions into natural‑language responses.

**3.2 Data Preprocessing & Graph Construction**

- **Data sources:** Historical passenger counts, real‑time sensor streams, weather, calendar events.
- **Graph definition:** Each station is a node; edges represent physical connectivity (rail lines) or passenger‑flow correlations. The adjacency matrix \(A\) is built from network topology.
- **Feature matrix:** Each node has a feature vector \(X_t \in \mathbb{R}^{F}\) at time \(t\), containing passenger flow, time‑of‑day, day‑of‑week, and external factors.

**3.3 Incremental Spatiotemporal Graph Neural Network (ISTGNN)**

**3.3.1 Spatial Module – Graph Convolution**
We use a two‑layer GCN to aggregate neighbor information:
\[
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} \Theta^{(l)}\right)
\]
where \(\tilde{A} = A + I\), \(\tilde{D}\) is the degree matrix, \(H^{(l)}\) is the node representation at layer \(l\), and \(\Theta^{(l)}\) are learnable weights.

**3.3.2 Temporal Module – Gated Recurrent Unit (GRU)**
The sequence of graph‑convolved features \(\{H*t\}\) is fed into a GRU:
\[
z_t = \sigma(W_z [H_t, h*{t-1}]),
\quad
r*t = \sigma(W_r [H_t, h*{t-1}]),
\quad
\tilde{h}_t = \tanh(W_h [H_t, r_t \odot h_{t-1}]),
\quad
h*t = (1-z_t) \odot h*{t-1} + z_t \odot \tilde{h}\_t.
\]

**3.3.3 Incremental Learning with Memory‑Aware Synapses (MAS)**
To adapt to evolving patterns without forgetting, we adopt MAS[reference:9]. For each parameter \(\theta*i\), an importance weight \(\Omega_i\) is estimated based on the sensitivity of the loss to changes in that parameter. During incremental updates, a regularization term is added:
\[
\mathcal{L}*{\text{total}} = \mathcal{L}\_{\text{pred}} + \lambda \sum*i \Omega_i (\theta_i - \theta_i^*)^2,
\]
where \(\theta*i^*\) is the parameter value before the update, and \(\lambda\) controls the trade‑off between plasticity and stability.

**3.4 LLM Integration for Natural‑Language Interface**
We fine‑tune a pre‑tained LLM (e.g., GPT‑2) on a custom dataset of (query, answer) pairs derived from historical predictions and operational reports. The model takes as input a user question (e.g., “What will be the passenger flow at Central Station tomorrow at 8 AM?”) and the latest ISTGNN forecast, and generates a fluent, informative response.

**3.5 Training & Implementation Details**

- **Training schedule:** The ISTGNN is first trained on historical data, then switched to incremental mode for online updates.
- **Hyperparameters:** Adam optimizer, initial learning rate 0.001, batch size 32, sequence length 12 time steps (1 h).
- **Infrastructure:** The pipeline is deployed on a Kubernetes cluster, with edge devices running TensorFlow Lite for local inference.

---

### **4. Evaluation Plan & Expected Results**

**4.1 Datasets**

- **Delhi Metro** (Oct 2012 – May 2017)[reference:10].
- **Nanjing Metro** (2020‑2025)[reference:11].
- A synthetic country‑wide dataset generated by scaling the above networks to 500+ stations.

**4.2 Baselines**

- **Statistical:** ARIMA, Prophet.
- **Deep‑learning:** LSTM, GCN, GCMN[reference:12].
- **Incremental:** IMGSN[reference:13].
- **LLM‑based:** GPT‑2 fine‑tuned on flow data[reference:14].

**4.3 Metrics**

- **Accuracy:** RMSE, MAE, MAPE, R².
- **Adaptability:** Performance drift over time (measured as increase in RMSE after 6 months without retraining).
- **Scalability:** Inference latency vs. number of stations, memory footprint.
- **User satisfaction:** BLEU score and human rating of LLM responses.

**4.4 Expected Outcomes**

- The ISTGNN should outperform GCMN and IMGSN by at least 15 % in RMSE on the Delhi dataset.
- The incremental learning mechanism should reduce performance drift by >50 % compared to a static hybrid model.
- The distributed pipeline should sustain sub‑second latency for up to 1,000 stations.
- The LLM interface should achieve a BLEU score >0.8 on a held‑out set of user queries.

**4.5 Ablation Studies**
We will ablate (i) the incremental component, (ii) the graph‑convolutional layer, and (iii) the LLM module to quantify their individual contributions.

---

### **5. Discussion**

**5.1 Interpretation of Expected Results**
The anticipated improvements stem from the synergy between spatial‑temporal modeling and continuous adaptation. The ISTGNN’s graph convolutions capture network‑wide dependencies, while the incremental mechanism allows it to track gradual pattern shifts (e.g., new residential areas, changed work schedules). The LLM interface democratizes access to predictions, making the system usable by non‑technical staff.

**5.2 Comparison with Prior Work**
Our framework extends the hybrid LSTM‑GCN concept[reference:15] by adding incremental learning, a feature shown critical for long‑term deployment[reference:16]. It also goes beyond existing LLM‑based predictors[reference:17] by integrating the LLM as a natural‑language front‑end to a robust adaptive model, rather than using the LLM as the sole predictor.

**5.3 Practical Implications**
Operators can use the system for dynamic scheduling, crowd management, and emergency response. The scalable architecture makes it feasible to roll out the solution across an entire country, with centralized monitoring and local edge processing.

**5.4 Limitations**

- **Data dependence:** The model requires high‑quality, continuous sensor data; missing or noisy data can degrade performance.
- **Computational cost:** The incremental learning step introduces overhead; however, we expect it to be manageable with modern cloud resources.
- **LLM hallucination:** The fine‑tuned LLM may occasionally generate plausible but incorrect statements; human‑in‑the‑loop verification is recommended for critical decisions.

---

### **6. Ethics & Societal Impact**

**Privacy:** All personally identifiable information (PII) must be stripped from sensor data before ingestion. Aggregated flow counts should be the only data stored long‑term.

**Bias:** The model may reflect historical biases in travel patterns (e.g., under‑serving certain neighborhoods). Regular audits should be conducted to ensure equitable service.

**Safety:** Predictions used for crowd control must have fail‑safe mechanisms; human operators should always have the final say.

**Environmental impact:** By optimizing transport efficiency, the framework can reduce congestion and lower carbon emissions.

**Transparency:** The LLM’s responses should include confidence scores and references to the underlying data sources.

---

### **7. Conclusion**

This dissertation proposes a novel hybrid framework for real‑time adaptive passenger‑flow prediction. The framework combines an Incremental Spatiotemporal Graph Neural Network (ISTGNN) for accurate and adaptive forecasting, a distributed edge‑cloud pipeline for scalability, and a fine‑tuned LLM for natural‑language interaction. We provide a complete mathematical formulation and a detailed evaluation plan. If successfully implemented, the framework could set a new standard for intelligent transport‑management systems, enabling more efficient, responsive, and user‑friendly public transportation on a country‑wide scale.

**Future work** includes exploring more efficient incremental‑learning algorithms, integrating multimodal data (e.g., social media feeds), and conducting real‑world pilot deployments.

---

### **References**

1.  Begisbayev, D. (2024). _Investigation of Hybrid Models and Architectures for Real‑Time Adaptive Passenger Flow Prediction_. IEEE Conference. [reference:18]
2.  Fu, X. et al. (2023). _A Hybrid Deep Learning Approach for Real‑Time Estimation of Passenger Traffic Flow in Urban Railway Systems_. MDPI Buildings. [reference:19]
3.  Wu, F. et al. (2025). _Adaptive Passenger Flow Prediction in Metro Systems: An Incremental Learning Framework for Evolving Patterns_. Journal of Transportation Engineering. [reference:20]
4.  Ma, C. et al. (2025). _Urban rail transit passenger flow prediction using large language model under multi‑source spatiotemporal data fusion_. ScienceDirect. [reference:21]
5.  Salas, E. et al. (2025). _An end‑to‑end distributed deep learning system for real‑time passenger flow measurement in transport interchanges_. Applied Intelligence. [reference:22]

---

## **Actionable Next Improvements (for the MSc Candidate)**

1.  **Data acquisition:** Secure access to at least two large‑scale metro datasets (e.g., Delhi, Nanjing) and ensure they include both historical and real‑time streaming formats.
2.  **Baseline implementation:** Reproduce the key baselines (GCMN, IMGSN, GPT‑2‑based predictor) to establish a solid comparison floor.
3.  **Prototype development:** Build a minimal viable pipeline (data ingestion → ISTGNN training → incremental update) on a cloud platform (e.g., AWS/GCP).
4.  **LLM fine‑tuning:** Collect or synthesize a QA dataset specific to passenger‑flow queries and fine‑tune an open‑source LLM (e.g., GPT‑2, LLaMA‑2).
5.  **Evaluation:** Run the full ablation study and scalability tests, documenting all metrics and latency figures.

By following these steps, you will transform the theoretical framework presented here into a concrete, evaluable MSc dissertation.
