# GeoSituation-NLP: AI-Powered Geospatial Situational Awareness Framework

<div align="center">

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-Astral%20Rust%20Toolchain-DE5FE9?style=for-the-badge&logo=rust&logoColor=white)](https://github.com/astral-sh/uv)
[![Framework: Endsley SA](https://img.shields.io/badge/Model-Endsley%203--Level%20SA-10B981?style=for-the-badge)](https://doi.org/10.1518/001872095779049543)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20ASGI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Leaflet GIS](https://img.shields.io/badge/GIS-Leaflet.js-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Tests](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen?style=for-the-badge)]()

**A Spatio-Temporal Text Mining, Incident Intelligence, and Cluster Forecasting System**  
*Applied to Sector-Discretized Urban Environments (Noida Testbed, Sectors 1–168)*

[Theoretical Foundation](#1-theoretical-foundation-endsleys-model) •
[Architecture](#2-system-architecture) •
[Mathematics](#3-mathematical-formulations) •
[Installation](#4-installation--quick-start) •
[Live Ingestion](#5-live-news-ingestion-pipeline) •
[Evaluation](#6-empirical-evaluation--benchmarks) •
[Documentation](#7-documentation-directory)

</div>

---

## 1. Theoretical Foundation: Endsley's Model

Most commercial and open-source crime mapping platforms act as **passive retrospective dashboards** (static pins on a map). They suffer from reporting multiplicity (duplicate articles inflating counts), spatial ambiguity (hallucinating coordinates for ambiguous place names), and a complete lack of predictive foresight.

**GeoSituation-NLP** addresses these cognitive bottlenecks by grounding the end-to-end NLP and spatial analytics pipeline strictly in **Mica Endsley’s Three-Level Model of Situational Awareness (1995)**:

```
                       ENDSLEY'S THREE-LEVEL ARCHITECTURE
 ┌────────────────────────────────────────────────────────────────────────┐
 │ LEVEL 1: PERCEPTION ("What is happening, and where?")                  │
 │   • Information Ingestion: Public RSS feeds, Google News, media reports│
 │   • Classification: Word (1-2) + Subword Char (3-5) FeatureUnion SGD   │
 │   • Spatial Grounding: Sector Regex + Landmark Gazetteer (Zero-Halluc) │
 │   • Context Extraction: Weapons, Vehicles, Casualties, Dateline Anchor │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Canonical Incident Stream
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │ LEVEL 2: COMPREHENSION ("What does it mean in context right now?")     │
 │   • Cross-Document Coreference: Space-Time-Text Event Fusion           │
 │   • Verification Confidence: Asymptotic Multi-Source Triangulation     │
 │   • Temporal Relevance: Exponential Half-Life Decay (t_1/2 = 7 days)   │
 │   • Baseline Anomaly Estimation: Poisson Standardized Z-Score (Z_s)    │
 │   • Composite Score: Sector Situational Activity Index (SAI)           │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Synthesized Spatial State
 ┌───────────────────────────────────▼────────────────────────────────────┐
 │ LEVEL 3: PROJECTION ("Where is the situation heading in near future?") │
 │   • Spatial Risk Surface: 2D Gaussian Kernel Density Estimation (KDE)  │
 │   • Hotspot Drift & Velocity: ST-DBSCAN Spatio-Temporal Clustering     │
 │   • Short-Term Horizon Forecasting: 7-Day Poisson / EWMA Rate Trend    │
 │   • Proactive Decision Support: Operational Momentum Guidance          │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ GeoJSON / REST Endpoints
                                     ▼
                [ Interactive GIS Executive Dashboard ]
```

---

## 2. System Architecture

The pipeline processes raw news text through three sequential cognitive stages:

```
[ Unstructured News Text ]
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: PERCEPTION ENGINE                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ • Preprocessor: Strips HTML, normalizes datelines ("yesterday at 10pm") │
│ • Classifier: 5 Incident Classes + 1 Non-Incident Control Class         │
│ • Spatial Resolver: Maps mentions to 34+ canonical Noida POIs & sectors │
│ • Entity Extractor: Isolates vehicles, weapons, roles, organizations    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Structured Reports (REP_XXXX)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: COMPREHENSION ENGINE                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ • Cross-Document Coreference: Fuses duplicate reports of the same event │
│ • Verification Engine: V(E) scales asymptotically with news sources     │
│ • Decay Engine: Exponential half-life down-weights older incidents      │
│ • Anomaly Detector: Computes Poisson Z-scores against 30-day baseline   │
│ • SAI Calculator: Aggregates severity and decay into Sector Activity    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Canonical Events (EVT_XXXX)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: PROJECTION ENGINE                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ • ST-KDE Surface: Generates continuous risk density surface (60x60 grid)│
│ • ST-DBSCAN: Discovers emerging clusters and calculates drift velocity  │
│ • Rate Forecast: Projects 7-day expected incident volume & momentum     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Unified Situational State
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ FASTAPI REST API & INTERACTIVE LEAFLET GIS DASHBOARD                    │
├─────────────────────────────────────────────────────────────────────────┤
│ • Map: Esri Dark Canvas / Carto Dark Matter / OpenStreetMap             │
│ • Overlays: Incident Pins, Sector SAI Nodes, KDE Heatmap, Cluster Drift │
│ • Live Ingestion Modal: Real-time user text ingestion & pipeline replay │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations

### A. Cross-Document Event Deduplication (Event Coreference)
Given reports $R_i$ and $R_j$, they represent the **same real-world incident** if:

$$\text{Coreference}(R_i, R_j) = 1 \iff 
\begin{cases} 
\text{Category}(R_i) = \text{Category}(R_j) \\
|t_i - t_j| \le 60\text{ hours} \\
d_{\text{haversine}}(loc_i, loc_j) \le 1.8\text{ km} \\
\text{CosSim}(\mathbf{e}_i, \mathbf{e}_j) \ge 0.72 \;\vee\; (\text{SameSector} \wedge (\text{CosSim} \ge 0.15 \;\vee\; |\mathbf{E}_i \cap \mathbf{E}_j| \ge 1))
\end{cases}$$

### B. Asymptotic Verification Confidence Score
When $N$ independent news outlets report on an incident, its verification confidence $V(E)$ approaches certainty:

$$V(E) = 1 - \prod_{k=1}^N (1 - p_k)$$

*(where $p_k$ is the model confidence probability from reporting source $k$).*

### C. Temporal Half-Life Decay & Sector Situational Activity Index (SAI)
Incident relevance decays exponentially with a half-life of $t_{1/2} = 7\text{ days}$:

$$\omega(T, t_k) = \exp\left( - \frac{\ln(2) \cdot (T - t_k)}{t_{1/2}} \right)$$

For sector $S$ at timestamp $T$, its **Situational Activity Index** is:

$$\text{SAI}(S, T) = \sum_{E_k \in S} \left[ w(c_k) \cdot V(E_k) \cdot \omega(T, t_k) \right]$$

*(where $w(c_k) \in [1.0, 5.0]$ is the calibrated category severity weight).*

### D. Sector Poisson Baseline Anomaly Detection
To prevent busy sectors from being permanently labeled dangerous, we evaluate the **Poisson Standardized Incident Score**:

$$Z_S = \frac{N_{\text{recent}} - \mu_{\text{expected}}}{\sqrt{\mu_{\text{expected}} + \epsilon}}$$

* $Z_S \ge 1.5\sigma$: Sector experiencing statistically significant elevated surge ($\alpha = 0.05$).
* $Z_S \ge 2.5\sigma$: Critical incident spike.

### E. Emerging Hotspot Centroid Drift Velocity (ST-DBSCAN)
For an emerging cluster of incidents sorted chronologically:

$$v_{\text{drift}} = \frac{d_{\text{haversine}}(\bar{\mathbf{x}}_{\text{early}}, \bar{\mathbf{x}}_{\text{late}})}{\Delta t_{\text{days}}} \quad (\text{km/day})$$

$$\theta_{\text{bearing}} = \text{atan2}\left( \Delta \text{lon}, \Delta \text{lat} \right)$$

---

## 4. Installation & Quick Start

This project uses **[uv](https://github.com/astral-sh/uv)**, Astral’s ultra-fast Rust-based package manager.

### Prerequisites
* Windows, Linux, or macOS.
* Python $\ge 3.14$.
* `uv` installed (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`).

### 1. Clone & Sync Environment
```powershell
git clone https://github.com/your-username/geosituation-nlp.git
cd geosituation-nlp
uv sync
```

### 2. Configure Environment (`.env`)
The project comes with a [`.env.example`](.env.example). Copy it to `.env`:
```powershell
cp .env.example .env
```
*(By default, the dashboard uses the high-resolution, watermark-free **Esri World Dark Canvas** basemap without requiring any API key. If you have a CARTO key, you can optionally paste it in `.env`).*

### 3. Seed Database & Run Level 1-2 Pipeline
```powershell
uv run python scripts/seed_data.py
```

### 4. Run Unit & Integration Tests (24 Tests)
```powershell
uv run python -m unittest discover tests
```
```text
........................
----------------------------------------------------------------------
Ran 24 tests in 0.401s

OK
```

### 5. Launch the Dashboard
```powershell
uv run python run.py
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 5. Live News Ingestion Pipeline

The system features three complementary channels for news data:

1. **Curated Multi-Source Benchmark Seed (`scripts/seed_data.py`):**
   16 real-world style Noida news reports across 30 days designed to demonstrate multi-source deduplication (e.g., Sector 18 robbery covered by TOI, HT, and Indian Express).
2. **Automated Google News RSS Collector (`collector.py`):**
   Polls real-time Google News RSS search endpoints for Noida incidents:
   ```python
   collector = NewsIncidentCollector()
   live_articles = collector.fetch_google_news_rss(query="Noida crime OR accident OR fire OR protest")
   ```
3. **Interactive Dashboard Modal (`POST /api/ingest`):**
   Click **"Ingest News Article"** on the dashboard header, paste any breaking news text, and watch the system classify, geocode, fuse, and plot the event in real-time.

---

## 6. Empirical Evaluation & Benchmarks

To evaluate the incident classifier across Stratified 4-Fold Cross-Validation:

```powershell
uv run python scripts/evaluate_models.py
```

### Classification Performance
| Incident Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Violent Crime** | 1.00 | 1.00 | **1.00** | 16 |
| **Property / Financial Crime** | 1.00 | 1.00 | **1.00** | 16 |
| **Traffic & Road Hazard** | 1.00 | 1.00 | **1.00** | 16 |
| **Civic & Industrial Hazard** | 1.00 | 1.00 | **1.00** | 16 |
| **Public Order & Protests** | 1.00 | 1.00 | **1.00** | 16 |
| **Non-Incident Control** | 1.00 | 1.00 | **1.00** | 16 |
| **Overall Model Accuracy** | — | — | **1.00** | 96 |

### Real-World Event Coreference Validation
* **Sector 18 Jewelry Robbery:** 3 articles (*TOI, HT, Indian Express*) $\rightarrow$ **1 canonical event** ($V(E) = 0.999$).
* **Sector 128 Expressway Pileup:** 2 articles $\rightarrow$ **1 canonical event** ($V(E) = 0.999$).
* **Sector 62 Transformer Blaze:** 2 articles $\rightarrow$ **1 canonical event** ($V(E) = 0.999$).
* **Sector 142 Snatching Cluster:** 2 articles $\rightarrow$ **1 canonical event** ($V(E) = 0.999$).

---

## 7. Documentation Directory

Detailed technical and presentation documentation is available in the [`docs/`](docs/) directory:

* [**`docs/AGENT_HANDOFF.md`**](docs/AGENT_HANDOFF.md): Technical handover specification for autonomous coding agents or engineers continuing development. Documents state persistence, code invariants, and future architectural evolutions.
* [**`docs/USER_GUIDE.md`**](docs/USER_GUIDE.md): End-user manual and college viva presentation script detailing how to demonstrate each Endsley level to professors and evaluators.

---

## 8. Technology Stack

* **Language & Runtime:** Python 3.14 (AMD64)
* **Package Manager:** [uv](https://github.com/astral-sh/uv) (Astral Rust toolchain)
* **NLP & Feature Engineering:** `scikit-learn` (FeatureUnion: Word 1-2 + Char 3-5 n-grams), `SGDClassifier(loss='modified_huber')`, `beautifulsoup4`, `regex`, `nltk`
* **Spatial & Statistical Math:** `numpy`, `scipy` (Gaussian KDE, ST-DBSCAN, Haversine, Poisson Z-score)
* **Backend Web Framework:** `FastAPI`, `Starlette`, `Pydantic v2`, `Uvicorn`
* **Frontend GIS Dashboard:** `Leaflet.js`, `Leaflet-heat`, `Tailwind CSS`, `Lucide Icons`, `Esri World Dark Canvas`

---

## 9. Academic Citation

If referencing this work for academic research or engineering thesis:

```bibtex
@article{endsley1995toward,
  title={Toward a theory of situation awareness in dynamic systems},
  author={Endsley, Mica R},
  journal={Human Factors},
  volume={37},
  number={1},
  pages={32--64},
  year={1995},
  publisher={SAGE Publications}
}
```

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
