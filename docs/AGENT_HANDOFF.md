# GeoSituation-NLP: Agent Handoff & Technical Specification

> **Target Audience:** Autonomous AI coding agents, NLP researchers, and backend engineers continuing or extending this project.
> **Last Verified State:** Python 3.14 (AMD64 Windows), `uv` package manager, 24/24 unit tests passing, full end-to-end pipeline operational.

---

## 1. System Mission & Theoretical Grounding

This project implements an AI-driven urban intelligence system modeled after **Mica Endsley’s Three-Level Model of Situational Awareness (1995)**. It transforms noisy, unstructured news headlines and journalistic articles into structured, geo-referenced, and temporally forecasted situational states across **Noida (Gautam Buddha Nagar, Sectors 1–168), Uttar Pradesh, India**.

### The Three Invariant Levels
1. **Level 1: Perception ("What is happening, and where?")**
   - Ingests raw news text via public RSS / manual ingestion.
   - Categorizes text into 5 operational incident classes (`VIOLENT_CRIME`, `PROPERTY_CRIME`, `TRAFFIC_INCIDENT`, `CIVIC_HAZARD`, `PUBLIC_DISRUPTION`) plus a `NON_INCIDENT` control class.
   - Grounds mentions to a canonical Noida sector or landmark POI using a two-tier gazetteer.
   - Extracts temporal datelines, casualties, weapons, vehicles, and organizations.
2. **Level 2: Comprehension ("What does it mean in context right now?")**
   - Solves multi-source report proliferation using **Cross-Document Spatio-Temporal Coreference** ($\Delta t \le 60\text{h}$, $d \le 1.8\text{km}$, text similarity $\ge 0.15$ or shared entities).
   - Computes an asymptotic **Verification Confidence Score** $V(E) = 1 - \prod (1 - p_i)$.
   - Applies an **exponential half-life decay function** ($t_{1/2} = 7\text{ days}$) to incident relevance.
   - Computes the **Sector Situational Activity Index ($\text{SAI}$)**.
   - Computes **Poisson Standardized Anomaly Z-scores ($Z_S$)** against 30-day rolling baselines to detect surges without biasing against inherently busy commercial sectors.
3. **Level 3: Projection ("Where is the situation heading in the near future?")**
   - Computes a continuous 2D **Gaussian Kernel Density Estimation (KDE)** surface ($60 \times 60$ grid) over Noida.
   - Runs **ST-DBSCAN** to detect emerging clusters and computes their **drift velocity ($km/\text{day}$)** and **directional heading**.
   - Projects 7-day sector incident momentum (`ACCELERATING_ACTIVITY`, `STABLE_MODERATE`, `LOW_STABLE`).

---

## 2. Directory Layout & Module Responsibilities

```
.
├── .env                        # Local environment secrets (CARTO_API_KEY, HOST, PORT)
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules (ignores .env, .venv, etc.)
├── pyproject.toml              # UV / PEP 621 project configuration & dependencies
├── uv.lock                     # Locked dependency tree
├── run.py                      # Application launcher (ASGI Uvicorn runner)
├── data/
│   ├── gazetteers/
│   │   └── noida_sectors.json  # 34+ canonical sectors, centroids, zones, aliases
│   └── processed/
│       ├── level1_reports.json    # Level 1 perceived raw incident records
│       ├── canonical_events.json  # Level 2 deduplicated, multi-verified events
│       └── situational_state.json # Full cached pipeline state (summary, KDE, hotspots)
├── models/
│   └── incident_classifier.joblib # Serialized FeatureUnion + SGDClassifier
├── src/
│   └── geosituation/
│       ├── config.py           # Bounds, categories, weights, decay half-life, thresholds
│       ├── pipeline.py         # End-to-end Three-Level orchestrator
│       ├── perception/         # LEVEL 1
│       │   ├── preprocessor.py # Text cleaner, HTML stripper, dateline temporal extractor
│       │   ├── gazetteer.py    # Two-tier deterministic regex + POI alias resolver
│       │   ├── classifier.py   # Word (1-2) + Char (3-5) TF-IDF FeatureUnion SGD
│       │   ├── ner_extractor.py# Extracts roles, vehicles, weapons, organizations
│       │   └── collector.py    # RSS news collector & Level 1 article coordinator
│       ├── comprehension/      # LEVEL 2
│       │   ├── coreference.py  # Cross-document spatio-temporal deduplication
│       │   ├── situational_index.py # Temporal decay & SAI score computation
│       │   └── anomaly_detector.py  # Poisson anomaly Z-score velocity analysis
│       ├── projection/         # LEVEL 3
│       │   ├── spatial_kde.py  # Continuous 2D Gaussian Kernel Density Estimation
│       │   ├── hotspot_dbscan.py # ST-DBSCAN cluster discovery & drift velocity
│       │   └── momentum.py     # 7-day rate trajectory & executive briefing
│       └── api/
│           ├── main.py         # FastAPI async endpoints (REST + Lifespan)
│           └── templates/
│               └── dashboard.html # Leaflet.js interactive dark GIS dashboard
├── scripts/
│   ├── generate_gazetteer.py   # Regenerates canonical Noida sector database
│   ├── seed_data.py            # Generates baseline seed corpus & runs pipeline
│   └── evaluate_models.py      # Stratified 4-fold CV evaluation benchmark
├── docs/
│   ├── AGENT_HANDOFF.md        # This file
│   └── USER_GUIDE.md           # End-user & viva presentation manual
└── tests/
    ├── test_perception.py      # L1 unit tests
    ├── test_comprehension.py   # L2 unit tests
    ├── test_projection.py      # L3 unit tests
    └── test_api_client.py      # API & dashboard endpoint tests
```

---

## 3. Toolchain & Runtime Rules

* **Python Version:** `>=3.14`.
* **Package Manager:** `uv` exclusively. Do not run raw `pip install`.
  - Add dependency: `uv add <package>`
  - Run scripts: `uv run python <script.py>`
  - Run test suite: `uv run python -m unittest discover tests`
  - Re-sync environment: `uv sync`
* **Virtual Environment Path:** Located at `.venv/`. Configured in `.vscode/settings.json` via `python.defaultInterpreterPath`.
* **Zero-Hallucination Invariant:**
  Under no circumstances should the system guess or geocode arbitrary locations outside Noida. If an article cannot be resolved to a known sector or gazetteer POI, `gazetteer.resolve_location()` returns `None`, and `collector.process_raw_article()` discards the record from the spatial database.
* **Ethics Invariant:**
  Level 3 Projection must strictly model **aggregate spatio-temporal reporting density and cluster drift**. Do not implement individual suspect prediction, facial recognition, or person-level predictive policing.

---

## 4. Key Mathematical Formulations & Implementations

### A. Deduplication & Verification Confidence (`src/geosituation/comprehension/coreference.py`)
Two reports $R_i$ and $R_j$ belong to the same physical event if:
$$\text{Category}(R_i) = \text{Category}(R_j) \quad \wedge \quad |t_i - t_j| \le 60\text{h} \quad \wedge \quad d_{\text{haversine}} \le 1.8\text{km}$$
AND one of:
1. $\text{CosSim}(\text{TFIDF}_i, \text{TFIDF}_j) \ge 0.72$
2. $\text{Sector}(R_i) = \text{Sector}(R_j) \;\wedge\; (\text{CosSim} \ge 0.15 \;\vee\; |\text{Entities}_i \cap \text{Entities}_j| \ge 1)$
3. $d_{\text{haversine}} \le 0.8\text{km} \;\wedge\; \text{CosSim} \ge 0.20$

Multi-source verification confidence:
$$V(E) = 1 - \prod_{k=1}^N (1 - p_k)$$

### B. Sector Situational Activity Index (`src/geosituation/comprehension/situational_index.py`)
$$\text{SAI}(S, T) = \sum_{E_k \in S} w(c_k) \cdot V(E_k) \cdot \exp\left( -\frac{\ln(2)(T - t_k)}{7.0} \right)$$
* $w(c_k) \in [1.0, 5.0]$: Severity weight (`VIOLENT_CRIME` = 5.0, `CIVIC_HAZARD` = 3.5, `PROPERTY_CRIME` = 2.5, `TRAFFIC` = 2.0, `DISRUPTION` = 2.0).

### C. Poisson Anomaly Z-Score (`src/geosituation/comprehension/anomaly_detector.py`)
$$Z_S = \frac{N_{\text{recent}} - \mu_{\text{expected}}}{\sqrt{\mu_{\text{expected}} + \epsilon}}$$
Where $\mu_{\text{expected}} = N_{\text{baseline}} \times \frac{7}{30}$. A sector is flagged as anomalous if $Z_S \ge 1.5\sigma$.

### D. ST-DBSCAN Cluster Drift Velocity (`src/geosituation/projection/hotspot_dbscan.py`)
For cluster members sorted chronologically, centroid drift velocity:
$$v_{\text{drift}} = \frac{d_{\text{haversine}}(\bar{x}_{\text{early}}, \bar{x}_{\text{late}})}{\Delta t_{\text{days}}} \quad (\text{km/day})$$
Bearing angle:
$$\theta = \text{atan2}(\Delta \text{lon}, \Delta \text{lat})$$

---

## 5. How to Continue Development (Roadmap for Next Agent)

1. **Option A: Add Deep Transformer / DistilBERT Fine-Tuning**
   - The current classifier uses a high-speed Word + Subword Char `FeatureUnion` with `SGDClassifier(loss='modified_huber')`.
   - To fine-tune a full Hugging Face transformer (`distilbert-base-uncased`), create `scripts/train_transformer.py` using `transformers` + `torch` or `onnxruntime`. The classifier interface in `src/geosituation/perception/classifier.py` already exposes `predict(text: str) -> Dict` which can wrap a PyTorch or ONNX model seamlessly.
2. **Option B: Connect Live Scraping Daemon**
   - Hook `NewsIncidentCollector.fetch_google_news_rss()` into an async background task or recurring cron script (`scripts/daemon_ingest.py`) that polls every hour and calls `pipeline.ingest_new_article()`.
3. **Option C: Add GeoJSON Sector Polygons**
   - Currently, sector nodes render as dynamic interactive circles scaled by SAI score. To render exact municipal boundary polygons, place a GeoJSON file into `data/geojson/noida_sectors.geojson` and bind it via `L.geoJSON()` in `dashboard.html`.

---

## 6. Verification Checklist Before Any Commit
```powershell
# 1. Run all unit tests
uv run python -m unittest discover tests

# 2. Check model benchmark
uv run python scripts/evaluate_models.py

# 3. Test ASGI server startup
uv run python run.py
```
Ensure all 24 tests pass before modifying core modules.
