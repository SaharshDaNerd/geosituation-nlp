# GeoSituation-NLP: User Manual & Viva Presentation Guide

> **Project:** AI-Powered Geospatial Situational Awareness System Using Endsley's Model  
> **Target Scope:** Noida (Gautam Buddha Nagar, Sectors 1–168), Uttar Pradesh, India

---

## 1. Quick Start Guide

### Step 1: Prerequisites
Ensure you have Python $\ge 3.14$ and `uv` installed. If you do not have `uv`, install it via PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Sync Dependencies
Clone or open the repository in terminal and sync the locked environment:
```powershell
uv sync
```

### Step 3: Seed the Incident Database
Populate the database with historical baseline reports and multi-source articles:
```powershell
uv run python scripts/seed_data.py
```

### Step 4: Launch the Dashboard
```powershell
uv run python run.py
```
Open your browser and navigate to:
**`http://127.0.0.1:8000`**

---

## 2. Interactive Dashboard Tour

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  [Top Bar]  GeoSituation-NLP  •  Endsley's Model  •  [+ Ingest News Article] [⟳] │
├─────────────────────────┬───────────────────────────────┬────────────────────────┤
│ LEFT PANE               │ CENTER PANE (GIS Map)         │ RIGHT PANE             │
│ Level 1: Perception     │ Interactive Leaflet Dark Map  │ Level 2: Comprehension │
│                         │ • Sector Nodes (SAI color)    │ • Top Sectors by SAI   │
│ • Category Filter Pills │ • Incident Markers            │ • Poisson Anomalies    │
│   (All, Violent, etc.)  │ • ST-KDE Heatmap Surface      │                        │
│ • Real-Time Article     │ • ST-DBSCAN Cluster Drift     │ Level 3: Projection    │
│   Perception Feed with  │                               │ • Hotspot Trajectories │
│   Multi-Source Badges   │ [Top Right: Layer Controls &  │ • 7-Day Forecast       │
│                         │  Basemap Theme Switcher]      │   Momentum Cards       │
└─────────────────────────┴───────────────────────────────┴────────────────────────┘
```

### A. Left Pane: Level 1 Perception Feed
* **Category Filters:** Filter incidents by category (`All`, `Violent`, `Property`, `Traffic`, `Hazard`).
* **Incident Cards:** Shows the headline, extracted sector, summary, model confidence, and **Multi-Source Badges** (e.g., `✓ 3 Sources` for verified incidents).
* **Click to Inspect:** Clicking any incident card automatically centers and zooms the map onto the event coordinates.

### B. Center Pane: Interactive Map & Layer Controls
* **Basemap Switcher (Top Right):**
  * `🌙 Dark Canvas (Esri)`: Clean, high-resolution dark GIS basemap (no key required, default).
  * `🛰️ Carto Dark Matter`: Modern dark theme with your optional Carto API key.
  * `🗺️ OpenStreetMap`: Standard light street map.
* **GIS Layer Toggles:**
  * **Incident Markers (L1):** Toggle individual incident pins colored by category.
  * **Sector Activity Choropleth (L2):** Toggle circular sector coverage circles color-coded by Situational Activity Index:
    * 🟢 Green: Normal Baseline ($\text{SAI} < 3.0$)
    * 🟡 Yellow: Moderate Activity ($3.0 \le \text{SAI} < 8.0$)
    * 🟠 Orange: Elevated Attention ($8.0 \le \text{SAI} < 15.0$)
    * 🔴 Red: Critical Surge ($\text{SAI} \ge 15.0$)
  * **KDE Continuous Risk Surface (L3):** Renders a smooth 2D Gaussian density gradient across Noida.
  * **ST-DBSCAN Cluster Drift (L3):** Renders dashed spatial bounding circles for active incident clusters, showing drift direction and velocity.

### C. Right Pane: Level 2 Comprehension & Level 3 Projection
* **Top Sectors by SAI Leaderboard:** Ranks sectors by their composite activity score incorporating temporal decay and severity.
* **Poisson Baseline Anomalies:** Displays statistical surge alerts (e.g., *"Sector 142 exhibits an anomaly Z-score of 2.24σ (2 reports in 7d vs 0.5 expected)"*).
* **Active Hotspot Drift Trajectories:** Explains directional drift (e.g., *"Emerging PROPERTY_CRIME cluster across Sector 137, Sector 142 with 3 incidents. Drift: South-East at 0.53 km/day"*).
* **7-Day Risk Momentum Forecast:** Projects expected incident volume over the next week (`ACCELERATING`, `STABLE`, or `DE-ESCALATING`).

---

## 3. How to Demonstrate in a College Viva / Presentation

Follow this exact 4-step script to showcase your engineering contributions:

### Demonstration Step 1: Explain the Theoretical Foundation
> *"Most existing crime maps are passive record-keepers. Our system is grounded in Mica Endsley's 1995 Model of Situational Awareness, dividing urban intelligence into Perception (What and Where), Comprehension (Context and Anomalies), and Projection (Trajectory and Momentum)."*

### Demonstration Step 2: Show Level 2 Deduplication (Event Coreference)
1. Point to **Sector 18** on the map.
2. Click on the red marker in Sector 18:
   * Show that it displays **"Verified across 3 sources"** (Times of India, Hindustan Times, Indian Express).
3. Explain the mathematics:
   > *"If three separate news outlets report the same robbery over 48 hours, a naive system places 3 pins on the map, distorting the statistics. Our Level 2 Coreference engine evaluates TF-IDF cosine distance, Haversine distance, and temporal intervals to fuse them into one canonical event, boosting verification confidence asymptotically to 99.9%."*

### Demonstration Step 3: Show Level 3 Spatial Projection & Cluster Drift
1. Turn on the **ST-DBSCAN Cluster Drift** layer.
2. Point out the cluster along the **Noida-Greater Noida Expressway** (Sectors 137 & 142).
3. Explain the trajectory summary:
   > *"Level 3 does not claim to predict individual human crimes. Instead, it tracks the spatial velocity and drift of active reporting clusters. Here, it projects an emerging property crime cluster drifting South-East along the Expressway at 0.53 km/day."*

### Demonstration Step 4: Live Article Ingestion Demo
1. Click **"Ingest News Article"** in the top navigation bar.
2. In the modal, paste a new headline and body:
   * **Headline:** `Delivery van overturned near Sector 62 Fortis hospital crossing`
   * **Body:** `Noida: An electrical delivery van collided with a road divider near Sector 62 Fortis crossing early morning. Traffic police cleared the road.`
   * **Source:** `Live Demonstration`
3. Click **"Ingest & Run Pipeline"**.
4. Show the evaluators:
   * The NLP classifier instantly perceives it as `TRAFFIC_INCIDENT`.
   * The gazetteer resolves it to **Sector 62** without hallucinating coordinates.
   * The map centers, the incident feed updates, and the Sector 62 SAI score recalibrates in real time.

---

## 4. Troubleshooting & FAQ

### Q1: The server won't start or reports a port conflict.
By default, the server binds to port 8000. If port 8000 is occupied by another service:
1. Open [`.env`](file:///c:/Users/saxen/OneDrive/Desktop/College/Minor/.env).
2. Change `PORT=8000` to `PORT=8080`.
3. Re-run `uv run python run.py`.

### Q2: How do I test the system offline without an internet connection?
The entire ML pipeline, gazetteer, PostGIS-style spatial math, and seed database run completely offline on local Python. For the map tiles, if offline, Leaflet will render all sector circles, incident markers, and cluster hulls on a dark canvas.

### Q3: How do I retrain or evaluate the NLP classifier?
Run the empirical 4-fold cross-validation script:
```powershell
uv run python scripts/evaluate_models.py
```
This generates an evaluation table showing Precision, Recall, Macro F1, and Accuracy across all classes for your project thesis.
