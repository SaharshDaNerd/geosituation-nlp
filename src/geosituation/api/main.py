from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
from pathlib import Path

from geosituation.pipeline import SituationalAwarenessPipeline
from geosituation.config import INCIDENT_CATEGORIES, NOIDA_BOUNDS, STATIC_DIR, CARTO_API_KEY

# Template and static assets directory
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Pipeline Instance
pipeline = SituationalAwarenessPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure pipeline state exists on startup
    try:
        pipeline.run_pipeline()
    except Exception as e:
        print(f"Warning during startup pipeline run: {e}")
    yield

app = FastAPI(
    title="GeoSituation-NLP",
    description="AI-Powered Geospatial Situational Awareness System Using Endsley's Cognitive Model",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

class ArticleIngestRequest(BaseModel):
    headline: str
    body: str
    source_url: Optional[str] = "Manual Ingestion / Web Portal"

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the interactive GIS situational awareness dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "noida_bounds": NOIDA_BOUNDS,
            "categories": INCIDENT_CATEGORIES,
            "carto_api_key": CARTO_API_KEY
        }
    )

@app.get("/api/status")
async def get_system_status():
    state = pipeline.run_pipeline()
    return {
        "status": "OPERATIONAL",
        "framework": "Endsley's Three-Level Model of Situational Awareness",
        "last_updated": state["last_updated"],
        "total_canonical_events": len(state["canonical_events"]),
        "total_sectors_monitored": len(state["sectors"]),
        "active_hotspots": len(state["hotspots"])
    }

@app.get("/api/summary")
async def get_executive_summary():
    state = pipeline.run_pipeline()
    return state["executive_summary"]

@app.get("/api/incidents")
async def get_incidents(category: Optional[str] = None, sector: Optional[str] = None):
    state = pipeline.run_pipeline()
    events = state["canonical_events"]
    if category and category != "ALL":
        events = [e for e in events if e.get("category") == category]
    if sector and sector != "ALL":
        events = [e for e in events if e.get("sector_id") == sector]
    return events

@app.get("/api/sectors")
async def get_sectors():
    state = pipeline.run_pipeline()
    return state["sectors"]

@app.get("/api/hotspots")
async def get_hotspots():
    state = pipeline.run_pipeline()
    return state["hotspots"]

@app.get("/api/heatmap")
async def get_heatmap():
    state = pipeline.run_pipeline()
    return state["kde_surface"]

@app.get("/api/trajectories")
async def get_trajectories():
    state = pipeline.run_pipeline()
    return state["trajectories"]

@app.post("/api/ingest")
async def ingest_article(req: ArticleIngestRequest):
    """
    Ingests and runs live perception, coreference deduplication, and projection
    on a user-submitted news article.
    """
    if not req.headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
        
    result_state = pipeline.ingest_new_article(
        headline=req.headline,
        body=req.body,
        source_url=req.source_url or "User Ingested"
    )
    if result_state is None:
        return JSONResponse(status_code=200, content={
            "success": False,
            "message": "Article parsed as NON_INCIDENT or could not be geocoded to a Noida sector."
        })

    return {
        "success": True,
        "message": "Article ingested, perceived, deduplicated, and mapped successfully!",
        "new_event_count": len(result_state["canonical_events"]),
        "executive_summary": result_state["executive_summary"]
    }
