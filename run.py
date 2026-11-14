import sys
import uvicorn
from pathlib import Path
from geosituation.config import SERVER_HOST, SERVER_PORT

def main():
    print("================================================================================")
    print("   GEOSITUATION-NLP: AI-POWERED GEOSPATIAL SITUATIONAL AWARENESS SYSTEM         ")
    print("   Theoretical Foundation: Endsley's Three-Level Model (1995)                   ")
    print("   Testbed Scope: Noida (Gautam Buddha Nagar, Sectors 1-168)                    ")
    print("================================================================================\n")
    print(f"Starting FastAPI ASGI Server at http://{SERVER_HOST}:{SERVER_PORT} ...")
    print(f"Access Dashboard: http://{SERVER_HOST}:{SERVER_PORT}\n")
    
    uvicorn.run("geosituation.api.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)

if __name__ == "__main__":
    main()
