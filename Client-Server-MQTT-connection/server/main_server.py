# server/main_server.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from server import db, clustering
import server.db as db_mod

app = FastAPI(title="Hazard Server")

# initialize DB
db.init_db()

class HazardIn(BaseModel):
    latitude: float
    longitude: float
    confidence: float
    hazard_type: str

@app.post("/upload_hazard")
async def upload_hazard(h: HazardIn):
    # Load existing hazards
    hazards = db.list_hazards()
    # find near duplicate within threshold (50m)
    nearby_id = clustering.find_nearby(hazards, h.latitude, h.longitude, threshold_meters=50)
    if nearby_id:
        db.update_hazard(nearby_id, h.latitude, h.longitude, h.confidence)
        return {"status": "merged", "hazard_id": nearby_id}
    else:
        new_id = db.insert_hazard(h.latitude, h.longitude, h.hazard_type, h.confidence)
        return {"status": "created", "hazard_id": new_id}

@app.get("/get_hazards")
async def get_hazards():
    return {"hazards": db.list_hazards()}

# Placeholder for federated learning endpoints
@app.post("/upload_model/{client_id}")
async def upload_model(client_id: str):
    # receive and store model weights (omitted)
    return {"status": "model_received", "client_id": client_id}

@app.get("/get_global_model")
async def get_global_model():
    # return latest global model weights or metadata (omitted)
    return {"status": "ok", "model_version": "v1-placeholder"}

if __name__ == "__main__":
    uvicorn.run("server.main_server:app", host="0.0.0.0", port=8000, reload=True)
