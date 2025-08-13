from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from predict_disease import predict_disease

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://medicaliot.netlify.app"],  # Adjust if frontend port differs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define expected request structure
class PatientData(BaseModel):
    data: list

@app.post("/predict")
def predict(patient: PatientData):
    try:
        print("Received data:", patient.data)  # Debug input
        result = predict_disease(patient.data)
        print("✅ Prediction result:", result)
        return result
    except Exception as e:
        print("🔥 Backend error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict_get():
    raise HTTPException(status_code=405, detail="Method Not Allowed: Use POST for predictions")

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Optional root endpoint to avoid 404 on GET /
@app.get("/")
async def root():
    return {"message": "Welcome to MedIoT API. Use POST /predict for predictions."}
