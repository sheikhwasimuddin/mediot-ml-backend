import requests

# Your FastAPI URL
url = "http://localhost:8000/predict"

# Sample patient input (match model input format)
sample_data = {
    "data": [
        {
            "Patient Number": 1,
            "Heart Rate (bpm)": 85,
            "SpO2 Level (%)": 95,
            "Systolic Blood Pressure (mmHg)": 130,
            "Diastolic Blood Pressure (mmHg)": 85,
            "Body Temperature (°C)": 36.9,
            "Fall Detection": "No",
            "Heart Rate Alert": "Normal",
            "SpO2 Level Alert": "Normal",
            "Blood Pressure Alert": "Normal",
            "Temperature Alert": "Normal",
            "Data Accuracy (%)": 95
        }
    ]
}

# Send POST request
response = requests.post(url, json=sample_data)

# Output result
if response.status_code == 200:
    print("✅ Prediction:", response.json())
else:
    print("❌ Error:", response.status_code, response.text)
