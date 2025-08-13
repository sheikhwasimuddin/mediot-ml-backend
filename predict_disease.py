def predict_disease(input_data: list):
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import load_model
    import joblib

    model = load_model('model/disease_prediction_model.keras')
    scaler = joblib.load('model/scaler.pkl')
    le = joblib.load('model/label_encoder.pkl')
    expected_features = joblib.load('model/feature_names.pkl')

    df = pd.DataFrame(input_data)
    df['BP_Difference'] = df['Systolic Blood Pressure (mmHg)'] - df['Diastolic Blood Pressure (mmHg)']
    df['Heart_Rate_SpO2_Ratio'] = df['Heart Rate (bpm)'] / df['SpO2 Level (%)']
    df = df.drop(['Patient Number', 'Data Accuracy (%)'], axis=1, errors='ignore')

    categorical_cols = ['Fall Detection', 'Heart Rate Alert', 'SpO2 Level Alert',
                        'Blood Pressure Alert', 'Temperature Alert']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    numerical_cols = ['Heart Rate (bpm)', 'SpO2 Level (%)', 'Systolic Blood Pressure (mmHg)',
                      'Diastolic Blood Pressure (mmHg)', 'Body Temperature (°C)',
                      'BP_Difference', 'Heart_Rate_SpO2_Ratio']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]

    preds = model.predict(df)
    confidences = np.max(preds, axis=1)
    pred_classes = np.argmax(preds, axis=1)
    pred_labels = le.inverse_transform(pred_classes)

    risk_score = float((confidences[0] * 100))
    risk_level = (
        "Critical" if risk_score < 50 else
        "High" if risk_score < 65 else
        "Medium" if risk_score < 80 else
        "Low"
    )

    alerts = []
    if risk_level == "Critical":
        alerts.append("Critical condition detected")
    if input_data[0]["Fall Detection"] == "Yes":
        alerts.append("Fall detected - high emergency")

    recommendations = [
        "Follow a healthy lifestyle",
        "Regular monitoring recommended",
        "Consult a doctor if symptoms persist"
    ]

    return {
        "disease": pred_labels[0],
        "confidence": float(confidences[0]),
        "riskScore": int(risk_score),
        "riskLevel": risk_level,
        "alerts": alerts,
        "recommendations": recommendations
    }