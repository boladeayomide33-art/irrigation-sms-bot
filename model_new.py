import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

os.makedirs('models', exist_ok=True)
np.random.seed(42)

class IrrigationModel:
    def __init__(self):
        self.classifier = None
        self.scaler = None
        self.encoders = {}
        self.load_or_train()
    
    def load_or_train(self):
        model_path = 'models/irrigation_classifier.pkl'
        
        if os.path.exists(model_path):
            try:
                self.classifier = joblib.load(model_path)
                self.scaler = joblib.load('models/scaler.pkl')
                self.encoders = joblib.load('models/label_encoders.pkl')
                print("✅ Model loaded successfully!")
                return
            except:
                print("⚠️ Corrupted model, retraining...")
        
        print("🛠 Training new model...")
        
        # Better training data
        data = {
            'Soil_Type': ['Clay', 'Silt', 'Sandy', 'Clay', 'Loamy', 'Silt', 'Sandy', 'Clay'],
            'Soil_Moisture': [36.48, 50.56, 40.07, 12.75, 25.0, 60.0, 15.0, 30.0],
            'Temperature_C': [21.9, 36.5, 41.83, 37.22, 28.0, 22.0, 35.0, 24.0],
            'Humidity': [31.19, 26.01, 76.41, 43.32, 65.0, 80.0, 45.0, 55.0],
            'Rainfall_mm': [1167.7, 831.28, 1844.45, 306.26, 500.0, 100.0, 1200.0, 200.0],
            'Wind_Speed_kmh': [1.97, 16.82, 19.03, 11.44, 5.0, 12.0, 8.0, 3.0],
            'Crop_Type': ['Wheat', 'Maize', 'Cotton', 'Wheat', 'Rice', 'Tomato', 'Cotton', 'Maize'],
            'Crop_Growth_Stage': ['Vegetative', 'Flowering', 'Harvest', 'Sowing', 'Flowering', 'Fruiting', 'Harvest', 'Sowing'],
            'Irrigation_Need': [0, 1, 0, 1, 1, 1, 0, 1]
        }
        
        df = pd.DataFrame(data)
        y = df['Irrigation_Need']
        
        cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        
        features = ['Soil_Moisture', 'Temperature_C', 'Humidity', 
                   'Rainfall_mm', 'Wind_Speed_kmh', 'Soil_Type', 
                   'Crop_Type', 'Crop_Growth_Stage']
        
        X = df[features]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.classifier = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        self.classifier.fit(X_scaled, y)
        
        joblib.dump(self.classifier, 'models/irrigation_classifier.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.encoders, 'models/label_encoders.pkl')
        print("✅ New model trained and saved!")
    
    # (predict method remains the same as before - use the fixed one I gave earlier)
    
    def predict(self, data: dict):
        try:
            input_data = {
                'Soil_Moisture': data.get('Soil_Moisture'),
                'Temperature_C': data.get('Temperature_C'),
                'Humidity': data.get('Humidity'),
                'Rainfall_mm': data.get('Rainfall_mm'),
                'Wind_Speed_kmh': data.get('Wind_Speed_kmh'),
                'Soil_Type': data.get('Soil_Type'),
                'Crop_Type': data.get('Crop_Type') or data.get('Crop Type'),
                'Crop_Growth_Stage': data.get('Crop_Growth_Stage')
            }
            
            input_df = pd.DataFrame([input_data])
            
            for col, le in self.encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
            
            features = ['Soil_Moisture', 'Temperature_C', 'Humidity', 
                       'Rainfall_mm', 'Wind_Speed_kmh', 'Soil_Type', 
                       'Crop_Type', 'Crop_Growth_Stage']
            
            X_input = input_df[features]
            X_scaled = self.scaler.transform(X_input)
            
            pred = self.classifier.predict(X_scaled)[0]
            prob = self.classifier.predict_proba(X_scaled)[0].max()
            
            moisture = data.get('Soil_Moisture', 0)
            crop = data.get('Crop_Type') or data.get('Crop Type', 'Unknown')
            rainfall = data.get('Rainfall_mm', 0)
            temp = data.get('Temperature_C', 25)
            
            base = {'Maize': 30, 'Rice': 40, 'Tomato': 25, 'Wheat': 28, 'Cotton': 25}.get(crop, 28)
            deficit = max(0, 40 - moisture)
            rain_factor = max(0.3, 1 - rainfall / 30)
            temp_factor = 1 + (temp - 25) / 40
            water_amount = round(base * (deficit / 40) * rain_factor * temp_factor, 1)
            
            if moisture < 25 or rainfall < 10:
                recommendation = f"Irrigation needed. Recommended water: {water_amount} liters per m²"
            else:
                recommendation = f"No significant irrigation needed. (Water need: {water_amount} liters/m²)"
            
            return {"recommendation": recommendation, "confidence": float(prob)}
            
        except Exception as e:
            return {"recommendation": f"Error: {str(e)}", "confidence": 0.0}

model = IrrigationModel()