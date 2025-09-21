import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestRegressor
import random

class HandwritingAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = "stroke_model.h5"
        self.scaler_path = "scaler.pkl"
        
    def extract_features(self, strokes):
        """Extract comprehensive features from handwriting strokes"""
        if not strokes:
            return np.zeros(50)
        
        features = []
        
        # Basic stroke features
        total_points = sum(len(stroke) for stroke in strokes)
        total_strokes = len(strokes)
        features.extend([total_points, total_strokes])
        
        # Stroke length analysis
        stroke_lengths = []
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            length = sum(np.linalg.norm(np.array(stroke[i]) - np.array(stroke[i-1])) 
                        for i in range(1, len(stroke)))
            stroke_lengths.append(length)
        
        if stroke_lengths:
            features.extend([
                np.mean(stroke_lengths),
                np.std(stroke_lengths),
                np.min(stroke_lengths),
                np.max(stroke_lengths)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Smoothness analysis
        smoothness_scores = []
        for stroke in strokes:
            if len(stroke) < 3:
                continue
            
            stroke_points = np.array(stroke)
            angles = []
            
            for i in range(1, len(stroke_points) - 1):
                v1 = stroke_points[i] - stroke_points[i-1]
                v2 = stroke_points[i+1] - stroke_points[i]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                smoothness = max(0, 1 - avg_angle / np.pi)
                smoothness_scores.append(smoothness)
        
        features.extend([
            np.mean(smoothness_scores) if smoothness_scores else 0,
            np.std(smoothness_scores) if smoothness_scores else 0
        ])
        
        # Spatial distribution features
        all_points = []
        for stroke in strokes:
            all_points.extend(stroke)
        
        if all_points:
            points = np.array(all_points)
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            features.extend([
                np.max(x_coords) - np.min(x_coords),  # width
                np.max(y_coords) - np.min(y_coords),  # height
                np.std(x_coords),
                np.std(y_coords),
                np.mean(x_coords),
                np.mean(y_coords)
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Temporal features
        timing_features = []
        for stroke in strokes:
            if len(stroke) > 1:
                # Simulate timing data (in real app, this would come from actual timing)
                stroke_duration = len(stroke) * random.uniform(0.01, 0.05)
                timing_features.append(stroke_duration)
        
        features.extend([
            np.mean(timing_features) if timing_features else 0,
            np.std(timing_features) if timing_features else 0
        ])
        
        # Consistency features
        if len(strokes) > 1:
            stroke_consistency = 1 - (np.std(stroke_lengths) / np.mean(stroke_lengths)) if np.mean(stroke_lengths) > 0 else 0
            features.append(max(0, stroke_consistency))
        else:
            features.append(0)
        
        # Directional features
        direction_changes = 0
        for stroke in strokes:
            if len(stroke) < 3:
                continue
            
            for i in range(1, len(stroke) - 1):
                v1 = np.array(stroke[i]) - np.array(stroke[i-1])
                v2 = np.array(stroke[i+1]) - np.array(stroke[i])
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if cos_angle < 0:  # Direction change
                        direction_changes += 1
        
        features.append(direction_changes)
        
        # Pressure simulation (in real app, this would come from actual pressure data)
        pressure_variation = random.uniform(0.3, 0.8)
        features.append(pressure_variation)
        
        # Completeness features
        total_distance = sum(stroke_lengths)
        bounding_box_area = (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords)) if all_points else 0
        density = total_distance / bounding_box_area if bounding_box_area > 0 else 0
        features.append(density)
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0)
        
        return np.array(features[:50])
    
    def generate_training_data(self, num_samples=1000):
        """Generate realistic training data with better quality variation"""
        X = []
        y = []
        
        for _ in range(num_samples):
            # Generate strokes with varying quality
            quality = random.uniform(0.1, 1.0)
            num_strokes = random.randint(1, 5)
            
            strokes = []
            for _ in range(num_strokes):
                stroke_length = random.randint(10, 50)
                stroke = []
                
                # Start point
                x, y_coord = random.uniform(0, 100), random.uniform(0, 100)
                stroke.append([x, y_coord])
                
                for _ in range(stroke_length - 1):
                    # Add noise based on quality
                    noise_level = (1 - quality) * random.uniform(0, 10)
                    
                    # Direction change
                    angle = random.uniform(-np.pi/4, np.pi/4)
                    distance = random.uniform(1, 5)
                    
                    dx = distance * np.cos(angle) + random.uniform(-noise_level, noise_level)
                    dy = distance * np.sin(angle) + random.uniform(-noise_level, noise_level)
                    
                    x += dx
                    y_coord += dy
                    stroke.append([x, y_coord])
                
                strokes.append(stroke)
            
            # Extract features
            features = self.extract_features(strokes)
            X.append(features)
            
            # Generate realistic score based on quality
            base_score = quality
            # Add some randomness but keep correlation with quality
            final_score = max(0, min(1, base_score + random.uniform(-0.2, 0.2)))
            y.append(float(final_score))  # Convert to Python float
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train the handwriting analysis model"""
        print("🔄 Generating training data...")
        X, y = self.generate_training_data(2000)
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📈 Score distribution: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🤖 Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"✅ Training completed!")
        print(f"📊 Training R² score: {train_score:.4f}")
        print(f"📊 Test R² score: {test_score:.4f}")
        
        # Save model and scaler
        self.save_model()
        
        return train_score, test_score
    
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"💾 Model saved to {self.model_path}")
            print(f"💾 Scaler saved to {self.scaler_path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(f"📂 Model loaded from {self.model_path}")
            print(f"📂 Scaler loaded from {self.scaler_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main function to train the model"""
    print("🚀 Starting Handwriting Analysis Model Training...")
    
    analyzer = HandwritingAnalyzer()
    
    # Check if model already exists
    if os.path.exists(analyzer.model_path) and os.path.exists(analyzer.scaler_path):
        print("📂 Found existing model. Loading...")
        if analyzer.load_model():
            print("✅ Model loaded successfully!")
            return
        else:
            print("⚠️ Failed to load existing model. Training new one...")
    
    # Train new model
    train_score, test_score = analyzer.train_model()
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Test R² Score: {test_score:.4f}")
    
    if test_score > 0.7:
        print("🌟 Model performance is excellent!")
    elif test_score > 0.5:
        print("👍 Model performance is good!")
    else:
        print("⚠️ Model performance could be improved. Consider adjusting parameters.")

if __name__ == "__main__":
    main()