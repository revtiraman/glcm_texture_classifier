import os
import joblib
from glcm_features import extract_glcm_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Data containers ===
X = []
y = []

# === Load dataset ===
base_dir = 'images'  # Images should be organized as images/class_name/image.png

# Get only folder names (labels)
labels = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for label in labels:
    folder_path = os.path.join(base_dir, label)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        try:
            features = extract_glcm_features(image_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"❌ Failed to process {image_path} → {e}")

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Save trained model ===
joblib.dump(model, "model.pkl")

# === Print accuracy ===
print("✅ Training Accuracy:", model.score(X_train, y_train))
print("✅ Testing Accuracy:", model.score(X_test, y_test))
