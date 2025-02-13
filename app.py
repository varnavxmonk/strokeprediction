from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# File paths
DATA_PATH = "healthcare-dataset-stroke-data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Define features and target
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Handle class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (RandomForestClassifier for better performance)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Store feature names
feature_names = X.columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Ensure input dictionary has all required features
    input_dict = {feature: 0 for feature in feature_names}  
    input_dict.update({
        'age': float(data['age']),
        'hypertension': int(data['hypertension']),
        'heart_disease': int(data['heart_disease']),
        'avg_glucose_level': float(data['avg_glucose_level']),
        'bmi': float(data['bmi']),
        'gender_Male': int(data.get('gender') == 'Male'),
        'gender_Other': int(data.get('gender') == 'Other'),
        'ever_married_Yes': int(data.get('ever_married') == 'Yes'),
        'work_type_Never_worked': int(data.get('work_type') == 'Never_worked'),
        'work_type_Private': int(data.get('work_type') == 'Private'),
        'work_type_Self-employed': int(data.get('work_type') == 'Self-employed'),
        'work_type_children': int(data.get('work_type') == 'children'),
        'Residence_type_Urban': int(data.get('Residence_type') == 'Urban'),
        'smoking_status_formerly smoked': int(data.get('smoking_status') == 'formerly smoked'),
        'smoking_status_never smoked': int(data.get('smoking_status') == 'never smoked'),
        'smoking_status_smokes': int(data.get('smoking_status') == 'smokes')
    })

    # Convert to DataFrame and scale
    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Adjusted Probability Thresholds
    if probability >= 0.2:
        prediction = "🚨 Extremely High Stroke Risk"
    elif probability >= 0.15:
        prediction = "⚠️ High Stroke Risk"
    elif probability >= 0.08:
        prediction = "⚖️ Moderate Stroke Risk"
    else:
        prediction = "✅ Low Stroke Risk"

    return jsonify({
        'stroke_prediction': prediction,
        'probability': round(probability, 4),
        'message': "⚠️ AI-based prediction. Consult a doctor for medical advice."
    })

if __name__ == '__main__':
    app.run(debug=True)
