**Customer Churn Prediction Pipeline**
A production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. Built with Scikit-learn's Pipeline API for maximum reproducibility and deployability.

---

**📊 Project Overview**
This project implements an end-to-end ML pipeline that:

Preprocesses raw customer data (handling missing values, encoding, scaling)

Trains and compares multiple classification models

Performs hyperparameter tuning with GridSearchCV

Exports a complete, reusable pipeline for production deployment

Provides a clean API for making predictions on new customer data

---

**🚀 Features**
Data Preprocessing: Automated handling of categorical encoding, feature scaling, and missing values

Model Comparison: Logistic Regression vs. Random Forest with comprehensive evaluation

Hyperparameter Tuning: Optimized model parameters using GridSearchCV

Production Ready: Serialized pipeline that can be integrated into web applications or batch processes

Class Imbalance Handling: Built-in support for handling imbalanced target variable

---
              
**🛠️ Installation & Setup**
Clone the repository
git clone Nasir-Raza-AI/End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API
cd Nasir-Raza-AI/End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API

---

**Install dependencies**

pip install scikit-learn pandas numpy joblib matplotlib
Download the dataset

Get the dataset from Kaggle

Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the project root directory

---

**🎯 Usage**
1. Training the Model
Run the Jupyter notebook to train and export the pipeline:

jupyter notebook Churn_Prediction.ipynb
The notebook will:

Load and preprocess the data

Train and compare models

Perform hyperparameter tuning

Export the best pipeline to customer_churn_pipeline.joblib

---

**2. Making Predictions**
Option A: Using the provided prediction function
python
from churn_predictor import predict_churn

sample_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    # ... (all other features)
}

result = predict_churn(sample_customer)
print(result)
Option B: Using the pipeline directly
python
import joblib
import pandas as pd

---

# Load the pipeline
pipeline = joblib.load('customer_churn_pipeline.joblib')

# Prepare new data (as DataFrame)
new_data = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    # ... all other features
}])

# Make prediction
prediction = pipeline.predict(new_data)
probability = pipeline.predict_proba(new_data)

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Churn Probability: {probability[0][1]:.2%}")
3. Integration Examples
Web Application (Flask Example)
python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('customer_churn_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return jsonify({
        'churn_prediction': bool(prediction),
        'churn_probability': float(probability),
        'churn_risk': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    })

if __name__ == '__main__':
    app.run(debug=True)
Batch Processing Script
python
import pandas as pd
import joblib

def batch_predict(input_csv, output_csv):
    """Process multiple customers from a CSV file"""
    model = joblib.load('customer_churn_pipeline.joblib')
    df = pd.read_csv(input_csv)
    
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    
    df['churn_prediction'] = predictions
    df['churn_probability'] = probabilities
    df['churn_risk'] = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities]
    
    df.to_csv(output_csv, index=False)
    return df

---

# Usage
batch_predict('new_customers.csv', 'predictions.csv')
📈 Model Performance
The current best model achieves:

Accuracy: ~81%

Precision (Churn): ~68%

Recall (Churn): ~52%

F1-Score (Churn): ~59%

Note: The class imbalance (73% No Churn vs. 27% Churn) makes predicting churn challenging. The model is better at identifying customers who will stay than those who will leave.

🔧 Customization
Adding New Models
Add new classifiers to the pipeline in the notebook:

python
from sklearn.svm import SVC
from xgboost import XGBClassifier

svc_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42))
])
Modifying Hyperparameter Search
Adjust the parameter grids in the notebook:

python
custom_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 6, 9]
}
📋 Requirements
Python 3.8+

scikit-learn >= 1.0

pandas >= 1.3

numpy >= 1.21

joblib >= 1.0

matplotlib >= 3.5 (for visualization)

---

**📄 License**
This project is licensed under the MIT License - see the LICENSE file for details.

---

**🆓 Dataset Attribution**
The Telco Customer Churn dataset is from Kaggle and is available under the CC0: Public Domain license.

---

**🎓 Skills Demonstrated**
End-to-end ML pipeline construction with Scikit-learn

Hyperparameter tuning with GridSearchCV

Model evaluation and selection

Production-ready model serialization with joblib

Handling class imbalance in classification problems

Creating reusable prediction interfaces

---

**📞 Support**
For questions or issues, please open an issue on GitHub or contact the maintainers.
