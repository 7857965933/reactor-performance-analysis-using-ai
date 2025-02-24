from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('reactor_performance_model.pkl')

# Load the scaler (if scaling is required)
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    form_data = request.form
    
    # Extract input features
    input_features = [
        float(form_data['Temperature']),
        float(form_data['Pressure']),
        float(form_data['FeedRate']),
        float(form_data['CatalystConcentration']),
        float(form_data['ReactionTime']),
        float(form_data['CoolingRate']),
        float(form_data['AgitationSpeed'])
    ]

    # Scale the input features (if scaling was used during training)
    scaled_input = scaler.fit_transform([input_features])

    # Make predictions
    predictions = model.predict(scaled_input)
    
    # Extract outputs
    conversion_efficiency, product_purity, energy_consumption = predictions[0]
    
    # Render results
    return render_template(
        'index.html',
        prediction=f"Conversion Efficiency: {conversion_efficiency:.2f}%, "
                   f"Product Purity: {product_purity:.2f}%, "
                   f"Energy Consumption: {energy_consumption:.2f} MJ"
    )

if __name__ == '__main__':
    app.run(debug=True)
