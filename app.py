from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('car_price_model.pkl', 'rb') as file:
    car_price_model = pickle.load(file)

# Load CSV data for dropdown options
df = pd.read_csv('cars.csv')

# Get unique values for 'Make', 'Model', and 'Fuel' and clean invalid ones
unique_makes = df['Make'].drop_duplicates().sort_values().tolist()

# Get unique fuels, ensuring no empty or invalid values
unique_fuels = df['Fuel'].dropna().unique().tolist()
valid_fuels = [fuel for fuel in unique_fuels if isinstance(fuel, str) and fuel.strip() != '' and not fuel.isdigit()]

# Create list of unique models (Make-Model pairs)
unique_models = df[['Make', 'Model']].drop_duplicates().sort_values(by=['Make', 'Model']).to_dict('records')


@app.route('/')
def index():
    return render_template('index.html', makes=unique_makes, fuels=valid_fuels)


@app.route('/get_models/<make>', methods=['GET'])
def get_models(make):
    # Get models based on the selected make
    models = [item['Model'] for item in unique_models if item['Make'].lower() == make.lower()]
    return jsonify(sorted(models))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    make = data['make']
    model_name = data['model']
    fuel = data['fuel']
    year = int(data['year'])
    mileage = int(data['mileage'])

    # Create a DataFrame with column names as expected by the model
    input_data = pd.DataFrame([[make, model_name, fuel, year, mileage]],
                              columns=['Make', 'Model', 'Fuel', 'Year', 'Mileage'])

    # Make prediction
    prediction = car_price_model.predict(input_data)[0]

    # Handle NaN or invalid predictions
    if prediction == 'NaN' or prediction is None or prediction < 0:
        return {"prediction": "Invalid data, unable to predict"}
    else:
        return {"prediction": f"{prediction:.2f}"}


if __name__ == '__main__':
    app.run(debug=True)
