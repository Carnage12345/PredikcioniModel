from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Učitaj dataset
df = pd.read_csv('enriched_cars.csv')

# Učitaj trenirani model
model = joblib.load('car_price_model.pkl')  # Trenirani pipeline model

# Dobavi jedinstvene vrijednosti za marke i goriva
unique_makes = df['Make'].dropna().unique().tolist()
valid_fuels = df['Fuel'].dropna().unique().tolist()

@app.route('/')
def index():
    return render_template('index.html', makes=unique_makes, fuels=valid_fuels)

@app.route('/get_models/<make>', methods=['GET'])
def get_models(make):
    models = df[df['Make'] == make]['Model'].dropna().unique().tolist()
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    # Dobavi podatke iz zahtjeva
    data = request.get_json()
    make = data['make']
    model_input = data['model']
    year = int(data['year'])
    mileage = int(data['mileage'])
    fuel = data['fuel']

    # Priprema ulaznih podataka za pipeline model
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model_input],
        'Year': [year],
        'Mileage': [mileage],
        'Fuel': [fuel]
    })

    # Predikcija cijene koristeći pipeline
    predicted_price = model.predict(input_data)[0]

    # Pronalaženje sličnih automobila u dataset-u
    similar_cars = df[
        (df['Make'] == make) &
        (df['Model'] == model_input) &
        (df['Fuel'] == fuel) &
        (np.abs(df['Year'] - year) <= 2)
    ]

    # Izračunaj minimalnu, maksimalnu i prosječnu cijenu
    lower_price = similar_cars['min_price'].min() if not similar_cars.empty else None
    higher_price = similar_cars['max_price'].max() if not similar_cars.empty else None

    result = {
        'prediction': f'{predicted_price:.2f}€',
        'options': {
            'lower_option': f'Niža cijena: {lower_price:.2f}€' if lower_price else 'Nema podataka za nižu cijenu',
            'higher_option': f'Viša cijena: {higher_price:.2f}€' if higher_price else 'Nema podataka za višu cijenu'
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
