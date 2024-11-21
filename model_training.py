from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

# Load and clean data
df = pd.read_csv('cars.csv')

# Inspect unique values in Mileage, Price, and Year
print("Unique Mileage values:", df['Mileage'].unique())
print("Unique Price values:", df['Price'].unique())
print("Unique Year values:", df['Year'].unique())

# Clean the Mileage, Price, and Year columns
# Handling potential non-numeric values and missing entries

# Clean Mileage
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=True).str.replace(',', '', regex=True)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce').fillna(0).astype(int)  # Convert to numeric, set errors to NaN, fill NaN with 0

# Clean Price
df['Price'] = df['Price'].str.replace('€', '', regex=True).str.replace(',', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0).astype(int)  # Convert to numeric, set errors to NaN, fill NaN with 0

# Clean Year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)  # Convert to numeric, set errors to NaN, fill NaN with 0

# Drop rows with invalid data if necessary
df = df[(df['Mileage'] > 0) & (df['Price'] > 0) & (df['Year'] > 1900)]  # Example criteria for valid data

# Features and target
X = df[['Make', 'Model', 'Fuel', 'Year', 'Mileage']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Make', 'Model', 'Fuel']),
        ('num', StandardScaler(), ['Year', 'Mileage'])
    ])

# Create pipeline with preprocessor and regression model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as car_price_model.pkl")