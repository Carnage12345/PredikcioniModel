from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import pickle

# Učitaj podatke
df = pd.read_csv('cars.csv')

# Čišćenje podataka
# Obradi Mileage kolonu
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=True).str.replace(',', '', regex=True)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce').fillna(0).astype(int)

# Obradi Price kolonu
df['Price'] = df['Price'].str.replace('€', '', regex=True).str.replace(',', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0).astype(int)

# Obradi Year kolonu
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

# Filtriraj redove s validnim podacima
df = df[(df['Mileage'] > 0) & (df['Price'] > 0) & (df['Year'] > 1900)]

# Dodaj statistiku cijena (min, max, prosjek) po Make-Model-Fuel-Year kombinaciji
price_stats = df.groupby(['Make', 'Model', 'Fuel', 'Year']).agg(
    min_price=('Price', 'min'),
    max_price=('Price', 'max'),
    avg_price=('Price', 'mean')
).reset_index()

# Dodaj ove statistike u originalni DataFrame
df = pd.merge(df, price_stats, on=['Make', 'Model', 'Fuel', 'Year'], how='left')

# Definiši ulazne varijable (X) i ciljnu varijablu (y)
X = df[['Make', 'Model', 'Fuel', 'Year', 'Mileage']]
y = df['Price']

# Podijeli podatke na trening i test setove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiši pipeline za obradu podataka
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Make', 'Model', 'Fuel']),  # One-hot kodiranje
        ('num', StandardScaler(), ['Year', 'Mileage'])  # Standardizacija numeričkih kolona
    ])

# Kreiraj pipeline za model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Trenira model
model.fit(X_train, y_train)

# Napravi predikcije na test podacima
y_pred = model.predict(X_test)

# Evaluacija modela
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Sačuvaj trenirani model
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Sačuvaj obogaćen dataset
df.to_csv('enriched_cars.csv', index=False)

print("Model treniran i sačuvan kao car_price_model.pkl")
print("Obogaćen dataset sačuvan kao enriched_cars.csv")
