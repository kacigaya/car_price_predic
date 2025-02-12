import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from gemini_integration import setup_gemini, get_gemini_prediction, combine_predictions


def prepare_data(csv_file):
    data = pd.read_csv(csv_file)
    
    data['Year'] = data['Year'].fillna(data['Year'].median())
    data['Mileage'] = data['Mileage'].fillna(data['Mileage'].median())
    data['Price'] = data['Price'].fillna(data['Price'].median())
    data['Condition'] = data['Condition'].fillna('Good')  
    data['Owners'] = data['Owners'].fillna(data['Owners'].median())  
    
    data['Age'] = 2025 - data['Year']
    data['Mileage_log'] = np.log1p(data['Mileage'])
    X = data.drop(['Id', 'Price', 'Year', 'Mileage'], axis=1)
    y = np.log(data['Price'])
    
    return X, y, data


def create_model():
    numeric_features = ['Mileage_log', 'Age']
    categorical_features = ['Make', 'Model', 'Condition']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, random_state=42))
    ])


async def predict_car_price(make, model, mileage, condition, age, trained_model, gemini_model=None):
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model],
        'Condition': [condition],
        'Mileage_log': [np.log1p(mileage)],
        'Age': [age]
    })
    
    log_predicted_price = trained_model.predict(input_data)[0]
    statistical_price = np.exp(log_predicted_price)
    
    if mileage > 300000:
        statistical_price = min(statistical_price, 5000)
    
    statistical_price = max(0, statistical_price)
    
    if gemini_model:
        year = 2025 - age
        gemini_price = await get_gemini_prediction(gemini_model, make, model, year, mileage, condition)
        return combine_predictions(statistical_price, gemini_price)
    
    return statistical_price


def validate_input(make, model, year, mileage, condition, data):
    if make not in data['Make'].unique():
        return f"Erreur : La marque '{make}' n'est pas dans la base de données."
    
    if model not in data['Model'].unique():
        return f"Erreur : Le modèle '{model}' n'est pas dans la base de données."
    
    if year < 1900 or year > 2025:
        return "Erreur : L'année de fabrication doit être entre 1900 et 2025."
    
    if mileage < 0:
        return "Erreur : Le kilométrage ne peut pas être négatif."
    
    if condition not in ['Excellent', 'Good', 'Fair', 'Poor']:
        return f"Erreur : L'état '{condition}' est invalide. Choisissez parmi : Excellent, Good, Fair, Poor."
    
    return None 


def predict_trend(data):
    trend_data = data.groupby('Year')['Price'].mean().reset_index()
    
    X_trend = trend_data['Year'].values.reshape(-1, 1)
    y_trend = trend_data['Price'].values
    
    trend_model = LinearRegression()
    trend_model.fit(X_trend, y_trend)
    
    future_years = np.arange(2025, 2031).reshape(-1, 1)
    future_prices = trend_model.predict(future_years)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(trend_data['Year'], trend_data['Price'], label="Prix moyen par année")
    plt.plot(future_years, future_prices, color='red', label="Tendance future (2025-2030)")
    plt.title("Tendances des prix des voitures en fonction des années")
    plt.xlabel("Année")
    plt.ylabel("Prix moyen (€)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nPrédictions des prix moyens pour les années futures :")
    for year, price in zip(future_years.flatten(), future_prices):
        print(f"Année {year}: {price:.2f} €")


def generate_visualizations(data):
    plt.figure(figsize=(8, 6))
    data['Condition'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des états des véhicules")
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(data['Mileage'], data['Price'], alpha=0.7)
    plt.title("Relation entre le prix et le kilométrage")
    plt.xlabel("Kilométrage")
    plt.ylabel("Prix")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    data['Year'].astype(int).value_counts().sort_index().plot(kind='bar')
    plt.title("Répartition des années de fabrication")
    plt.xlabel("Année")
    plt.ylabel("Nombre de véhicules")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def generate_owners_visualization(data):
    owners_by_make = data.groupby('Make')['Owners'].sum().reset_index()
    
    owners_by_make = owners_by_make.sort_values(by='Owners', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(owners_by_make['Make'], owners_by_make['Owners'], color='skyblue')
    plt.title("Nombre de propriétaires par marque")
    plt.xlabel("Marque")
    plt.ylabel("Nombre de propriétaires")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


async def main():
    X, y, data = prepare_data('data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(np.exp(y_test), np.exp(y_pred))
    print(f"Model Mean Squared Error: {mse:.2f}")
    
    generate_visualizations(data)
    generate_owners_visualization(data)  
    predict_trend(data)  
    
    print("\nBienvenue dans l'application de prédiction de prix de voiture")
    make = input("Entrez la marque de la voiture: ")
    model_name = input("Entrez le modèle de la voiture: ")
    year = int(input("Entrez l'année de la voiture: "))
    mileage = float(input("Entrez le kilométrage de la voiture: "))
    condition = input("Entrez l'état de la voiture (Excellent, Good, Fair, Poor): ")
    
    validation_error = validate_input(make, model_name, year, mileage, condition, data)
    if validation_error:
        print(validation_error)
        return  
    
    age = 2025 - year
    
    # Initialize Gemini model (you'll need to set your API key)
    api_key = input("Enter your Gemini API key (press Enter to skip Gemini integration): ").strip()
    gemini_model = setup_gemini(api_key) if api_key else None
    
    predicted_price = await predict_car_price(
        make, model_name, mileage, condition, age, model, gemini_model
    )
    
    print(f"\nLe prix prédit de la voiture est: {predicted_price:.2f} €")


if __name__ == "__main__":
    asyncio.run(main())