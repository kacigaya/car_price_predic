import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
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

X, y, data = prepare_data('data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = create_model()
model.fit(X_train, y_train)

def run_gui():
    async def predict_async():
        try:
            make = make_entry.get()
            model_name = model_entry.get()
            year = int(year_entry.get())
            mileage = float(mileage_entry.get())
            condition = condition_combobox.get()

            if make not in data['Make'].unique():
                messagebox.showerror("Erreur", f"La marque '{make}' n'est pas reconnue.")
                return
            if model_name not in data['Model'].unique():
                messagebox.showerror("Erreur", f"Le modèle '{model_name}' n'est pas reconnu.")
                return
            if condition not in ['Excellent', 'Good', 'Fair', 'Poor']:
                messagebox.showerror("Erreur", "Veuillez choisir un état valide.")
                return
            if year < 1900 or year > 2025:
                messagebox.showerror("Erreur", "L'année doit être comprise entre 1900 et 2025.")
                return

            age = 2025 - year
            
            # Initialize Gemini model if API key is provided
            api_key = api_key_entry.get().strip()
            gemini_model = setup_gemini(api_key) if api_key else None
            
            predicted_price = await predict_car_price(make, model_name, mileage, condition, age, model, gemini_model)
            result_label.config(text=f"Prix estimé : {predicted_price:.2f} €")

        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue : {str(e)}")
    
    def on_predict():
        asyncio.run(predict_async())

    root = tk.Tk()
    root.title("Prédiction du Prix des Voitures")

    tk.Label(root, text="Marque :").grid(row=0, column=0, padx=10, pady=10)
    make_entry = tk.Entry(root)
    make_entry.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Modèle :").grid(row=1, column=0, padx=10, pady=10)
    model_entry = tk.Entry(root)
    model_entry.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Année :").grid(row=2, column=0, padx=10, pady=10)
    year_entry = tk.Entry(root)
    year_entry.grid(row=2, column=1, padx=10, pady=10)

    tk.Label(root, text="Kilométrage :").grid(row=3, column=0, padx=10, pady=10)
    mileage_entry = tk.Entry(root)
    mileage_entry.grid(row=3, column=1, padx=10, pady=10)

    tk.Label(root, text="État :").grid(row=4, column=0, padx=10, pady=10)
    condition_combobox = ttk.Combobox(root, values=['Excellent', 'Good', 'Fair', 'Poor'])
    condition_combobox.grid(row=4, column=1, padx=10, pady=10)
    condition_combobox.current(0)

    tk.Label(root, text="Gemini API Key :").grid(row=5, column=0, padx=10, pady=10)
    api_key_entry = tk.Entry(root)
    api_key_entry.grid(row=5, column=1, padx=10, pady=10)

    predict_button = tk.Button(root, text="Prédire", command=on_predict)
    predict_button.grid(row=6, column=0, columnspan=2, pady=20)

    result_label = tk.Label(root, text="", font=("Arial", 14))
    result_label.grid(row=6, column=0, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
