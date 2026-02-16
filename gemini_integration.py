import asyncio
from typing import Any
from google import genai

def setup_gemini(api_key: str):
    return genai.Client(api_key=api_key)

def format_car_data(make: str, model: str, year: int, mileage: float, condition: str) -> str:
    return f"""Analyze the following car details and predict a fair market price:
    - Make: {make}
    - Model: {model}
    - Year: {year}
    - Mileage: {mileage}
    - Condition: {condition}
    Consider current market trends, regional pricing variations, and seasonal patterns.
    Provide only the predicted price as a number without any currency symbols or formatting."""

async def get_gemini_prediction(model: Any, make: str, car_model: str, year: int, 
                              mileage: float, condition: str) -> float:
    try:
        prompt = format_car_data(make, car_model, year, mileage, condition)
        response = await asyncio.to_thread(
            model.models.generate_content,
            model='gemini-2.5-flash',
            contents=prompt
        )
        if not response.text:
            return None
        price_str = response.text.strip().replace('€', '').replace('$', '').replace(',', '')
        return float(price_str)
    except Exception as e:
        print(f"Error getting Gemini prediction: {e}")
        return None

def combine_predictions(statistical_price: float, gemini_price: float, 
                       confidence_weight: float = 0.7) -> float:
    """Combine predictions from statistical model and Gemini AI
    Args:
        statistical_price: Price predicted by the statistical model
        gemini_price: Price predicted by Gemini AI
        confidence_weight: Weight given to statistical model (0-1)
    Returns:
        Combined prediction
    """
    if gemini_price is None:
        return statistical_price
    
    return (statistical_price * confidence_weight + 
            gemini_price * (1 - confidence_weight))
