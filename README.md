# Car Price Prediction Application

An application that predicts car prices using machine learning and optionally integrates with Google Gemini for a second estimate.

## Prerequisites

Requirements:
- Python 3.10 or higher
- pip

## Installation

1. Clone the repository or download the source code.
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

4. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### GUI version

```bash
python gui.py
```

### Command-line version

```bash
python main.py
```

## Features

- Predicts car prices based on make, model, year, mileage, and condition.
- Optional Gemini integration for a blended estimate.
- GUI and command-line interfaces.
- Basic data visualization and trend plotting.

## Refactoring and modernization review

The following updates were applied and should be used as coding guidelines:

### Outdated patterns identified

1. Hard-coded year values (`2025`) in prediction and validation logic.
2. Very old minimum dependency versions in `requirements.txt`.
3. Deprecated Gemini SDK (`google-generativeai`) usage.
4. README formatting with emojis that reduced clarity for technical documentation.

### Changes implemented

1. Replaced hard-coded year values with runtime constants (`CURRENT_YEAR` and `MIN_YEAR`) in:
   - `/home/runner/work/car_price_predic/car_price_predic/main.py`
   - `/home/runner/work/car_price_predic/car_price_predic/gui.py`
2. Upgraded dependency baselines in:
   - `/home/runner/work/car_price_predic/car_price_predic/requirements.txt`
3. Migrated Gemini integration from deprecated `google-generativeai` to `google-genai` in:
   - `/home/runner/work/car_price_predic/car_price_predic/gemini_integration.py`
4. Rewrote this README to remove emojis and improve maintainability.

### Best practices for cleaner, maintainable code

- Prefer named constants over magic numbers.
- Keep user-facing validation messages generated from constants to avoid drift.
- Keep dependency baselines modern and reviewed regularly.
- Keep documentation concise, plain, and task-oriented.

### Specific example of the applied refactor

Before:

```python
if year < 1900 or year > 2025:
    return "Erreur : L'année de fabrication doit être entre 1900 et 2025."
```

After:

```python
if year < MIN_YEAR or year > CURRENT_YEAR:
    return f"Erreur : L'année de fabrication doit être entre {MIN_YEAR} et {CURRENT_YEAR}."
```

## Note

If you use Gemini integration, provide a valid Gemini API key when prompted.
