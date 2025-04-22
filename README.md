# Student Performance Prediction Web App

A web-based application that predicts student performance based on various demographic and academic factors using machine learning models.

## Features

- Predict student performance based on:
  - Gender
  - Race/Ethnicity
  - Parental Education Level
  - Lunch Type
  - Test Preparation Course
  - Reading Scores
  - Writing Scores
- User-friendly web interface
- Real-time predictions
- Built with Flask and machine learning models

## Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: CatBoost, XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Flask

## Project Structure

```
mlproject/
├── app.py                  # Main Flask application
├── catboost_info/          # CatBoost model information
├── logs/                   # Application logs
├── notebook/               # Jupyter notebooks for analysis
├── src/                    # Source code
├── static/                 # Static files (CSS, JS)
├── templates/              # HTML templates
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## Usage

1. Open the web application in your browser
2. Fill in the student's information:
   - Personal details (gender, race/ethnicity)
   - Academic background (parental education, lunch type)
   - Test scores (reading and writing)
3. Click "Predict" to get the predicted performance score

## Acknowledgments

- Thanks to the Flask community for their excellent web framework
- Special thanks to the ML community for their contributions to the field
- Inspired by educational machine learning project by Krish Naik
