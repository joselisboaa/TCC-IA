from flask import Flask, request, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

def load_data():
    df = pd.read_excel("data/Numeros de PCD no IFCE.xlsx", parse_dates=["Data"])
    
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y", dayfirst=True)
        df.set_index("Data", inplace=True)
    else:
        raise ValueError("A coluna 'Data' não foi encontrada no arquivo.")
    
    df["Total_PCDs"] = df.sum(axis=1)
    
    return df

def train_model(df):
    model = ARIMA(df["Total_PCDs"], order=(5,1,0))
    model_fit = model.fit()
    return model_fit

@app.route('/data', methods=['GET'])
def get_data():
    meses = int(request.args.get("meses", 12))
    df = load_data()
    
    model_fit = train_model(df)
    future_dates = [df.index[-1] + timedelta(days=30*i) for i in range(1, meses+1)]
    forecast = model_fit.forecast(steps=meses)
    
    historical_data = {str(date.date()): int(value) for date, value in df["Total_PCDs"].items()}
    predicted_data = {str(date.date()): int(forecast[i]) for i, date in enumerate(future_dates)}
    
    return jsonify({
        "historical": historical_data,
        "predicted": predicted_data
    })

if __name__ == '__main__':
    app.run(debug=True)
