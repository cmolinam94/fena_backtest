import pandas as pd
import base64
import io

def get_download_link(df, filename, text):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def parse_tickers_and_weights(tickers_str, weights_str):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    weights = [float(w) for w in weights_str.split(",") if w.strip()]
    
    if len(tickers) != len(weights):
        raise ValueError("El número de tickers no coincide con el número de pesos")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Los pesos deben sumar 1.0")
    
    return tickers, weights

def create_portfolio_config(
    tickers_str, weights_str, initial_amount, monthly_contribution, 
    rebalance_strategy, tx_cost=0.0, reb_threshold=0.0
):
    tickers, weights = parse_tickers_and_weights(tickers_str, weights_str)
    
    return {
        "tickers": tickers,
        "weights": weights,
        "initial_amount": initial_amount,
        "monthly": monthly_contribution,
        "reb": rebalance_strategy,
        "tx_cost": tx_cost,
        "reb_threshold": reb_threshold
    }

def get_historical_scenarios():
    return {
        "Crisis 2008": {"start_year": 2007, "end_year": 2009, "description": "Crisis financiera global"},
        "Burbuja Tecnológica": {"start_year": 1999, "end_year": 2002, "description": "Burbuja de las puntocom"},
        "Crisis COVID": {"start_year": 2020, "end_year": 2020, "description": "Pánico por el COVID-19"},
        "Recuperación 2009-2010": {"start_year": 2009, "end_year": 2010, "description": "Recuperación post-crisis"},
        "Subida Tipos 2022": {"start_year": 2021, "end_year": 2022, "description": "Políticas restrictivas de bancos centrales"}
    } 