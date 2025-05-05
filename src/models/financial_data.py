import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit as st

@st.cache_data(show_spinner=False)
def download_daily_data(tickers, start_date, end_date, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            adjusted_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
            adjusted_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
            
            data = yf.download(tickers, start=adjusted_start, end=adjusted_end, interval="1d", progress=False)
            
            if data.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError(f"No se pudieron obtener datos para los tickers: {tickers}")
            
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            adj_close = data[col]
            
            if isinstance(adj_close, pd.Series):
                adj_close = adj_close.to_frame()
            
            adj_close = adj_close.loc[start_date:end_date]
            
            missing_data = adj_close.isna().sum()
            significant_missing = missing_data[missing_data > len(adj_close) * 0.1]
            
            if not significant_missing.empty:
                st.warning(f"Los siguientes tickers tienen más del 10% de datos faltantes: {significant_missing.index.tolist()}")
            
            return adj_close.dropna(how="all")
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise ValueError(f"Error al descargar datos: {str(e)}")

def validate_tickers(tickers, test_period_days=30):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=test_period_days)).strftime("%Y-%m-%d")
    
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 5:
                invalid_tickers.append(ticker)
            else:
                valid_tickers.append(ticker)
        except:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers 