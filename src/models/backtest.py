import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from src.models.financial_data import download_daily_data, validate_tickers

def rebalance_portfolio(weights, value, prices, prev_shares=None, tx_cost=0.0):
    w = np.array(weights)
    alloc = value * w
    new_shares = {t: a / prices[t] for t, a in zip(prices.index, alloc)}
    
    tx_costs = 0.0
    if prev_shares and tx_cost > 0:
        for t in new_shares:
            if t in prev_shares:
                shares_diff = abs(new_shares[t] - prev_shares[t])
                tx_costs += shares_diff * prices[t] * tx_cost
    
    if tx_costs > 0:
        remaining_value = value - tx_costs
        alloc = remaining_value * w
        new_shares = {t: a / prices[t] for t, a in zip(prices.index, alloc)}
    
    return new_shares, tx_costs

def portfolio_value_from_shares(shares, prices):
    return sum(n * prices[t] for t, n in shares.items())

def backtest_portfolio(tickers, weights, start_year, end_year, initial_amount, monthly, reb, bench, tx_cost=0.0, reb_threshold=0.0):
    start, end = f"{start_year}-01-01", f"{end_year}-12-31"
    
    all_symbols = list(tickers) + [bench]
    valid_tickers, invalid_tickers = validate_tickers(all_symbols)
    
    if invalid_tickers:
        st.warning(f"Los siguientes tickers pueden no ser válidos: {invalid_tickers}")
        if bench in invalid_tickers:
            raise ValueError(f"El benchmark {bench} no es válido o no tiene datos recientes.")
    
    data = download_daily_data(all_symbols, start, end)
    
    if bench not in data.columns:
        raise ValueError(f"El benchmark {bench} no tiene datos para el período seleccionado.")
    
    bench_px, prices = data[bench], data.drop(columns=[bench])
    
    if prices.empty:
        raise ValueError("No hay datos suficientes para los tickers seleccionados.")
    
    pv, bv = initial_amount, initial_amount
    shares = None
    p_vals, b_vals = [], []
    pm, py = None, None
    tx_costs_total = 0.0
    tx_costs_hist = []
    reb_dates = []
    
    for i, d in enumerate(prices.index):
        rebalance_now = False
        
        if i == 0:
            rebalance_now = True
        elif reb == "annual" and d.year != py:
            rebalance_now = True
        elif reb == "monthly" and d.month != pm:
            rebalance_now = True
        
        if not rebalance_now and reb_threshold > 0 and shares:
            current_values = {t: shares[t] * prices.loc[d, t] for t in shares}
            total_value = sum(current_values.values())
            current_weights = {t: v/total_value for t, v in current_values.items()}
            
            max_deviation = max(abs(current_weights[t] - w) for t, w in zip(current_weights, weights))
            if max_deviation > reb_threshold:
                rebalance_now = True
        
        if rebalance_now:
            if i > 0:
                reb_dates.append(d)
            shares, tx_cost = rebalance_portfolio(weights, pv, prices.loc[d], shares, tx_cost)
            tx_costs_total += tx_cost
        
        pv = portfolio_value_from_shares(shares, prices.loc[d])
        
        if i > 0 and d.month != pm:
            pv += monthly
            if reb != "none":
                shares, tx_cost = rebalance_portfolio(weights, pv, prices.loc[d], shares, tx_cost)
                tx_costs_total += tx_cost
                reb_dates.append(d)
        
        if i > 0:
            br = bench_px.loc[d] / bench_px.iloc[i-1] - 1
            bv = bv * (1 + br) + (monthly if d.month != pm else 0)
        
        p_vals.append(pv)
        b_vals.append(bv)
        tx_costs_hist.append(tx_costs_total)
        
        pm, py = d.month, d.year
    
    results = pd.DataFrame({
        "Portfolio": p_vals, 
        "Benchmark": b_vals,
        "TX_Costs": tx_costs_hist
    }, index=prices.index)
    
    if reb_dates:
        reb_info = pd.Series(1, index=reb_dates)
        results["Rebalance"] = 0
        results.loc[reb_info.index, "Rebalance"] = 1
    
    return results 