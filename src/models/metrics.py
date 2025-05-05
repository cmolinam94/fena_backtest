import pandas as pd
import numpy as np

def calculate_advanced_metrics(returns, benchmark_returns=None, rf=0.0):
    if returns.empty:
        return {}
    
    total_return = returns.iloc[-1] / returns.iloc[0] - 1
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1/years) - 1 if years else np.nan
    
    daily_returns = returns.pct_change().dropna()
    excess_returns = daily_returns - (((1 + rf) ** (1/252)) - 1)
    
    volatility = daily_returns.std() * np.sqrt(252)
    
    drawdown = (returns / returns.cummax() - 1).min()
    
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else np.nan
    
    downside_returns = daily_returns[daily_returns < 0]
    sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if not downside_returns.empty and downside_returns.std() > 0 else np.nan
    
    beta = alpha = r_squared = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_daily_returns = benchmark_returns.pct_change().dropna()
        aligned_returns = pd.concat([daily_returns, bench_daily_returns], axis=1).dropna()
        if len(aligned_returns) > 0 and aligned_returns.iloc[:, 1].std() > 0:
            cov_matrix = aligned_returns.cov()
            beta = cov_matrix.iloc[0, 1] / aligned_returns.iloc[:, 1].var() if aligned_returns.iloc[:, 1].var() > 0 else np.nan
            
            port_excess = (aligned_returns.iloc[:, 0] - (rf/252)).mean() * 252
            market_excess = (aligned_returns.iloc[:, 1] - (rf/252)).mean() * 252
            alpha = port_excess - beta * market_excess
            
            corr = aligned_returns.corr().iloc[0, 1]
            r_squared = corr**2 if not np.isnan(corr) else np.nan
    
    calmar = abs(cagr / drawdown) if drawdown < 0 else np.nan
    
    ann_return = daily_returns.mean() * 252
    
    return {
        "CAGR": f"{cagr:.2%}",
        "Return": f"{total_return:.2%}",
        "Ann. Return": f"{ann_return:.2%}",
        "Volatility": f"{volatility:.2%}",
        "Máx DD": f"{drawdown:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Beta": f"{beta:.2f}",
        "Alpha": f"{alpha:.2%}" if not np.isnan(alpha) else "–",
        "R²": f"{r_squared:.2f}" if not np.isnan(r_squared) else "–",
        "Calmar": f"{calmar:.2f}" if not np.isnan(calmar) else "–"
    }

def calculate_rolling_metrics(series, window=21):
    daily_returns = series.pct_change().dropna()
    
    rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)
    
    rolling_returns = (1 + daily_returns).rolling(window=window).apply(lambda x: np.prod(x) - 1)
    
    rolling_dd = series / series.rolling(window=window).max() - 1
    
    return pd.DataFrame({
        "Volatilidad": rolling_vol,
        "Retornos": rolling_returns,
        "Drawdown": rolling_dd
    }, index=series.index).dropna()

def run_stress_test(portfolio_config, stress_scenarios, start_year, end_year, bench):
    from src.models.backtest import backtest_portfolio
    
    results = {}
    
    for scenario_name, scenario_config in stress_scenarios.items():
        scenario_start = scenario_config.get("start_year", start_year)
        scenario_end = scenario_config.get("end_year", end_year)
        
        try:
            result = backtest_portfolio(
                portfolio_config["tickers"], 
                portfolio_config["weights"],
                scenario_start, scenario_end,
                portfolio_config["initial_amount"],
                portfolio_config["monthly"],
                portfolio_config["reb"],
                bench,
                portfolio_config.get("tx_cost", 0.0),
                portfolio_config.get("reb_threshold", 0.0)
            )
            metrics = calculate_advanced_metrics(result["Portfolio"], result["Benchmark"])
            results[scenario_name] = metrics
        except Exception as e:
            results[scenario_name] = {"Error": str(e)}
    
    return results

def monte_carlo_simulation(returns, initial_value, horizon_days, simulations=1000, percentiles=[5, 25, 50, 75, 95]):
    mu = returns.mean()
    sigma = returns.std()
    
    simulation_results = np.zeros((horizon_days, simulations))
    simulation_results[0] = initial_value
    
    for i in range(1, horizon_days):
        random_returns = np.random.normal(mu, sigma, simulations)
        simulation_results[i] = simulation_results[i-1] * (1 + random_returns)
    
    percentile_results = np.percentile(simulation_results, percentiles, axis=1)
    
    return {
        "simulations": simulation_results,
        "percentiles": percentile_results,
        "percentile_labels": percentiles
    } 