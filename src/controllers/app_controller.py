import streamlit as st
import pandas as pd
from src.models.backtest import backtest_portfolio
from src.models.metrics import calculate_advanced_metrics, calculate_rolling_metrics, run_stress_test, monte_carlo_simulation
from src.views.main_view import MainView

class AppController:
    def __init__(self):
        self.view = MainView()
    
    def run(self):
        user_inputs = self.view.render()
        
        if user_inputs["run_backtest"]:
            self._run_backtest(user_inputs)
    
    def _run_backtest(self, user_inputs):
        portfolios_cfg = st.session_state.portfolios_cfg
        global_params = st.session_state.global_params
        advanced_options = st.session_state.advanced_options if "advanced_options" in st.session_state else {}
        
        for name, cfg in portfolios_cfg.items():
            if len(cfg["tickers"]) != len(cfg["weights"]):
                st.error(f"{name}: número de tickers ≠ número de pesos.")
                st.stop()
            if abs(sum(cfg["weights"]) - 1.0) > 1e-6:
                st.error(f"{name}: los pesos no suman 1.0.")
                st.stop()
        
        progress_container = st.empty()
        with progress_container.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        combined = pd.DataFrame()
        metrics = {}
        
        try:
            for i, (name, cfg) in enumerate(portfolios_cfg.items()):
                status_text.text(f"Procesando {name}...")
                progress_bar.progress((i / len(portfolios_cfg)) / 2)
                
                res = backtest_portfolio(
                    cfg["tickers"], cfg["weights"],
                    global_params["y0"], global_params["y1"], 
                    cfg["initial_amount"], cfg["monthly"], 
                    cfg["reb"], global_params["bench"],
                    cfg["tx_cost"], cfg["reb_threshold"]
                )
                
                combined[name] = res["Portfolio"]
                if "Benchmark" not in combined.columns:
                    combined["Benchmark"] = res["Benchmark"]
                    
                metrics[name] = calculate_advanced_metrics(
                    res["Portfolio"], res["Benchmark"], 
                    global_params["rf"]
                )
        except ValueError as e:
            progress_container.empty()
            st.error(str(e))
            st.stop()
        
        if combined.empty:
            progress_container.empty()
            st.error("Sin datos que mostrar.")
            st.stop()
        
        progress_bar.progress(0.5)
        status_text.text("Calculando métricas adicionales...")
        
        metrics_df = pd.DataFrame(metrics).T
        
        scenarios_results = {}
        if advanced_options.get("run_scenarios", False) and advanced_options.get("selected_scenarios", []):
            status_text.text("Analizando escenarios históricos...")
            progress_bar.progress(0.7)
            
            for name, cfg in portfolios_cfg.items():
                selected_historical_scenarios = {
                    k: advanced_options["historical_scenarios"][k] 
                    for k in advanced_options["selected_scenarios"]
                }
                
                scenarios_results[name] = run_stress_test(
                    cfg, selected_historical_scenarios, 
                    global_params["y0"], global_params["y1"], 
                    global_params["bench"]
                )
        
        rolling_metrics = {}
        if advanced_options.get("run_rolling", False):
            status_text.text("Calculando métricas rodantes...")
            progress_bar.progress(0.8)
            
            for col in combined.columns:
                if col != "Benchmark":
                    rolling_metrics[col] = calculate_rolling_metrics(
                        combined[col], 
                        advanced_options.get("rolling_window", 63)
                    )
        
        monte_carlo_results = {}
        if advanced_options.get("run_monte_carlo", False):
            status_text.text("Ejecutando simulación de Monte Carlo...")
            progress_bar.progress(0.9)
            
            for col in combined.columns:
                if col != "Benchmark":
                    daily_returns = combined[col].pct_change().dropna()
                    last_value = combined[col].iloc[-1]
                    monte_carlo_results[col] = monte_carlo_simulation(
                        daily_returns, last_value, 
                        advanced_options.get("mc_days", 252), 
                        advanced_options.get("mc_sims", 1000)
                    )
        
        progress_container.empty()
        
        results = {
            "combined": combined,
            "metrics_df": metrics_df,
            "scenarios_results": scenarios_results,
            "rolling_metrics": rolling_metrics,
            "monte_carlo_results": monte_carlo_results
        }
        
        self.view.display_results(results, {
            "portfolios_cfg": portfolios_cfg,
            "global_params": global_params
        }, {
            "show_raw_data": user_inputs["show_raw_data"],
            "rolling_window": advanced_options.get("rolling_window", 63)
        }) 