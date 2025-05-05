import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.utils.helpers import get_download_link

class ResultsView:
    def render(self, results, config, options):
        if not results or "combined" not in results:
            return
        
        combined = results["combined"]
        metrics_df = results.get("metrics_df", pd.DataFrame())
        scenarios_results = results.get("scenarios_results", {})
        rolling_metrics = results.get("rolling_metrics", {})
        monte_carlo_results = results.get("monte_carlo_results", {})
        show_raw_data = options.get("show_raw_data", False)
        
        self._render_metrics_summary(metrics_df)
        self._render_portfolio_evolution(combined)
        self._render_drawdown_analysis(combined)
        
        if rolling_metrics:
            self._render_rolling_analysis(rolling_metrics, options.get("rolling_window", 63))
        
        if scenarios_results:
            self._render_scenarios_analysis(scenarios_results)
        
        if monte_carlo_results:
            self._render_monte_carlo_analysis(monte_carlo_results, combined)
        
        if show_raw_data:
            self._render_raw_data(combined)
    
    def _render_metrics_summary(self, metrics_df):
        if metrics_df.empty:
            return
            
        st.subheader("Resumen de Desempeño")
        st.dataframe(
            metrics_df.rename(columns={
                "CAGR": "CAGR", 
                "Return": "Retorno Total", 
                "Ann. Return": "Retorno Anualizado",
                "Volatility": "Volatilidad",
                "Máx DD": "Máx Drawdown", 
                "Sharpe": "Sharpe",
                "Sortino": "Sortino",
                "Beta": "Beta",
                "Alpha": "Alpha",
                "R²": "R²",
                "Calmar": "Calmar"
            })
        )
        
        st.markdown(get_download_link(metrics_df, "metricas_carteras.csv", "📥 Descargar métricas completas (CSV)"), unsafe_allow_html=True)
    
    def _render_portfolio_evolution(self, combined):
        st.subheader("Evolución del Valor")
        
        fig = px.line(
            combined, 
            title="Valor de las carteras", 
            labels={"value": "Valor ($)", "variable": "Cartera", "date": "Fecha"}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_analysis(self, combined):
        st.subheader("Análisis de Drawdown")
        
        drawdown_df = pd.DataFrame()
        for col in combined.columns:
            drawdown_df[col] = combined[col] / combined[col].cummax() - 1
        
        fig_dd = px.line(
            drawdown_df, 
            title="Drawdown a lo largo del tiempo",
            labels={"value": "Drawdown (%)", "variable": "Cartera", "date": "Fecha"}
        )
        fig_dd.update_layout(hovermode="x unified")
        fig_dd.add_hline(y=-0.1, line_dash="dash", line_color="yellow", annotation_text="-10%")
        fig_dd.add_hline(y=-0.2, line_dash="dash", line_color="orange", annotation_text="-20%")
        fig_dd.add_hline(y=-0.3, line_dash="dash", line_color="red", annotation_text="-30%")
        
        st.plotly_chart(fig_dd, use_container_width=True)
    
    def _render_rolling_analysis(self, rolling_metrics, window):
        st.subheader(f"Métricas Rodantes ({window} días)")
        
        for portfolio, metrics in rolling_metrics.items():
            with st.expander(f"Métricas de {portfolio}", expanded=False):
                fig_vol = px.line(
                    metrics["Volatilidad"],
                    title=f"Volatilidad Rodante - {portfolio}",
                    labels={"value": "Volatilidad Anualizada", "date": "Fecha"}
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                fig_ret = px.line(
                    metrics["Retornos"],
                    title=f"Retornos Rodantes - {portfolio}",
                    labels={"value": "Retorno Acumulado", "date": "Fecha"}
                )
                st.plotly_chart(fig_ret, use_container_width=True)
    
    def _render_scenarios_analysis(self, scenarios_results):
        st.subheader("Análisis de Escenarios Históricos")
        
        for portfolio, scenarios in scenarios_results.items():
            with st.expander(f"Escenarios para {portfolio}", expanded=False):
                scenarios_df = pd.DataFrame(scenarios).T
                st.dataframe(scenarios_df)
                
                if not scenarios_df.empty and "Return" in scenarios_df.columns[0]:
                    returns_by_scenario = {}
                    for scenario, metrics in scenarios.items():
                        if "Return" in metrics:
                            try:
                                returns_by_scenario[scenario] = float(metrics["Return"].strip("%")) / 100
                            except:
                                continue
                    
                    if returns_by_scenario:
                        returns_df = pd.DataFrame(list(returns_by_scenario.items()), 
                                                columns=["Escenario", "Retorno"])
                        fig_scenarios = px.bar(
                            returns_df, 
                            x="Escenario", 
                            y="Retorno",
                            title=f"Retornos por Escenario - {portfolio}",
                            labels={"Retorno": "Retorno Total"}
                        )
                        fig_scenarios.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_scenarios, use_container_width=True)
    
    def _render_monte_carlo_analysis(self, monte_carlo_results, combined):
        st.subheader("Simulación de Monte Carlo")
        
        for portfolio, sim_results in monte_carlo_results.items():
            with st.expander(f"Proyección para {portfolio}", expanded=False):
                mc_df = pd.DataFrame()
                labels = sim_results["percentile_labels"]
                
                for i, percentile in enumerate(labels):
                    mc_df[f"P{percentile}"] = sim_results["percentiles"][i]
                
                last_date = combined.index[-1]
                future_dates = pd.date_range(start=last_date, periods=len(mc_df.index)+1)[1:]
                mc_df.index = future_dates
                
                historical = pd.Series([combined[portfolio].iloc[-1]], index=[last_date])
                full_series = pd.concat([historical, mc_df])
                
                fig_mc = go.Figure()
                
                fig_mc.add_trace(
                    go.Scatter(
                        x=combined.index, 
                        y=combined[portfolio],
                        name="Histórico",
                        line=dict(color="blue", width=2)
                    )
                )
                
                colors = ["rgba(255,0,0,0.3)", "rgba(255,165,0,0.3)", 
                         "rgba(0,128,0,0.4)", "rgba(255,165,0,0.3)", "rgba(255,0,0,0.3)"]
                
                for i, percentile in enumerate(labels):
                    fig_mc.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=sim_results["percentiles"][i],
                            name=f"P{percentile}",
                            line=dict(color=colors[i])
                        )
                    )
                
                fig_mc.update_layout(
                    title=f"Proyección Monte Carlo - {portfolio} ({sim_results['simulations'].shape[1]} simulaciones)",
                    xaxis_title="Fecha",
                    yaxis_title="Valor Proyectado ($)",
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                st.write("#### Probabilidades en el horizonte final")
                final_values = sim_results["simulations"][-1]
                current_value = combined[portfolio].iloc[-1]
                
                prob_gain = ((final_values > current_value).sum() / len(final_values)) * 100
                prob_10pct = ((final_values > current_value * 1.1).sum() / len(final_values)) * 100
                prob_20pct = ((final_values > current_value * 1.2).sum() / len(final_values)) * 100
                prob_loss = ((final_values < current_value).sum() / len(final_values)) * 100
                prob_10pct_loss = ((final_values < current_value * 0.9).sum() / len(final_values)) * 100
                prob_20pct_loss = ((final_values < current_value * 0.8).sum() / len(final_values)) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Prob. de ganancia", f"{prob_gain:.1f}%")
                col2.metric("Prob. de +10%", f"{prob_10pct:.1f}%")
                col3.metric("Prob. de +20%", f"{prob_20pct:.1f}%")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Prob. de pérdida", f"{prob_loss:.1f}%")
                col2.metric("Prob. de -10%", f"{prob_10pct_loss:.1f}%")
                col3.metric("Prob. de -20%", f"{prob_20pct_loss:.1f}%")
    
    def _render_raw_data(self, combined):
        with st.expander("Datos diarios", expanded=True):
            st.dataframe(combined)
            st.markdown(get_download_link(combined, "datos_backtest.csv", "📥 Descargar datos completos (CSV)"), unsafe_allow_html=True) 