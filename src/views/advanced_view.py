import streamlit as st
from src.utils.helpers import get_historical_scenarios

class AdvancedView:
    def render(self):
        st.subheader("Opciones Avanzadas")
        
        with st.expander("Análisis de Escenarios", expanded=False):
            st.write("Ejecute su estrategia en diferentes escenarios históricos")
            run_scenarios = st.checkbox("Analizar escenarios históricos", value=False, key="run_scenarios")
            
            historical_scenarios = get_historical_scenarios()
            
            selected_scenarios = []
            cols = st.columns(3)
            for i, (name, details) in enumerate(historical_scenarios.items()):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.checkbox(f"{name} ({details['start_year']}-{details['end_year']})", value=True, key=f"scenario_{name}"):
                        selected_scenarios.append(name)
        
        with st.expander("Simulación de Monte Carlo", expanded=False):
            st.write("Proyección de posibles resultados futuros")
            run_monte_carlo = st.checkbox("Realizar simulación de Monte Carlo", value=False, key="run_monte_carlo")
            mc_days = st.slider("Horizonte de proyección (días)", 21, 504, 252, key="mc_days")
            mc_sims = st.slider("Número de simulaciones", 100, 5000, 1000, key="mc_sims")
        
        with st.expander("Análisis de Métricas Rodantes", expanded=False):
            st.write("Seguimiento de métricas a lo largo del tiempo")
            run_rolling = st.checkbox("Calcular métricas rodantes", value=False, key="run_rolling")
            rolling_window = st.select_slider("Ventana de cálculo (días)", 
                                             options=[21, 63, 126, 252], value=63, key="rolling_window")
        
        st.session_state.advanced_options = {
            "run_scenarios": run_scenarios,
            "selected_scenarios": selected_scenarios,
            "historical_scenarios": historical_scenarios,
            "run_monte_carlo": run_monte_carlo,
            "mc_days": mc_days,
            "mc_sims": mc_sims,
            "run_rolling": run_rolling,
            "rolling_window": rolling_window
        }
        
        return st.session_state.advanced_options 