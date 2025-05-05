import streamlit as st
from src.views.configuration_view import ConfigurationView
from src.views.advanced_view import AdvancedView
from src.views.about_view import AboutView
from src.views.results_view import ResultsView

class MainView:
    def __init__(self):
        self.config_view = ConfigurationView()
        self.advanced_view = AdvancedView()
        self.about_view = AboutView()
        self.results_view = ResultsView()
    
    def render(self):
        st.title("📈 Back-testing de Múltiples Carteras")
        
        tabs = st.tabs(["Configuración", "Avanzado", "Acerca del Backtesting"])
        
        with tabs[0]:
            self.config_view.render()
        
        with tabs[1]:
            self.advanced_view.render()
        
        with tabs[2]:
            self.about_view.render()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            run_backtest = st.button("🎬 Ejecutar Back-test", type="primary", use_container_width=True)
        with col2:
            show_raw_data = st.checkbox("Mostrar datos crudos", value=False)
        
        return {
            "run_backtest": run_backtest,
            "show_raw_data": show_raw_data
        }
    
    def display_results(self, results, config, options):
        self.results_view.render(results, config, options) 