"""
Aplicación principal para Backtest de Carteras.
Este es el punto de entrada de la aplicación Streamlit.
"""
import streamlit as st
from src.controllers.app_controller import AppController

def main():
    """Función principal que inicializa la aplicación Streamlit."""
    st.set_page_config(page_title="Backtest de Carteras", layout="wide")
    
    controller = AppController()
    
    controller.run()

if __name__ == "__main__":
    main() 