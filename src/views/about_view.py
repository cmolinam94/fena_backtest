import streamlit as st

class AboutView:
    def render(self):
        st.subheader("Limitaciones y Consideraciones")
        
        st.info("""
        ### Sesgo de Supervivencia
        - Este backtesting utiliza datos actuales, lo que significa que solo incluye compañías que han sobrevivido hasta hoy.
        - Las carteras históricas reales podrían haber incluido empresas que quebraron o fueron adquiridas.
        
        ### Exactitud de los Datos
        - Los datos de Yahoo Finance pueden tener imprecisiones o inconsistencias, especialmente para tickers antiguos.
        - Los ajustes por dividendos y splits están incluidos en los precios "Adj Close", pero pueden no ser perfectos.
        
        ### Slippage y Liquidez
        - El modelo no considera completamente problemas de liquidez o deslizamiento (slippage) en momentos de alta volatilidad.
        - Para activos de baja capitalización, los costos reales de transacción podrían ser mayores.
        
        ### Impuestos
        - Este modelo no considera el impacto de los impuestos, que pueden afectar significativamente los rendimientos reales.
        
        ### Otros Factores
        - No se consideran factores como cambios regulatorios, eventos geopolíticos o cambios estructurales en los mercados.
        """) 