import streamlit as st

class ConfigurationView:
    def render(self):
        st.subheader("Parámetros Globales")
        gl_c1, gl_c2 = st.columns([2,1])
        
        with gl_c1:
            n_port = st.number_input("Número de carteras a comparar", 1, 6, 2, step=1, key="n_port")
            bench = st.text_input("Ticker Benchmark", "^GSPC", key="bench")
        
        with gl_c2:
            y0 = st.number_input("Año inicio", 1980, 2025, 2010, key="y0")
            y1 = st.number_input("Año fin", y0, 2025, 2024, key="y1")
            rf = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 20.0, 0.0, key="rf") / 100
        
        portfolios_cfg = {}
        for idx in range(int(n_port)):
            with st.expander(f"Cartera {idx+1}", expanded=(idx==0)):
                col1, col2 = st.columns(2)
                with col1:
                    tk = st.text_input("Tickers (coma)", value="SPY,QQQ,TLT", key=f"tk{idx}")
                    wt = st.text_input("Pesos (coma)", value="0.4,0.4,0.2", key=f"wt{idx}")
                    init = st.number_input("Monto inicial", 0, 10_000_000, 10_000, step=1000, key=f"init{idx}")
                with col2:
                    mon = st.number_input("Aporte mensual", 0, 1_000_000, 0, step=100, key=f"mon{idx}")
                    reb = st.selectbox("Rebalanceo", ["none", "annual", "monthly"], index=1, key=f"reb{idx}")
                    tx_cost = st.number_input("Costo de transacción (%)", 0.0, 5.0, 0.1, step=0.05, key=f"tx{idx}") / 100
                    reb_threshold = st.number_input("Umbral de rebalanceo", 0.0, 0.5, 0.0, step=0.01, key=f"thresh{idx}")
                
                portfolios_cfg[f"Cartera_{idx+1}"] = {
                    "tickers": [t.strip().upper() for t in tk.split(",") if t.strip()],
                    "weights": [float(w) for w in wt.split(",") if w.strip()],
                    "initial_amount": init,
                    "monthly": mon,
                    "reb": reb,
                    "tx_cost": tx_cost,
                    "reb_threshold": reb_threshold
                }
        
        st.session_state.portfolios_cfg = portfolios_cfg
        st.session_state.global_params = {
            "n_port": n_port,
            "bench": bench,
            "y0": y0,
            "y1": y1,
            "rf": rf
        }
        
        return {
            "portfolios_cfg": portfolios_cfg,
            "global_params": st.session_state.global_params
        } 