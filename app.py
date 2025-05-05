import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import io
import base64

st.set_page_config(page_title="Backtest de Carteras", layout="wide")

# ------------- Funciones de negocio mejoradas ----------------
@st.cache_data(show_spinner=False)
def download_daily_data(tickers, start_date, end_date, max_retries=3, retry_delay=2):
    """Descarga datos con reintentos y validación mejorada"""
    for attempt in range(max_retries):
        try:
            # Ampliar el rango de fechas para evitar problemas con días no comerciales
            adjusted_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
            adjusted_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
            
            data = yf.download(tickers, start=adjusted_start, end=adjusted_end, interval="1d", progress=False)
            
            if data.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError(f"No se pudieron obtener datos para los tickers: {tickers}")
            
            # Verificar qué columnas están disponibles
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            adj_close = data[col]
            
            # Convertir a DataFrame si es una Serie
            if isinstance(adj_close, pd.Series):
                adj_close = adj_close.to_frame()
            
            # Recortar al rango de fechas original
            adj_close = adj_close.loc[start_date:end_date]
            
            # Verificar datos faltantes por ticker
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
    """Valida que los tickers existan y tengan datos recientes"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=test_period_days)).strftime("%Y-%m-%d")
    
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 5:  # Al menos 5 días de trading
                invalid_tickers.append(ticker)
            else:
                valid_tickers.append(ticker)
        except:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

def rebalance_portfolio(weights, value, prices, prev_shares=None, tx_cost=0.0):
    """Rebalancea el portafolio considerando costos de transacción"""
    w = np.array(weights)
    alloc = value * w
    new_shares = {t: a / prices[t] for t, a in zip(prices.index, alloc)}
    
    # Calcular costos de transacción si hay acciones previas
    tx_costs = 0.0
    if prev_shares and tx_cost > 0:
        for t in new_shares:
            if t in prev_shares:
                shares_diff = abs(new_shares[t] - prev_shares[t])
                tx_costs += shares_diff * prices[t] * tx_cost
    
    # Ajustar las nuevas acciones si hay costos de transacción
    if tx_costs > 0:
        remaining_value = value - tx_costs
        alloc = remaining_value * w
        new_shares = {t: a / prices[t] for t, a in zip(prices.index, alloc)}
    
    return new_shares, tx_costs

def portfolio_value_from_shares(shares, prices):
    """Calcula el valor del portafolio a partir de las acciones"""
    return sum(n * prices[t] for t, n in shares.items())

def calculate_advanced_metrics(returns, benchmark_returns=None, rf=0.0):
    """Calcula métricas avanzadas de rendimiento y riesgo"""
    if returns.empty:
        return {}
    
    # Métricas básicas
    total_return = returns.iloc[-1] / returns.iloc[0] - 1
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1/years) - 1 if years else np.nan
    
    # Cálculos diarios
    daily_returns = returns.pct_change().dropna()
    excess_returns = daily_returns - (((1 + rf) ** (1/252)) - 1)
    
    # Volatilidad
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Drawdown
    drawdown = (returns / returns.cummax() - 1).min()
    
    # Ratio de Sharpe
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else np.nan
    
    # Ratio de Sortino (penaliza sólo los retornos negativos)
    downside_returns = daily_returns[daily_returns < 0]
    sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if not downside_returns.empty and downside_returns.std() > 0 else np.nan
    
    # Métricas en relación al benchmark
    beta = alpha = r_squared = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_daily_returns = benchmark_returns.pct_change().dropna()
        # Alinear los índices
        aligned_returns = pd.concat([daily_returns, bench_daily_returns], axis=1).dropna()
        if len(aligned_returns) > 0 and aligned_returns.iloc[:, 1].std() > 0:
            # Beta y Alpha
            cov_matrix = aligned_returns.cov()
            beta = cov_matrix.iloc[0, 1] / aligned_returns.iloc[:, 1].var() if aligned_returns.iloc[:, 1].var() > 0 else np.nan
            
            # Alpha anualizado (Jensen's Alpha)
            port_excess = (aligned_returns.iloc[:, 0] - (rf/252)).mean() * 252
            market_excess = (aligned_returns.iloc[:, 1] - (rf/252)).mean() * 252
            alpha = port_excess - beta * market_excess
            
            # R² (Coeficiente de determinación)
            corr = aligned_returns.corr().iloc[0, 1]
            r_squared = corr**2 if not np.isnan(corr) else np.nan
    
    # Calmar Ratio (Return / Max DD)
    calmar = abs(cagr / drawdown) if drawdown < 0 else np.nan
    
    # Retornos anualizados
    ann_return = daily_returns.mean() * 252
    
    # Resultados
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
    """Calcula métricas rodantes para análisis de riesgo"""
    daily_returns = series.pct_change().dropna()
    
    # Volatilidad rodante (21 días ≈ 1 mes de trading)
    rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)
    
    # Retornos acumulados rodantes
    rolling_returns = (1 + daily_returns).rolling(window=window).apply(lambda x: np.prod(x) - 1)
    
    # Drawdown
    rolling_dd = series / series.rolling(window=window).max() - 1
    
    return pd.DataFrame({
        "Volatilidad": rolling_vol,
        "Retornos": rolling_returns,
        "Drawdown": rolling_dd
    }, index=series.index).dropna()

def perf(series, benchmark=None, rf=0.0):
    """Función de compatibilidad con el código original"""
    if series.empty:
        return {"CAGR":"–", "Return":"–", "Máx DD":"–", "Sharpe":"–"}
    
    return calculate_advanced_metrics(series, benchmark, rf)

def get_download_link(df, filename, text):
    """Genera un link para descargar un DataFrame como CSV"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def backtest_portfolio(tickers, weights, start_year, end_year,
                      initial_amount, monthly, reb, bench, 
                      tx_cost=0.0, reb_threshold=0.0):
    """
    Backtesting de cartera mejorado con:
    - Costos de transacción
    - Umbral de rebalanceo
    - Mejor manejo de errores
    - Seguimiento de costos
    """
    start, end = f"{start_year}-01-01", f"{end_year}-12-31"
    
    # Validar tickers antes del backtest
    all_symbols = list(tickers) + [bench]
    valid_tickers, invalid_tickers = validate_tickers(all_symbols)
    
    if invalid_tickers:
        st.warning(f"Los siguientes tickers pueden no ser válidos: {invalid_tickers}")
        if bench in invalid_tickers:
            raise ValueError(f"El benchmark {bench} no es válido o no tiene datos recientes.")
    
    # Descargar datos con mejoras
    data = download_daily_data(all_symbols, start, end)
    
    if bench not in data.columns:
        raise ValueError(f"El benchmark {bench} no tiene datos para el período seleccionado.")
    
    bench_px, prices = data[bench], data.drop(columns=[bench])
    
    if prices.empty:
        raise ValueError("No hay datos suficientes para los tickers seleccionados.")
    
    # Inicializar variables
    pv, bv = initial_amount, initial_amount
    shares = None
    p_vals, b_vals = [], []
    pm, py = None, None
    tx_costs_total = 0.0
    tx_costs_hist = []
    reb_dates = []
    
    for i, d in enumerate(prices.index):
        # Rebalanceo inicial o programado
        rebalance_now = False
        
        if i == 0:
            rebalance_now = True
        elif reb == "annual" and d.year != py:
            rebalance_now = True
        elif reb == "monthly" and d.month != pm:
            rebalance_now = True
        
        # Comprobar si es necesario rebalancear debido a desviación
        if not rebalance_now and reb_threshold > 0 and shares:
            # Calcular pesos actuales
            current_values = {t: shares[t] * prices.loc[d, t] for t in shares}
            total_value = sum(current_values.values())
            current_weights = {t: v/total_value for t, v in current_values.items()}
            
            # Comprobar desviación máxima
            max_deviation = max(abs(current_weights[t] - w) for t, w in zip(current_weights, weights))
            if max_deviation > reb_threshold:
                rebalance_now = True
        
        # Realizar el rebalanceo si es necesario
        if rebalance_now:
            if i > 0:  # No es el primer día
                reb_dates.append(d)
            shares, tx_cost = rebalance_portfolio(weights, pv, prices.loc[d], shares, tx_cost)
            tx_costs_total += tx_cost
        
        # Calcular valor del portafolio
        pv = portfolio_value_from_shares(shares, prices.loc[d])
        
        # Aportes mensuales
        if i > 0 and d.month != pm:
            pv += monthly
            if reb != "none":
                shares, tx_cost = rebalance_portfolio(weights, pv, prices.loc[d], shares, tx_cost)
                tx_costs_total += tx_cost
                reb_dates.append(d)
        
        # Benchmark
        if i > 0:
            br = bench_px.loc[d] / bench_px.iloc[i-1] - 1
            bv = bv * (1 + br) + (monthly if d.month != pm else 0)
        
        # Guardar valores
        p_vals.append(pv)
        b_vals.append(bv)
        tx_costs_hist.append(tx_costs_total)
        
        # Actualizar mes y año previos
        pm, py = d.month, d.year
    
    # Crear DataFrame con resultados
    results = pd.DataFrame({
        "Portfolio": p_vals, 
        "Benchmark": b_vals,
        "TX_Costs": tx_costs_hist
    }, index=prices.index)
    
    # Añadir información de rebalanceo
    if reb_dates:
        reb_info = pd.Series(1, index=reb_dates)
        results["Rebalance"] = 0
        results.loc[reb_info.index, "Rebalance"] = 1
    
    return results

def run_stress_test(portfolio_config, stress_scenarios, start_year, end_year, bench):
    """Ejecuta pruebas de estrés basadas en escenarios históricos"""
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
            # Calcular métricas para este escenario
            metrics = calculate_advanced_metrics(result["Portfolio"], result["Benchmark"])
            results[scenario_name] = metrics
        except Exception as e:
            results[scenario_name] = {"Error": str(e)}
    
    return results

def monte_carlo_simulation(returns, initial_value, horizon_days, simulations=1000, percentiles=[5, 25, 50, 75, 95]):
    """Realiza simulación de Monte Carlo para proyectar posibles resultados futuros"""
    # Parámetros de la distribución log-normal
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulaciones
    simulation_results = np.zeros((horizon_days, simulations))
    simulation_results[0] = initial_value
    
    # Generar caminos aleatorios
    for i in range(1, horizon_days):
        random_returns = np.random.normal(mu, sigma, simulations)
        simulation_results[i] = simulation_results[i-1] * (1 + random_returns)
    
    # Calcular percentiles
    percentile_results = np.percentile(simulation_results, percentiles, axis=1)
    
    return {
        "simulations": simulation_results,
        "percentiles": percentile_results,
        "percentile_labels": percentiles
    }

# ──────────── Interfaz ────────────
st.title("📈 Back-testing de Múltiples Carteras")

# Tabs para organizar la interfaz
tabs = st.tabs(["Configuración", "Avanzado", "Acerca del Backtesting"])

with tabs[0]:  # Tab Configuración
    # Parámetros globales
    st.subheader("Parámetros Globales")
    gl_c1, gl_c2 = st.columns([2,1])
    with gl_c1:
        n_port = st.number_input("Número de carteras a comparar", 1, 6, 2, step=1)
        bench = st.text_input("Ticker Benchmark", "^GSPC")
    with gl_c2:
        y0 = st.number_input("Año inicio", 1980, 2025, 2010)
        y1 = st.number_input("Año fin", y0, 2025, 2024)
        rf = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 20.0, 0.0) / 100

    # Controles por cartera
    portfolios_cfg = {}
    for idx in range(int(n_port)):
        with st.expander(f"Cartera {idx+1}", expanded=(idx==0)):
            col1, col2 = st.columns(2)
            with col1:
                tk = st.text_input("Tickers (coma)", value="SPY,QQQ,TLT", key=f"tk{idx}")
                wt = st.text_input("Pesos (coma)", value="0.4,0.4,0.2", key=f"wt{idx}")
                init = st.number_input("Monto inicial", 0, 10_000_000, 10_000,
                                      step=1000, key=f"init{idx}")
            with col2:
                mon = st.number_input("Aporte mensual", 0, 1_000_000, 0,
                                    step=100, key=f"mon{idx}")
                reb = st.selectbox("Rebalanceo", ["none", "annual", "monthly"],
                                 index=1, key=f"reb{idx}")
                tx_cost = st.number_input("Costo de transacción (%)", 0.0, 5.0, 0.1, 
                                         step=0.05, key=f"tx{idx}") / 100
                reb_threshold = st.number_input("Umbral de rebalanceo", 0.0, 0.5, 0.0, 
                                              step=0.01, key=f"thresh{idx}")
                
            portfolios_cfg[f"Cartera_{idx+1}"] = {
                "tickers": [t.strip().upper() for t in tk.split(",") if t.strip()],
                "weights": [float(w) for w in wt.split(",") if w.strip()],
                "initial_amount": init,
                "monthly": mon,
                "reb": reb,
                "tx_cost": tx_cost,
                "reb_threshold": reb_threshold
            }

with tabs[1]:  # Tab Avanzado
    st.subheader("Opciones Avanzadas")
    
    # Análisis de escenarios históricos
    with st.expander("Análisis de Escenarios", expanded=False):
        st.write("Ejecute su estrategia en diferentes escenarios históricos")
        run_scenarios = st.checkbox("Analizar escenarios históricos", value=False)
        
        # Escenarios predefinidos (pueden personalizarse)
        historical_scenarios = {
            "Crisis 2008": {"start_year": 2007, "end_year": 2009, "description": "Crisis financiera global"},
            "Burbuja Tecnológica": {"start_year": 1999, "end_year": 2002, "description": "Burbuja de las puntocom"},
            "Crisis COVID": {"start_year": 2020, "end_year": 2020, "description": "Pánico por el COVID-19"},
            "Recuperación 2009-2010": {"start_year": 2009, "end_year": 2010, "description": "Recuperación post-crisis"},
            "Subida Tipos 2022": {"start_year": 2021, "end_year": 2022, "description": "Políticas restrictivas de bancos centrales"}
        }
        
        selected_scenarios = []
        cols = st.columns(3)
        for i, (name, details) in enumerate(historical_scenarios.items()):
            col_idx = i % 3
            with cols[col_idx]:
                if st.checkbox(f"{name} ({details['start_year']}-{details['end_year']})", value=True):
                    selected_scenarios.append(name)
    
    # Simulación de Monte Carlo
    with st.expander("Simulación de Monte Carlo", expanded=False):
        st.write("Proyección de posibles resultados futuros")
        run_monte_carlo = st.checkbox("Realizar simulación de Monte Carlo", value=False)
        mc_days = st.slider("Horizonte de proyección (días)", 21, 504, 252)
        mc_sims = st.slider("Número de simulaciones", 100, 5000, 1000)

    # Análisis de métricas rodantes
    with st.expander("Análisis de Métricas Rodantes", expanded=False):
        st.write("Seguimiento de métricas a lo largo del tiempo")
        run_rolling = st.checkbox("Calcular métricas rodantes", value=False)
        rolling_window = st.select_slider("Ventana de cálculo (días)", 
                                         options=[21, 63, 126, 252], value=63)

with tabs[2]:  # Tab Acerca del Backtesting
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

# Botón principal
col1, col2 = st.columns([3, 1])
with col1:
    run_backtest = st.button("🎬 Ejecutar Back-test", type="primary", use_container_width=True)
with col2:
    show_raw_data = st.checkbox("Mostrar datos crudos", value=False)

if run_backtest:
    # Validaciones rápidas
    for name, cfg in portfolios_cfg.items():
        if len(cfg["tickers"]) != len(cfg["weights"]):
            st.error(f"{name}: número de tickers ≠ número de pesos.")
            st.stop()
        if abs(sum(cfg["weights"]) - 1.0) > 1e-6:
            st.error(f"{name}: los pesos no suman 1.0.")
            st.stop()

    # Crear un contenedor para mostrar el progreso
    progress_container = st.empty()
    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
    # Back-tests
    combined = pd.DataFrame()
    metrics = {}
    
    try:
        # Ejecutar backtests para cada cartera
        for i, (name, cfg) in enumerate(portfolios_cfg.items()):
            status_text.text(f"Procesando {name}...")
            progress_bar.progress((i / len(portfolios_cfg)) / 2)  # Primera mitad para backtests principales
            
            res = backtest_portfolio(
                cfg["tickers"], cfg["weights"],
                y0, y1, cfg["initial_amount"],
                cfg["monthly"], cfg["reb"], bench,
                cfg["tx_cost"], cfg["reb_threshold"]
            )
            
            combined[name] = res["Portfolio"]
            if "Benchmark" not in combined.columns:
                combined["Benchmark"] = res["Benchmark"]
                
            # Calcular métricas de rendimiento con mejoras
            metrics[name] = calculate_advanced_metrics(res["Portfolio"], res["Benchmark"], rf)
    except ValueError as e:
        progress_container.empty()
        st.error(str(e))
        st.stop()

    if combined.empty:
        progress_container.empty()
        st.error("Sin datos que mostrar.")
        st.stop()
    
    # Métricas resumen
    progress_bar.progress(0.5)  # 50% completado
    status_text.text("Calculando métricas adicionales...")
    
    # Convertir métricas a DataFrame y mostrar
    metrics_df = pd.DataFrame(metrics).T

    # Escenarios históricos
    scenarios_results = {}
    if 'run_scenarios' in locals() and run_scenarios and selected_scenarios:
        status_text.text("Analizando escenarios históricos...")
        progress_bar.progress(0.7)  # 70% completado
        
        for name, cfg in portfolios_cfg.items():
            selected_historical_scenarios = {k: historical_scenarios[k] for k in selected_scenarios}
            scenarios_results[name] = run_stress_test(
                cfg, selected_historical_scenarios, y0, y1, bench
            )
    
    # Métricas rodantes
    rolling_metrics = {}
    if 'run_rolling' in locals() and run_rolling:
        status_text.text("Calculando métricas rodantes...")
        progress_bar.progress(0.8)  # 80% completado
        
        for col in combined.columns:
            if col != "Benchmark":
                rolling_metrics[col] = calculate_rolling_metrics(combined[col], rolling_window)
    
    # Monte Carlo
    monte_carlo_results = {}
    if 'run_monte_carlo' in locals() and run_monte_carlo:
        status_text.text("Ejecutando simulación de Monte Carlo...")
        progress_bar.progress(0.9)  # 90% completado
        
        for col in combined.columns:
            if col != "Benchmark":
                daily_returns = combined[col].pct_change().dropna()
                last_value = combined[col].iloc[-1]
                monte_carlo_results[col] = monte_carlo_simulation(
                    daily_returns, last_value, mc_days, mc_sims
                )
    
    # Finalizar progreso
    progress_container.empty()
    
    # Mostrar resultados
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
    
    # Opción para descargar resultados
    st.markdown(get_download_link(metrics_df, "metricas_carteras.csv", "📥 Descargar métricas completas (CSV)"), unsafe_allow_html=True)
    
    # Gráfico evolutivo
    st.subheader("Evolución del Valor")
    
    # Crear gráfico interactivo mejorado
    fig = px.line(
        combined, 
        title="Valor de las carteras", 
        labels={"value": "Valor ($)", "variable": "Cartera", "date": "Fecha"}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de Drawdown
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
    # Añadir líneas horizontales para referencia
    fig_dd.add_hline(y=-0.1, line_dash="dash", line_color="yellow", annotation_text="-10%")
    fig_dd.add_hline(y=-0.2, line_dash="dash", line_color="orange", annotation_text="-20%")
    fig_dd.add_hline(y=-0.3, line_dash="dash", line_color="red", annotation_text="-30%")
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Mostrar métricas rodantes si fueron calculadas
    if 'run_rolling' in locals() and run_rolling and rolling_metrics:
        st.subheader(f"Métricas Rodantes ({rolling_window} días)")
        
        for portfolio, metrics in rolling_metrics.items():
            with st.expander(f"Métricas de {portfolio}", expanded=False):
                # Gráficos de volatilidad rodante
                fig_vol = px.line(
                    metrics["Volatilidad"],
                    title=f"Volatilidad Rodante - {portfolio}",
                    labels={"value": "Volatilidad Anualizada", "date": "Fecha"}
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Gráficos de retornos rodantes
                fig_ret = px.line(
                    metrics["Retornos"],
                    title=f"Retornos Rodantes - {portfolio}",
                    labels={"value": "Retorno Acumulado", "date": "Fecha"}
                )
                st.plotly_chart(fig_ret, use_container_width=True)
    
    # Mostrar resultados de escenarios históricos
    if 'run_scenarios' in locals() and run_scenarios and scenarios_results:
        st.subheader("Análisis de Escenarios Históricos")
        
        for portfolio, scenarios in scenarios_results.items():
            with st.expander(f"Escenarios para {portfolio}", expanded=False):
                # Convertir a DataFrame para mejor visualización
                scenarios_df = pd.DataFrame(scenarios).T
                st.dataframe(scenarios_df)
                
                # Gráfico comparativo de retornos en escenarios
                if not scenarios_df.empty and "Return" in scenarios_df.columns[0]:
                    returns_by_scenario = {}
                    for scenario, metrics in scenarios.items():
                        if "Return" in metrics:
                            # Convertir de string a número quitando el símbolo %
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
    
    # Mostrar resultados de Monte Carlo
    if 'run_monte_carlo' in locals() and run_monte_carlo and monte_carlo_results:
        st.subheader("Simulación de Monte Carlo")
        
        for portfolio, sim_results in monte_carlo_results.items():
            with st.expander(f"Proyección para {portfolio}", expanded=False):
                # Crear un DataFrame con los percentiles
                mc_df = pd.DataFrame()
                labels = sim_results["percentile_labels"]
                
                for i, percentile in enumerate(labels):
                    mc_df[f"P{percentile}"] = sim_results["percentiles"][i]
                
                # Añadir fechas proyectadas
                last_date = combined.index[-1]
                future_dates = pd.date_range(start=last_date, periods=mc_days+1)[1:]
                mc_df.index = future_dates
                
                # Añadir el último valor conocido como punto inicial
                historical = pd.Series([combined[portfolio].iloc[-1]], index=[last_date])
                full_series = pd.concat([historical, mc_df])
                
                # Graficar
                fig_mc = go.Figure()
                
                # Añadir histórico
                fig_mc.add_trace(
                    go.Scatter(
                        x=combined.index, 
                        y=combined[portfolio],
                        name="Histórico",
                        line=dict(color="blue", width=2)
                    )
                )
                
                # Añadir líneas para cada percentil
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
                
                # Personalizar el gráfico
                fig_mc.update_layout(
                    title=f"Proyección Monte Carlo - {portfolio} ({mc_sims} simulaciones)",
                    xaxis_title="Fecha",
                    yaxis_title="Valor Proyectado ($)",
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Probabilidades de alcanzar ciertos objetivos
                st.write("#### Probabilidades en el horizonte final")
                final_values = sim_results["simulations"][-1]
                current_value = combined[portfolio].iloc[-1]
                
                prob_gain = np.mean(final_values > current_value) * 100
                prob_10pct = np.mean(final_values > current_value * 1.1) * 100
                prob_20pct = np.mean(final_values > current_value * 1.2) * 100
                prob_loss = np.mean(final_values < current_value) * 100
                prob_10pct_loss = np.mean(final_values < current_value * 0.9) * 100
                prob_20pct_loss = np.mean(final_values < current_value * 0.8) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Prob. de ganancia", f"{prob_gain:.1f}%")
                col2.metric("Prob. de +10%", f"{prob_10pct:.1f}%")
                col3.metric("Prob. de +20%", f"{prob_20pct:.1f}%")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Prob. de pérdida", f"{prob_loss:.1f}%")
                col2.metric("Prob. de -10%", f"{prob_10pct_loss:.1f}%")
                col3.metric("Prob. de -20%", f"{prob_20pct_loss:.1f}%")
    
    # Tabla de datos crudos
    if show_raw_data:
        with st.expander("Datos diarios", expanded=True):
            st.dataframe(combined)
            st.markdown(get_download_link(combined, "datos_backtest.csv", "📥 Descargar datos completos (CSV)"), unsafe_allow_html=True) 