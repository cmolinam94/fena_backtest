# Backtesting de Carteras

Esta aplicación permite realizar backtesting de múltiples carteras de inversión, comparando su rendimiento contra un benchmark. Incluye análisis avanzados, visualizaciones interactivas y escenarios de estrés.

## Características

- Análisis de hasta 6 carteras simultáneamente
- Parámetros configurables:
  - Tickers y pesos por cartera
  - Monto inicial
  - Aportes mensuales
  - Estrategia de rebalanceo (mensual, anual, ninguno)
  - Costos de transacción
  - Umbral de rebalanceo (% de desviación para rebalancear)

### Métricas de rendimiento avanzadas
- CAGR (Tasa de Crecimiento Anual Compuesta)
- Retorno total y anualizado
- Volatilidad
- Máximo drawdown
- Ratio de Sharpe
- Ratio de Sortino (penaliza solo retornos negativos)
- Beta y Alpha
- R²
- Ratio de Calmar

### Análisis avanzados
- Simulación de Monte Carlo
- Análisis de escenarios históricos (Crisis 2008, Burbuja Tecnológica, etc.)
- Métricas rodantes (volatilidad, retornos, drawdown)
- Visualización de drawdowns
- Probabilidades en proyecciones futuras

### Interfaz mejorada
- Visualizaciones interactivas con Plotly
- Exportación de datos a CSV
- Información sobre limitaciones del backtesting
- Barra de progreso durante cálculos intensivos
- Interfaz organizada en pestañas

## Instalación

1. Clonar este repositorio
2. Crear un entorno virtual:
   ```
   python -m venv venv
   ```
3. Activar el entorno virtual:
   - En Windows:
     ```
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```
     source venv/bin/activate
     ```
4. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

Para ejecutar la aplicación (con el entorno virtual activado):

```
streamlit run app.py
```

## Guía de uso

### Configuración básica
1. Define el número de carteras a comparar (1-6)
2. Establece el benchmark (por defecto: ^GSPC - S&P 500)
3. Configura el rango de tiempo para el análisis
4. Para cada cartera:
   - Ingresa tickers separados por comas (ej: "SPY,QQQ,TLT")
   - Define pesos para cada ticker (ej: "0.4,0.4,0.2")
   - Configura monto inicial de inversión
   - Establece aporte mensual (opcional)
   - Selecciona estrategia de rebalanceo
   - Configura costos de transacción y umbral de rebalanceo

### Análisis avanzados
- **Escenarios históricos**: Verifica cómo habría funcionado tu cartera en eventos como la Crisis de 2008
- **Monte Carlo**: Proyecta posibles resultados futuros con miles de simulaciones
- **Métricas rodantes**: Observa cómo cambian métricas como volatilidad y drawdown a lo largo del tiempo

### Interpretación de resultados
- **Métricas de rendimiento**: Compara CAGR, Sharpe, Sortino, etc. entre carteras
- **Gráficos de evolución**: Visualiza el crecimiento del valor de cada cartera
- **Drawdowns**: Analiza las caídas desde máximos y períodos de recuperación
- **Probabilidades de Monte Carlo**: Evalúa las probabilidades de diferentes resultados en el futuro

## Limitaciones
- Sesgo de supervivencia
- Precisión de datos históricos
- No considera completamente la liquidez y slippage
- No incluye impuestos
- Simplificación de eventos corporativos

## Mejoras futuras
- Optimización de carteras
- Backtesting con datos intradiarios
- Inclusión de más clases de activos
- Análisis de correlaciones
- Implementación de más estrategias de trading 