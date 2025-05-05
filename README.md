# Backtesting de Carteras

Esta aplicación permite realizar backtesting de múltiples carteras de inversión, comparando su rendimiento contra un benchmark.

## Características

- Análisis de hasta 6 carteras simultáneamente
- Parámetros configurables:
  - Tickers y pesos por cartera
  - Monto inicial
  - Aportes mensuales
  - Estrategia de rebalanceo (mensual, anual, ninguno)
- Métricas de rendimiento:
  - CAGR (Tasa de Crecimiento Anual Compuesta)
  - Retorno total
  - Máximo drawdown
  - Ratio de Sharpe
- Gráfico interactivo de evolución de carteras

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

## Ejemplo de uso

1. Configura el número de carteras a comparar (1-6)
2. Define el benchmark (por defecto: ^GSPC - S&P 500)
3. Establece el rango de tiempo para el análisis
4. Para cada cartera, configura:
   - Tickers separados por comas (ej: "SPY,QQQ,TLT")
   - Pesos para cada ticker (ej: "0.4,0.4,0.2")
   - Monto inicial de inversión
   - Aporte mensual (opcional)
   - Estrategia de rebalanceo
5. Haz clic en "Ejecutar Back-test" para visualizar los resultados 