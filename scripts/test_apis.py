"""
test_apis.py
Valida conexión y respuesta de Yahoo Finance y FRED API.
Ejecutar antes de construir los scripts de ingesta.
"""
import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()


def test_yahoo_equities():
    print("\n" + "="*55)
    print("TEST 1 — Yahoo Finance: Renta Variable")
    print("="*55)

    tickers = ["SPY", "GLD", "TLT"]

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start="2024-01-01",
                end="2024-01-10",
                auto_adjust=False,
                progress=False,
                multi_level_index=False
            )

            if data.empty:
                print(f"  ⚠️  {ticker}: respuesta vacía")
                continue

            adj   = float(data["Adj Close"].iloc[0])
            vol   = int(data["Volume"].iloc[0])
            fecha = data.index[0].date()

            print(f"  ✅ {ticker}")
            print(f"     Filas: {len(data)} | Fecha inicio: {fecha}")
            print(f"     Adj Close: {adj:.4f} | Volume: {vol:,}")

        except Exception as e:
            print(f"  ❌ {ticker}: {e}")


def test_yahoo_macro():
    print("\n" + "="*55)
    print("TEST 2 — Yahoo Finance: Indicadores Macro")
    print("="*55)

    macro = {
        "^VIX":     "VIX — Índice de volatilidad",
        "DX-Y.NYB": "DXY — Índice del dólar",
        "CL=F":     "WTI — Petróleo Futures",
    }

    for ticker, nombre in macro.items():
        try:
            data = yf.download(
                ticker,
                start="2024-01-01",
                end="2024-01-10",
                auto_adjust=False,
                progress=False,
                multi_level_index=False
            )

            if data.empty:
                print(f"  ⚠️  {ticker}: respuesta vacía")
                continue

            close = float(data["Close"].iloc[-1])
            print(f"  ✅ {ticker} ({nombre})")
            print(f"     Filas: {len(data)} | Último Close: {close:.4f}")

        except Exception as e:
            print(f"  ❌ {ticker}: {e}")


def test_fred():
    print("\n" + "="*55)
    print("TEST 3 — FRED API: Series Macroeconómicas")
    print("="*55)

    api_key = os.getenv("FRED_API_KEY")

    if not api_key or "tu_api" in api_key:
        print("  ❌ FRED_API_KEY no configurada en .env")
        print("     → https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    fred = Fred(api_key=api_key)

    series = {
        "FEDFUNDS": "Tipo interés FED (mensual)",
        "DGS2":     "Bono tesoro 2 años (diario)",
        "DGS10":    "Bono tesoro 10 años (diario)",
        "CPIAUCSL": "IPC EE.UU. Inflación (mensual)",
        "M2SL":     "Oferta monetaria M2 (mensual)",
        "UNRATE":   "Tasa desempleo EE.UU. (mensual)",
        "ICSA":     "Peticiones desempleo semanal",
        "NAPM":     "PMI Manufacturero ISM (mensual)",
    }

    for code, nombre in series.items():
        try:
            data = fred.get_series(
                code,
                observation_start="2024-01-01",
                observation_end="2024-06-01"
            ).dropna()

            if data.empty:
                print(f"  ⚠️  {code}: sin datos en el rango")
                continue

            ultimo_val   = float(data.iloc[-1])
            ultima_fecha = data.index[-1].date()

            print(f"  ✅ {code} — {nombre}")
            print(f"     {len(data)} obs | Último: {ultimo_val:.4f} ({ultima_fecha})")

        except Exception as e:
            print(f"  ❌ {code}: {e}")


if __name__ == "__main__":
    print("🔌 Iniciando validación de APIs externas...")
    test_yahoo_equities()
    test_yahoo_macro()
    test_fred()
    print("\n" + "="*55)
    print("🎯 Validación completada")
    print("="*55)