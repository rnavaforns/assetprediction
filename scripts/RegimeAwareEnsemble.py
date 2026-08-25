import numpy as np
import pandas as pd

class RegimeAwareEnsemble:

  def __init__(self, tft_weight_floor=0.1, cat_weight_floor=0.1):
    self.tft_floor = tft_weight_floor
    self.cat_floor = cat_weight_floor

  def _get_regime_weights(self, row: pd.Series) -> tuple[float, float]:
    """Calcula los pesos dinámicos basándose en los umbrales empíricos extraídos del árbol exploratorio de regímenes y comportamiento de VIX.

    """
    vix = row.get('regime_vix_ma_5', row.get('regime_vix', 15.0))
    spy_60d = row.get('regime_spy_return_60d', 0.0)
    spy_sma50_dist = row.get('regime_spy_distance_sma50', 0.0)

    # 1. Régimen de Estrés / Rotura a la baja (VIX alto + S&P500 bajo su media de 50)
    if vix > 20.4 and spy_sma50_dist <= 0.0043:
      w_tft = 0.80

    # 2. Régimen de Rebote de Volatilidad (VIX alto + S&P500 sobre su media de 50)
    elif vix > 20.4 and spy_sma50_dist > 0.0043:
      w_tft = 0.50

    # 3. Sobreventa Profunda / Suelo de Mercado (SPY 60d <= -6.75%)
    elif spy_60d <= -0.0675:
      w_tft = 0.30

    # 4. Inercia Alcista Fuerte (SPY 60d > +6.97%)
    elif spy_60d > 0.0697:
      w_tft = 0.60

    # 5. Mercado Estable / Baja Volatilidad (Zona de confort de CatBoost)
    else:
      w_tft = 0.30

    # Garantizar límites mínimos de diversidad
    w_tft = np.clip(w_tft, self.tft_floor, 1.0 - self.cat_floor)
    w_cat = 1.0 - w_tft

    return w_tft, w_cat

  def predict(self, df_eval: pd.DataFrame) -> pd.DataFrame:
    """Recibe un DataFrame con las columnas:

    - 'pred_tft' - 'pred_catboost' - Variables de régimen ('regime_vix_ma_5',
    'regime_spy_return_60d', 'regime_spy_distance_sma50')

    Retorna el DataFrame enriquecido con los pesos y la predicción final del
    ensamble.
    """
    df = df_eval.copy()

    # 1. Obtener pesos por fila segun régimen
    weights = df.apply(self._get_regime_weights, axis=1)
    df['w_tft'] = [w[0] for w in weights]
    df['w_cat'] = [w[1] for w in weights]

    # 2. Ponderación base
    df['pred_ensemble_raw'] = (
        df['w_tft'] * df['pred_tft'] + df['w_cat'] * df['pred_catboost']
    )

    # 3. Asimetría Direccional:
    # CatBoost no es fiable en señales negativas. Si la predicción base apunta a caída,
    # otorgamos peso dominante a TFT (0.85) o requerimos doble confirmación.
    is_negative_signal = df['pred_ensemble_raw'] < 0

    df['w_tft_final'] = np.where(is_negative_signal, 0.85, df['w_tft'])
    df['w_cat_final'] = 1.0 - df['w_tft_final']

    df['pred_ensemble'] = (
        df['w_tft_final'] * df['pred_tft']
        + df['w_cat_final'] * df['pred_catboost']
    )

    return df

# ==========================================
# EJEMPLO DE USO E INTEGRACIÓN EN EL PIPELINE
# ==========================================
if __name__ == '__main__':
  # Simulación de un dataset de predicciones out-of-sample
  data = {
      'pred_tft': [0.015, -0.020, 0.008, -0.012],
      'pred_catboost': [0.018, -0.005, 0.010, 0.002],
      'regime_vix_ma_5': [15.2, 24.5, 18.1, 22.0],
      'regime_spy_return_60d': [0.03, -0.08, 0.08, -0.02],
      'regime_spy_distance_sma50': [0.01, -0.03, 0.02, -0.01],
  }

  df_predictions = pd.DataFrame(data)
  ensemble = RegimeAwareEnsemble()
  df_result = ensemble.predict(df_predictions)

  print(
      df_result[
          ['pred_tft', 'pred_catboost', 'w_tft_final', 'pred_ensemble']
      ].to_string()
  )