-- Capa Bronze: Añadimos las probabilidades y el std básico
ALTER TABLE bronze.sentiment_data
  ADD COLUMN IF NOT EXISTS sentiment_pos NUMERIC(5, 4) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_neg NUMERIC(5, 4) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_neu NUMERIC(5, 4) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_std NUMERIC(5, 4) DEFAULT NULL;

-- Capa Silver: Reflejamos la estructura factual (quitamos el DEFAULT 0.0 para permitir NULLs en el histórico)
ALTER TABLE silver.fact_sentiment
  ADD COLUMN IF NOT EXISTS sentiment_pos NUMERIC(5, 4) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_neg NUMERIC(5, 4) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_neu NUMERIC(5, 4) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_std NUMERIC(5, 4) DEFAULT NULL;

-- Capa Gold: Incorporamos los predictores base + la Puntuación Ponderada por Volumen
ALTER TABLE gold.training_dataset
  ADD COLUMN IF NOT EXISTS sentiment_pos DOUBLE PRECISION DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_neg DOUBLE PRECISION DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_neu DOUBLE PRECISION DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_std DOUBLE PRECISION DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_weighted DOUBLE PRECISION DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_ema_3 DOUBLE PRECISION DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS sentiment_ema_5 DOUBLE PRECISION DEFAULT NULL;