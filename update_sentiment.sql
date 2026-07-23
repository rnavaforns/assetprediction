    -- Capa Bronze: Añadimos probabilidades de FinBERT y la desviación estándar (desacuerdo)
ALTER TABLE bronze.sentiment_data 
  ADD COLUMN IF NOT EXISTS sentiment_pos NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_neg NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_neu NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_std NUMERIC(5, 4) NOT NULL DEFAULT 0.0;

-- Capa Silver: Reflejamos la misma estructura factual
ALTER TABLE silver.fact_sentiment 
  ADD COLUMN IF NOT EXISTS sentiment_pos NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_neg NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_neu NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_std NUMERIC(5, 4) NOT NULL DEFAULT 0.0;

-- Capa Gold: Incorporamos los nuevos predictores para XGBoost
ALTER TABLE gold.training_dataset 
  ADD COLUMN IF NOT EXISTS sentiment_pos NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_neg NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_neu NUMERIC(5, 4) NOT NULL DEFAULT 0.0,
  ADD COLUMN IF NOT EXISTS sentiment_std NUMERIC(5, 4) NOT NULL DEFAULT 0.0;