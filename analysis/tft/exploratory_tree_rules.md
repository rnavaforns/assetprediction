### regime-analysis-124

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_return_60d <= -0.0148
|   |--- origin_spy_volatility_20d <= 0.1513
|   |   |--- class: 1
|   |--- origin_spy_volatility_20d >  0.1513
|   |   |--- class: 1
|--- origin_spy_return_60d >  -0.0148
|   |--- origin_vix_pct_change_5d <= 0.0676
|   |   |--- origin_spy_distance_sma200 <= 0.1129
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma200 >  0.1129
|   |   |   |--- class: 0
|   |--- origin_vix_pct_change_5d >  0.0676
|   |   |--- origin_market_breadth_252d <= 0.9762
|   |   |   |--- class: 0
|   |   |--- origin_market_breadth_252d >  0.9762
|   |   |   |--- class: 1

### regime-analysis-123

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_return_60d <= -0.0372
|   |--- class: 1
|--- origin_spy_return_60d >  -0.0372
|   |--- origin_spy_return_120d <= 0.1665
|   |   |--- origin_spy_trend_ma50_200 <= 0.0724
|   |   |   |--- class: 0
|   |   |--- origin_spy_trend_ma50_200 >  0.0724
|   |   |   |--- class: 1
|   |--- origin_spy_return_120d >  0.1665
|   |   |--- origin_spy_return_5d <= 0.0034
|   |   |   |--- class: 0
|   |   |--- origin_spy_return_5d >  0.0034
|   |   |   |--- class: 0

### regime-analysis-122

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1211
|   |--- origin_market_breadth_20d <= 0.6429
|   |   |--- origin_spy_vol_change_20d <= 0.0057
|   |   |   |--- class: 0
|   |   |--- origin_spy_vol_change_20d >  0.0057
|   |   |   |--- class: 0
|   |--- origin_market_breadth_20d >  0.6429
|   |   |--- origin_spy_volatility_60d <= 0.1084
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.1084
|   |   |   |--- class: 1
|--- origin_spy_volatility_60d >  0.1211
|   |--- origin_spy_return_60d <= 0.0119
|   |   |--- origin_spy_return_120d <= -0.0150
|   |   |   |--- class: 1
|   |   |--- origin_spy_return_120d >  -0.0150
|   |   |   |--- class: 1
|   |--- origin_spy_return_60d >  0.0119
|   |   |--- origin_spy_trend_ma50_200 <= 0.0658
|   |   |   |--- class: 0
|   |   |--- origin_spy_trend_ma50_200 >  0.0658
|   |   |   |--- class: 1

### regime-analysis-121

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_distance_sma50 <= -0.0036
|   |--- origin_spy_trend_ma50_200 <= 0.0316
|   |   |--- class: 1
|   |--- origin_spy_trend_ma50_200 >  0.0316
|   |   |--- origin_market_breadth_above_sma200 <= 0.9286
|   |   |   |--- class: 1
|   |   |--- origin_market_breadth_above_sma200 >  0.9286
|   |   |   |--- class: 1
|--- origin_spy_distance_sma50 >  -0.0036
|   |--- origin_spy_return_120d <= 0.1651
|   |   |--- origin_spy_trend_ma50_200 <= 0.0724
|   |   |   |--- class: 0
|   |   |--- origin_spy_trend_ma50_200 >  0.0724
|   |   |   |--- class: 1
|   |--- origin_spy_return_120d >  0.1651
|   |   |--- origin_cross_asset_return_20d_dispersion <= 0.0452
|   |   |   |--- class: 0
|   |   |--- origin_cross_asset_return_20d_dispersion >  0.0452
|   |   |   |--- class: 0

### regime-analysis-120

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_distance_sma200 <= 0.0233
|   |--- origin_spy_trend_ma50_200 <= 0.0316
|   |   |--- class: 1
|   |--- origin_spy_trend_ma50_200 >  0.0316
|   |   |--- class: 1
|--- origin_spy_distance_sma200 >  0.0233
|   |--- origin_spy_trend_ma50_200 <= 0.0816
|   |   |--- origin_spy_volatility_60d <= 0.1083
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.1083
|   |   |   |--- class: 0
|   |--- origin_spy_trend_ma50_200 >  0.0816
|   |   |--- origin_spy_distance_sma50 <= 0.0119
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma50 >  0.0119
|   |   |   |--- class: 1

### regime-analysis-119

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1080
|   |--- origin_vix_pct_change_20d <= 0.1058
|   |   |--- origin_vix_trend_ma <= 0.0283
|   |   |   |--- class: 1
|   |   |--- origin_vix_trend_ma >  0.0283
|   |   |   |--- class: 0
|   |--- origin_vix_pct_change_20d >  0.1058
|   |   |--- origin_vix_distance_ma20 <= 0.0753
|   |   |   |--- class: 0
|   |   |--- origin_vix_distance_ma20 >  0.0753
|   |   |   |--- class: 0
|--- origin_spy_volatility_60d >  0.1080
|   |--- origin_market_breadth_above_sma200 <= 0.9762
|   |   |--- origin_spy_volatility_60d <= 0.1150
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.1150
|   |   |   |--- class: 0
|   |--- origin_market_breadth_above_sma200 >  0.9762
|   |   |--- origin_spy_distance_sma50 <= 0.0155
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma50 >  0.0155
|   |   |   |--- class: 0

### regime-analysis-118

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_return_60d <= -0.0372
|   |--- class: 1
|--- origin_spy_return_60d >  -0.0372
|   |--- origin_spy_return_60d <= 0.0089
|   |   |--- origin_cross_asset_return_20d_dispersion <= 0.0488
|   |   |   |--- class: 0
|   |   |--- origin_cross_asset_return_20d_dispersion >  0.0488
|   |   |   |--- class: 0
|   |--- origin_spy_return_60d >  0.0089
|   |   |--- origin_spy_distance_sma50 <= 0.0075
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma50 >  0.0075
|   |   |   |--- class: 0

### regime-analysis-117

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_return_60d <= -0.0372
|   |--- class: 1
|--- origin_spy_return_60d >  -0.0372
|   |--- origin_spy_return_60d <= 0.0089
|   |   |--- origin_cross_asset_return_20d_dispersion <= 0.0488
|   |   |   |--- class: 0
|   |   |--- origin_cross_asset_return_20d_dispersion >  0.0488
|   |   |   |--- class: 0
|   |--- origin_spy_return_60d >  0.0089
|   |   |--- origin_spy_distance_sma50 <= 0.0075
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma50 >  0.0075
|   |   |   |--- class: 0

### regime-analysis-116

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_distance_sma50 <= -0.0233
|   |--- origin_market_breadth_252d <= 0.9762
|   |   |--- class: 1
|   |--- origin_market_breadth_252d >  0.9762
|   |   |--- class: 1
|--- origin_spy_distance_sma50 >  -0.0233
|   |--- origin_market_breadth_above_sma200 <= 0.9762
|   |   |--- origin_spy_vol_change_20d <= -0.0956
|   |   |   |--- class: 0
|   |   |--- origin_spy_vol_change_20d >  -0.0956
|   |   |   |--- class: 0
|   |--- origin_market_breadth_above_sma200 >  0.9762
|   |   |--- origin_spy_volatility_60d <= 0.1081
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.1081
|   |   |   |--- class: 1

### regime-analysis-115

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_distance_sma50 <= 0.0716
|   |--- origin_spy_distance_sma200 <= 0.0262
|   |   |--- origin_spy_trend_ma50_200 <= 0.0316
|   |   |   |--- class: 1
|   |   |--- origin_spy_trend_ma50_200 >  0.0316
|   |   |   |--- class: 1
|   |--- origin_spy_distance_sma200 >  0.0262
|   |   |--- origin_vix_pct_change_5d <= 0.0623
|   |   |   |--- class: 0
|   |   |--- origin_vix_pct_change_5d >  0.0623
|   |   |   |--- class: 1
|--- origin_spy_distance_sma50 >  0.0716
|   |--- class: 0

### regime-analysis-114

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1095
|   |--- origin_spy_return_5d <= 0.0103
|   |   |--- origin_vix_trend_ma <= 0.0190
|   |   |   |--- class: 0
|   |   |--- origin_vix_trend_ma >  0.0190
|   |   |   |--- class: 0
|   |--- origin_spy_return_5d >  0.0103
|   |   |--- origin_spy_return_20d <= 0.0293
|   |   |   |--- class: 0
|   |   |--- origin_spy_return_20d >  0.0293
|   |   |   |--- class: 1
|--- origin_spy_volatility_60d >  0.1095
|   |--- origin_spy_return_60d <= 0.0903
|   |   |--- origin_spy_distance_sma200 <= -0.0032
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma200 >  -0.0032
|   |   |   |--- class: 1
|   |--- origin_spy_return_60d >  0.0903
|   |   |--- origin_vix_trend_ma <= -0.0637
|   |   |   |--- class: 0
|   |   |--- origin_vix_trend_ma >  -0.0637
|   |   |   |--- class: 0

### regime-analysis-113

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1080
|   |--- origin_spy_return_5d <= 0.0103
|   |   |--- origin_spy_distance_sma200 <= 0.1072
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma200 >  0.1072
|   |   |   |--- class: 0
|   |--- origin_spy_return_5d >  0.0103
|   |   |--- origin_spy_trend_ma50_200 <= 0.0767
|   |   |   |--- class: 0
|   |   |--- origin_spy_trend_ma50_200 >  0.0767
|   |   |   |--- class: 1
|--- origin_spy_volatility_60d >  0.1080
|   |--- origin_market_breadth_above_sma200 <= 0.9762
|   |   |--- origin_spy_volatility_60d <= 0.1216
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.1216
|   |   |   |--- class: 1
|   |--- origin_market_breadth_above_sma200 >  0.9762
|   |   |--- origin_spy_return_60d <= 0.0346
|   |   |   |--- class: 1
|   |   |--- origin_spy_return_60d >  0.0346
|   |   |   |--- class: 1

### regime-analysis-112

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1095
|   |--- origin_spy_return_5d <= 0.0103
|   |   |--- origin_spy_distance_sma200 <= 0.1072
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma200 >  0.1072
|   |   |   |--- class: 0
|   |--- origin_spy_return_5d >  0.0103
|   |   |--- origin_spy_distance_sma200 <= 0.1127
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma200 >  0.1127
|   |   |   |--- class: 1
|--- origin_spy_volatility_60d >  0.1095
|   |--- origin_market_breadth_above_sma200 <= 0.9762
|   |   |--- origin_spy_volatility_60d <= 0.1216
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.1216
|   |   |   |--- class: 1
|   |--- origin_market_breadth_above_sma200 >  0.9762
|   |   |--- origin_spy_return_5d <= 0.0085
|   |   |   |--- class: 1
|   |   |--- origin_spy_return_5d >  0.0085
|   |   |   |--- class: 0

### regime-analysis-111

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1080
|   |--- origin_spy_distance_sma50 <= 0.0313
|   |   |--- origin_spy_vol_change_20d <= 0.2512
|   |   |   |--- class: 0
|   |   |--- origin_spy_vol_change_20d >  0.2512
|   |   |   |--- class: 0
|   |--- origin_spy_distance_sma50 >  0.0313
|   |   |--- origin_vix_change_1d <= 0.0350
|   |   |   |--- class: 0
|   |   |--- origin_vix_change_1d >  0.0350
|   |   |   |--- class: 0
|--- origin_spy_volatility_60d >  0.1080
|   |--- origin_spy_distance_sma50 <= 0.0716
|   |   |--- origin_spy_vol_change_20d <= 0.5791
|   |   |   |--- class: 1
|   |   |--- origin_spy_vol_change_20d >  0.5791
|   |   |   |--- class: 0
|   |--- origin_spy_distance_sma50 >  0.0716
|   |   |--- class: 0

### regime-analysis-110

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1080
|   |--- origin_spy_return_5d <= 0.0068
|   |   |--- origin_vix_percentile_252 <= 0.4901
|   |   |   |--- class: 0
|   |   |--- origin_vix_percentile_252 >  0.4901
|   |   |   |--- class: 0
|   |--- origin_spy_return_5d >  0.0068
|   |   |--- origin_spy_trend_ma50_200 <= 0.0679
|   |   |   |--- class: 0
|   |   |--- origin_spy_trend_ma50_200 >  0.0679
|   |   |   |--- class: 1
|--- origin_spy_volatility_60d >  0.1080
|   |--- origin_spy_distance_sma50 <= 0.0077
|   |   |--- origin_spy_return_60d <= 0.0351
|   |   |   |--- class: 1
|   |   |--- origin_spy_return_60d >  0.0351
|   |   |   |--- class: 0
|   |--- origin_spy_distance_sma50 >  0.0077
|   |   |--- origin_spy_vol_change_20d <= -0.3824
|   |   |   |--- class: 1
|   |   |--- origin_spy_vol_change_20d >  -0.3824
|   |   |   |--- class: 0

### regime-analysis-109

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1087
|   |--- origin_spy_return_5d <= 0.0103
|   |   |--- origin_spy_distance_sma200 <= 0.1072
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma200 >  0.1072
|   |   |   |--- class: 0
|   |--- origin_spy_return_5d >  0.0103
|   |   |--- origin_spy_distance_sma200 <= 0.1097
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma200 >  0.1097
|   |   |   |--- class: 1
|--- origin_spy_volatility_60d >  0.1087
|   |--- origin_spy_distance_sma50 <= 0.0074
|   |   |--- origin_spy_return_20d <= 0.0028
|   |   |   |--- class: 1
|   |   |--- origin_spy_return_20d >  0.0028
|   |   |   |--- class: 0
|   |--- origin_spy_distance_sma50 >  0.0074
|   |   |--- origin_spy_volatility_20d <= 0.1047
|   |   |   |--- class: 1
|   |   |--- origin_spy_volatility_20d >  0.1047
|   |   |   |--- class: 0

### regime-analysis-108

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_trend_ma50_200 <= 0.0555
|   |--- origin_spy_trend_ma50_200 <= 0.0381
|   |   |--- origin_spy_volatility_60d <= 0.1478
|   |   |   |--- class: 1
|   |   |--- origin_spy_volatility_60d >  0.1478
|   |   |   |--- class: 0
|   |--- origin_spy_trend_ma50_200 >  0.0381
|   |   |--- origin_cross_asset_return_20d_dispersion <= 0.0480
|   |   |   |--- class: 0
|   |   |--- origin_cross_asset_return_20d_dispersion >  0.0480
|   |   |   |--- class: 0
|--- origin_spy_trend_ma50_200 >  0.0555
|   |--- origin_spy_return_60d <= 0.0478
|   |   |--- origin_market_breadth_20d <= 0.8810
|   |   |   |--- class: 1
|   |   |--- origin_market_breadth_20d >  0.8810
|   |   |   |--- class: 0
|   |--- origin_spy_return_60d >  0.0478
|   |   |--- origin_spy_trend_ma50_200 <= 0.0627
|   |   |   |--- class: 1
|   |   |--- origin_spy_trend_ma50_200 >  0.0627
|   |   |   |--- class: 0

### regime-analysis-107

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_return_5d <= 0.0058
|   |--- origin_spy_vol_change_20d <= 0.0125
|   |   |--- origin_spy_distance_sma200 <= 0.1039
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma200 >  0.1039
|   |   |   |--- class: 0
|   |--- origin_spy_vol_change_20d >  0.0125
|   |   |--- origin_vix_pct_change_5d <= 0.0385
|   |   |   |--- class: 0
|   |   |--- origin_vix_pct_change_5d >  0.0385
|   |   |   |--- class: 1
|--- origin_spy_return_5d >  0.0058
|   |--- origin_spy_return_60d <= 0.0564
|   |   |--- origin_spy_return_60d <= 0.0248
|   |   |   |--- class: 0
|   |   |--- origin_spy_return_60d >  0.0248
|   |   |   |--- class: 1
|   |--- origin_spy_return_60d >  0.0564
|   |   |--- origin_vix_change_1d <= -0.5950
|   |   |   |--- class: 0
|   |   |--- origin_vix_change_1d >  -0.5950
|   |   |   |--- class: 0

### regime-analysis-106

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_distance_sma50 <= 0.0151
|   |--- origin_spy_volatility_60d <= 0.1084
|   |   |--- class: 0
|   |--- origin_spy_volatility_60d >  0.1084
|   |   |--- origin_spy_trend_ma50_200 <= 0.0280
|   |   |   |--- class: 1
|   |   |--- origin_spy_trend_ma50_200 >  0.0280
|   |   |   |--- class: 1
|--- origin_spy_distance_sma50 >  0.0151
|   |--- origin_spy_return_120d <= 0.1705
|   |   |--- origin_spy_distance_sma50 <= 0.0716
|   |   |   |--- class: 0
|   |   |--- origin_spy_distance_sma50 >  0.0716
|   |   |   |--- class: 0
|   |--- origin_spy_return_120d >  0.1705
|   |   |--- origin_spy_volatility_60d <= 0.0883
|   |   |   |--- class: 0
|   |   |--- origin_spy_volatility_60d >  0.0883
|   |   |   |--- class: 0

### regime-analysis-105

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_distance_sma200 <= -0.0032
|   |--- class: 1
|--- origin_spy_distance_sma200 >  -0.0032
|   |--- origin_spy_trend_ma50_200 <= 0.0571
|   |   |--- origin_spy_vol_change_20d <= -0.3363
|   |   |   |--- class: 1
|   |   |--- origin_spy_vol_change_20d >  -0.3363
|   |   |   |--- class: 0
|   |--- origin_spy_trend_ma50_200 >  0.0571
|   |   |--- origin_spy_distance_sma200 <= 0.1127
|   |   |   |--- class: 1
|   |   |--- origin_spy_distance_sma200 >  0.1127
|   |   |   |--- class: 0

### regime-analysis-104

==================================================
ÁRBOL EXPLORATORIO DE REGÍMENES TFT
==================================================

AVISO: análisis exploratorio sobre las predicciones de esta ejecución.
No utilizar estas reglas como estrategia final sin validación temporal independiente.

|--- origin_spy_volatility_60d <= 0.1080
|   |--- origin_spy_distance_sma200 <= 0.1000
|   |   |--- origin_spy_vol_change_20d <= 0.2152
|   |   |   |--- class: 0
|   |   |--- origin_spy_vol_change_20d >  0.2152
|   |   |   |--- class: 0
|   |--- origin_spy_distance_sma200 >  0.1000
|   |   |--- origin_vix_trend_ma <= 0.0246
|   |   |   |--- class: 1
|   |   |--- origin_vix_trend_ma >  0.0246
|   |   |   |--- class: 0
|--- origin_spy_volatility_60d >  0.1080
|   |--- origin_spy_distance_sma50 <= 0.0074
|   |   |--- origin_market_breadth_above_sma200 <= 0.8333
|   |   |   |--- class: 0
|   |   |--- origin_market_breadth_above_sma200 >  0.8333
|   |   |   |--- class: 1
|   |--- origin_spy_distance_sma50 >  0.0074
|   |   |--- origin_spy_volatility_20d <= 0.1050
|   |   |   |--- class: 1
|   |   |--- origin_spy_volatility_20d >  0.1050
|   |   |   |--- class: 0