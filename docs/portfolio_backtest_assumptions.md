# Supuestos de la simulación financiera

El backtest convierte predicciones de retorno a cinco sesiones en una política de cartera diaria reproducible. Sus resultados sirven para investigación y comparación de reglas; no se envían órdenes.

## Ejecución y horizonte

- La señal se considera disponible al cierre de su fecha de origen y se ejecuta al cierre de la siguiente sesión disponible.
- TFT usa solo `horizon_step = 1`, una predicción negociable por activo y sesión. Los otros cuatro pasos son pronósticos alternativos emitidos desde el mismo origen y no se cuentan como nuevas decisiones de cartera.
- En CatBoost la etiqueta guardada cubre origen a origen+5, pero la cartera espera una sesión para ejecutar y vuelve a medir el retorno entre los cierres reales de compra y venta; por eso su retorno realizado puede diferir del `y_true` del artifact.
- Cada señal elegible abre un tramo de cartera que se mantiene cinco sesiones y se liquida al cierre de la quinta sesión. Los tramos diarios se solapan; cada nuevo tramo recibe como máximo una quinta parte del capital. Esto modela una estrategia que inicia una nueva cohorte cada sesión sin atribuir cinco veces el mismo capital.
- TFT usa solo `horizon_step = 1`, una predicción negociable por activo y sesión. Los otros cuatro pasos son pronósticos alternativos emitidos desde el mismo origen y no se cuentan como nuevas decisiones de cartera.
- La cantidad de acciones y las tarifas se estiman con `close`; la evolución de la posición se marca con `adj_close` para incluir distribuciones y otros ajustes de retorno total. El rendimiento realizado de señal usa el cociente de `adj_close` entre entrada y salida.
- Los rendimientos de cartera se reconstruyen desde precios de Gold entre ejecución y liquidación. No se usan como PnL los targets repetidos en los artifacts.

## Perfil de riesgo simulado

- Solo posiciones largas; sin margen ni ventas en corto.
- Exposición bruta máxima del 100 %; el efectivo no obtiene interés en este escenario.
- Límite máximo del 10 % del NAV por activo, aplicado también a las posiciones solapadas.
- Dentro de las señales elegibles, el capital de cada tramo se pondera inversamente a la volatilidad realizada de 60 sesiones. Se reduce la nueva exposición cuando la matriz de covarianza histórica proyecta que la volatilidad anualizada del libro superaría el objetivo del 10 %.
- El objetivo del 10 %, el límite del 10 % por activo, la quinta parte por tramo y el capital inicial de 100.000 USD describen un escenario configurable conservador para explorar el comportamiento. No son niveles universales recomendados por la literatura ni una recomendación para una cuenta individual.

La reducción de riesgo cuando sube la volatilidad sigue la familia de estrategias de volatilidad gestionada estudiadas por Moreira y Muir. El paper no prescribe un objetivo universal de volatilidad ni garantiza que esta regla vaya a mejorar una cartera concreta: [Moreira y Muir, “Volatility-Managed Portfolios”, Journal of Finance (2017)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513).

## Broker y costes

El universo actual del dataset contiene ETF cotizados en EE. UU. y precios en USD. Se usa como referencia IBKR Pro tiered para acciones/ETF estadounidenses, en el tramo de hasta 300.000 acciones al mes: 0,0035 USD por acción, mínimo 0,35 USD por orden y máximo del 1 % del valor de operación. La tabla de IBKR también publica la comisión SEC sobre ventas, FINRA TAF y CAT, NSCC/DTC y las repercusiones porcentuales de FINRA/NYSE; el backtest las incorpora. Las tarifas de exchange/ECN dependen de la ruta y de si la orden añade o retira liquidez; el artifact no contiene ejecuciones ni venues para reconstruirlas.

- Comisión estimada por orden: `min(1 % del nominal, max(0,35 USD, 0,0035 USD × acciones))`.
- En ventas: se añaden las tarifas SEC, FINRA TAF, FINRA CAT, NSCC/DTC y las pequeñas tasas de repercusión sobre comisión descritas en la tabla estadounidense de IBKR.
- Se simulan deslizamientos de 5, 10 y 20 puntos básicos por lado. El escenario principal es 10 pb por lado. El deslizamiento es una hipótesis de ejecución, no una comisión publicada por el broker; se incluye sensibilidad porque la horquilla e impacto reales dependen del activo, tamaño y tipo de orden.
- Se usan acciones enteras para reflejar una ejecución conservadora y hacer que el mínimo por orden dependa del tamaño de cuenta.
- Los resultados se expresan en USD. No se modela el rendimiento EUR/USD ni una conversión de divisa; un inversor cuyo capital esté en EUR debe añadir ambos antes de evaluar resultados en euros.
- No se modelan impuestos, suscripciones de datos, gastos del ETF, intereses de efectivo ni cargos mensuales de cuenta. Tampoco se ajusta el deslizamiento con horquillas históricas o profundidad de mercado. La serie de costes es un escenario de referencia, no una liquidación de broker.

Las tarifas de IBKR cambian y las tasas de exchange dependen de la ejecución. La tabla oficial indica, para acciones y ETF de EE. UU., 0,0035 USD/acción y mínimo de 0,35 USD en Tiered hasta el tramo de volumen indicado, junto con las tasas regulatorias/clearing aplicables: [tabla de comisiones de IBKR Irlanda](https://www.interactivebrokers.ie/en/pricing/commissions-stocks.php?region=americas). La configuración conserva las tasas FINRA publicadas en esa tabla y debe actualizarse cuando el broker modifique precios.

IBKR documenta TWS API para Python, envío/modificación/cancelación de órdenes y una cuenta paper accesible por API. Su paper trading simula fills desde la parte superior del libro y puede diferir del mercado real; es adecuado para una fase de comprobación operativa, no para asumir fills reales: [curso oficial TWS API Python](https://www.interactivebrokers.com/campus/trading-course/python-tws-api/), [cuenta paper](https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/) y [limitaciones de paper trading](https://www.interactivebrokers.com/docs/tws-api/doc/notes-limitations/limitations/paper-trading). Se escoge como referencia por el universo de ETF estadounidenses y su API, no como recomendación de broker.

## Métricas e incertidumbre

- Se publican Sharpe neto convencional y Sharpe neto HAC anualizado sobre retornos diarios de cartera. El intervalo del 95 % remuestrea bloques móviles de cinco sesiones. Ajustar por autocorrelación importa: el Sharpe anualizado con raíz del tiempo puede ser engañoso cuando los retornos tienen dependencia serial (Lo, 2002, [The Statistics of Sharpe Ratios](https://traders.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf)).
- DSR aproximado por año, usando el número de pares modelo-vintage incluidos en esa ventana como ensayos, y corrigiendo el PSR por asimetría, curtosis y tamaño efectivo estimado. Es deliberadamente marcado como aproximado: no incorpora la dispersión completa entre ensayos ni cada iteración de código, umbral o experimento manual, y los árboles tampoco son independientes. El DSR original trata explícitamente selección múltiple y no normalidad (Bailey y López de Prado, 2014, [paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)).
- La hit rate de las señales positivas y su precisión coinciden aquí: ambas son la proporción de compras con retorno positivo realizado. Se comparan con la proporción de retornos positivos del mismo universo, vintage, fechas de ejecución y ventana. La hit rate direccional de todas las predicciones también se publica frente a la regla base de mayoría direccional. Los intervalos de mejora remuestrean fechas en bloques, manteniendo juntos los activos de cada fecha.
- Se publican benchmarks SPY, 60/40 SPY/IEF, cartera larga de todos los ETF y el predictor positivo sin árbol.
- Tres ventanas futuras y dos que pasan conservan el papel de filtro de descubrimiento. No sustituyen un periodo cronológico intacto; dado que los artifacts ya se han inspeccionado, una validación final exige nuevas fechas no usadas para elegir reglas y una fase paper.
- `passes_financial_window` exige que el límite inferior del intervalo de Sharpe HAC sea positivo, que el DSR aproximado supere el valor configurable y que el límite inferior del lift de precisión sea positivo. No impone un mínimo de cobertura; sí publica frecuencia y exposición para que el inversor determine restricciones operativas.

Los estudios de aprendizaje automático financiero evalúan estrategias mediante carteras y criterios de implementabilidad, incluyendo costes de transacción: Gu, Kelly y Xiu (2020), [Machine Learning and the Implementable Efficient Frontier](https://academic.oup.com/rfs/article/33/5/2223/5758276). El ajuste de exposición inversamente relacionado con volatilidad se inspira en Moreira y Muir (2017), [Volatility-Managed Portfolios](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513); el artículo no prescribe el objetivo configurado aquí.

## Parámetros que se pueden ajustar

Edita `config/portfolio_backtest.json` para cambiar tamaño de cuenta, horizonte, desfase de ejecución, exposición bruta, límite por activo, objetivo de volatilidad, ponderación, costes y umbrales inferenciales. El `run_manifest.json` guarda una copia de la configuración aplicada con cada artifact.
