"""Cost-aware daily portfolio replay for cross-window model predictions.

This is a research backtest, not an order router. Inputs are frozen predictions,
their saved tree decisions, and adjusted close prices from the Gold dataset.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


KEYS = ["model", "model_vintage_fold"]
STRATEGIES = ("tree_gated", "model_positive", "always_long_risk_managed")


def _commission(notional: float, shares: int, config: dict) -> float:
    raw = max(
        float(config["minimum_commission_per_order_usd"]),
        float(config["commission_per_share_usd"]) * shares,
    )
    return min(raw, notional * float(config["maximum_commission_pct_of_order"]))


def _trade_cost(
    side: str, price: float, shares: int, config: dict, slippage_bps: float
) -> float:
    if shares <= 0 or not np.isfinite(price) or price <= 0:
        return 0.0
    notional = price * shares
    commission = _commission(notional, shares, config)
    pass_through = commission * (
        float(config["finra_commission_pass_through_pct"])
        + float(config["nyse_commission_pass_through_pct"])
    )
    clearing = shares * (
        float(config["finra_cat_fee_per_share_usd"])
        + float(config["nscc_dtc_fee_per_share_usd"])
    )
    regulatory = 0.0
    if side == "sell":
        regulatory = (
            notional * float(config["sec_transaction_fee_pct_of_sales"])
            + shares * float(config["finra_taf_per_share_sold_usd"])
        )
    slippage = notional * slippage_bps / 10000.0
    return float(commission + pass_through + clearing + regulatory + slippage)


def _price_matrices(price_data: pd.DataFrame):
    required = {"ticker", "trade_date"}
    if not required.issubset(price_data.columns):
        raise ValueError("Gold necesita las columnas ticker y trade_date.")
    if "adj_close" not in price_data and "close" not in price_data:
        raise ValueError("Gold necesita adj_close o close para simular PnL.")
    price_column = "adj_close" if "adj_close" in price_data else "close"
    selected_columns = list(dict.fromkeys(
        column for column in ("ticker", "trade_date", price_column, "close")
        if column in price_data
    ))
    frame = price_data[selected_columns].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    if "close" not in frame:
        frame["close"] = frame[price_column]
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "ticker", price_column, "close"])
    frame = frame[(frame[price_column] > 0) & (frame["close"] > 0)].drop_duplicates(
        ["trade_date", "ticker"], keep="last"
    )
    prices = frame.pivot(index="trade_date", columns="ticker", values=price_column).sort_index()
    trade_prices = frame.pivot(index="trade_date", columns="ticker", values="close").sort_index()
    # Use the SPY exchange calendar when available. ETF prices are forward-filled
    # only after their first observation; a missing quote is not a zero return.
    if "SPY" in prices:
        calendar = prices.index[prices["SPY"].notna()]
        prices = prices.loc[calendar]
        trade_prices = trade_prices.reindex(calendar)
    prices = prices.ffill()
    trade_prices = trade_prices.ffill()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return prices, trade_prices, returns


def _prepare_predictions(predictions: pd.DataFrame, prices: pd.DataFrame, config: dict):
    if predictions.empty:
        return pd.DataFrame()
    required = {"model", "model_vintage_fold", "ticker", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Predicciones incompletas; faltan: " + ", ".join(missing))
    frame = predictions.copy()
    if "evaluation_year" not in frame:
        if "validation_year" in frame:
            frame["evaluation_year"] = frame["validation_year"]
        else:
            date = next((c for c in ("target_date", "prediction_date", "origin_date") if c in frame), None)
            if date is None:
                raise ValueError("Predicciones sin evaluation_year ni fecha de origen.")
            frame["evaluation_year"] = pd.to_datetime(frame[date], errors="coerce").dt.year
    if "tree_discovery_year" not in frame:
        frame["tree_discovery_year"] = pd.to_numeric(
            frame.get("model_vintage_year", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
    frame["model_vintage_fold"] = frame["model_vintage_fold"].map(
        lambda value: str(int(float(value))) if pd.notna(value) and str(value).replace(".", "", 1).isdigit() else str(value)
    )
    frame["ticker"] = frame["ticker"].astype(str)
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame["evaluation_year"] = pd.to_numeric(frame["evaluation_year"], errors="coerce")
    frame["tree_discovery_year"] = pd.to_numeric(frame["tree_discovery_year"], errors="coerce")
    if "_tree_predicted_hit_class" not in frame:
        frame["_tree_predicted_hit_class"] = 1
    frame["_tree_predicted_hit_class"] = pd.to_numeric(
        frame["_tree_predicted_hit_class"], errors="coerce"
    ).fillna(0).astype(int)
    # TFT emits five decoder targets from a single origin. Only the first
    # decoder step is a decision for the next session; later steps are not
    # treated as extra independent opportunities.
    if "horizon_step" in frame:
        frame = frame[
            pd.to_numeric(frame["horizon_step"], errors="coerce")
            == int(config.get("tft_decoder_step", 1))
        ].copy()
    frame = frame[
        frame["evaluation_year"].notna()
        & frame["tree_discovery_year"].notna()
        & (frame["evaluation_year"] > frame["tree_discovery_year"])
        & frame["y_pred"].notna()
        & frame["ticker"].isin(prices.columns)
    ].copy()
    # TFT's target_date is the first decoder session and is the next session
    # after the signal origin. CatBoost predictions are timestamped at origin.
    entries = []
    sessions_by_ticker = {}
    for ticker in frame["ticker"].dropna().unique():
        ticker_sessions = prices.index[prices[ticker].notna()]
        sessions_by_ticker[ticker] = ticker_sessions
    for row in frame.itertuples(index=False):
        ticker = str(getattr(row, "ticker"))
        sessions = sessions_by_ticker.get(ticker, pd.DatetimeIndex([]))
        if len(sessions) == 0:
            entries.append(-1)
            continue
        signal_date = getattr(row, "origin_date", pd.NaT)
        if pd.isna(signal_date):
            signal_date = getattr(row, "prediction_date", pd.NaT)
        if pd.isna(signal_date):
            entries.append(-1)
            continue
        signal_date = pd.Timestamp(signal_date).normalize()
        execution_lag = int(config.get("execution_lag_sessions", 1))
        if execution_lag < 0:
            raise ValueError("execution_lag_sessions no puede ser negativo.")
        if execution_lag == 0:
            entry_idx = int(sessions.searchsorted(signal_date, side="left"))
        else:
            first_future = int(sessions.searchsorted(signal_date, side="right"))
            entry_idx = first_future + execution_lag - 1
        entries.append(entry_idx if entry_idx < len(sessions) else -1)
    frame["_entry_index"] = entries
    frame["_entry_date"] = pd.NaT
    frame["_exit_date"] = pd.NaT
    frame["_realized_return"] = np.nan
    holding = int(config["holding_sessions"])
    for index, row in frame.iterrows():
        sessions = sessions_by_ticker[row["ticker"]]
        entry_index = int(row["_entry_index"])
        exit_index = entry_index + holding
        if entry_index < 0 or exit_index >= len(sessions):
            continue
        entry_date, exit_date = sessions[entry_index], sessions[exit_index]
        entry_price = prices.at[entry_date, row["ticker"]]
        exit_price = prices.at[exit_date, row["ticker"]]
        if not (np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price > 0):
            continue
        frame.at[index, "_entry_date"] = entry_date
        frame.at[index, "_exit_date"] = exit_date
        frame.at[index, "_realized_return"] = exit_price / entry_price - 1.0
    frame = frame.dropna(subset=["_entry_date", "_exit_date", "_realized_return"])
    frame["_entry_date"] = pd.to_datetime(frame["_entry_date"]).dt.normalize()
    frame["_exit_date"] = pd.to_datetime(frame["_exit_date"]).dt.normalize()
    frame["source_evaluation_year"] = frame["evaluation_year"]
    # Attribute an execution and its realized five-session return to the year
    # in which the position was actually opened.
    frame["evaluation_year"] = frame["_entry_date"].dt.year
    frame["_model_positive"] = frame["y_pred"] > 0
    frame["_tree_gated"] = frame["_model_positive"] & (frame["_tree_predicted_hit_class"] == 1)
    frame = frame.drop_duplicates(
        ["model", "model_vintage_fold", "evaluation_year", "ticker", "_entry_date"],
        keep="last",
    )
    return frame


def _annual_covariance(returns: pd.DataFrame, date: pd.Timestamp, tickers: list[str], config: dict):
    lookback = int(config["risk_lookback_sessions"])
    min_obs = int(config["minimum_covariance_observations"])
    history = returns.loc[returns.index < date, tickers].tail(lookback)
    vol_floor = float(config["minimum_asset_volatility"])
    if history.notna().sum().min() < min_obs:
        daily_vol = history.std(ddof=1).fillna(vol_floor / np.sqrt(252)).to_numpy()
        annual_vol = np.maximum(daily_vol * np.sqrt(252), vol_floor)
        variances = np.square(annual_vol)
        return np.diag(variances)
    covariance = history.cov(min_periods=min_obs).fillna(0.0).to_numpy() * 252.0
    covariance = (covariance + covariance.T) / 2.0
    eig_min = float(np.linalg.eigvalsh(covariance).min()) if len(covariance) else 0.0
    if eig_min < 1e-8:
        covariance += np.eye(len(tickers)) * (1e-8 - eig_min)
    return covariance


def _risk_scale(
    holdings: dict[str, float], candidate_dollars: dict[str, float], prices: pd.DataFrame,
    returns: pd.DataFrame, date: pd.Timestamp, nav: float, config: dict,
) -> float:
    if nav <= 0 or not candidate_dollars:
        return 0.0
    tickers = sorted(set(holdings) | set(candidate_dollars))
    covariance = _annual_covariance(returns, date, tickers, config)
    price_row = prices.loc[date]
    existing = np.array([holdings.get(t, 0) * float(price_row.get(t, 0.0)) / nav for t in tickers])
    additions = np.array([candidate_dollars.get(t, 0.0) / nav for t in tickers])
    target = float(config["target_annual_volatility"])

    def projected(scale: float) -> float:
        weights = existing + scale * additions
        return float(np.sqrt(max(0.0, weights @ covariance @ weights)))

    if projected(0.0) >= target:
        return 0.0
    if projected(1.0) <= target:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(32):
        middle = (low + high) / 2.0
        if projected(middle) <= target:
            low = middle
        else:
            high = middle
    return low


def _weights_for_cohort(
    eligible: list[str], date: pd.Timestamp, returns: pd.DataFrame, prices: pd.DataFrame,
    holdings: dict[str, float], nav: float, config: dict, tranche_budget: float,
) -> dict[str, float]:
    if not eligible or nav <= 0:
        return {}
    lookback = int(config["risk_lookback_sessions"])
    history = returns.loc[returns.index < date, eligible].tail(lookback)
    weighting = str(config.get("weighting", "inverse_volatility"))
    if weighting == "inverse_volatility":
        vol = history.std(ddof=1).replace(0, np.nan).fillna(
            float(config["minimum_asset_volatility"]) / np.sqrt(252)
        )
        weights = 1.0 / vol.clip(
            lower=float(config["minimum_asset_volatility"]) / np.sqrt(252)
        )
    elif weighting == "equal_weight":
        weights = pd.Series(1.0, index=eligible)
    else:
        raise ValueError(f"Ponderación de cartera no soportada: {weighting}")
    raw = weights / weights.sum()
    cap = float(config["max_single_asset_weight"]) * nav
    price_row = prices.loc[date]
    budgets = {}
    for ticker in eligible:
        current_value = holdings.get(ticker, 0) * float(price_row[ticker])
        budgets[ticker] = max(0.0, min(float(raw[ticker]) * tranche_budget, cap - current_value))
    total = sum(budgets.values())
    gross_room = max(0.0, float(config["max_gross_exposure"]) * nav - sum(
        shares * float(price_row.get(ticker, 0.0)) for ticker, shares in holdings.items()
    ))
    if total > gross_room > 0:
        budgets = {ticker: value * gross_room / total for ticker, value in budgets.items()}
    elif gross_room <= 0:
        return {}
    risk_scale = _risk_scale(holdings, budgets, prices, returns, date, nav, config)
    return {ticker: value * risk_scale for ticker, value in budgets.items() if value > 0}


def _simulate_strategy(
    signal_rows: pd.DataFrame, strategy: str, prices: pd.DataFrame, trade_prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: dict, slippage_bps: float, always_long: bool = False,
    window_rows: pd.DataFrame | None = None,
):
    holding = int(config["holding_sessions"])
    initial = float(config["initial_equity_usd"])
    all_tickers = [str(t) for t in prices.columns]
    context_rows = window_rows if window_rows is not None else signal_rows
    if context_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    last_entry_date = context_rows["_entry_date"].max()
    start_date = context_rows["_entry_date"].min()
    end_date = context_rows["_exit_date"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame(), pd.DataFrame()
    dates = prices.index[(prices.index >= start_date) & (prices.index <= end_date)]
    if len(dates) == 0:
        return pd.DataFrame(), pd.DataFrame()
    date_to_index = {date: i for i, date in enumerate(prices.index)}
    holdings: dict[str, float] = {}
    lots = []
    cash = initial
    previous_nav = initial
    daily_rows, trade_rows = [], []
    signals_by_entry = {}
    if not always_long:
        for date, group in signal_rows.groupby("_entry_date", sort=True):
            signals_by_entry[pd.Timestamp(date)] = group

    for date in dates:
        date = pd.Timestamp(date)
        cash += max(0.0, cash) * float(config.get("cash_return_annual", 0.0)) / 252.0
        price_row = prices.loc[date]
        valid_price = {ticker: float(price_row[ticker]) for ticker in all_tickers if pd.notna(price_row[ticker])}
        # Mark existing positions to the adjusted close before closing and opening lots.
        marked = sum(shares * valid_price[ticker] for ticker, shares in holdings.items() if ticker in valid_price)
        idx = date_to_index[date]

        due = [lot for lot in lots if lot["exit_index"] == idx]
        for lot in due:
            ticker, shares, units = lot["ticker"], lot["shares"], lot["units"]
            price = valid_price.get(ticker)
            if price is None:
                continue
            proceeds = units * price
            execution_price = float(trade_prices.at[date, ticker])
            cost = _trade_cost("sell", execution_price, shares, config, slippage_bps)
            cash += proceeds - cost
            holdings[ticker] = holdings.get(ticker, 0.0) - units
            if holdings[ticker] <= 1e-10:
                holdings.pop(ticker, None)
            trade_rows.append({"date": date, "ticker": ticker, "side": "sell", "shares": shares,
                               "notional_usd": shares * execution_price, "cost_usd": cost, "strategy": strategy})
        lots = [lot for lot in lots if lot["exit_index"] != idx]

        if always_long:
            eligible = [ticker for ticker in all_tickers if ticker in valid_price]
            evaluation_year = date.year
            model = context_rows["model"].iloc[0]
            vintage = context_rows["model_vintage_fold"].iloc[0]
            if date > last_entry_date:
                eligible = []
        else:
            group = signals_by_entry.get(date)
            eligible = []
            evaluation_year = date.year
            model = context_rows["model"].iloc[0]
            vintage = context_rows["model_vintage_fold"].iloc[0]
            if group is not None:
                eligible = [str(t) for t in group["ticker"].drop_duplicates() if str(t) in valid_price]

        nav = cash + sum(shares * valid_price[t] for t, shares in holdings.items() if t in valid_price)
        if eligible:
            tranche = nav * float(config["cohort_fraction_of_equity"])
            allocations = _weights_for_cohort(
                eligible, date, returns, prices, holdings, nav, config, tranche
            )
            session_index = date_to_index[date]
            for ticker, budget in allocations.items():
                adjusted_price = valid_price[ticker]
                price = float(trade_prices.at[date, ticker])
                if price <= 0 or adjusted_price <= 0 or budget <= 0:
                    continue
                shares = int(budget // price) if config.get("whole_shares_only", True) else int(np.floor(budget / price))
                if shares <= 0:
                    continue
                notional = shares * price
                cost = _trade_cost("buy", price, shares, config, slippage_bps)
                while shares > 0 and notional + cost > cash:
                    shares -= 1
                    notional = shares * price
                    cost = _trade_cost("buy", price, shares, config, slippage_bps)
                if shares <= 0:
                    continue
                cash -= notional + cost
                units = notional / adjusted_price
                holdings[ticker] = holdings.get(ticker, 0.0) + units
                exit_index = session_index + holding
                lots.append({"ticker": ticker, "shares": shares, "units": units, "exit_index": exit_index})
                trade_rows.append({"date": date, "ticker": ticker, "side": "buy", "shares": shares,
                                   "notional_usd": notional, "cost_usd": cost, "strategy": strategy})

        nav = cash + sum(shares * valid_price[t] for t, shares in holdings.items() if t in valid_price)
        daily_return = nav / previous_nav - 1.0 if previous_nav > 0 else 0.0
        exposure = sum(shares * valid_price[t] for t, shares in holdings.items() if t in valid_price)
        trades = [trade for trade in trade_rows if trade["date"] == date]
        daily_rows.append({
            "date": date, "model": model, "model_vintage_fold": vintage,
            "evaluation_year": int(evaluation_year), "strategy": strategy,
            "slippage_bps_per_side": float(slippage_bps), "nav_usd": nav,
            "cash_usd": cash,
            "daily_net_return": daily_return, "gross_exposure_usd": exposure,
            "gross_exposure_fraction": exposure / nav if nav > 0 else np.nan,
            "daily_turnover_usd": float(sum(trade["notional_usd"] for trade in trades)),
            "daily_cost_usd": float(sum(trade["cost_usd"] for trade in trades)),
            "active_positions": int(len(holdings)),
        })
        previous_nav = nav

    # Liquidate any residual position at the last available close so costs and
    # terminal NAV reflect an actually closed five-session book.
    if len(dates):
        final_date = pd.Timestamp(dates[-1])
        final_prices = prices.loc[final_date]
        for ticker, units in list(holdings.items()):
            units = float(units)
            adjusted_price = float(final_prices[ticker])
            execution_price = float(trade_prices.at[final_date, ticker])
            actual_shares = sum(lot["shares"] for lot in lots if lot["ticker"] == ticker)
            cost = _trade_cost("sell", execution_price, actual_shares, config, slippage_bps)
            proceeds = units * adjusted_price
            cash += proceeds - cost
            holdings.pop(ticker, None)
            trade_rows.append({"date": final_date, "ticker": ticker, "side": "sell_terminal",
                               "shares": actual_shares, "notional_usd": actual_shares * execution_price, "cost_usd": cost,
                               "strategy": strategy})
        if daily_rows:
            daily_rows[-1]["daily_cost_usd"] += sum(
                trade["cost_usd"] for trade in trade_rows
                if trade["date"] == final_date and trade["side"] == "sell_terminal"
            )
            daily_rows[-1]["nav_usd"] = cash
            daily_rows[-1]["daily_net_return"] = cash / previous_nav - 1.0 if previous_nav > 0 else 0.0
            daily_rows[-1]["gross_exposure_usd"] = 0.0
            daily_rows[-1]["gross_exposure_fraction"] = 0.0
    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _simulate_buy_hold(
    dates: pd.DatetimeIndex, prices: pd.DataFrame, trade_prices: pd.DataFrame,
    ticker_weights: dict[str, float],
    strategy: str, model: str, vintage: str, config: dict, slippage_bps: float,
):
    if not len(dates) or any(ticker not in prices for ticker in ticker_weights):
        return pd.DataFrame(), pd.DataFrame()
    initial = float(config["initial_equity_usd"])
    cash = initial
    holdings = {}
    rows, trades = [], []
    previous_nav = initial
    first, last = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    for ticker, weight in ticker_weights.items():
        price = float(trade_prices.at[first, ticker])
        adjusted_price = float(prices.at[first, ticker])
        budget = initial * float(weight)
        shares = int(budget // price)
        if shares <= 0:
            continue
        notional = shares * price
        cost = _trade_cost("buy", price, shares, config, slippage_bps)
        cash -= notional + cost
        holdings[ticker] = {"shares": shares, "units": notional / adjusted_price}
        trades.append({"date": first, "ticker": ticker, "side": "buy", "shares": shares,
                       "notional_usd": notional, "cost_usd": cost, "strategy": strategy})
    for date in dates:
        date = pd.Timestamp(date)
        cash += max(0.0, cash) * float(config.get("cash_return_annual", 0.0)) / 252.0
        marked = sum(position["units"] * float(prices.at[date, ticker]) for ticker, position in holdings.items())
        nav = cash + marked
        if date == last:
            for ticker, position in list(holdings.items()):
                price = float(trade_prices.at[date, ticker])
                proceeds = position["units"] * float(prices.at[date, ticker])
                cost = _trade_cost("sell", price, position["shares"], config, slippage_bps)
                cash += proceeds - cost
                trades.append({"date": date, "ticker": ticker, "side": "sell_terminal", "shares": position["shares"],
                               "notional_usd": position["shares"] * price, "cost_usd": cost, "strategy": strategy})
            nav = cash
        daily_return = nav / previous_nav - 1.0 if previous_nav > 0 else 0.0
        daily_trades = [trade for trade in trades if trade["date"] == date]
        rows.append({
            "date": date, "model": model, "model_vintage_fold": vintage,
            "evaluation_year": date.year, "strategy": strategy,
            "slippage_bps_per_side": float(slippage_bps), "nav_usd": nav,
            "cash_usd": cash,
            "daily_net_return": daily_return,
            "gross_exposure_usd": max(0.0, nav - cash),
            "gross_exposure_fraction": max(0.0, nav - cash) / nav if nav > 0 else np.nan,
            "daily_turnover_usd": float(sum(trade["notional_usd"] for trade in daily_trades)),
            "daily_cost_usd": float(sum(trade["cost_usd"] for trade in daily_trades)),
            "active_positions": 0 if date == last else len(holdings),
        })
        previous_nav = nav
    return pd.DataFrame(rows), pd.DataFrame(trades)


def _hac_sharpe(values: np.ndarray, lags: int) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    mean = float(values.mean())
    centered = values - mean
    gamma0 = float(centered @ centered / len(values))
    long_run = gamma0
    for lag in range(1, min(int(lags), len(values) - 1) + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / len(values))
        weight = 1.0 - lag / (int(lags) + 1.0)
        long_run += 2.0 * weight * covariance
    if long_run <= 1e-16:
        return np.nan
    return mean / np.sqrt(long_run) * np.sqrt(252.0)


def _bootstrap_sharpe_ci(values: np.ndarray, config: dict, rng: np.random.Generator):
    values = values[np.isfinite(values)]
    n = len(values)
    reps = int(config["block_bootstrap_repetitions"])
    block = max(1, int(config["block_bootstrap_length_sessions"]))
    if n < 20 or reps <= 0:
        return np.nan, np.nan
    results = []
    for _ in range(reps):
        sample_idx = []
        while len(sample_idx) < n:
            start = int(rng.integers(0, n))
            sample_idx.extend((start + offset) % n for offset in range(block))
        results.append(_hac_sharpe(values[np.asarray(sample_idx[:n])], config["hac_lags"]))
    finite = np.asarray(results, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _dsr_probability(values: np.ndarray, n_trials: int) -> float:
    """Approximate DSR using a HAC effective sample size and Bailey PSR form."""
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 20:
        return np.nan
    mean, std = float(values.mean()), float(values.std(ddof=1))
    if std <= 1e-12:
        return np.nan
    daily_sr = mean / std
    centered = values - mean
    variance = float(np.mean(centered**2))
    rho_sum = 0.0
    max_lag = min(20, n - 1)
    for lag in range(1, max_lag + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        rho_sum += (1.0 - lag / (max_lag + 1.0)) * covariance / variance
    effective_n = float(np.clip(n / max(1e-6, 1.0 + 2.0 * rho_sum), 1.0, n))
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurtosis() + 3.0)
    trials = max(1, int(n_trials))
    if trials <= 1:
        threshold = 0.0
    else:
        normal = NormalDist()
        gamma = 0.5772156649015329
        z1 = normal.inv_cdf(np.clip(1.0 - 1.0 / trials, 1e-8, 1.0 - 1e-8))
        z2 = normal.inv_cdf(np.clip(1.0 - 1.0 / (trials * np.e), 1e-8, 1.0 - 1e-8))
        expected_max = (1.0 - gamma) * z1 + gamma * z2
        threshold = expected_max / np.sqrt(max(1.0, effective_n))
    denominator = 1.0 - skew * daily_sr + ((kurtosis - 1.0) / 4.0) * daily_sr**2
    if denominator <= 0:
        return np.nan
    z_score = (daily_sr - threshold) * np.sqrt(max(1.0, effective_n - 1.0)) / np.sqrt(denominator)
    return float(NormalDist().cdf(z_score))


def _portfolio_window_metrics(daily: pd.DataFrame, config: dict, trial_counts: dict):
    if daily.empty:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(42042)
    group_cols = ["model", "model_vintage_fold", "evaluation_year", "strategy", "slippage_bps_per_side"]
    for keys, group in daily.groupby(group_cols, dropna=False, sort=True):
        model, vintage, year, strategy, slip = keys
        values = pd.to_numeric(group["daily_net_return"], errors="coerce").dropna().to_numpy()
        n = len(values)
        if not n:
            continue
        wealth = np.cumprod(1.0 + values)
        total_return = float(wealth[-1] - 1.0)
        cagr = float(wealth[-1] ** (252.0 / n) - 1.0) if wealth[-1] > 0 else -1.0
        vol = float(np.std(values, ddof=1) * np.sqrt(252.0)) if n > 1 else np.nan
        sharpe = float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0)) if n > 1 and np.std(values, ddof=1) > 1e-12 else np.nan
        running_max = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        drawdown = wealth / running_max - 1.0
        ci_low, ci_high = _bootstrap_sharpe_ci(values, config, rng)
        signal_dates = group["date"].nunique()
        max_assets = group["active_positions"].max()
        rows.append({
            "model": model, "model_vintage_fold": vintage, "evaluation_year": int(year),
            "strategy": strategy, "slippage_bps_per_side": float(slip),
            "trading_sessions": int(n), "total_return_net": total_return, "cagr_net": cagr,
            "annualized_volatility_net": vol, "sharpe_net_annualized": sharpe,
            "sharpe_net_hac_annualized": _hac_sharpe(values, config["hac_lags"]),
            "sharpe_hac_block_ci95_low": ci_low, "sharpe_hac_block_ci95_high": ci_high,
            "max_drawdown": float(drawdown.min()),
            "average_gross_exposure": float(group["gross_exposure_fraction"].mean()),
            "average_daily_turnover_usd": float(group["daily_turnover_usd"].mean()),
            "total_cost_usd": float(group["daily_cost_usd"].sum()),
            "active_position_count_max": int(max_assets), "unique_sessions": int(signal_dates),
            "dsr_probability_approx": (
                _dsr_probability(values, trial_counts.get(int(year), 1))
                if strategy == "tree_gated" and float(slip) == float(config["slippage_bps_per_side_base"])
                else np.nan
            ),
            "dsr_tree_trials_counted": trial_counts.get(int(year), np.nan)
            if strategy == "tree_gated" and float(slip) == float(config["slippage_bps_per_side_base"])
            else np.nan,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    base_slip = float(config["slippage_bps_per_side_base"])
    gated = result[
        (result["strategy"] == "tree_gated")
        & (result["slippage_bps_per_side"] == base_slip)
    ]
    result["passes_financial_window"] = False
    eligible = (
        (gated["sharpe_hac_block_ci95_low"] > 0)
        & (gated["dsr_probability_approx"] >= float(config["minimum_dsr_probability"]))
        & (gated["unique_sessions"] >= int(config["minimum_dates_for_lift_inference"]))
    )
    passing_keys = set(zip(
        gated.loc[eligible, "model"], gated.loc[eligible, "model_vintage_fold"],
        gated.loc[eligible, "evaluation_year"],
    ))
    result["passes_financial_window"] = [
        (row.strategy == "tree_gated" and float(row.slippage_bps_per_side) == base_slip
         and (row.model, row.model_vintage_fold, row.evaluation_year) in passing_keys)
        for row in result.itertuples(index=False)
    ]
    return result


def _block_lift_ci(date_rows: pd.DataFrame, config: dict, rng):
    n_dates = len(date_rows)
    if n_dates < int(config["minimum_dates_for_lift_inference"]):
        return np.nan, np.nan
    n_dates = int(n_dates)
    block = max(1, int(config["block_bootstrap_length_sessions"]))
    reps = int(config["block_bootstrap_repetitions"])
    def lift(sample):
        selected_n = sample[:, 1].sum()
        if selected_n <= 0:
            return np.nan
        selected_rate = sample[:, 0].sum() / selected_n
        all_n = sample[:, 3].sum()
        baseline = sample[:, 2].sum() / all_n if all_n > 0 else np.nan
        return selected_rate - baseline

    # columns 0/1 are selected positives/count; 2/3 are all-universe positives/count.
    values = date_rows[["selected_up", "selected_count", "all_up", "all_count"]].to_numpy(dtype=float)
    estimates = []
    for _ in range(reps):
        indexes = []
        while len(indexes) < n_dates:
            start = int(rng.integers(0, n_dates))
            indexes.extend((start + step) % n_dates for step in range(block))
        estimate = lift(values[np.asarray(indexes[:n_dates])])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return np.nan, np.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _block_directional_lift_ci(date_rows: pd.DataFrame, config: dict, rng):
    n_dates = len(date_rows)
    if n_dates < int(config["minimum_dates_for_lift_inference"]):
        return np.nan, np.nan
    block = max(1, int(config["block_bootstrap_length_sessions"]))
    reps = int(config["block_bootstrap_repetitions"])
    values = date_rows[["direction_hit", "all_up", "all_count"]].to_numpy(dtype=float)
    estimates = []
    for _ in range(reps):
        indexes = []
        while len(indexes) < n_dates:
            start = int(rng.integers(0, n_dates))
            indexes.extend((start + offset) % n_dates for offset in range(block))
        sample = values[np.asarray(indexes[:n_dates])]
        all_count = sample[:, 2].sum()
        if all_count <= 0:
            continue
        hit_rate = sample[:, 0].sum() / all_count
        up_rate = sample[:, 1].sum() / all_count
        estimates.append(hit_rate - max(up_rate, 1.0 - up_rate))
    if not estimates:
        return np.nan, np.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _signal_validation(predictions: pd.DataFrame, config: dict):
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(112233)
    for keys, group in predictions.groupby(KEYS + ["evaluation_year"], sort=True, dropna=False):
        model, vintage, year = keys
        date_rows = []
        for date, daily in group.groupby("_entry_date", sort=True):
            all_up = int((daily["_realized_return"] > 0).sum())
            all_count = int(len(daily))
            direction_hit = int(((daily["y_pred"] > 0) == (daily["_realized_return"] > 0)).sum())
            # Two rows share the same matched-universe baseline and date blocks.
            for strategy, mask in (
                ("model_positive", daily["_model_positive"]),
                ("tree_gated", daily["_tree_gated"]),
            ):
                selected = daily.loc[mask]
                selected_count = int(len(selected))
                selected_up = int((selected["_realized_return"] > 0).sum())
                date_rows.append({
                    "date": date, "strategy": strategy,
                    "selected_up": selected_up, "selected_count": selected_count,
                    "all_up": all_up, "all_count": all_count,
                    "direction_hit": direction_hit,
                })
        per_date = pd.DataFrame(date_rows)
        for strategy, dates in per_date.groupby("strategy", sort=False):
            selected_n = int(dates["selected_count"].sum())
            selected_up = int(dates["selected_up"].sum())
            all_n = int(dates["all_count"].sum())
            all_up = int(dates["all_up"].sum())
            precision = selected_up / selected_n if selected_n else np.nan
            baseline = all_up / all_n if all_n else np.nan
            ci_low, ci_high = _block_lift_ci(dates, config, rng)
            directional_ci_low, directional_ci_high = _block_directional_lift_ci(
                dates, config, rng
            )
            directional_hit = (
                float(dates["direction_hit"].sum() / all_n) if all_n else np.nan
            )
            majority_baseline = (
                max(baseline, 1.0 - baseline) if np.isfinite(baseline) else np.nan
            )
            rows.append({
                "model": model, "model_vintage_fold": vintage, "evaluation_year": int(year),
                "strategy": strategy, "signal_dates": int(dates["date"].nunique()),
                "selected_signals": selected_n, "matched_universe_predictions": all_n,
                "positive_signal_coverage": selected_n / all_n if all_n else np.nan,
                "positive_precision": precision, "matched_universe_up_rate": baseline,
                "selected_signal_hit_rate": precision,
                "selected_hit_rate_lift_vs_matched_universe": precision - baseline
                if np.isfinite(precision) and np.isfinite(baseline) else np.nan,
                "precision_lift_vs_matched_universe": precision - baseline
                if np.isfinite(precision) and np.isfinite(baseline) else np.nan,
                "precision_lift_block_ci95_low": ci_low,
                "precision_lift_block_ci95_high": ci_high,
                "directional_hit_rate_all_predictions": directional_hit,
                "matched_majority_direction_baseline": majority_baseline,
                "directional_hit_lift_vs_majority_baseline": directional_hit - majority_baseline
                if np.isfinite(directional_hit) and np.isfinite(majority_baseline) else np.nan,
                "directional_hit_lift_block_ci95_low": directional_ci_low,
                "directional_hit_lift_block_ci95_high": directional_ci_high,
                "confidence_inference_available": bool(
                    dates["date"].nunique() >= int(config["minimum_dates_for_lift_inference"])
                ),
            })
    return pd.DataFrame(rows)


def _candidate_summary(metrics: pd.DataFrame, validation: pd.DataFrame, config: dict):
    if metrics.empty:
        return pd.DataFrame()
    base_slip = float(config["slippage_bps_per_side_base"])
    gated = metrics[
        (metrics["strategy"] == "tree_gated")
        & (metrics["slippage_bps_per_side"] == base_slip)
    ].copy()
    if gated.empty:
        return pd.DataFrame()
    if not validation.empty:
        lifts = validation[validation["strategy"] == "tree_gated"][[
            *KEYS, "evaluation_year", "precision_lift_vs_matched_universe",
            "precision_lift_block_ci95_low", "precision_lift_block_ci95_high",
            "positive_signal_coverage", "positive_precision", "matched_universe_up_rate",
        ]]
        gated = gated.merge(lifts, on=KEYS + ["evaluation_year"], how="left")
    else:
        gated["precision_lift_block_ci95_low"] = np.nan
    gated["passes_financial_window"] &= gated["precision_lift_block_ci95_low"] > 0
    rows = []
    for keys, group in gated.groupby(KEYS, sort=True):
        passed = int(group["passes_financial_window"].sum())
        window_count = int(group["evaluation_year"].nunique())
        minimum = int(config["minimum_future_windows"])
        enough_windows = window_count >= minimum
        candidate = enough_windows and passed >= int(config["minimum_passing_future_windows"])
        rows.append({
            "model": keys[0], "model_vintage_fold": keys[1],
            "future_windows_evaluated": window_count,
            "financial_windows_passed": passed,
            "minimum_future_windows": minimum,
            "minimum_passing_future_windows": int(config["minimum_passing_future_windows"]),
            "discovery_candidate": bool(candidate),
            "mean_net_hac_sharpe": float(group["sharpe_net_hac_annualized"].mean()),
            "worst_net_hac_sharpe": float(group["sharpe_net_hac_annualized"].min()),
            "minimum_dsr_probability": float(group["dsr_probability_approx"].min()),
            "median_precision_lift": float(group["precision_lift_vs_matched_universe"].median()),
            "median_gross_exposure": float(group["average_gross_exposure"].median()),
            "exposure_is_reported_not_a_quality_gate": True,
            "validation_status": "discovery_only_requires_untouched_chronological_holdout",
        })
    return pd.DataFrame(rows)


def simulate_financial_analysis(
    predictions: pd.DataFrame, price_data: pd.DataFrame, config: dict
) -> dict[str, pd.DataFrame]:
    """Replay each frozen vintage against later windows and return analysis tables."""
    prices, trade_prices, returns = _price_matrices(price_data)
    prepared = _prepare_predictions(predictions, prices, config)
    if prepared.empty:
        return {
            "financial_strategy_window_metrics": pd.DataFrame(),
            "financial_signal_validation": pd.DataFrame(),
            "financial_tree_candidates": pd.DataFrame(),
            "financial_daily_returns": pd.DataFrame(),
            "financial_trades": pd.DataFrame(),
        }

    daily_frames, trade_frames = [], []
    # The number of strategies inspected in each calendar window is the trial
    # count for DSR; it is still a lower bound on all manual/code experiments.
    trial_counts = prepared.groupby("evaluation_year")[KEYS].apply(
        lambda values: values.drop_duplicates().shape[0]
    ).to_dict()
    slippages = [float(value) for value in config["slippage_bps_per_side_scenarios"]]
    for (model, vintage), vintage_rows in prepared.groupby(KEYS, sort=True):
        strategy_rows = {
            "tree_gated": vintage_rows[vintage_rows["_tree_gated"]],
            "model_positive": vintage_rows[vintage_rows["_model_positive"]],
            "always_long_risk_managed": vintage_rows,
        }
        for slippage in slippages:
            for strategy in STRATEGIES:
                daily, trades = _simulate_strategy(
                    strategy_rows[strategy], strategy, prices, trade_prices, returns, config, slippage,
                    always_long=(strategy == "always_long_risk_managed"),
                    window_rows=vintage_rows,
                )
                if not daily.empty:
                    daily_frames.append(daily)
                if not trades.empty:
                    trades["model"] = model
                    trades["model_vintage_fold"] = vintage
                    trades["slippage_bps_per_side"] = slippage
                    trade_frames.append(trades)
            all_dates = prices.index[
                (prices.index >= vintage_rows["_entry_date"].min())
                & (prices.index <= vintage_rows["_exit_date"].max())
            ]
            for strategy, weights in (
                ("SPY_buy_hold", {config["benchmarks"]["market_ticker"]: 1.0}),
                ("60_40_SPY_IEF_buy_hold", {
                    config["benchmarks"]["balanced_equity_ticker"]: float(config["benchmarks"]["balanced_equity_weight"]),
                    config["benchmarks"]["balanced_bond_ticker"]: float(config["benchmarks"]["balanced_bond_weight"]),
                }),
            ):
                daily, trades = _simulate_buy_hold(
                    all_dates, prices, trade_prices, weights, strategy, model, vintage, config, slippage
                )
                if not daily.empty:
                    daily_frames.append(daily)
                if not trades.empty:
                    trades["model"] = model
                    trades["model_vintage_fold"] = vintage
                    trades["slippage_bps_per_side"] = slippage
                    trade_frames.append(trades)

    daily_returns = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True, sort=False) if trade_frames else pd.DataFrame()
    validation = _signal_validation(prepared, config)
    metrics = _portfolio_window_metrics(daily_returns, config, trial_counts)
    if not metrics.empty and not validation.empty:
        base_slip = float(config["slippage_bps_per_side_base"])
        financial_lifts = validation[validation["strategy"] == "tree_gated"].set_index(
            KEYS + ["evaluation_year"]
        )["precision_lift_block_ci95_low"]
        for index, row in metrics.iterrows():
            if row["strategy"] != "tree_gated" or float(row["slippage_bps_per_side"]) != base_slip:
                continue
            key = (row["model"], row["model_vintage_fold"], row["evaluation_year"])
            lift_low = financial_lifts.get(key, np.nan)
            metrics.at[index, "passes_financial_window"] = bool(
                row["passes_financial_window"] and pd.notna(lift_low) and lift_low > 0
            )
    candidates = _candidate_summary(metrics, validation, config)
    return {
        "financial_strategy_window_metrics": metrics,
        "financial_signal_validation": validation,
        "financial_tree_candidates": candidates,
        "financial_daily_returns": daily_returns,
        "financial_trades": trades,
    }
