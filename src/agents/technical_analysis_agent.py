import math
import json
import ast
from typing import Dict, Any

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage

from graph.state import TradingAgentState, show_agent_reasoning
from tools.api import fetch_prices, convert_prices_to_dataframe


##### Technical Analysis Agent #####
def technical_analysis_agent(state: TradingAgentState):
    """
    A sophisticated technical analysis system that combines multiple trading strategies:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]

    # 1. Fetch and convert historical price data
    prices = fetch_prices(
        ticker=data["ticker"],
        start_date=start_date,
        end_date=end_date,
    )
    prices_df = convert_prices_to_dataframe(prices)

    # 2. Compute strategy signals
    trend_signals = compute_trend_signals(prices_df)
    mean_reversion_signals = compute_mean_reversion_signals(prices_df)
    momentum_signals = compute_momentum_signals(prices_df)
    volatility_signals = compute_volatility_signals(prices_df)
    stat_arb_signals = compute_stat_arb_signals(prices_df)

    # 3. Combine signals using a weighted ensemble approach
    strategy_weights = {
        "trend": 0.25,
        "mean_reversion": 0.20,
        "momentum": 0.25,
        "volatility": 0.15,
        "stat_arb": 0.15,
    }
    combined_signal = combine_signals_weighted(
        {
            "trend": trend_signals,
            "mean_reversion": mean_reversion_signals,
            "momentum": momentum_signals,
            "volatility": volatility_signals,
            "stat_arb": stat_arb_signals,
        },
        strategy_weights,
    )

    # 4. Generate detailed analysis report
    analysis_report = {
        "signal": combined_signal["signal"],
        "confidence_level": round(combined_signal["confidence"] * 100),
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals["signal"],
                "confidence_level": round(trend_signals["confidence"] * 100),
                "metrics": normalize_pandas_object(trend_signals["metrics"]),
            },
            "mean_reversion": {
                "signal": mean_reversion_signals["signal"],
                "confidence_level": round(mean_reversion_signals["confidence"] * 100),
                "metrics": normalize_pandas_object(mean_reversion_signals["metrics"]),
            },
            "momentum": {
                "signal": momentum_signals["signal"],
                "confidence_level": round(momentum_signals["confidence"] * 100),
                "metrics": normalize_pandas_object(momentum_signals["metrics"]),
            },
            "volatility": {
                "signal": volatility_signals["signal"],
                "confidence_level": round(volatility_signals["confidence"] * 100),
                "metrics": normalize_pandas_object(volatility_signals["metrics"]),
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals["signal"],
                "confidence_level": round(stat_arb_signals["confidence"] * 100),
                "metrics": normalize_pandas_object(stat_arb_signals["metrics"]),
            },
        },
    }

    # 5. Create the final technical analysis message
    message = HumanMessage(
        content=json.dumps(analysis_report),
        name="technical_analysis_agent",
    )

    # 6. Show reasoning if requested
    if show_reasoning:
        show_agent_reasoning(analysis_report, "Technical Analysis Agent")

    # 7. Store signal in data["analyst_signals"]
    state["data"]["analyst_signals"]["technical_analysis_agent"] = {
        "signal": analysis_report["signal"],
        "confidence_level": analysis_report["confidence_level"],
        "reasoning": analysis_report["strategy_signals"],
    }

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


##### Helper Functions #####
def compute_trend_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Advanced trend-following strategy using multiple timeframes and indicators
    """
    ema_8 = compute_ema(prices_df, 8)
    ema_21 = compute_ema(prices_df, 21)
    ema_55 = compute_ema(prices_df, 55)
    adx = compute_adx(prices_df, 14)

    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence_level = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence_level = trend_strength
    else:
        signal = "neutral"
        confidence_level = 0.5

    return {
        "signal": signal,
        "confidence": confidence_level,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def compute_mean_reversion_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mean reversion strategy using Bollinger Bands and z-score
    """
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    bb_upper, bb_lower = compute_bollinger_bands(prices_df)
    rsi_14 = compute_rsi(prices_df, 14)
    rsi_28 = compute_rsi(prices_df, 28)

    price_vs_bb = 0.5
    if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) != 0:
        price_vs_bb = (
            (prices_df["close"].iloc[-1] - bb_lower.iloc[-1])
            / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        )

    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence_level = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence_level = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence_level = 0.5

    return {
        "signal": signal,
        "confidence": confidence_level,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def compute_momentum_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Multi-factor momentum strategy
    """
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence_level = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence_level = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence_level = 0.5

    return {
        "signal": signal,
        "confidence": confidence_level,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def compute_volatility_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Volatility-based trading strategy
    """
    returns = prices_df["close"].pct_change()

    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    atr_series = compute_atr(prices_df)
    atr_ratio = atr_series / prices_df["close"]
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"
        confidence_level = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"
        confidence_level = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence_level = 0.5

    return {
        "signal": signal,
        "confidence": confidence_level,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def compute_stat_arb_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Statistical arbitrage signals based on price action analysis
    """
    returns = prices_df["close"].pct_change()
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()
    hurst = compute_hurst_exponent(prices_df["close"])

    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence_level = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence_level = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence_level = 0.5

    return {
        "signal": signal,
        "confidence": confidence_level,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def combine_signals_weighted(
    signals: Dict[str, Dict[str, Any]],
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Combine multiple trading signals using a weighted approach
    """
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    weighted_sum = 0.0
    total_confidence = 0.0

    for strategy_name, strategy_signal in signals.items():
        numeric_signal = signal_values[strategy_signal["signal"]]
        strategy_weight = weights[strategy_name]
        confidence_level = strategy_signal["confidence"]

        weighted_sum += numeric_signal * strategy_weight * confidence_level
        total_confidence += strategy_weight * confidence_level

    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0.0

    if final_score > 0.2:
        final_signal = "bullish"
    elif final_score < -0.2:
        final_signal = "bearish"
    else:
        final_signal = "neutral"

    return {
        "signal": final_signal,
        "confidence": abs(final_score),
    }


def normalize_pandas_object(obj: Any) -> Any:
    """Convert pandas Series/DataFrames to primitive Python types."""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas_object(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas_object(item) for item in obj]
    return obj


def compute_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(
    prices_df: pd.DataFrame, window: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Compute Bollinger Bands for a given rolling window."""
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def compute_ema(prices_df: pd.DataFrame, window: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA)."""
    return prices_df["close"].ewm(span=window, adjust=False).mean()


def compute_adx(prices_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index (ADX)."""
    prices_df["high_low"] = prices_df["high"] - prices_df["low"]
    prices_df["high_close"] = abs(prices_df["high"] - prices_df["close"].shift())
    prices_df["low_close"] = abs(prices_df["low"] - prices_df["close"].shift())
    prices_df["tr"] = prices_df[["high_low", "high_close", "low_close"]].max(axis=1)

    prices_df["up_move"] = prices_df["high"] - prices_df["high"].shift()
    prices_df["down_move"] = prices_df["low"].shift() - prices_df["low"]

    prices_df["plus_dm"] = np.where(
        (prices_df["up_move"] > prices_df["down_move"]) & (prices_df["up_move"] > 0),
        prices_df["up_move"],
        0,
    )
    prices_df["minus_dm"] = np.where(
        (prices_df["down_move"] > prices_df["up_move"]) & (prices_df["down_move"] > 0),
        prices_df["down_move"],
        0,
    )

    prices_df["+di"] = 100 * (
        prices_df["plus_dm"].ewm(span=period).mean()
        / prices_df["tr"].ewm(span=period).mean()
    )
    prices_df["-di"] = 100 * (
        prices_df["minus_dm"].ewm(span=period).mean()
        / prices_df["tr"].ewm(span=period).mean()
    )
    prices_df["dx"] = 100 * abs(prices_df["+di"] - prices_df["-di"]) / (
        prices_df["+di"] + prices_df["-di"]
    )
    prices_df["adx"] = prices_df["dx"].ewm(span=period).mean()

    return prices_df[["adx", "+di", "-di"]]


def compute_atr(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high_low = prices_df["high"] - prices_df["low"]
    high_close = abs(prices_df["high"] - prices_df["close"].shift())
    low_close = abs(prices_df["low"] - prices_df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def compute_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Compute Hurst Exponent to detect long-term memory in a time series.
    H < 0.5 => Mean-reverting
    H = 0.5 => Random walk
    H > 0.5 => Trending
    """
    lags = range(2, max_lag)
    tau = []

    for lag in lags:
        diff = price_series[lag:] - price_series[:-lag]
        std_dev = np.std(diff) if diff.size > 0 else 1e-8
        tau.append(max(std_dev, 1e-8))

    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        return 0.5  # Default to 0.5 if calculation fails
