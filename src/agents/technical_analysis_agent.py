import math
import json
import ast
from typing import Dict

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage

from agents.state import TradingAgentState, show_agent_reasoning
from tools.api import convert_prices_to_dataframe,fetch_prices



def technical_analysis_agent(state: TradingAgentState):
##### Technical Analysis Agent #####
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
    # Get the historical price data
    prices = fetch_prices(
        ticker=data["ticker"], 
        start_date=start_date, 
        end_date=end_date,
    )
    # Convert prices to a DataFrame    prices_df = convert_prices_to_dataframe(prices)

    # Compute indicators
    macd_line, signal_line = compute_macd(convert_prices_to_dataframe)
    rsi_values = compute_rsi(convert_prices_to_dataframe)
    upper_band, lower_band = compute_bollinger_bands(convert_prices_to_dataframe)
    obv_values = compute_obv(convert_prices_to_dataframe)

    # Generate individual signals
    signals = []

    # MACD signal
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append("bullish")
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # RSI signal
    if rsi_values.iloc[-1] < 30:
        signals.append("bullish")
    elif rsi_values.iloc[-1] > 70:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # Bollinger Bands signal
    current_price = convert_prices_to_dataframe["close"].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append("bullish")
    elif current_price > upper_band.iloc[-1]:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # OBV signal
    obv_slope = obv_values.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append("bullish")
    elif obv_slope < 0:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # Build reasoning
    reasoning = {
        "MACD": {
            "signal": signals[0],
            "details": (
                "MACD Line crossed above Signal Line"
                if signals[0] == "bullish"
                else "MACD Line crossed below Signal Line"
                if signals[0] == "bearish"
                else "No crossover"
            ),
        },
        "RSI": {
            "signal": signals[1],
            "details": (
                f"RSI is {rsi_values.iloc[-1]:.2f} (oversold)"
                if signals[1] == "bullish"
                else f"RSI is {rsi_values.iloc[-1]:.2f} (overbought)"
                if signals[1] == "bearish"
                else f"RSI is {rsi_values.iloc[-1]:.2f} (neutral)"
            ),
        },
        "Bollinger": {
            "signal": signals[2],
            "details": (
                "Price is below lower band"
                if signals[2] == "bullish"
                else "Price is above upper band"
                if signals[2] == "bearish"
                else "Price is within bands"
            ),
        },
        "OBV": {
            "signal": signals[3],
            "details": f"OBV slope is {obv_slope:.2f} ({signals[3]})",
        },
    }

    # Determine overall signal + confidence
    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    total_signals = len(signals)
    confidence_score = max(bullish_signals, bearish_signals) / total_signals

    # Base message content (simpler aggregated view)
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_score * 100)}%",
        "reasoning": reasoning,
    }

    # Extended Strategies
    trend_signals = compute_trend_signals(convert_prices_to_dataframe)
    mean_reversion_signals = compute_mean_reversion_signals(convert_prices_to_dataframe)
    momentum_signals = compute_momentum_signals(convert_prices_to_dataframe)
    volatility_signals = compute_volatility_signals(convert_prices_to_dataframe)
    stat_arb_signals = compute_stat_arb_signals(convert_prices_to_dataframe)

    # Weighted ensemble approach
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

    analysis_report = {
        "signal": combined_signal["signal"],
        "confidence": f"{round(combined_signal['confidence'] * 100)}%",
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals["signal"],
                "confidence": f"{round(trend_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas_object(trend_signals["metrics"]),
            },
            "mean_reversion": {
                "signal": mean_reversion_signals["signal"],
                "confidence": f"{round(mean_reversion_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas_object(mean_reversion_signals["metrics"]),
            },
            "momentum": {
                "signal": momentum_signals["signal"],
                "confidence": f"{round(momentum_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas_object(momentum_signals["metrics"]),
            },
            "volatility": {
                "signal": volatility_signals["signal"],
                "confidence": f"{round(volatility_signals["confidence"] * 100)}%",
                "metrics": normalize_pandas_object(volatility_signals["metrics"]),
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals["signal"],
                "confidence": f"{round(stat_arb_signals["confidence"] * 100)}%",
                "metrics": normalize_pandas_object(stat_arb_signals["metrics"]),
            },
        },
    }

    # Create the final technical analysis message
    message = HumanMessage(
        content=json.dumps(analysis_report),
        name="technical_analysis_agent",
    )

    if show_reasoning:
        show_agent_reasoning(analysis_report, "Technical Analysis Agent")

    return {
        "messages": [message],
        "data": data,
    }


def compute_trend_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """Advanced trend following strategy using multiple timeframes and indicators."""
    ema_8 = compute_ema(prices_df, 8)
    ema_21 = compute_ema(prices_df, 21)
    ema_55 = compute_ema(prices_df, 55)
    adx = compute_adx(prices_df, 14)
    ichimoku = compute_ichimoku(prices_df)

    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
            # "ichimoku": ichimoku  # If needed
        },
    }


def compute_mean_reversion_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """Mean reversion strategy using Bollinger Bands and z-score calculations."""
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50
    bb_upper, bb_lower = compute_bollinger_bands(prices_df)
    rsi_14 = compute_rsi(prices_df, 14)
    rsi_28 = compute_rsi(prices_df, 28)

    extreme_z = abs(z_score.iloc[-1]) > 2
    range_bb = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / range_bb if range_bb != 0 else 0

    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def compute_momentum_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """Multi-factor momentum strategy using price and volume momentum."""
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    momentum_score = 0.4 * mom_1m.iloc[-1] + 0.3 * mom_3m.iloc[-1] + 0.3 * mom_6m.iloc[-1]
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def compute_volatility_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """Volatility-based trading strategy using historical vol and ATR."""
    returns = prices_df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    atr = compute_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def compute_stat_arb_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """Statistical arbitrage signals based on time-series analysis."""
    returns = prices_df["close"].pct_change()
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()
    hurst = compute_hurst_exponent(prices_df["close"])

    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def combine_signals_weighted(signals: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    """Combine multiple trading signals using a weighted approach."""
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}
    weighted_sum = 0.0
    total_confidence = 0.0

    for strategy_name, strategy_signal in signals.items():
        numeric_signal = signal_values[strategy_signal["signal"]]
        strategy_weight = weights[strategy_name]
        confidence = strategy_signal["confidence"]

        weighted_sum += numeric_signal * strategy_weight * confidence
        total_confidence += strategy_weight * confidence

    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "confidence": abs(final_score),
    }


def normalize_pandas_object(obj: Any) -> Any:
    """
    Convert Pandas Series/DataFrames or nested structures to primitive Python types
    for JSON serialization.
    """
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas_object(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas_object(item) for item in obj]
    return obj


def compute_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute MACD (Moving Average Convergence Divergence) and its signal line."""
    ema_12 = prices_df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def compute_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """Compute Bollinger Bands for a given rolling window."""
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def compute_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA) for a given window."""
    return df["close"].ewm(span=window, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index (ADX)."""
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    df["+di"] = 100 * (
        df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["-di"] = 100 * (
        df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean()
    )
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def compute_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute Ichimoku Cloud indicators."""
    period9_high = df["high"].rolling(window=9).max()
    period9_low = df["low"].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    period26_high = df["high"].rolling(window=26).max()
    period26_low = df["low"].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    period52_high = df["high"].rolling(window=52).max()
    period52_low = df["low"].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    chikou_span = df["close"].shift(-26)

    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def compute_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Compute Hurst Exponent to detect long-term memory in a time series.
    H < 0.5 => Mean reverting series
    H = 0.5 => Random walk
    H > 0.5 => Trending series
    """
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        shifted = price_series[lag:]
        original = price_series[:-lag]
        diff = shifted - original
        # Use small epsilon to avoid log(0)
        std_dev = np.std(diff) if diff.size > 0 else 1e-8
        tau.append(max(1e-8, std_dev))

    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]  # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        return 0.5  # default to random walk if calculation fails


def compute_obv(prices_df: pd.DataFrame) -> pd.Series:
    """Compute On-Balance Volume (OBV)."""
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df["close"].iloc[i] > prices_df["close"].iloc[i - 1]:
            obv.append(obv[-1] + prices_df["volume"].iloc[i])
        elif prices_df["close"].iloc[i] < prices_df["close"].iloc[i - 1]:
            obv.append(obv[-1] - prices_df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df["OBV"] = obv
    return prices_df["OBV"]
