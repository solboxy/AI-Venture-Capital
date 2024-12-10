import os
import requests
import pandas as pd
from typing import Dict, Union
from tavily import TavilyClient


def fetch_prices(ticker: str, start_date: str, end_date: str):
    """Fetch price data from an external API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    prices = data.get("prices")
    if not prices:
        raise ValueError("No price data returned")
    return prices


def convert_prices_to_dataframe(prices):
    """Convert fetched prices to a Pandas DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def fetch_price_data(ticker: str, start_date: str, end_date: str):
    """Utility function that fetches raw prices and converts them into a DataFrame."""
    prices = fetch_prices(ticker, start_date, end_date)
    return convert_prices_to_dataframe(prices)


def fetch_financial_metrics(ticker: str, report_period: str, period: str = "ttm", limit: int = 1):
    """Fetch financial metrics from the external API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching financial metrics: {response.status_code} - {response.text}"
        )
    data = response.json()
    financial_metrics = data.get("financial_metrics")
    if not financial_metrics:
        raise ValueError("No financial metrics returned")
    return financial_metrics


def fetch_news(query: str, end_date: str, max_results: int = 5) -> Union[Dict, str]:
    """
    Perform a web search using the Tavily API to retrieve news articles.

    Returns up-to-date results filtered by end_date, ensuring only items published on or before end_date are included.
    """
    from datetime import datetime

    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = client.search(query, topic="news", max_results=max_results)

    # Convert end_date string to datetime object
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Filter results by published_date if present
    if "results" in response:
        filtered_results = []
        for result in response["results"]:
            if "published_date" in result:
                pub_date = datetime.strptime(result["published_date"], "%a, %d %b %Y %H:%M:%S %Z")
                if pub_date.date() <= end_date_dt.date():
                    filtered_results.append(result)

        response["results"] = filtered_results

    return response


def compute_confidence_level(signals: Dict[str, float]) -> float:
    """Compute a confidence level based on the difference between SMAs (Simple Moving Averages)."""
    sma_diff_prev = abs(signals["sma_5_prev"] - signals["sma_20_prev"])
    sma_diff_curr = abs(signals["sma_5_curr"] - signals["sma_20_curr"])
    diff_change = sma_diff_curr - sma_diff_prev
    # Normalize confidence between 0 and 1
    confidence = min(max(diff_change / signals["current_price"], 0), 1)
    return confidence


def compute_macd(prices_df: pd.DataFrame):
    """Compute MACD (Moving Average Convergence Divergence) and its signal line."""
    ema_12 = prices_df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def compute_rsi(prices_df: pd.DataFrame, period: int = 14):
    """Compute RSI (Relative Strength Index)."""
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(prices_df: pd.DataFrame, window: int = 20):
    """Compute Bollinger Bands with a given window size."""
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def compute_obv(prices_df: pd.DataFrame):
    """Compute OBV (On-Balance Volume)."""
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
