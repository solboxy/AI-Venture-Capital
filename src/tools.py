import os
import requests
import pandas as pd
from typing import Dict, Any, List


def fetch_financial_metrics(
    ticker: str,
    report_period: str,
    period: str = "ttm",
    max_results: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetch financial metrics from an external API.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={max_results}"
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


def fetch_cash_flow_statements(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    max_results: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetch cash flow statements from an external API.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financials/cash-flow-statements/"
        f"?ticker={ticker}"
        f"&report_period_lte={end_date}"
        f"&limit={max_results}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching cash flow statements: {response.status_code} - {response.text}"
        )
    data = response.json()
    cash_flow_statements = data.get("cash_flow_statements")
    if not cash_flow_statements:
        raise ValueError("No cash flow statements returned")
    return cash_flow_statements


def fetch_insider_trades(
    ticker: str,
    start_date: str,
    end_date: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch insider trades for a given ticker and date range.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_gte={start_date}"
        f"&filing_date_lte={end_date}"
        f"&limit={max_results}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching insider trades: {response.status_code} - {response.text}"
        )
    data = response.json()
    insider_trades = data.get("insider_trades")
    if not insider_trades:
        raise ValueError("No insider trades returned")
    return insider_trades


def fetch_market_cap(
    ticker: str,
) -> float:
    """
    Fetch the market cap for the given ticker from an external API.
    Returns the float value of the market cap.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = f"https://api.financialdatasets.ai/company/facts?ticker={ticker}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching market cap data: {response.status_code} - {response.text}"
        )
    data = response.json()
    company_facts = data.get("company_facts")
    if not company_facts or "market_cap" not in company_facts:
        raise ValueError("No market cap data found")
    return company_facts["market_cap"]


def fetch_prices(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """
    Fetch price data from an external API for a given ticker and date range.
    """
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
            f"Error fetching price data: {response.status_code} - {response.text}"
        )
    data = response.json()
    prices = data.get("prices")
    if not prices:
        raise ValueError("No price data returned")
    return prices


def convert_prices_to_dataframe(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the fetched price data into a pandas DataFrame.
    """
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def compute_confidence_level(signals: Dict[str, Any]) -> float:
    """
    Calculate a confidence level based on the difference between SMAs (Simple Moving Averages).
    Normalizes the confidence to a value between 0 and 1.
    """
    sma_diff_prev = abs(signals["sma_5_prev"] - signals["sma_20_prev"])
    sma_diff_curr = abs(signals["sma_5_curr"] - signals["sma_20_curr"])
    diff_change = sma_diff_curr - sma_diff_prev

    confidence = min(max(diff_change / signals["current_price"], 0), 1)
    return confidence


def compute_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence) and its signal line from a prices DataFrame.
    """
    ema_12 = prices_df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def compute_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index) for a given period from a prices DataFrame.
    """
    delta = prices_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(
    prices_df: pd.DataFrame,
    window: int = 20
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Bollinger Bands with a specified rolling window from a prices DataFrame.
    Returns upper and lower bands.
    """
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def compute_obv(prices_df: pd.DataFrame) -> pd.Series:
    """
    Compute OBV (On-Balance Volume) from a prices DataFrame.
    """
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


def compute_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5
) -> float:
    """
    Compute the discounted cash flow (DCF) / intrinsic value of a company
    based on the current free cash flow, growth rate, discount rate,
    and terminal growth.
    """
    # Estimate future cash flows
    projected_cash_flows = [
        free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)
    ]

    # Calculate present values
    present_values = []
    for i in range(num_years):
        present_value = projected_cash_flows[i] / ((1 + discount_rate) ** (i + 1))
        present_values.append(present_value)

    # Terminal value
    terminal_value = (
        projected_cash_flows[-1] * (1 + terminal_growth_rate)
        / (discount_rate - terminal_growth_rate)
    )
    terminal_present_value = terminal_value / ((1 + discount_rate) ** num_years)

    # Sum present values + terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value
