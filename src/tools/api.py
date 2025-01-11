import os
from typing import Dict, Any, List
import pandas as pd
import requests

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


def fetch_line_items(
    ticker: str,
    line_items: List[str],
    period: str = "ttm",
    max_results: int = 1
) -> List[Dict[str, Any]]:
    """
    Fetch specific financial statement line items (e.g. free_cash_flow) for a given ticker.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "period": period,
        "limit": max_results
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching line items: {response.status_code} - {response.text}"
        )
    data = response.json()
    search_results = data.get("search_results")
    if not search_results:
        raise ValueError("No search results returned")
    return search_results


def fetch_insider_trades(
    ticker: str,
    end_date: str,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetch insider trades for a given ticker up to a specified end date.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
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


def fetch_market_cap(ticker: str) -> float:
    """
    Fetch the market cap for the given ticker from an external API.
    Returns a float representing the market cap.
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
    Fetch price data for a given ticker and date range from an external API.
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
    Convert fetched price data into a pandas DataFrame.
    """
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def fetch_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Utility function that fetches raw price data, then converts it into a DataFrame.
    """
    prices = fetch_prices(ticker, start_date, end_date)
    return convert_prices_to_dataframe(prices)
