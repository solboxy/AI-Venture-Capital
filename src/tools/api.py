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
    Fetches financial metrics for a specified ticker from an external API.

    Args:
        ticker (str): Stock ticker (e.g. "AAPL").
        report_period (str): The financial report period limit (e.g. "2022-12-31").
        period (str, optional): Period type (e.g. "annual", "quarter", "ttm"). Defaults to "ttm".
        max_results (int, optional): Maximum number of records to fetch. Defaults to 1.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing key-value pairs of financial metrics.

    Raises:
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If no financial metrics are returned by the API.
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
    Fetches specific financial statement line items (e.g., free_cash_flow) for a given ticker.

    Args:
        ticker (str): Stock ticker symbol.
        line_items (List[str]): A list of line item names to retrieve (e.g. "free_cash_flow").
        period (str, optional): Period type (e.g. "annual", "quarter", "ttm"). Defaults to "ttm".
        max_results (int, optional): The maximum number of records to fetch. Defaults to 1.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the requested line item data.

    Raises:
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If no search results are returned.
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
    Fetches insider trades for a given ticker up to a specified end date.

    Args:
        ticker (str): The stock ticker symbol.
        end_date (str): The cut-off date for insider trades (e.g. "2023-01-01").
        max_results (int, optional): Maximum number of trade records to fetch. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries describing each insider trade event.

    Raises:
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If no insider trades data is returned.
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
    Fetches the market cap for the given ticker from an external API.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        float: The market capitalization of the given ticker.

    Raises:
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If the returned data does not include 'market_cap'.
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
    Fetches daily price data for a given ticker and date range from an external API.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the price data in "YYYY-MM-DD" format.
        end_date (str): The end date for the price data in "YYYY-MM-DD" format.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing date, open, close, high, low, and volume data.

    Raises:
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If no price data is returned by the API.
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
    Converts a list of price data dictionaries into a pandas DataFrame, 
    setting the DateTime index and sorting by date.

    Args:
        prices (List[Dict[str, Any]]): The list of price data dictionaries.

    Returns:
        pd.DataFrame: DataFrame indexed by datetime, containing columns for open, close, high, low, and volume.
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
    Fetches raw price data for a given date range and converts it into a pandas DataFrame.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the price data (YYYY-MM-DD).
        end_date (str): The end date for the price data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A pandas DataFrame containing the open, close, high, low, and volume data 
                      indexed by date for the specified ticker and date range.
    """
    prices = fetch_prices(ticker, start_date, end_date)
    return convert_prices_to_dataframe(prices)
