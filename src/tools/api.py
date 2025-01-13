import os
from typing import Dict, Any, List, Optional
import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)

def _get_api_key() -> str:
    """Retrieves the API key from environment variables, raising a descriptive error if not found."""
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing FINANCIAL_DATASETS_API_KEY environment variable.")
    return api_key


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
        EnvironmentError: If the API key is not found in environment variables.
        requests.exceptions.RequestException: If there's a network issue.
        Exception: If the HTTP request fails with non-200 status code.
        ValueError: If no financial metrics are returned by the API.
    """
    headers = {"X-API-KEY": _get_api_key()}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={max_results}"
        f"&period={period}"
    )

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Network or request error while fetching financial metrics: %s", e)
        raise

    data = response.json() or {}
    financial_metrics = data.get("financial_metrics")
    if not financial_metrics:
        raise ValueError(f"No financial metrics returned for ticker={ticker}, report_period={report_period}.")
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
        EnvironmentError: If the API key is missing.
        requests.exceptions.RequestException: If there's a network error.
        Exception: If the HTTP request fails with non-200 status code.
        ValueError: If no search results are returned.
    """
    headers = {"X-API-KEY": _get_api_key()}
    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "period": period,
        "limit": max_results
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Network or request error while fetching line items: %s", e)
        raise

    data = response.json() or {}
    search_results = data.get("search_results")
    if not search_results:
        raise ValueError(f"No line item results returned for ticker={ticker}, line_items={line_items}.")
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
        EnvironmentError: If the API key is missing.
        requests.exceptions.RequestException: If there's a network error.
        Exception: If the HTTP request fails with non-200 status code.
        ValueError: If no insider trades data is returned.
    """
    headers = {"X-API-KEY": _get_api_key()}
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_lte={end_date}"
        f"&limit={max_results}"
    )

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Network or request error while fetching insider trades: %s", e)
        raise

    data = response.json() or {}
    insider_trades = data.get("insider_trades")
    if not insider_trades:
        raise ValueError(f"No insider trades returned for ticker={ticker} up to {end_date}.")
    return insider_trades


def fetch_market_cap(ticker: str) -> float:
    """
    Fetches the market cap for the given ticker from an external API.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        float: The market capitalization of the given ticker.

    Raises:
        EnvironmentError: If the API key is missing.
        requests.exceptions.RequestException: If there's a network or request error.
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If the returned data does not include 'market_cap'.
    """
    headers = {"X-API-KEY": _get_api_key()}
    url = f"https://api.financialdatasets.ai/company/facts?ticker={ticker}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Network or request error while fetching market cap: %s", e)
        raise

    data = response.json() or {}
    company_facts = data.get("company_facts")
    if not company_facts or "market_cap" not in company_facts:
        raise ValueError(f"No 'market_cap' found in company facts for ticker={ticker}.")
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
        EnvironmentError: If the API key is missing.
        requests.exceptions.RequestException: If there's a network or request error.
        Exception: If the HTTP request fails (non-200 status code).
        ValueError: If no price data is returned by the API.
    """
    headers = {"X-API-KEY": _get_api_key()}
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Network or request error while fetching price data: %s", e)
        raise

    data = response.json() or {}
    prices = data.get("prices")
    if not prices:
        raise ValueError(f"No price data returned for ticker={ticker}, range={start_date} to {end_date}.")
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
    if not prices:
        logger.warning("Received an empty list of prices to convert; returning an empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(prices)
    if "time" not in df.columns:
        logger.error("Missing 'time' field in price data. Returning empty DataFrame.")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["time"], errors='coerce')
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning("Expected column '%s' not found in price data. Setting to NaN.", col)
            df[col] = float('nan')

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
    try:
        prices = fetch_prices(ticker, start_date, end_date)
    except (EnvironmentError, requests.exceptions.RequestException, ValueError) as e:
        logger.error("Failed to fetch prices for %s (%s to %s): %s", ticker, start_date, end_date, e)
        # Return an empty DataFrame to allow the calling function to handle gracefully
        return pd.DataFrame()

    return convert_prices_to_dataframe(prices)
