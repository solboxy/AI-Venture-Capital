from datetime import datetime

from agents.state import TradingAgentState
from langchain_openai.chat_models import ChatOpenAI
from tools.api import fetch_line_items, fetch_financial_metrics, fetch_insider_trades, fetch_market_cap, fetch_prices


def gather_market_data_agent(state: TradingAgentState):
    """Responsible for gathering and preprocessing market data."""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime("%Y-%m-%d")
    if not data["start_date"]:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        if end_date_obj.month > 3:
            start_date_obj = end_date_obj.replace(month=end_date_obj.month - 3)
        else:
            start_date_obj = end_date_obj.replace(
                year=end_date_obj.year - 1, month=end_date_obj.month + 9
            )
        start_date = start_date_obj.strftime("%Y-%m-%d")
    else:
        start_date = data["start_date"]

    # Fetch historical price data
    prices = fetch_prices(
        ticker=data["ticker"],
        start_date=start_date,
        end_date=end_date,
    )

    # Fetch financial metrics
    financial_metrics = fetch_financial_metrics(
        ticker=data["ticker"],
        report_period=end_date,
        period="ttm",
        max_results=1,
    )

    # Fetch insider trades
    insider_trades = fetch_insider_trades(
        ticker=data["ticker"],
        end_date=end_date,
        max_results=5,
    )

    # Fetch market cap
    market_cap = fetch_market_cap(
        ticker=data["ticker"],
    )

    # Fetch specific line items (e.g., free cash flow)
    financial_line_items = fetch_line_items(
        ticker=data["ticker"],
        line_items=["free_cash_flow"],
        period="ttm",
        max_results=1,
    )

    return {
        "messages": messages,
        "data": {
            **data,
            "prices": prices,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "insider_trades": insider_trades,
            "market_cap": market_cap,
            "financial_line_items": financial_line_items,
        },
    }