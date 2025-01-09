from datetime import datetime

from agents.state import TradingAgentState
from langchain_openai.chat_models import ChatOpenAI
from tools.api import fetch_line_items, fetch_financial_metrics, fetch_insider_trades, fetch_market_cap, fetch_prices



def gather_market_data_agent(state: TradingAgentState):
    """Responsible for gathering and preprocessing market data."""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    start_date = data["start_date"]
    end_date = data["end_date"]

   

    # Fetch insider trades
    insider_trades = fetch_insider_trades(
        ticker=data["ticker"],
        end_date=end_date,
        max_results=5,
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
            "start_date": start_date,
            "end_date": end_date,
            "insider_trades": insider_trades,
            "financial_line_items": financial_line_items,
        },
    }