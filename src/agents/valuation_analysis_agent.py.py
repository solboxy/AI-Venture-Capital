import math
import json

from langchain_core.messages import HumanMessage

from agents.state import TradingAgentState, show_agent_reasoning
from tools.api import fetch_financial_metrics


##### Valuation Analysis Agent #####
def valuation_analysis_agent(state: TradingAgentState):
    """
    Performs a detailed valuation analysis using multiple methodologies:
    1. Owner Earnings (Buffett Method)
    2. DCF Valuation
    Then compares both valuations to the market cap.
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    current_line_item = data["financial_line_items"][0]
    previous_line_item = data["financial_line_items"][1]
    market_cap = data["market_cap"]
    end_date = data["end_date"]
    # Get the financial metrics
    financial_metrics = fetch_financial_metrics(
        ticker=data["ticker"], 
        report_period=end_date, 
        period='ttm', 
        limit=1,
    )


    metrics = financial_metrics[0]

    # Calculate working capital change
    working_cap_change = (current_line_item.get('working_capital') or 0) - (previous_line_item.get('working_capital') or 0)


    # 1. Owner Earnings Valuation (Buffett Method)
    owner_earnings_value = compute_owner_earnings_value(
        net_income=current_line_item.get("net_income"),
        depreciation=current_line_item.get("depreciation_and_amortization"),
        capex=current_line_item.get("capital_expenditure"),
        working_capital_change=working_cap_change,
        growth_rate=metrics["earnings_growth"],
        required_return=0.15,
        margin_of_safety=0.25
    )

    # 2. DCF Valuation
    dcf_value = compute_intrinsic_value(
        free_cash_flow=current_line_item.get("free_cash_flow"),
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    # Calculate valuation gaps (relative to market cap)
    dcf_gap = (dcf_value - market_cap) / market_cap
    owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap
    valuation_gap = (dcf_gap + owner_earnings_gap) / 2

    # Determine overall signal
    if valuation_gap > 0.15:      # More than 15% undervalued
        signal = "bullish"
    elif valuation_gap < -0.15:  # More than 15% overvalued
        signal = "bearish"
    else:
        signal = "neutral"
        
    # Create the reasoning
    reasoning = {}

    # Build reasoning
    reasoning["dcf_analysis"] = {
        "signal": "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral",
        "details": f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}",
    }
    reasoning["owner_earnings_analysis"] = {
        "signal": "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral",
        "details": f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}",
    }

    message_content = {
        "signal": signal,
        "confidence": f"{abs(valuation_gap):.0%}",
        "reasoning": reasoning
    }

    # Create the valuation analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_analysis_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")

    return {
        "messages": [message],
        "data": data,
    }


def compute_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5
) -> float:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income
                     + Depreciation/Amortization
                     - Capital Expenditures
                     - Working Capital Changes
    """
    # Validate inputs
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
        return 0

    # Calculate initial owner earnings
    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    # Project future owner earnings
    future_values = []
    for year in range(1, num_years + 1):
        projected = owner_earnings * (1 + growth_rate) ** year
        discounted = projected / ((1 + required_return) ** year)
        future_values.append(discounted)

    # Compute terminal value (using perpetuity growth)
    terminal_growth = min(growth_rate, 0.03)
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / ((1 + required_return) ** num_years)

    intrinsic_value = sum(future_values) + terminal_value_discounted
    return intrinsic_value * (1 - margin_of_safety)


def compute_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5
) -> float:
    """
    Computes the discounted cash flow (DCF) to estimate the intrinsic value of a stock.
    """
    if not isinstance(free_cash_flow, (int, float)):
        return 0.0

    projected_cf = [
        free_cash_flow * ((1 + growth_rate) ** i) for i in range(num_years)
    ]
    present_values = [
        cf / ((1 + discount_rate) ** (idx + 1)) for idx, cf in enumerate(projected_cf)
    ]
    terminal_value = (
        projected_cf[-1] * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)
    terminal_value_discounted = terminal_value / ((1 + discount_rate) ** num_years)

    return sum(present_values) + terminal_value_discounted


def compute_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change => more capital is tied up (cash outflow),
    A negative change => less capital is tied up (cash inflow).
    """
    return current_working_capital - previous_working_capital
