import math
import json

from langchain_core.messages import HumanMessage
from graph.state import TradingAgentState, show_agent_reasoning
from tools.api import (
    fetch_financial_metrics,
    fetch_market_cap,
    fetch_line_items,
)

##### Valuation Analysis Agent #####
def valuation_analysis_agent(state: TradingAgentState):
    """
    Performs a detailed valuation analysis using multiple methodologies:
    
    1. Owner Earnings (Buffett Method)
    2. DCF (Discounted Cash Flow) Valuation
    
    The results are compared to the company's market cap to determine a 
    valuation gap and an overall signal (bullish, neutral, or bearish).

    Args:
        state (TradingAgentState): The shared agent state containing metadata and data fields.
            Relevant fields include:
            - data["ticker"]: The stock ticker symbol.
            - data["end_date"]: The date indicating the latest financial period to consider.
            - data["analyst_signals"]: A dictionary to store signals from various analysts.
            - metadata["show_reasoning"]: If True, prints reasoning to the console.

    Returns:
        Dict[str, Any]: A dictionary with updated "messages" and "data" after valuation analysis.
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    end_date = data["end_date"]

    # Ensure the "analyst_signals" dict exists
    data.setdefault("analyst_signals", {})

    # 1. Fetch financial metrics
    financial_metrics = fetch_financial_metrics(
        ticker=data["ticker"],
        report_period=end_date,
        period="ttm",
        max_results=1,
    )
    metrics = financial_metrics[0]

    # 2. Fetch line items for valuation
    financial_line_items = fetch_line_items(
        ticker=data["ticker"],
        line_items=[
            "free_cash_flow",
            "net_income",
            "depreciation_and_amortization",
            "capital_expenditure",
            "working_capital",
        ],
        period="ttm",
        max_results=2,
    )
    if len(financial_line_items) < 2:
        # Safety check if the API doesn't return enough records
        return {
            "messages": [],
            "data": data
        }

    current_financial_line_item = financial_line_items[0]
    previous_financial_line_item = financial_line_items[1]

    # 3. Compute working capital change
    working_capital_change = (
        (current_financial_line_item.get("working_capital") or 0)
        - (previous_financial_line_item.get("working_capital") or 0)
    )

    # 4. Owner Earnings Valuation (Buffett Method)
    owner_earnings_value = compute_owner_earnings_value(
        net_income=current_financial_line_item.get("net_income"),
        depreciation=current_financial_line_item.get("depreciation_and_amortization"),
        capex=current_financial_line_item.get("capital_expenditure"),
        working_capital_change=working_capital_change,
        growth_rate=metrics["earnings_growth"],
        required_return=0.15,
        margin_of_safety=0.25,
    )

    # 5. DCF Valuation
    dcf_value = compute_intrinsic_value(
        free_cash_flow=current_financial_line_item.get("free_cash_flow"),
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    # 6. Fetch market cap
    market_cap = fetch_market_cap(ticker=data["ticker"])

    # 7. Calculate valuation gap
    dcf_gap = (dcf_value - market_cap) / market_cap if market_cap else 0
    owner_earnings_gap = (owner_earnings_value - market_cap) / market_cap if market_cap else 0
    valuation_gap = (dcf_gap + owner_earnings_gap) / 2

    # 8. Determine overall signal
    if valuation_gap > 0.15:
        signal = "bullish"
    elif valuation_gap < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    # 9. Build reasoning
    dcf_details = f"Intrinsic Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {dcf_gap:.1%}"
    owner_earnings_details = f"Owner Earnings Value: ${owner_earnings_value:,.2f}, Market Cap: ${market_cap:,.2f}, Gap: {owner_earnings_gap:.1%}"
    reasoning = {
        "dcf_analysis": {
            "signal": (
                "bullish" if dcf_gap > 0.15 else "bearish" if dcf_gap < -0.15 else "neutral"
            ),
            "details": dcf_details,
        },
        "owner_earnings_analysis": {
            "signal": (
                "bullish" if owner_earnings_gap > 0.15 else "bearish" if owner_earnings_gap < -0.15 else "neutral"
            ),
            "details": owner_earnings_details,
        },
    }

    # 10. Calculate confidence_level
    confidence_level = round(abs(valuation_gap) * 100, 2)

    message_content = {
        "signal": signal,
        "confidence_level": confidence_level,
        "reasoning": reasoning,
    }

    # 11. Create the valuation analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_analysis_agent",
    )

    # 12. Show reasoning if requested
    if show_reasoning:
        show_agent_reasoning(message_content, "Valuation Analysis Agent")

    # 13. Store signals in data["analyst_signals"]
    data["analyst_signals"]["valuation_analysis_agent"] = {
        "signal": signal,
        "confidence_level": confidence_level,
        "reasoning": reasoning,
    }

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
    num_years: int = 5,
) -> float:
    """
    Calculates intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income + Depreciation/Amortization 
                     - Capital Expenditures - Working Capital Changes

    Then it projects owner earnings for `num_years` at a given growth_rate, 
    discounts them at a required_return, computes a terminal value, and 
    applies a margin of safety at the end.

    Args:
        net_income (float): The company's net income.
        depreciation (float): Depreciation and amortization expense.
        capex (float): Capital expenditures.
        working_capital_change (float): Change in working capital between periods.
        growth_rate (float, optional): Annual earnings growth rate. Defaults to 0.05 (5%).
        required_return (float, optional): The discount rate or required rate of return. Defaults to 0.15.
        margin_of_safety (float, optional): Fraction to reduce the intrinsic value by for safety. Defaults to 0.25.
        num_years (int, optional): Number of projection years. Defaults to 5.

    Returns:
        float: The estimated intrinsic value per the Owner Earnings method.
    """
    if not all(
        isinstance(x, (int, float))
        for x in [net_income, depreciation, capex, working_capital_change]
    ):
        return 0

    # 1. Calculate initial owner earnings
    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    # 2. Project future owner earnings
    future_values = []
    for year in range(1, num_years + 1):
        projected = owner_earnings * (1 + growth_rate) ** year
        discounted = projected / ((1 + required_return) ** year)
        future_values.append(discounted)

    # 3. Compute terminal value with perpetuity growth
    terminal_growth = min(growth_rate, 0.03)
    terminal_value = (
        future_values[-1] * (1 + terminal_growth)
    ) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / ((1 + required_return) ** num_years)

    # 4. Sum all values and apply margin of safety
    intrinsic_value = sum(future_values) + terminal_value_discounted
    return intrinsic_value * (1 - margin_of_safety)


def compute_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) value for a company 
    using a multi-year projection plus terminal value.

    Args:
        free_cash_flow (float): The current or most recent FCF.
        growth_rate (float, optional): Annualized growth rate for FCF. Defaults to 0.05 (5%).
        discount_rate (float, optional): The discount rate or required rate of return. Defaults to 0.10 (10%).
        terminal_growth_rate (float, optional): Growth rate used in terminal value calculations. Defaults to 0.02.
        num_years (int, optional): Number of years to project FCF. Defaults to 5.

    Returns:
        float: The estimated intrinsic value based on the DCF model.
    """
    if not isinstance(free_cash_flow, (int, float)):
        return 0.0

    # 1. Estimate future cash flows
    projected_cf = [
        free_cash_flow * ((1 + growth_rate) ** year)
        for year in range(num_years)
    ]

    # 2. Calculate present values of projected cash flows
    present_values = [
        cf / ((1 + discount_rate) ** (idx + 1))
        for idx, cf in enumerate(projected_cf)
    ]

    # 3. Calculate terminal value (perpetuity with terminal_growth_rate)
    terminal_value = (
        projected_cf[-1] * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)
    terminal_value_discounted = terminal_value / (
        (1 + discount_rate) ** num_years
    )

    # 4. Sum up present values and terminal value
    return sum(present_values) + terminal_value_discounted


def compute_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculates the absolute change in working capital between two periods.

    Args:
        current_working_capital (float): The current period's working capital.
        previous_working_capital (float): The previous period's working capital.

    Returns:
        float: The difference (current - previous).
    """
    return current_working_capital - previous_working_capital
