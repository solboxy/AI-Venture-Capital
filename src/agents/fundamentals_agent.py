from langchain_core.messages import HumanMessage
from agents.state import TradingAgentState, show_agent_reasoning
import json


def fundamental_analysis_agent(state: TradingAgentState):
    """Analyzes fundamental data (profitability, growth, financial health, price ratios, intrinsic value)."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    financial_line_item = data["financial_line_items"][0]
    market_cap = data["market_cap"]

    signals = []
    reasoning = {}

    # 1. Profitability
    profitability_score = 0
    if metrics["return_on_equity"] > 0.15:
        profitability_score += 1
    if metrics["net_margin"] > 0.20:
        profitability_score += 1
    if metrics["operating_margin"] > 0.15:
        profitability_score += 1

    profit_signal = (
        "bullish"
        if profitability_score >= 2
        else "bearish"
        if profitability_score == 0
        else "neutral"
    )
    signals.append(profit_signal)
    reasoning["Profitability"] = {
        "signal": profit_signal,
        "details": (
            f"ROE: {metrics['return_on_equity']:.2%}, "
            f"Net Margin: {metrics['net_margin']:.2%}, "
            f"Operating Margin: {metrics['operating_margin']:.2%}"
        ),
    }

    # 2. Growth
    growth_score = 0
    if metrics["revenue_growth"] > 0.10:
        growth_score += 1
    if metrics["earnings_growth"] > 0.10:
        growth_score += 1
    if metrics["book_value_growth"] > 0.10:
        growth_score += 1

    growth_signal = (
        "bullish"
        if growth_score >= 2
        else "bearish"
        if growth_score == 0
        else "neutral"
    )
    signals.append(growth_signal)
    reasoning["Growth"] = {
        "signal": growth_signal,
        "details": (
            f"Revenue Growth: {metrics['revenue_growth']:.2%}, "
            f"Earnings Growth: {metrics['earnings_growth']:.2%}"
        ),
    }

    # 3. Financial Health
    health_score = 0
    if metrics["current_ratio"] > 1.5:
        health_score += 1
    if metrics["debt_to_equity"] < 0.5:
        health_score += 1
    if (
        metrics["free_cash_flow_per_share"]
        > metrics["earnings_per_share"] * 0.8
    ):
        health_score += 1

    health_signal = (
        "bullish"
        if health_score >= 2
        else "bearish"
        if health_score == 0
        else "neutral"
    )
    signals.append(health_signal)
    reasoning["Financial_Health"] = {
        "signal": health_signal,
        "details": (
            f"Current Ratio: {metrics['current_ratio']:.2f}, "
            f"D/E: {metrics['debt_to_equity']:.2f}"
        ),
    }

    # 4. Price Ratios
    pe_ratio = metrics["price_to_earnings_ratio"]
    pb_ratio = metrics["price_to_book_ratio"]
    ps_ratio = metrics["price_to_sales_ratio"]
    ratio_score = 0

    if pe_ratio < 25:
        ratio_score += 1
    if pb_ratio < 3:
        ratio_score += 1
    if ps_ratio < 5:
        ratio_score += 1

    ratio_signal = (
        "bullish"
        if ratio_score >= 2
        else "bearish"
        if ratio_score == 0
        else "neutral"
    )
    signals.append(ratio_signal)
    reasoning["Price_Ratios"] = {
        "signal": ratio_signal,
        "details": (
            f"P/E: {pe_ratio:.2f}, "
            f"P/B: {pb_ratio:.2f}, "
            f"P/S: {ps_ratio:.2f}"
        ),
    }

    # 5. Intrinsic Value
    free_cash_flow = financial_line_item.get("free_cash_flow")
    intrinsic_value = compute_intrinsic_value(
        free_cash_flow=free_cash_flow,
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    if market_cap < intrinsic_value:
        signals.append("bullish")
    else:
        signals.append("bearish")

    reasoning["Intrinsic_Value"] = {
        "signal": signals[4],
        "details": f"Intrinsic Value: ${intrinsic_value:,.2f}, Market Cap: ${market_cap:,.2f}",
    }

    # Overall fundamental signal
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

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_score * 100)}%",
        "reasoning": reasoning,
    }
    message = HumanMessage(content=str(message_content), name="fundamentals_agent")

    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")

    return {"messages": [message], "data": data}