import json

from langchain_core.messages import HumanMessage
from agents.agent_state_utils import TradingAgentState, show_agent_reasoning


##### Fundamental Analysis Agent #####
def fundamental_analysis_agent(state: TradingAgentState):
    """
    Analyzes fundamental data (profitability, growth, financial health, price ratios)
    and generates trading signals.
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]

    # Track individual signals for each fundamental aspect
    analysis_signals = []
    reasoning = {}
    
    return_on_equity = metrics.get("return_on_equity", None)
    net_margin = metrics.get("net_margin", None)
    operating_margin = metrics.get("operating_margin", None)

    # 1. Profitability
    profitability_score = 0
    
   
    if metrics["return_on_equity"] and metrics["return_on_equity"] > 0.15:  # Strong ROE above 15%
        profitability_score += 1
    if net_margin and net_margin > 0.20:  # Healthy profit margins
        profitability_score += 1
    if operating_margin and operating_margin > 0.15:  # Strong operating efficiency
        profitability_score += 1

    profitability_signal = (
        "bullish"
        if profitability_score >= 2
        else "bearish"
        if profitability_score == 0
        else "neutral"
    )
    analysis_signals.append(profitability_signal)
    reasoning["profitability"] = {
        "signal": profitability_signal,
        "details": (
            f"ROE: {metrics['return_on_equity']:.2%}" if metrics["return_on_equity"] else "ROE: N/A"
        )
        + ", "
        + (
            f"Net Margin: {metrics['net_margin']:.2%}" if metrics["net_margin"] else "Net Margin: N/A"
        )
        + ", "
        + (
            f"Op Margin: {metrics['operating_margin']:.2%}" if metrics["operating_margin"] else "Op Margin: N/A"
        ),
    }

    # 2. Growth
    growth_score = 0
    if metrics["revenue_growth"] and metrics["revenue_growth"] > 0.10:
        growth_score += 1
    if metrics["earnings_growth"] and metrics["earnings_growth"] > 0.10:
        growth_score += 1
    if metrics["book_value_growth"] and metrics["book_value_growth"] > 0.10:
        growth_score += 1

    growth_signal = (
        "bullish"
        if growth_score >= 2
        else "bearish"
        if growth_score == 0
        else "neutral"
    )
    analysis_signals.append(growth_signal)
    reasoning["growth"] = {
        "signal": growth_signal,
        "details": (
            f"Revenue Growth: {metrics['revenue_growth']:.2%}" if metrics["revenue_growth"] else "Revenue Growth: N/A"
        )
        + ", "
        + (
            f"Earnings Growth: {metrics['earnings_growth']:.2%}" if metrics["earnings_growth"] else "Earnings Growth: N/A"
        ),
    }

    # 3. Financial Health
    health_score = 0
    if metrics["current_ratio"] and metrics["current_ratio"] > 1.5:
        health_score += 1
    if metrics["debt_to_equity"] and metrics["debt_to_equity"] < 0.5:
        health_score += 1
    if (
        metrics.get("free_cash_flow_per_share")
        and metrics.get("earnings_per_share")
        and metrics["free_cash_flow_per_share"] > metrics["earnings_per_share"] * 0.8
    ):
        health_score += 1

    financial_health_signal = (
        "bullish"
        if health_score >= 2
        else "bearish"
        if health_score == 0
        else "neutral"
    )
    analysis_signals.append(financial_health_signal)
    reasoning["financial_health"] = {
        "signal": financial_health_signal,
        "details": (
            f"Current Ratio: {metrics['current_ratio']:.2f}" if metrics["current_ratio"] else "Current Ratio: N/A"
        )
        + ", "
        + (
            f"D/E: {metrics['debt_to_equity']:.2f}" if metrics["debt_to_equity"] else "D/E: N/A"
        ),
    }

    # 4. Price Ratios
    pe_ratio = metrics.get("price_to_earnings_ratio", None)
    pb_ratio = metrics.get("price_to_book_ratio", None)
    ps_ratio = metrics.get("price_to_sales_ratio", None)

    price_ratios_score = 0
    if pe_ratio and pe_ratio < 25:
        price_ratios_score += 1
    if pb_ratio and pb_ratio < 3:
        price_ratios_score += 1
    if ps_ratio and ps_ratio < 5:
        price_ratios_score += 1

    price_ratios_signal = (
        "bullish"
        if price_ratios_score >= 2
        else "bearish"
        if price_ratios_score == 0
        else "neutral"
    )
    analysis_signals.append(price_ratios_signal)
    reasoning["price_ratios"] = {
        "signal": price_ratios_signal,
        "details": (
            f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A"
        )
        + ", "
        + (
            f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A"
        )
        + ", "
        + (
            f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"
        ),
    }

    # Determine overall signal
    bullish_signals = analysis_signals.count("bullish")
    bearish_signals = analysis_signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Calculate confidence
    total_signals = len(analysis_signals)
    if total_signals > 0:
        confidence_level = max(bullish_signals, bearish_signals) / total_signals
    else:
        confidence_level = 0.0

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_level * 100)}%",
        "reasoning": reasoning,
    }

    # Build the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamental_analysis_agent",
    )

    # Print the reasoning if requested
    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")

    return {
        "messages": [message],
        "data": data,
    }
