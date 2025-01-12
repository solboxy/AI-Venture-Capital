import json

from langchain_core.messages import HumanMessage
from graph.state import TradingAgentState, show_agent_reasoning
from tools.api import fetch_financial_metrics

##### Fundamental Analysis Agent #####
def fundamental_analysis_agent(state: TradingAgentState):
    """
    Analyzes fundamental data (profitability, growth, financial health, price ratios)
    and generates trading signals.
    """
    show_reasoning = state["metadata"].get("show_reasoning", False)
    data = state["data"]
    end_date = data["end_date"]

    # Ensure "analyst_signals" exists
    data.setdefault("analyst_signals", {})

    # 1. Fetch the financial metrics
    financial_metrics = fetch_financial_metrics(
        ticker=data["ticker"],
        report_period=end_date,
        period="ttm",
        max_results=1,
    )
    # Pull the most recent metrics
    metrics = financial_metrics[0]

    # Initialize analysis_signals for different fundamental aspects
    analysis_signals = []
    reasoning = {}

    # -------------------------
    # 2. Profitability Analysis
    # -------------------------
    return_on_equity = metrics.get("return_on_equity")
    net_margin = metrics.get("net_margin")
    operating_margin = metrics.get("operating_margin")

    # The thresholds for "good" metrics
    thresholds_p = [
        (return_on_equity, 0.15),  # e.g., strong ROE above 15%
        (net_margin, 0.20),       # healthy net margins
        (operating_margin, 0.15), # strong operating margin
    ]
    profitability_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds_p
    )

    if profitability_score >= 2:
        profitability_signal = "bullish"
    elif profitability_score == 0:
        profitability_signal = "bearish"
    else:
        profitability_signal = "neutral"

    analysis_signals.append(profitability_signal)
    reasoning["profitability"] = {
        "signal": profitability_signal,
        "details": f"ROE: {return_on_equity}, Net Margin: {net_margin}, Op Margin: {operating_margin}"
    }

    # -------------
    # 3. Growth Analysis
    # -------------
    revenue_growth = metrics.get("revenue_growth")
    earnings_growth = metrics.get("earnings_growth")
    book_value_growth = metrics.get("book_value_growth")

    thresholds_g = [
        (revenue_growth, 0.10),
        (earnings_growth, 0.10),
        (book_value_growth, 0.10),
    ]
    growth_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds_g
    )

    if growth_score >= 2:
        growth_signal = "bullish"
    elif growth_score == 0:
        growth_signal = "bearish"
    else:
        growth_signal = "neutral"

    analysis_signals.append(growth_signal)
    reasoning["growth"] = {
        "signal": growth_signal,
        "details": f"Revenue Growth: {revenue_growth}, Earnings Growth: {earnings_growth}, Book Value Growth: {book_value_growth}"
    }

    # -------------------
    # 4. Financial Health
    # -------------------
    current_ratio = metrics.get("current_ratio")
    debt_to_equity = metrics.get("debt_to_equity")
    free_cash_flow_per_share = metrics.get("free_cash_flow_per_share")
    earnings_per_share = metrics.get("earnings_per_share")

    health_score = 0
    if current_ratio and current_ratio > 1.5:
        health_score += 1
    if debt_to_equity is not None and debt_to_equity < 0.5:
        health_score += 1
    # If free cash flow per share is at least 80% of EPS, consider strong
    if (
        free_cash_flow_per_share
        and earnings_per_share
        and free_cash_flow_per_share > 0.8 * earnings_per_share
    ):
        health_score += 1

    if health_score >= 2:
        financial_health_signal = "bullish"
    elif health_score == 0:
        financial_health_signal = "bearish"
    else:
        financial_health_signal = "neutral"

    analysis_signals.append(financial_health_signal)
    reasoning["financial_health"] = {
        "signal": financial_health_signal,
        "details": f"Current Ratio: {current_ratio}, D/E: {debt_to_equity}, FCF/Share: {free_cash_flow_per_share}, EPS: {earnings_per_share}"
    }

    # ----------------------
    # 5. Price Ratios (Valuation)
    # ----------------------
    pe_ratio = metrics.get("price_to_earnings_ratio")
    pb_ratio = metrics.get("price_to_book_ratio")
    ps_ratio = metrics.get("price_to_sales_ratio")

    # We'll consider "lower = better" approach for these ratios
    # If the ratio is above a threshold => it's 'expensive'
    # We'll invert the logic: if ratio < threshold => "bullish" else => "bearish/neutral"
    # E.g. P/E < 25, P/B < 3, P/S < 5 might be "reasonable"

    price_ratios_score = 0
    if pe_ratio and pe_ratio < 25:
        price_ratios_score += 1
    if pb_ratio and pb_ratio < 3:
        price_ratios_score += 1
    if ps_ratio and ps_ratio < 5:
        price_ratios_score += 1

    # We'll say if at least 2 pass => bullish
    if price_ratios_score >= 2:
        price_ratios_signal = "bullish"
    elif price_ratios_score == 0:
        price_ratios_signal = "bearish"
    else:
        price_ratios_signal = "neutral"

    analysis_signals.append(price_ratios_signal)
    reasoning["price_ratios"] = {
        "signal": price_ratios_signal,
        "details": f"P/E: {pe_ratio}, P/B: {pb_ratio}, P/S: {ps_ratio}"
    }

    # -----------------------------
    # 6. Determine Overall Signal
    # -----------------------------
    bullish_signals = analysis_signals.count("bullish")
    bearish_signals = analysis_signals.count("bearish")

    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # We'll define a confidence based on how many of our sub-signals are
    # "bullish" or "bearish" vs neutral.
    total_signals = len(analysis_signals)
    if total_signals > 0:
        confidence = max(bullish_signals, bearish_signals) / total_signals
    else:
        confidence = 0.0

    # Build final message content
    message_content = {
        "signal": overall_signal,
        "confidence_level": f"{round(confidence * 100)}%",
        "reasoning": reasoning,
    }

    # Optionally show reasoning
    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")

    # Create the agent message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamental_analysis_agent",
    )

    # Store signals
    data["analyst_signals"]["fundamental_analysis_agent"] = {
        "signal": overall_signal,
        "confidence_level": f"{round(confidence * 100)}%",
        "reasoning": reasoning,
    }

    return {
        "messages": [message],
        "data": data,
    }
