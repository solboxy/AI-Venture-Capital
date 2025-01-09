import json

from langchain_core.messages import HumanMessage
from graph.state import TradingAgentState, show_agent_reasoning

# Updated tool function import
from tools.api import fetch_financial_metrics

##### Fundamental Analysis Agent #####
def fundamental_analysis_agent(state: TradingAgentState):
    """
    Analyzes fundamental data (profitability, growth, financial health, price ratios)
    and generates trading signals.
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    end_date = data["end_date"]

    # Fetch the financial metrics
    financial_metrics = fetch_financial_metrics(
        ticker=data["ticker"],
        report_period=end_date,
        period="ttm",
        limit=1,
    )

    # Pull the most recent financial metrics
    metrics = financial_metrics[0]

    # Initialize analysis_signals list for different fundamental aspects
    analysis_signals = []
    reasoning = {}

    # 1. Profitability Analysis
    return_on_equity = metrics.get("return_on_equity")
    net_margin = metrics.get("net_margin")
    operating_margin = metrics.get("operating_margin")

    thresholds = [
        (return_on_equity, 0.15),  # Strong ROE above 15%
        (net_margin, 0.20),       # Healthy profit margins
        (operating_margin, 0.15)  # Strong operating efficiency
    ]
    profitability_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )

    profitability_signal = (
        "bullish"
        if profitability_score >= 2
        else "bearish"
        if profitability_score == 0
        else "neutral"
    )
    analysis_signals.append(profitability_signal)
    reasoning["profitability_signal"] = {
        "signal": profitability_signal,
        "details": (
            f"ROE: {metrics['return_on_equity']:.2%}"
            if metrics["return_on_equity"]
            else "ROE: N/A"
        )
        + ", "
        + (
            f"Net Margin: {metrics['net_margin']:.2%}"
            if metrics["net_margin"]
            else "Net Margin: N/A"
        )
        + ", "
        + (
            f"Op Margin: {metrics['operating_margin']:.2%}"
            if metrics["operating_margin"]
            else "Op Margin: N/A"
        ),
    }

    # 2. Growth Analysis
    revenue_growth = metrics.get("revenue_growth")
    earnings_growth = metrics.get("earnings_growth")
    book_value_growth = metrics.get("book_value_growth")

    thresholds = [
        (revenue_growth, 0.10),    # 10% revenue growth
        (earnings_growth, 0.10),   # 10% earnings growth
        (book_value_growth, 0.10)  # 10% book value growth
    ]
    growth_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )

    growth_signal = (
        "bullish"
        if growth_score >= 2
        else "bearish"
        if growth_score == 0
        else "neutral"
    )
    analysis_signals.append(growth_signal)
    reasoning["growth_signal"] = {
        "signal": growth_signal,
        "details": (
            f"Revenue Growth: {metrics['revenue_growth']:.2%}"
            if metrics["revenue_growth"]
            else "Revenue Growth: N/A"
        )
        + ", "
        + (
            f"Earnings Growth: {metrics['earnings_growth']:.2%}"
            if metrics["earnings_growth"]
            else "Earnings Growth: N/A"
        ),
    }

    # 3. Financial Health
    current_ratio = metrics.get("current_ratio")
    debt_to_equity = metrics.get("debt_to_equity")
    free_cash_flow_per_share = metrics.get("free_cash_flow_per_share")
    earnings_per_share = metrics.get("earnings_per_share")

    health_score = 0
    if current_ratio and current_ratio > 1.5:  # Strong liquidity
        health_score += 1
    if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
        health_score += 1
    if (
        free_cash_flow_per_share
        and earnings_per_share
        and free_cash_flow_per_share > earnings_per_share * 0.8
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
    reasoning["financial_health_signal"] = {
        "signal": financial_health_signal,
        "details": (
            f"Current Ratio: {metrics['current_ratio']:.2f}"
            if metrics["current_ratio"]
            else "Current Ratio: N/A"
        )
        + ", "
        + (
            f"D/E: {metrics['debt_to_equity']:.2f}"
            if metrics["debt_to_equity"]
            else "D/E: N/A"
        ),
    }

    # 4. Price to X Ratios
    pe_ratio = metrics.get("price_to_earnings_ratio")
    pb_ratio = metrics.get("price_to_book_ratio")
    ps_ratio = metrics.get("price_to_sales_ratio")

    thresholds = [
        (pe_ratio, 25),  # Reasonable P/E ratio
        (pb_ratio, 3),   # Reasonable P/B ratio
        (ps_ratio, 5),   # Reasonable P/S ratio
    ]
    price_ratios_score = sum(
        metric is not None and metric > threshold
        for metric, threshold in thresholds
    )

    price_ratios_signal = (
        "bullish"
        if price_ratios_score >= 2
        else "bearish"
        if price_ratios_score == 0
        else "neutral"
    )
    analysis_signals.append(price_ratios_signal)
    reasoning["price_ratios_signal"] = {
        "signal": price_ratios_signal,
        "details": (
            f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A"
        ) + ", " + (
            f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A"
        ) + ", " + (
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

    # Calculate confidence_level
    total_signals = len(analysis_signals)
    confidence_level = 0.0
    if total_signals > 0:
        confidence_level = max(bullish_signals, bearish_signals) / total_signals

    # Build the
