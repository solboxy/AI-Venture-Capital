from langchain_core.messages import HumanMessage

from agents.state import TradingAgentState, show_agent_reasoning
import json


##### Sentiment Analysis Agent #####
def sentiment_analysis_agent(state: TradingAgentState):
    """
    Analyzes insider trades to generate a market sentiment signal (buy => bullish, sell => bearish).
    """
    data = state["data"]
    insider_trades = data["insider_trades"]
    show_reasoning = state["metadata"]["show_reasoning"]

    signals = []
    for trade in insider_trades:
        transaction_shares = trade.get("transaction_shares")
        if transaction_shares is None:
            # Skip if no transaction_shares info
            continue
        if transaction_shares < 0:
            signals.append("bearish")
        else:
            signals.append("bullish")

    # Determine overall sentiment signal
    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Calculate confidence based on proportion of bull/bear signals
    total_signals = len(signals)
    confidence_value = 0.0
    if total_signals > 0:
        confidence_value = max(bullish_signals, bearish_signals) / total_signals

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_value * 100)}%",
        "reasoning": f"Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}",
    }

    # Print the reasoning if the user requests it
    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    # Create the sentiment analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_analysis_agent",
    )

    return {
        "messages": [message],
        "data": data,
    }
