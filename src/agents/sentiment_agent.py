import pandas as pd
import numpy as np
import json

from langchain_core.messages import HumanMessage
from graph.state import TradingAgentState, show_agent_reasoning

# Updated tool function import
from tools.api import fetch_insider_trades

##### Sentiment Analysis Agent #####
def sentiment_analysis_agent(state: TradingAgentState):
    """
    Analyzes market sentiment and generates trading signals.
    Specifically looks at insider trades:
    - Negative transaction_shares => 'bearish'
    - Positive transaction_shares => 'bullish'
    """
    data = state["data"]
    end_date = data["end_date"]
    show_reasoning = state["metadata"]["show_reasoning"]

    # Fetch the insider trades
    insider_trades = fetch_insider_trades(
        ticker=data["ticker"],
        end_date=end_date,
        limit=5,
    )

    # Convert transaction_shares to a Series, dropping NaN values
    transaction_shares = pd.Series(
        [trade["transaction_shares"] for trade in insider_trades]
    ).dropna()

    # Vectorized approach: negative => 'bearish', else => 'bullish'
    analysis_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

    # Determine overall sentiment signal
    bullish_signals = analysis_signals.count("bullish")
    bearish_signals = analysis_signals.count("bearish")

    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Calculate confidence_level based on proportion
    total_signals = len(analysis_signals)
    if total_signals > 0:
        confidence_level = max(bullish_signals, bearish_signals) / total_signals
    else:
        confidence_level = 0.0

    reasoning_text = (
        f"Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}"
    )

    message_content = {
        "signal": overall_signal,
        "confidence_level": f"{round(confidence_level * 100)}%",
        "reasoning": reasoning_text,
    }

    # Print reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    # Create the sentiment analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_analysis_agent",
    )

    # Add the signal to the analyst_signals dictionary
    state["data"].setdefault("analyst_signals", {})
    state["data"]["analyst_signals"]["sentiment_analysis_agent"] = {
        "signal": overall_signal,
        "confidence_level": f"{round(confidence_level * 100)}%",
        "reasoning": reasoning_text,
    }

    return {
        "messages": [message],
        "data": data,
    }
