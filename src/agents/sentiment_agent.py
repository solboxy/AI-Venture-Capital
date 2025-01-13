import pandas as pd
import numpy as np
import json

from langchain_core.messages import HumanMessage
from graph.state import TradingAgentState, show_agent_reasoning
from tools.api import fetch_insider_trades

##### Sentiment Analysis Agent #####
def sentiment_analysis_agent(state: TradingAgentState):
    """
    Analyzes market sentiment and generates trading signals based on insider trades.
    
    For each insider trade, if 'transaction_shares' is negative, it is considered bearish. 
    If 'transaction_shares' is positive, it is considered bullish.

    The final signal is 'bullish' if bullish trades > bearish trades, 
    'bearish' if bearish trades > bullish trades, or 'neutral' otherwise.

    Args:
        state (TradingAgentState): The shared agent state which should include:
            - data["ticker"]: Ticker symbol.
            - data["end_date"]: Date through which to fetch insider data.
            - data["analyst_signals"]: Dictionary to store this agent's results.
            - metadata["show_reasoning"]: Boolean for printing reasoning to the console.

    Returns:
        Dict[str, Any]: A dictionary containing updated "messages" and "data" after sentiment analysis.
    """
    data = state.get("data", {})
    show_reasoning = state["metadata"].get("show_reasoning", False)
    end_date = data.get("end_date")
    ticker = data.get("ticker")

    # Ensure "analyst_signals" exists
    data.setdefault("analyst_signals", {})

    # Fetch the insider trades
    insider_trades = fetch_insider_trades(
        ticker=ticker,
        end_date=end_date,
        max_results=5,
    )

    # Convert transaction_shares to a Series, dropping NaN values
    transaction_shares = pd.Series(
        [t.get("transaction_shares") for t in insider_trades]
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
        confidence = max(bullish_signals, bearish_signals) / total_signals
    else:
        confidence = 0.0

    reasoning_text = f"Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}"

    message_content = {
        "signal": overall_signal,
        "confidence_level": f"{round(confidence * 100)}%",
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
    data["analyst_signals"]["sentiment_analysis_agent"] = {
        "signal": overall_signal,
        "confidence_level": f"{round(confidence * 100)}%",
        "reasoning": reasoning_text,
    }

    return {
        "messages": [message],
        "data": data,
    }
