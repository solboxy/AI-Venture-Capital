from langchain_core.messages import HumanMessage

from graph.state import TradingAgentState, show_agent_reasoning
import json


##### Sentiment Analysis Agent #####
def sentiment_analysis_agent(state: TradingAgentState):
    """
    Analyzes insider trades for market sentiment signals.
    If transaction_shares is negative => 'bearish'
    If transaction_shares is positive => 'bullish'
    """
    data = state["data"]
    insider_trades = data["insider_trades"]
    show_reasoning = state["metadata"]["show_reasoning"]

    # Extract transaction_shares and drop NaN values
    transaction_shares_series = pd.Series([trade["trade"] for trade in insider_trades]).dropna()

    # Vectorized approach to assign sentiment signals
    # Condition: Negative => 'bearish'; Else => 'bullish'
    signals_array = np.where(transaction_shares_series < 0, "bearish", "bullish").tolist()

    # Determine overall sentiment signal
    bullish_signals = signals_array.count("bullish")
    bearish_signals = signals_array.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Calculate confidence level based on proportion of bullish vs bearish signals
    total_signals = len(signals_array)
    if total_signals > 0:
        confidence_value = max(bullish_signals, bearish_signals) / total_signals
    else:
        confidence_value = 0.0

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_value * 100)}%",
        "reasoning": f"Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}",
    }

    # If show_reasoning is True, print the sentiment analysis details
    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    # Create the final message for the sentiment analysis agent
    message = HumanMessage(
        content=json.dumps(message_content),
        name="sentiment_analysis_agent",
    )

    return {
        "messages": [message],
        "data": data,
    }
