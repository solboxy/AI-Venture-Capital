from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
import argparse
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph, START

from tools import (
    fetch_prices,
    convert_prices_to_dataframe,
    compute_bollinger_bands,
    compute_macd,
    compute_obv,
    compute_rsi
)

llm = ChatOpenAI(model="gpt-4o")


# Define the agent state
class TradingAgentState(TypedDict):
    agent_messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_data: Dict[str, Any]


##### 1. Data Gathering Agent #####
def gather_market_data_agent(state: TradingAgentState):
    """Responsible for gathering and preprocessing market data."""
    messages = state["agent_messages"]
    data = state["agent_data"]

    # Get the historical price data
    prices = fetch_prices(
        data["ticker"], data["start_date"], data["end_date"]
    )

    return {
        "agent_messages": messages,
        "agent_data": {**data, "prices": prices}
    }


##### 2. Technical Analysis Agent #####
def technical_analysis_agent(state: TradingAgentState):
    """Analyzes technical indicators and generates trading signals."""
    data = state["agent_data"]
    prices = data["prices"]
    prices_df = convert_prices_to_dataframe(prices)
    
    # Calculate indicators
    macd_line, signal_line = compute_macd(prices_df)
    rsi = compute_rsi(prices_df)
    upper_band, lower_band = compute_bollinger_bands(prices_df)
    obv = compute_obv(prices_df)
    
    # Generate individual signals
    signals = []
    
    # MACD signal
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append('bullish')
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # RSI signal
    if rsi.iloc[-1] < 30:
        signals.append('bullish')
    elif rsi.iloc[-1] > 70:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # Bollinger Bands signal
    current_price = prices_df['close'].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append('bullish')
    elif current_price > upper_band.iloc[-1]:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # OBV signal
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append('bullish')
    elif obv_slope < 0:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals

    message_content = f"""
    Trading Signal: {overall_signal}
    Confidence (0-1, higher is better): {confidence:.2f}
    """
    message = HumanMessage(
        content=message_content.strip(),
        name="technical_analysis_agent",
    )
    
    return {
        "agent_messages": state["agent_messages"] + [message],
        "agent_data": data
    }


##### 3. Risk Evaluation Agent #####
def risk_evaluation_agent(state: TradingAgentState):
    """Evaluates portfolio risk and sets position limits."""
    portfolio = state["agent_messages"][0].additional_kwargs["portfolio"]
    last_message = state["agent_messages"][-1]

    risk_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a risk management specialist.
                Your job is to take a look at the trading analysis and
                evaluate portfolio exposure and recommend position sizing.
                Provide the following in your output (not as a JSON):
                - max_position_size: <float greater than 0>,
                - risk_score: <integer between 1 and 10>"""
            ),
            MessagesPlaceholder(variable_name="agent_messages"),
            (
                "human",
                f"""Based on the trading analysis below, provide your risk assessment.

                Trading Analysis: {last_message.content}

                Here is the current portfolio:
                Portfolio:
                Cash: ${portfolio['cash']:.2f}
                Current Position: {portfolio['stock']} shares
                
                Only include the max position size and risk score in your output.
                """
            ),
        ]
    )
    chain = risk_prompt | llm
    result = chain.invoke(state).content
    message = HumanMessage(
        content=f"Here is the risk management recommendation: {result}",
        name="risk_evaluation",
    )
    return {"agent_messages": state["agent_messages"] + [message]}


##### 4. Final Decision Agent #####
def final_decision_agent(state: TradingAgentState):
    """Makes final trading decisions and generates orders."""
    portfolio = state["agent_messages"][0].additional_kwargs["portfolio"]
    data = state["agent_data"]
    ticker = data["ticker"]
    last_message = state["agent_messages"][-1]

    portfolio_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the risk management data.
                Provide the following in your output as a JSON:
                - "action": "buy" | "sell" | "hold",
                - "quantity": <positive integer>
                - "ticker": <string>
                Only buy if you have available cash.
                The quantity that you buy must be less than or equal to the max position size.
                Only sell if you have shares in the portfolio to sell.
                The quantity that you sell must be less than or equal to the current position."""
            ),
            MessagesPlaceholder(variable_name="agent_messages"),
            (
                "human",
                f"""Based on the risk management data below, make your trading decision.

                Risk Management Data: {last_message.content}

                Here is the ticker: {ticker}

                Here is the current portfolio:
                Portfolio:
                Cash: ${portfolio['cash']:.2f}
                Current Position: {portfolio['stock']} shares
                
                Only include the action, quantity, and ticker in your output as JSON.
                """
            ),
        ]
    )

    chain = portfolio_prompt | llm
    result = chain.invoke(state).content
    return {"agent_messages": [HumanMessage(content=result, name="final_decision")]}


##### Run the Trading System #####
def run_trading_system(ticker: str, start_date: str, end_date: str, portfolio: dict):
    final_state = compiled_graph.invoke(
        {
            "agent_messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                    additional_kwargs={
                        "ticker": ticker,
                        "start_date": start_date,
                        "end_date": end_date,
                        "portfolio": portfolio
                    },
                )
            ],
            "agent_data": {"ticker": ticker, "start_date": start_date, "end_date": end_date}
        },
        config={"configurable": {"thread_id": 42}},
    )
    return final_state["agent_messages"][-1].content


# Build the graph
trading_graph = StateGraph(TradingAgentState)

# Add nodes
trading_graph.add_node("gather_market_data_agent", gather_market_data_agent)
trading_graph.add_node("technical_analysis_agent", technical_analysis_agent)
trading_graph.add_node("risk_evaluation_agent", risk_evaluation_agent)
trading_graph.add_node("final_decision_agent", final_decision_agent)

# Define workflow
trading_graph.add_edge(START, "gather_market_data_agent")
trading_graph.add_edge("gather_market_data_agent", "technical_analysis_agent")
trading_graph.add_edge("technical_analysis_agent", "risk_evaluation_agent")
trading_graph.add_edge("risk_evaluation_agent", "final_decision_agent")
trading_graph.add_edge("final_decision_agent", END)

compiled_graph = trading_graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading system")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

    sample_portfolio = {
        "cash": 100000.0,
        "stock": 0
    }
    
    result = run_trading_system(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=sample_portfolio
    )
    print(result)
