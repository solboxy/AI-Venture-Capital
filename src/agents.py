from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.tools import (
    fetch_prices,
    convert_prices_to_dataframe,
    compute_bollinger_bands,
    compute_macd,
    compute_obv,
    compute_rsi
)

import argparse
from datetime import datetime

llm = ChatOpenAI(model="gpt-4o")


# Define agent state
class TradingAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Dict[str, Any]


##### 1. Market Data Agent #####
def gather_market_data_agent(state: TradingAgentState):
    """Gather and preprocess market data."""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    if not data["start_date"]:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        # Adjust for month wrap-around
        if end_date_obj.month > 3:
            start_date_obj = end_date_obj.replace(month=end_date_obj.month - 3)
        else:
            start_date_obj = end_date_obj.replace(year=end_date_obj.year - 1, month=end_date_obj.month + 9)
        start_date = start_date_obj.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Fetch historical price data
    prices = fetch_prices(data["ticker"], start_date, end_date)

    return {
        "messages": messages,
        "data": {**data, "prices": prices, "start_date": start_date, "end_date": end_date}
    }


##### 2. Technical Analysis Agent #####
def technical_analysis_agent(state: TradingAgentState):
    """Analyze technical indicators and generate trading signals."""
    show_decisions = state["messages"][0].additional_kwargs["show_decisions"]

    data = state["data"]
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

    # Create the technical analysis agent's message
    message_content = (
        f"Quant Trading Signal: {overall_signal}\n"
        f"Confidence (0-1, higher is better): {confidence:.2f}"
    )
    message = HumanMessage(content=message_content.strip(), name="technical_analysis_agent")

    # Print the decision if the flag is set
    if show_decisions:
        show_agent_decision(message.content, "Technical Analysis Agent")

    return {"messages": state["messages"] + [message], "data": data}


##### 3. Risk Evaluation Agent #####
def risk_evaluation_agent(state: TradingAgentState):
    """Evaluate portfolio risk and set position limits."""
    show_decisions = state["messages"][0].additional_kwargs["show_decisions"]
    portfolio = state["messages"][0].additional_kwargs["portfolio"]
    last_message = state["messages"][-1]

    risk_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a risk management specialist.
                Your job is to look at the trading analysis and evaluate 
                portfolio exposure and recommend position sizing.
                Provide the following in your output (not as a JSON):
                Max Position Size: <float greater than 0>,
                Risk Score: <integer between 1 and 10>"""
            ),
            MessagesPlaceholder(variable_name="messages"),
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
    message_content = f"Risk Management Signal: {result}"
    message = HumanMessage(content=message_content.strip(), name="risk_evaluation_agent")

    # Print the decision if the flag is set
    if show_decisions:
        show_agent_decision(message.content, "Risk Evaluation Agent")

    return {"messages": state["messages"] + [message]}


##### 4. Final Decision Agent #####
def final_decision_agent(state: TradingAgentState):
    """Make final trading decisions and generate orders."""
    show_decisions = state["messages"][0].additional_kwargs["show_decisions"]
    portfolio = state["messages"][0].additional_kwargs["portfolio"]
    last_message = state["messages"][-1]

    portfolio_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the risk data.
                Provide the following in your output:
                - "action": "buy" | "sell" | "hold",
                - "quantity": <positive integer>
                Only buy if you have available cash.
                The quantity that you buy must be less than or equal to the max position size.
                Only sell if you have shares to sell.
                The quantity that you sell must be less than or equal to the current position."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human",
                f"""Based on the risk management data below, make your trading decision.

                Risk Management Data: {last_message.content}

                Here is the current portfolio:
                Portfolio:
                Cash: ${portfolio['cash']:.2f}
                Current Position: {portfolio['stock']} shares

                Only include the action and quantity in your output as JSON.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash.
                You can only sell if you have shares in the portfolio to sell.
                """
            ),
        ]
    )

    chain = portfolio_prompt | llm
    result = chain.invoke(state).content
    message = HumanMessage(content=result, name="final_decision_agent")

    # Print the decision if the flag is set
    if show_decisions:
        show_agent_decision(message.content, "Final Decision Agent")

    return {"messages": state["messages"] + [message]}


def show_agent_decision(output, agent_name):
    print(f"\n{'=' * 5} {agent_name.center(28)} {'=' * 5}")
    print(output)
    print("=" * 40)


##### Run the Trading System #####
def run_trading_system(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_decisions: bool = False
):
    final_state = compiled_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                    additional_kwargs={
                        "portfolio": portfolio,
                        "show_decisions": show_decisions,
                    },
                )
            ],
            "data": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date
            },
        },
    )
    return final_state["messages"][-1].content


# Define the workflow
trading_graph = StateGraph(TradingAgentState)

# Add nodes
trading_graph.add_node("gather_market_data_agent", gather_market_data_agent)
trading_graph.add_node("technical_analysis_agent", technical_analysis_agent)
trading_graph.add_node("risk_evaluation_agent", risk_evaluation_agent)
trading_graph.add_node("final_decision_agent", final_decision_agent)

# Define the workflow sequence
trading_graph.set_entry_point("gather_market_data_agent")
trading_graph.add_edge("gather_market_data_agent", "technical_analysis_agent")
trading_graph.add_edge("technical_analysis_agent", "risk_evaluation_agent")
trading_graph.add_edge("risk_evaluation_agent", "final_decision_agent")
trading_graph.add_edge("final_decision_agent", END)

compiled_graph = trading_graph.compile()


# Add this at the bottom of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading system")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today"
    )
    parser.add_argument(
        "--show-decisions",
        action="store_true",
        help="Show decisions from each agent"
    )
    args = parser.parse_args()

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Sample portfolio - you might want to make this configurable
    sample_portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0         # No initial stock position
    }

    result = run_trading_system(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=sample_portfolio,
        show_decisions=args.show_decisions
    )
    print("\nFinal Result:")
    print(result)
