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
import json

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
    show_reasoning = state["messages"][0].additional_kwargs["show_reasoning"]

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
        signals.append("bullish")
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # RSI signal
    if rsi.iloc[-1] < 30:
        signals.append("bullish")
    elif rsi.iloc[-1] > 70:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # Bollinger Bands signal
    current_price = prices_df["close"].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append("bullish")
    elif current_price > upper_band.iloc[-1]:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # OBV signal
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append("bullish")
    elif obv_slope < 0:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # Collect reasoning details
    reasoning = {
        "MACD": {
            "signal": signals[0],
            "details": (
                "MACD Line crossed above Signal Line"
                if signals[0] == "bullish"
                else "MACD Line crossed below Signal Line"
                if signals[0] == "bearish"
                else "No crossover"
            ),
        },
        "RSI": {
            "signal": signals[1],
            "details": (
                f"RSI is {rsi.iloc[-1]:.2f} (oversold)"
                if signals[1] == "bullish"
                else f"RSI is {rsi.iloc[-1]:.2f} (overbought)"
                if signals[1] == "bearish"
                else f"RSI is {rsi.iloc[-1]:.2f} (neutral)"
            ),
        },
        "Bollinger": {
            "signal": signals[2],
            "details": (
                "Price is below lower band"
                if signals[2] == "bullish"
                else "Price is above upper band"
                if signals[2] == "bearish"
                else "Price is within bands"
            ),
        },
        "OBV": {
            "signal": signals[3],
            "details": f"OBV slope is {obv_slope:.2f} ({signals[3]})",
        },
    }

    # Determine overall signal
    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals

    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
    }

    # Create the technical analysis message
    message = HumanMessage(
        content=json.dumps(message_content),  # Convert dict to JSON string
        name="technical_analysis_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Technical Analysis Agent")

    return {"messages": state["messages"] + [message], "data": data}


##### 3. Risk Evaluation Agent #####
def risk_evaluation_agent(state: TradingAgentState):
    """Evaluate portfolio risk and set position limits."""
    show_reasoning = state["messages"][0].additional_kwargs["show_reasoning"]
    portfolio = state["messages"][0].additional_kwargs["portfolio"]
    technical_message = state["messages"][-1]

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a risk evaluation specialist.
                Your job is to assess the trading analysis and recommend position sizing.
                Return the following in JSON (no markdown):
                "max_position_size": <float>,
                "risk_rating": <integer between 1 and 10>,
                "trading_action": <"buy"|"sell"|"hold">,
                "reasoning": <concise explanation>"""
            ),
            (
                "human",
                """Quant Analysis:
                {technical_message}

                Current Portfolio:
                - Cash: {portfolio_cash}
                - Position: {portfolio_stock} shares

                Please provide the max position size, risk rating, recommended trading action,
                and reasoning in JSON, without markdown. For example:
                {
                  "max_position_size": 1234.56,
                  "risk_rating": 7,
                  "trading_action": "buy",
                  "reasoning": "Your short reasoning here"
                }
                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "technical_message": technical_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"],
        }
    )

    # Invoke the LLM
    result = llm.invoke(prompt)
    message = HumanMessage(content=result.content, name="risk_evaluation_agent")

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Risk Evaluation Agent")

    return {"messages": state["messages"] + [message]}


##### 4. Final Decision Agent #####
def final_decision_agent(state: TradingAgentState):
    """Make final trading decisions and generate orders."""
    show_reasoning = state["messages"][0].additional_kwargs["show_reasoning"]
    portfolio = state["messages"][0].additional_kwargs["portfolio"]
    risk_message = state["messages"][-1]
    technical_message = state["messages"][-2]

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                You will consolidate the analysis from the technical and risk evaluation teams.
                Return your decision in JSON with:
                {
                  "action": "buy"|"sell"|"hold",
                  "quantity": <positive integer>,
                  "reasoning": "<brief explanation>"
                }
                Only buy if you have enough cash, and keep quantity <= max position size.
                Only sell if you have shares, and keep quantity <= current position."""
            ),
            (
                "human",
                """Technical Analysis: {technical_message}
                Risk Evaluation: {risk_message}

                Portfolio:
                Cash: {portfolio_cash}
                Position: {portfolio_stock} shares

                Return your decision as JSON only (no markdown). For example:
                {
                  "action": "buy",
                  "quantity": 100,
                  "reasoning": "example"
                }"""
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "technical_message": technical_message.content,
            "risk_message": risk_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"],
        }
    )
    # Invoke the LLM
    result = llm.invoke(prompt)

    # Create the final decision message
    message = HumanMessage(content=result.content, name="final_decision_agent")

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Final Decision Agent")

    return {"messages": state["messages"] + [message]}


def show_agent_reasoning(output, agent_name):
    """Utility function for printing agent decisions."""
    print(f"\n{'=' * 10} {agent_name.center(40)} {'=' * 10}")
    if isinstance(output, (dict, list)):
        # If output is a dictionary or list, just pretty print it
        print(json.dumps(output, indent=2))
    else:
        try:
            # Attempt to parse string as JSON, then pretty print
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Otherwise, just print raw string
            print(output)
    print("=" * 60)


##### Run the Trading System #####
def run_trading_system(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False
):
    final_state = compiled_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                    additional_kwargs={
                        "portfolio": portfolio,
                        "show_reasoning": show_reasoning,
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

# Define the workflow
trading_graph.set_entry_point("gather_market_data_agent")
trading_graph.add_edge("gather_market_data_agent", "technical_analysis_agent")
trading_graph.add_edge("technical_analysis_agent", "risk_evaluation_agent")
trading_graph.add_edge("risk_evaluation_agent", "final_decision_agent")
trading_graph.add_edge("final_decision_agent", END)

compiled_graph = trading_graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
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
        "--show-reasoning",
        action="store_true",
        help="Show reasoning from each agent"
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

    # Sample portfolio
    sample_portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0         # No initial stock position
    }

    final_decision = run_trading_system(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=sample_portfolio,
        show_reasoning=args.show_reasoning
    )
    print("\nFinal Result:")
    print(final_decision)
