from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
import argparse
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

# Updated tool imports with “v” style naming
from src.tools import (
    fetch_prices,
    fetch_financial_metrics,
    fetch_insider_trades,
    fetch_market_cap,
    fetch_line_items,
    convert_prices_to_dataframe,
    compute_bollinger_bands,
    compute_intrinsic_value,
    compute_macd,
    compute_obv,
    compute_rsi
)

llm = ChatOpenAI(model="gpt-4o")


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Utility function for merging two dictionaries."""
    return {**a, **b}


# Define agent state
class TradingAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]


##### 1. Gather Market Data Agent #####
def gather_market_data_agent(state: TradingAgentState):
    """Gather and preprocess market data (prices, financial metrics, insider trades, etc.)."""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime("%Y-%m-%d")
    if not data["start_date"]:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        if end_date_obj.month > 3:
            start_date_obj = end_date_obj.replace(month=end_date_obj.month - 3)
        else:
            start_date_obj = end_date_obj.replace(
                year=end_date_obj.year - 1, month=end_date_obj.month + 9
            )
        start_date = start_date_obj.strftime("%Y-%m-%d")
    else:
        start_date = data["start_date"]

    # Fetch historical price data
    prices = fetch_prices(
        ticker=data["ticker"],
        start_date=start_date,
        end_date=end_date,
    )

    # Fetch financial metrics
    financial_metrics = fetch_financial_metrics(
        ticker=data["ticker"],
        report_period=end_date,
        period="ttm",
        max_results=1,
    )

    # Fetch insider trades
    insider_trades = fetch_insider_trades(
        ticker=data["ticker"],
        end_date=end_date,
        max_results=5,
    )

    # Fetch market cap
    market_cap = fetch_market_cap(ticker=data["ticker"])

    # Fetch specific line items (e.g., free cash flow)
    financial_line_items = fetch_line_items(
        ticker=data["ticker"],
        line_items=["free_cash_flow"],
        period="ttm",
        max_results=1,
    )

    return {
        "messages": messages,
        "data": {
            **data,
            "prices": prices,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "insider_trades": insider_trades,
            "market_cap": market_cap,
            "financial_line_items": financial_line_items,
        },
    }


##### 2. Technical Analysis Agent #####
def technical_analysis_agent(state: TradingAgentState):
    """Analyze technical indicators (MACD, RSI, Bollinger Bands, OBV) and generate signals."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    prices_df = convert_prices_to_dataframe(data["prices"])

    # Compute indicators
    macd_line, signal_line = compute_macd(prices_df)
    rsi_values = compute_rsi(prices_df)
    upper_band, lower_band = compute_bollinger_bands(prices_df)
    obv_values = compute_obv(prices_df)

    # Generate signals
    signals = []

    # MACD
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append("bullish")
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # RSI
    if rsi_values.iloc[-1] < 30:
        signals.append("bullish")
    elif rsi_values.iloc[-1] > 70:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # Bollinger Bands
    current_price = prices_df["close"].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append("bullish")
    elif current_price > upper_band.iloc[-1]:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # OBV
    obv_slope = obv_values.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append("bullish")
    elif obv_slope < 0:
        signals.append("bearish")
    else:
        signals.append("neutral")

    # Build reasoning
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
                f"RSI is {rsi_values.iloc[-1]:.2f} (oversold)"
                if signals[1] == "bullish"
                else f"RSI is {rsi_values.iloc[-1]:.2f} (overbought)"
                if signals[1] == "bearish"
                else f"RSI is {rsi_values.iloc[-1]:.2f} (neutral)"
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

    # Determine overall signal + confidence
    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    total_signals = len(signals)
    confidence_score = max(bullish_signals, bearish_signals) / total_signals

    # Compile final content
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_score * 100)}%",
        "reasoning": reasoning,
    }
    message = HumanMessage(content=str(message_content), name="quant_agent")

    if show_reasoning:
        show_agent_reasoning(message_content, "Quant Agent")

    return {
        "messages": [message],
        "data": data,
    }


##### 3. Fundamental Analysis Agent #####
def fundamental_analysis_agent(state: TradingAgentState):
    """Analyze fundamental data (profitability, growth, health, ratios, intrinsic value) and generate signals."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    financial_line_item = data["financial_line_items"][0]
    market_cap = data["market_cap"]

    signals = []
    reasoning = {}

    # 1. Profitability
    profitability_score = 0
    if metrics["return_on_equity"] > 0.15:
        profitability_score += 1
    if metrics["net_margin"] > 0.20:
        profitability_score += 1
    if metrics["operating_margin"] > 0.15:
        profitability_score += 1

    profit_signal = (
        "bullish"
        if profitability_score >= 2
        else "bearish"
        if profitability_score == 0
        else "neutral"
    )
    signals.append(profit_signal)
    reasoning["Profitability"] = {
        "signal": profit_signal,
        "details": (
            f"ROE: {metrics['return_on_equity']:.2%}, "
            f"Net Margin: {metrics['net_margin']:.2%}, "
            f"Operating Margin: {metrics['operating_margin']:.2%}"
        ),
    }

    # 2. Growth
    growth_score = 0
    if metrics["revenue_growth"] > 0.10:
        growth_score += 1
    if metrics["earnings_growth"] > 0.10:
        growth_score += 1
    if metrics["book_value_growth"] > 0.10:
        growth_score += 1

    growth_signal = (
        "bullish"
        if growth_score >= 2
        else "bearish"
        if growth_score == 0
        else "neutral"
    )
    signals.append(growth_signal)
    reasoning["Growth"] = {
        "signal": growth_signal,
        "details": (
            f"Revenue Growth: {metrics['revenue_growth']:.2%}, "
            f"Earnings Growth: {metrics['earnings_growth']:.2%}"
        ),
    }

    # 3. Financial Health
    health_score = 0
    if metrics["current_ratio"] > 1.5:
        health_score += 1
    if metrics["debt_to_equity"] < 0.5:
        health_score += 1
    if (
        metrics["free_cash_flow_per_share"]
        > metrics["earnings_per_share"] * 0.8
    ):
        health_score += 1

    health_signal = (
        "bullish"
        if health_score >= 2
        else "bearish"
        if health_score == 0
        else "neutral"
    )
    signals.append(health_signal)
    reasoning["Financial_Health"] = {
        "signal": health_signal,
        "details": (
            f"Current Ratio: {metrics['current_ratio']:.2f}, "
            f"D/E: {metrics['debt_to_equity']:.2f}"
        ),
    }

    # 4. Price Ratios
    pe_ratio = metrics["price_to_earnings_ratio"]
    pb_ratio = metrics["price_to_book_ratio"]
    ps_ratio = metrics["price_to_sales_ratio"]
    ratio_score = 0

    if pe_ratio < 25:
        ratio_score += 1
    if pb_ratio < 3:
        ratio_score += 1
    if ps_ratio < 5:
        ratio_score += 1

    ratio_signal = (
        "bullish"
        if ratio_score >= 2
        else "bearish"
        if ratio_score == 0
        else "neutral"
    )
    signals.append(ratio_signal)
    reasoning["Price_Ratios"] = {
        "signal": ratio_signal,
        "details": (
            f"P/E: {pe_ratio:.2f}, "
            f"P/B: {pb_ratio:.2f}, "
            f"P/S: {ps_ratio:.2f}"
        ),
    }

    # 5. Intrinsic Value
    free_cash_flow = financial_line_item.get("free_cash_flow")
    intrinsic_value = compute_intrinsic_value(
        free_cash_flow=free_cash_flow,
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )

    if market_cap < intrinsic_value:
        signals.append("bullish")
    else:
        signals.append("bearish")

    reasoning["Intrinsic_Value"] = {
        "signal": signals[4],
        "details": f"Intrinsic Value: ${intrinsic_value:,.2f}, Market Cap: ${market_cap:,.2f}",
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

    total_signals = len(signals)
    confidence_score = max(bullish_signals, bearish_signals) / total_signals

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_score * 100)}%",
        "reasoning": reasoning,
    }
    message = HumanMessage(content=str(message_content), name="fundamentals_agent")

    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")

    return {"messages": [message], "data": data}


##### 4. Sentiment Analysis Agent #####
def sentiment_analysis_agent(state: TradingAgentState):
    """Analyze insider trades for sentiment signals (buy => bullish, sell => bearish)."""
    data = state["data"]
    show_reasoning = state["metadata"]["show_reasoning"]
    insider_trades = data["insider_trades"]

    signals = []
    for trade in insider_trades:
        if trade["transaction_shares"] < 0:
            signals.append("bearish")
        else:
            signals.append("bullish")

    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    total_signals = len(signals)
    confidence_score = max(bullish_signals, bearish_signals) / total_signals

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence_score * 100)}%",
        "reasoning": f"Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}",
    }

    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    message = HumanMessage(content=str(message_content), name="sentiment_agent")
    return {"messages": [message], "data": data}


##### 5. Risk Evaluation Agent #####
def risk_evaluation_agent(state: TradingAgentState):
    """Evaluate portfolio risk and set position limits (max size, risk score, trading action)."""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    technical_msg = next(m for m in state["messages"] if m.name == "quant_agent")
    fundamental_msg = next(m for m in state["messages"] if m.name == "fundamentals_agent")
    sentiment_msg = next(m for m in state["messages"] if m.name == "sentiment_agent")

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a risk evaluation specialist.
                Return JSON:
                {
                  "max_position_size": <float>,
                  "risk_score": <1..10>,
                  "trading_action": "buy"|"sell"|"hold",
                  "reasoning": "concise explanation"
                }
                """
            ),
            (
                "human",
                """Technical Analysis: {technical_message}
                Fundamental Analysis: {fundamental_message}
                Sentiment Analysis: {sentiment_message}

                Portfolio:
                Cash: {portfolio_cash}
                Stock: {portfolio_stock}

                Return only JSON with "max_position_size", "risk_score", "trading_action", "reasoning".
                """
            ),
        ]
    )

    prompt = template.invoke(
        {
            "technical_message": technical_msg.content,
            "fundamental_message": fundamental_msg.content,
            "sentiment_message": sentiment_msg.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"],
        }
    )

    result = llm.invoke(prompt)
    message = HumanMessage(content=result.content, name="risk_management_agent")

    if show_reasoning:
        show_agent_reasoning(message.content, "Risk Management Agent")

    return {"messages": state["messages"] + [message]}


##### 6. Final Decision Agent #####
def final_decision_agent(state: TradingAgentState):
    """Make final trading decisions (action, quantity, confidence, signals, reasoning)."""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    technical_msg = next(m for m in state["messages"] if m.name == "quant_agent")
    fundamental_msg = next(m for m in state["messages"] if m.name == "fundamentals_agent")
    sentiment_msg = next(m for m in state["messages"] if m.name == "sentiment_agent")
    risk_msg = next(m for m in state["messages"] if m.name == "risk_management_agent")

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a portfolio manager making final trading decisions.
                RISK MANAGEMENT CONSTRAINTS (Hard):
                  - MUST NOT exceed max_position_size from risk manager
                  - MUST follow the recommended trading_action in risk manager
                Weighted Approach:
                  - Fundamental Analysis = 50%
                  - Technical Analysis = 35%
                  - Sentiment Analysis = 15%
                Output (JSON only):
                {
                  "action": "buy"|"sell"|"hold",
                  "quantity": <positive integer>,
                  "confidence": <0..1>,
                  "agent_signals": <list of agent signals with name, signal, confidence>,
                  "reasoning": "concise explanation"
                }
                """
            ),
            (
                "human",
                """Technical Analysis: {quant_message}
                Fundamental Analysis: {fundamental_message}
                Sentiment Analysis: {sentiment_message}
                Risk Evaluation: {risk_message}

                Portfolio:
                Cash: {portfolio_cash}
                Stock: {portfolio_stock}

                Return only JSON with "action", "quantity", "confidence", "agent_signals", "reasoning". No markdown.
                """
            ),
        ]
    )

    prompt = template.invoke(
        {
            "quant_message": technical_msg.content,
            "fundamental_message": fundamental_msg.content,
            "sentiment_message": sentiment_msg.content,
            "risk_message": risk_msg.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"],
        }
    )

    result = llm.invoke(prompt)
    message = HumanMessage(content=result.content, name="portfolio_management")

    if show_reasoning:
        try:
            parsed_output = json.loads(result.content)
            show_agent_reasoning(parsed_output, "Final Decision Agent")
        except json.JSONDecodeError:
            show_agent_reasoning(result.content, "Final Decision Agent")

    return {"messages": state["messages"] + [message]}


def show_agent_reasoning(output, agent_name):
    """Utility function for console debugging and printing agent decisions."""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    if isinstance(output, (dict, list)):
        print(json.dumps(output, indent=2))
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            print(output)
    print("=" * 48)


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
                HumanMessage(content="Make a trading decision based on the provided data.")
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            },
        }
    )
    return final_state["messages"][-1].content


# Define the workflow
trading_graph = StateGraph(TradingAgentState)

trading_graph.add_node("gather_market_data_agent", gather_market_data_agent)
trading_graph.add_node("technical_analysis_agent", technical_analysis_agent)
trading_graph.add_node("fundamental_analysis_agent", fundamental_analysis_agent)
trading_graph.add_node("sentiment_analysis_agent", sentiment_analysis_agent)
trading_graph.add_node("risk_evaluation_agent", risk_evaluation_agent)
trading_graph.add_node("final_decision_agent", final_decision_agent)

trading_graph.set_entry_point("gather_market_data_agent")
trading_graph.add_edge("gather_market_data_agent", "technical_analysis_agent")
trading_graph.add_edge("gather_market_data_agent", "fundamental_analysis_agent")
trading_graph.add_edge("gather_market_data_agent", "sentiment_analysis_agent")
trading_graph.add_edge("technical_analysis_agent", "risk_evaluation_agent")
trading_graph.add_edge("fundamental_analysis_agent", "risk_evaluation_agent")
trading_graph.add_edge("sentiment_analysis_agent", "risk_evaluation_agent")
trading_graph.add_edge("risk_evaluation_agent", "final_decision_agent")
trading_graph.add_edge("final_decision_agent", END)

compiled_graph = trading_graph.compile()


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

    sample_portfolio = {
        "cash": 100000.0,
        "stock": 0
    }

    final_result = run_trading_system(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=sample_portfolio,
        show_reasoning=args.show_reasoning
    )
    print("\nFinal Result:")
    print(final_result)
