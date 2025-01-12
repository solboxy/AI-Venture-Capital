import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

# Agents
from agents.fundamentals_agent import fundamental_analysis_agent
from agents.technical_analysis_agent import technical_analysis_agent
from agents.sentiment_agent import sentiment_analysis_agent
from agents.risk_evaluation_agent import risk_evaluation_agent
from agents.decision_agent import final_decision_agent
from agents.valuation_analysis_agent import valuation_analysis_agent

# State and utilities
from graph.state import TradingAgentState
from tabulate import tabulate


##### Define the Trading Workflow #####
trading_workflow = StateGraph(TradingAgentState)

def gather_market_data_agent(state: TradingAgentState) -> TradingAgentState:
    """
    Placeholder agent that can load/initialize data if needed.
    Currently returns the state unchanged.
    """
    return state

# Register the agent nodes to the workflow
trading_workflow.add_node("gather_market_data_agent", gather_market_data_agent)
trading_workflow.add_node("technical_analysis_agent", technical_analysis_agent)
trading_workflow.add_node("fundamental_analysis_agent", fundamental_analysis_agent)
trading_workflow.add_node("sentiment_analysis_agent", sentiment_analysis_agent)
trading_workflow.add_node("valuation_analysis_agent", valuation_analysis_agent)
trading_workflow.add_node("risk_evaluation_agent", risk_evaluation_agent)
trading_workflow.add_node("final_decision_agent", final_decision_agent)

# Define the workflow edges
trading_workflow.set_entry_point("gather_market_data_agent")
trading_workflow.add_edge("gather_market_data_agent", "technical_analysis_agent")
trading_workflow.add_edge("gather_market_data_agent", "fundamental_analysis_agent")
trading_workflow.add_edge("gather_market_data_agent", "sentiment_analysis_agent")
trading_workflow.add_edge("gather_market_data_agent", "valuation_analysis_agent")

trading_workflow.add_edge("technical_analysis_agent", "risk_evaluation_agent")
trading_workflow.add_edge("fundamental_analysis_agent", "risk_evaluation_agent")
trading_workflow.add_edge("sentiment_analysis_agent", "risk_evaluation_agent")
trading_workflow.add_edge("valuation_analysis_agent", "risk_evaluation_agent")

trading_workflow.add_edge("risk_evaluation_agent", "final_decision_agent")
trading_workflow.add_edge("final_decision_agent", END)

# Compile the workflow
trading_app = trading_workflow.compile()


def run_trading_system(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: Dict[str, Any],
    show_reasoning: bool = False
) -> str:
    """
    Orchestrates the multi-agent workflow: gather market data, run technical/fundamental/
    sentiment analyses, evaluate risk, and produce a final trading decision.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        portfolio (dict): Dictionary representing the portfolio, e.g. {"cash": 100000, "stock": 50}.
        show_reasoning (bool): If True, each agent prints additional reasoning.

    Returns:
        str: The final decision from the system in JSON format (action, quantity, etc.).
    """
    try:
        final_state = trading_app.invoke(
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
                "metadata": {"show_reasoning": show_reasoning},
            }
        )
        # Return the final agent's message content (the decision in JSON).
        return final_state["messages"][-1].content
    except Exception as e:
        # Log or raise an error if the workflow fails
        error_msg = f"[ERROR] Trading system failed: {e}"
        print(error_msg)
        # You might choose to re-raise or return a fallback JSON response
        raise RuntimeError(error_msg) from e


def main() -> None:
    """
    Command-line entry point for running the trading system in a single-shot mode
    (without backtesting).
    """
    parser = argparse.ArgumentParser(description="Run the trading system")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD). Defaults to 3 months before end date")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument(
        "--show_reasoning",
        action="store_true",
        help="Show reasoning from each agent"
    )
    args = parser.parse_args()

    # Validate or default the end date
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    # Default the start date to 3 months before the end date if none provided
    if not args.start_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Example portfolio
    portfolio_example = {"cash": 100000.0, "stock": 0}

    # Run the trading system
    try:
        final_result_json = run_trading_system(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_example,
            show_reasoning=args.show_reasoning
        )
        print("\nFinal Decision (JSON):")
        print(final_result_json)
    except RuntimeError as err:
        print(f"[FATAL] The trading system encountered an error: {err}")


if __name__ == "__main__":
    main()
