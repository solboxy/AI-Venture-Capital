import argparse
import logging
from datetime import datetime
from typing import Any, Dict

from dateutil.relativedelta import relativedelta

from agents.decision_agent import final_decision_agent
from agents.fundamentals_agent import fundamental_analysis_agent
from agents.risk_evaluation_agent import risk_evaluation_agent
from agents.sentiment_agent import sentiment_analysis_agent
from agents.technical_analysis_agent import technical_analysis_agent
from agents.valuation_analysis_agent import valuation_analysis_agent

from graph.state import TradingAgentState
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def gather_market_data_agent(state: TradingAgentState) -> TradingAgentState:
    """
    Placeholder agent to fetch or validate market data and incorporate it 
    into the 'data' field of the TradingAgentState. In production, replace
    the placeholder code with a real data-fetching process (e.g., from an API).

    Args:
        state (TradingAgentState): The current shared state containing 'data'.

    Returns:
        TradingAgentState: The state updated with newly fetched/validated data.
    """
    ticker = state["data"].get("ticker")
    if not ticker:
        raise ValueError("No ticker specified; cannot fetch market data.")

    logger.info("Gathering market data for ticker=%s", ticker)
    # ======= Replace with Real Data-Fetching Logic =======
    # For example:
    # market_data = fetch_market_data(ticker, ...)
    # state["data"]["market_data"] = market_data
    # ============================================
    return state


def create_trading_workflow() -> StateGraph:
    """
    Creates and configures a multi-agent workflow (StateGraph) for trading decisions.

    The workflow includes:
    1. gather_market_data_agent
    2. technical_analysis_agent
    3. fundamental_analysis_agent
    4. sentiment_analysis_agent
    5. valuation_analysis_agent
    6. risk_evaluation_agent
    7. final_decision_agent

    Returns:
        StateGraph: A compiled StateGraph that defines the multi-agent process flow.
    """
    trading_workflow = StateGraph(TradingAgentState)

    # Register agent nodes
    trading_workflow.add_node("gather_market_data_agent", gather_market_data_agent)
    trading_workflow.add_node("technical_analysis_agent", technical_analysis_agent)
    trading_workflow.add_node("fundamental_analysis_agent", fundamental_analysis_agent)
    trading_workflow.add_node("sentiment_analysis_agent", sentiment_analysis_agent)
    trading_workflow.add_node("valuation_analysis_agent", valuation_analysis_agent)
    trading_workflow.add_node("risk_evaluation_agent", risk_evaluation_agent)
    trading_workflow.add_node("final_decision_agent", final_decision_agent)

    # Define data flow edges
    trading_workflow.set_entry_point("gather_market_data_agent")

    # gather_market_data_agent feeds four agents
    trading_workflow.add_edge("gather_market_data_agent", "technical_analysis_agent")
    trading_workflow.add_edge("gather_market_data_agent", "fundamental_analysis_agent")
    trading_workflow.add_edge("gather_market_data_agent", "sentiment_analysis_agent")
    trading_workflow.add_edge("gather_market_data_agent", "valuation_analysis_agent")

    # All feed into risk evaluation
    trading_workflow.add_edge("technical_analysis_agent", "risk_evaluation_agent")
    trading_workflow.add_edge("fundamental_analysis_agent", "risk_evaluation_agent")
    trading_workflow.add_edge("sentiment_analysis_agent", "risk_evaluation_agent")
    trading_workflow.add_edge("valuation_analysis_agent", "risk_evaluation_agent")

    # Then final decision
    trading_workflow.add_edge("risk_evaluation_agent", "final_decision_agent")
    trading_workflow.add_edge("final_decision_agent", END)

    # Compile the workflow
    compiled_workflow = trading_workflow.compile()
    return compiled_workflow


def run_trading_system(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: Dict[str, Any],
    show_reasoning: bool = False
) -> str:
    """
    Orchestrates the multi-agent workflow to produce a trading decision:
      1) Gathers market data
      2) Runs technical, fundamental, sentiment, valuation analyses
      3) Evaluates risk
      4) Produces a final trading decision.

    Args:
        ticker (str): Stock ticker (e.g., "AAPL").
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        portfolio (Dict[str, Any]): Current portfolio (e.g., {"cash": 100_000, "stock": 0}).
        show_reasoning (bool, optional): Whether to show verbose agent reasoning logs.

    Returns:
        str: JSON string containing the final decision, e.g. {"action":"buy","quantity":10,"confidence":0.9}.
    """
    if not ticker:
        raise ValueError("A valid ticker is required to run the trading system.")

    logger.info("Running trading system for %s from %s to %s", ticker, start_date, end_date)

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as ve:
        raise ValueError("Invalid date format. Use 'YYYY-MM-DD'.") from ve

    # Ensure portfolio has at least the mandatory fields
    if "cash" not in portfolio or "stock" not in portfolio:
        raise ValueError("Portfolio must contain 'cash' and 'stock' fields.")

    trading_app = create_trading_workflow()  # Build the multi-agent workflow

    try:
        # Invoke the compiled workflow
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
                "metadata": {
                    "show_reasoning": show_reasoning,
                },
            },
        )
        logger.info("Trading system workflow completed successfully.")
        return final_state["messages"][-1].content

    except Exception as e:
        logger.error("Trading system workflow failed: %s", e, exc_info=True)
        raise RuntimeError("Trading system encountered an error.") from e


def main() -> None:
    """
    CLI entry point for running the trading system in a single-run mode.

    Usage Examples:
        python main.py --ticker AAPL --show_reasoning
        python main.py --ticker TSLA --start-date 2023-01-01 --end-date 2023-04-01
    """
    parser = argparse.ArgumentParser(description="Run the multi-agent trading system.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD). Defaults to 3 months before end date")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show_reasoning", action="store_true", help="Show agent reasoning logs")
    args = parser.parse_args()

    # Validate or fallback end_date to "today"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    # If no start_date is provided, default to 3 months before end_date
    if not args.start_date:
        try:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
        except ValueError as ve:
            logger.error("Invalid end date format '%s': %s", end_date, ve)
            return
    else:
        start_date = args.start_date

    # Example portfolio – in production, load from a config or user input
    portfolio_example = {
        "cash": 100_000.0,
        "stock": 0
    }

    # Run the system
    try:
        final_json = run_trading_system(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio_example,
            show_reasoning=args.show_reasoning
        )
        logger.info("Final system decision: %s", final_json)
    except RuntimeError as err:
        logger.critical("System failed to produce a decision: %s", err, exc_info=True)


if __name__ == "__main__":
    main()
