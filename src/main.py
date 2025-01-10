from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

# Updated agent imports with “v” style naming
from agents.fundamentals_agent import fundamental_analysis_agent
from agents.technical_analysis_agent import technical_analysis_agent
from agents.sentiment_agent import sentiment_analysis_agent
from agents.risk_evaluation_agent import risk_evaluation_agent
from agents.decision_agent import final_decision_agent
from agents.valuation_analysis_agent import valuation_analysis_agent
from graph.state import TradingAgentState

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta



##### Run the Trading System #####
def run_trading_system(
    ticker: str, 
    start_date: str, 
    end_date: str, 
    portfolio: dict, 
    show_reasoning: bool = False
):
    """
    Orchestrates the multi-agent workflow: gather market data, run technical/fundamental/
    sentiment analyses, evaluate risk, and produce a final trading decision.
    """
    final_state = trading_app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data."
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content


##### Define the Trading Workflow #####
trading_workflow = StateGraph(TradingAgentState)

def gather_market_data_agent(state: TradingAgentState):
    """Initialize the workflow with the input message (placeholder or real data-gathering code)."""
    return state

# Add nodes (using “v” style agent names)
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
        "--show_reasoning",
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

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Example portfolio (configurable if desired)
    portfolio_example = {
        "cash": 100000.0,
        "stock": 0
    }

    # Run the trading system
    final_result = run_trading_system(
        ticker=args.ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio_example,
        show_reasoning=args.show_reasoning
    )
    print("\nFinal Result:")
    print(final_result)
