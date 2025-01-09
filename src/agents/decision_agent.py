import math
from typing import Annotated, Any, Dict, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from graph.state import TradingAgentState, show_agent_reasoning


##### Final Decision Agent #####
def final_decision_agent(state: TradingAgentState):
    """
    Makes final trading decisions and generates orders, strictly adhering to risk management constraints.
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Retrieve relevant messages
    technical_analysis_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analysis_agent"
    )
    fundamental_analysis_message = next(
        msg for msg in state["messages"] if msg.name == "fundamental_analysis_agent"
    )
    sentiment_analysis_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_analysis_agent"
    )
    risk_evaluation_message = next(
        msg for msg in state["messages"] if msg.name == "risk_evaluation_agent"
    )

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the team's analysis while strictly adhering
                to risk management constraints.

                RISK MANAGEMENT CONSTRAINTS:
                - You MUST NOT exceed the max_position_size specified by the risk manager
                - You MUST follow the trading_action (buy/sell/hold) recommended by risk management
                - These are hard constraints that cannot be overridden by other signals

                When weighing the different signals for direction and timing:
                1. Fundamental Analysis (50% weight)
                   - Primary driver of trading decisions
                   - Should determine overall direction
                
                2. Technical Analysis (35% weight)
                   - Secondary confirmation
                   - Helps with entry/exit timing
                
                3. Sentiment Analysis (15% weight)
                   - Final consideration
                   - Can influence sizing within risk limits
                
                The decision process should be:
                1. First check risk management constraints
                2. Then evaluate fundamental outlook
                3. Use technical analysis for timing
                4. Consider sentiment for final adjustment
                
                Provide the following in your output:
                - "action": "buy" | "sell" | "hold",
                - "quantity": <positive integer>
                - "confidence": <float between 0 and 1>
                - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
                - "reasoning": <concise explanation of the decision including how you weighted the signals>

                Trading Rules:
                - Never exceed risk management position limits
                - Only buy if you have available cash
                - Only sell if you have shares to sell
                - Quantity must be ≤ current position for sells
                - Quantity must be ≤ max_position_size from risk management
                """
            ),
            (
                "human",
                """Based on the team's analysis below, make your trading decision.

                Technical Analysis Trading Signal: {technical_message}
                Fundamental Analysis Trading Signal: {fundamentals_message}
                Sentiment Analysis Trading Signal: {sentiment_message}
                Risk Management Trading Signal: {risk_message}

                Here is the current portfolio:
                Portfolio:
                Cash: {portfolio_cash}
                Current Position: {portfolio_stock} shares

                Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  
                Do not include any JSON markdown.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash.
                You can only sell if you have shares in the portfolio to sell.
                """
            ),
        ]
    )

    # Generate the prompt with the updated references
    prompt = template.invoke(
        {
            "technical_message": technical_analysis_message.content,
            "fundamentals_message": fundamental_analysis_message.content,
            "sentiment_message": sentiment_analysis_message.content,
            "risk_message": risk_evaluation_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"]
        }
    )

    # Invoke the LLM
    llm = ChatOpenAI(model="gpt-4o")
    result = llm.invoke(prompt)

    # Create the final decision message
    message = HumanMessage(
        content=result.content,
        name="final_decision_agent",  # Updated agent name
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Final Decision Agent")

    return {"messages": state["messages"] + [message]}
