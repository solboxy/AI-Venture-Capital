import math
import json
import ast

from langchain_core.messages import HumanMessage
from agents.state import TradingAgentState, show_agent_reasoning
from tools.api import convert_prices_to_dataframe, fetch_prices

##### Risk Evaluation Agent #####
def risk_evaluation_agent(state: TradingAgentState):
    """
    Evaluates portfolio risk and sets position limits based on a comprehensive risk analysis.
    """
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    # Get the historical price data
    prices = fetch_prices(
        ticker=data["ticker"], 
        start_date=start_date, 
        end_date=end_date,
    )

    prices_df = convert_prices_to_dataframe(prices)
    # Fetch messages from other agents
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analysis_agent"
    )
    fundamental_message = next(
        msg for msg in state["messages"] if msg.name == "fundamental_analysis_agent"
    )
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_analysis_agent"
    )

    try:
        fundamental_signals = json.loads(fundamental_message.content)
        technical_signals = json.loads(technical_message.content)
        sentiment_signals = json.loads(sentiment_message.content)
    except Exception:
        # Fallback to literal_eval if JSON parse fails
        fundamental_signals = ast.literal_eval(fundamental_message.content)
        technical_signals = ast.literal_eval(technical_message.content)
        sentiment_signals = ast.literal_eval(sentiment_message.content)

    agent_signals = {
        "fundamental": fundamental_signals,
        "technical": technical_signals,
        "sentiment": sentiment_signals,
    }

    # 1. Calculate Risk Metrics
    returns = prices_df["close"].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)  # Annualized volatility approximation
    var_95 = returns.quantile(0.05)        # Simple historical VaR at 95% confidence
    max_drawdown = (prices_df["close"] / prices_df["close"].cummax() - 1).min()

    # 2. Market Risk Assessment
    market_risk_score = 0
    # Volatility scoring
    if volatility > 0.30:  # High volatility
        market_risk_score += 2
    elif volatility > 0.20:  # Moderate volatility
        market_risk_score += 1

    # VaR scoring (var_95 is typically negative)
    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    # Max Drawdown scoring
    if max_drawdown < -0.20:  # Severe drawdown
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    # 3. Position Size Limits
    current_stock_value = portfolio["stock"] * prices_df["close"].iloc[-1]
    total_portfolio_value = portfolio["cash"] + current_stock_value
    base_position_size = total_portfolio_value * 0.25  # Start with 25%

    if market_risk_score >= 4:
        max_position_size = base_position_size * 0.5  # High risk
    elif market_risk_score >= 2:
        max_position_size = base_position_size * 0.75  # Moderate risk
    else:
        max_position_size = base_position_size         # Low risk

    # 4. Stress Testing
    stress_test_scenarios = {
        "market_crash": -0.20,
        "moderate_decline": -0.10,
        "slight_decline": -0.05,
    }
    stress_test_results = {}
    current_position_value = current_stock_value

    for scenario, decline in stress_test_scenarios.items():
        potential_loss = current_position_value * decline
        denominator = portfolio["cash"] + current_position_value
        portfolio_impact = potential_loss / denominator if denominator != 0 else math.nan
        stress_test_results[scenario] = {
            "potential_loss": potential_loss,
            "portfolio_impact": portfolio_impact,
        }

    # 5. Risk-Adjusted Signals Analysis
    def parse_confidence(conf_str: str) -> float:
        return float(conf_str.replace("%", "")) / 100.0

    low_confidence = any(
        parse_confidence(signal["confidence"]) < 0.30 for signal in agent_signals.values()
    )

    # Check for signal divergence (increased uncertainty if all three differ)
    unique_signals = set(signal["signal"] for signal in agent_signals.values())
    signal_divergence = 2 if len(unique_signals) == 3 else 0

    risk_score = market_risk_score * 2  # Market risk can contribute up to ~6 points total
    if low_confidence:
        risk_score += 4  # Add penalty if any confidence < 30%
    risk_score += signal_divergence

    # Cap risk score at 10
    risk_score = min(round(risk_score), 10)

    # 6. Generate Trading Action
    if risk_score >= 8:
        trading_action = "hold"
    elif risk_score >= 6:
        trading_action = "reduce"
    else:
        trading_action = agent_signals["fundamental"]["signal"]

    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score,
            "stress_test_results": stress_test_results,
        },
        "reasoning": (
            f"Risk Score {risk_score}/10: Market Risk={market_risk_score}, "
            f"Volatility={volatility:.2%}, VaR={var_95:.2%}, "
            f"Max Drawdown={max_drawdown:.2%}"
        ),
    }

    # Build the risk evaluation message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_evaluation_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Evaluation Agent")

    return {"messages": state["messages"] + [message]}
