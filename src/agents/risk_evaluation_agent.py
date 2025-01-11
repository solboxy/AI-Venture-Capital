import math
import json
import ast

from langchain_core.messages import HumanMessage
from graph.state import TradingAgentState, show_agent_reasoning
from tools.api import convert_prices_to_dataframe, fetch_prices

##### Risk Evaluation Agent #####
def risk_evaluation_agent(state: TradingAgentState):
    """
    Evaluates portfolio risk and sets position limits based on a comprehensive risk analysis.
    """
    show_reasoning = state["metadata"].get("show_reasoning", False)
    data = state["data"]
    portfolio = data["portfolio"]

    start_date = data["start_date"]
    end_date = data["end_date"]

    # Ensure "analyst_signals" exists
    data.setdefault("analyst_signals", {})

    # Get the historical price data
    prices = fetch_prices(
        ticker=data["ticker"], 
        start_date=start_date, 
        end_date=end_date,
    )
    prices_df = convert_prices_to_dataframe(prices)

    # Fetch messages from other agents
    # (They must exist in state["messages"], or handle exceptions.)
    technical_message = next(
        (msg for msg in state["messages"] if msg.name == "technical_analysis_agent"), None
    )
    fundamental_message = next(
        (msg for msg in state["messages"] if msg.name == "fundamental_analysis_agent"), None
    )
    sentiment_message = next(
        (msg for msg in state["messages"] if msg.name == "sentiment_analysis_agent"), None
    )

    # Parse content
    def robust_parse(msg):
        if not msg:
            return {}
        try:
            return json.loads(msg.content)
        except:
            try:
                return ast.literal_eval(msg.content)
            except:
                return {}

    fundamental_signals = robust_parse(fundamental_message)
    technical_signals = robust_parse(technical_message)
    sentiment_signals = robust_parse(sentiment_message)

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
    current_price = prices_df["close"].iloc[-1]
    current_stock_value = portfolio["stock"] * current_price
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
    for scenario, decline in stress_test_scenarios.items():
        potential_loss = current_stock_value * decline
        denominator = portfolio["cash"] + current_stock_value
        portfolio_impact = potential_loss / denominator if denominator != 0 else math.nan
        stress_test_results[scenario] = {
            "potential_loss": potential_loss,
            "portfolio_impact": portfolio_impact,
        }

    # 5. Risk-Adjusted Signals Analysis
    def parse_confidence(conf_str: str) -> float:
        # Expects something like "50%" or numeric str
        if conf_str.endswith("%"):
            return float(conf_str.replace("%", "")) / 100.0
        try:
            return float(conf_str)
        except:
            return 0.0

    low_confidence = any(
        parse_confidence(signal.get("confidence_level", "0%"))
        < 0.30
        for signal in agent_signals.values()
    )

    # Check for signal divergence (increased uncertainty if all three differ)
    signals_set = set(signal.get("signal", "") for signal in agent_signals.values())
    # If you have 3 different signals (bullish, neutral, bearish), that's major divergence
    signal_divergence = 2 if len(signals_set) == 3 else 0

    risk_score = market_risk_score * 2  # Market risk can be up to ~6 points
    if low_confidence:
        risk_score += 4  # Add penalty if any confidence < 30%
    risk_score += signal_divergence
    # Cap risk score at 10
    risk_score = min(round(risk_score), 10)

    # 6. Generate Trading Action (for illustration)
    if risk_score >= 8:
        trading_action = "hold"
    elif risk_score >= 6:
        trading_action = "reduce"
    else:
        trading_action = fundamental_signals.get("signal", "neutral")

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

    # Optionally show reasoning
    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Evaluation Agent")

    # Store the result in analyst_signals
    data["analyst_signals"]["risk_evaluation_agent"] = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "reasoning": message_content["reasoning"],
    }

    return {"messages": state["messages"] + [message], "data": data}
