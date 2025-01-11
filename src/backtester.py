from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
import questionary


# Updated import to match your new naming convention
from main import run_trading_system
from tools.api import fetch_price_data


class TradingBacktester:
    """
    A backtesting class that simulates trades over a specified date range,
    using a multi-agent trading system.
    """

    def __init__(self, trading_agent, ticker, start_date, end_date, initial_capital, selected_analysts=None):
    
        self.trading_agent = trading_agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.selected_analysts = selected_analysts

        # Portfolio structure
        self.portfolio = {"cash": initial_capital, "stock": 0}
        self.portfolio_values = []

    def parse_agent_decision(self, agent_output):
        """
        Parse the JSON output from the agent's final decision.
        Formerly 'parse_agent_response'
        """
        try:
            decision = json.loads(agent_output)
            return decision["action"], decision["quantity"]
        except Exception:
            print(f"Error parsing action: {agent_output}")
            return "hold", 0

    def execute_agent_trade(self, action, quantity, current_price):
        """
        Validate and execute trades based on portfolio constraints.
        Formerly 'execute_trade'
        """
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.portfolio["stock"])
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0
        return 0

    def run_agent_backtest(self):
        """
        Run the backtesting loop, simulating trades over a range of dates.
        Formerly 'run_backtest'
        """
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        print("\nStarting backtest...")
        print(
            f"{'Date':<12} {'Ticker':<6} {'Action':<6} {'Quantity':>8} "
            f"{'Price':>8} {'Cash':>12} {'Stock':>8} {'Total Value':>12}"
        )
        print("-" * 90)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            # Call the trading agent
            agent_output = self.trading_agent(
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
            )

            # Parse the final decision
            action, quantity = self.parse_agent_decision(agent_output)

            # Fetch current price data
            df = fetch_price_data(self.ticker, lookback_start, current_date_str)
            current_price = df.iloc[-1]["close"]

            # Execute the trade with validation
            executed_quantity = self.execute_agent_trade(
                action, quantity, current_price
            )

            # Update total portfolio value
            total_value = self.portfolio["cash"] + self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value

            # Log the current state with executed quantity
            print(
                f"{current_date_str:<12} {self.ticker:<6} {action:<6} "
                f"{executed_quantity:>8} {current_price:>8.2f} {self.portfolio['cash']:>12.2f} "
                f"{self.portfolio['stock']:>8} {total_value:>12.2f}"
            )

            # Record the portfolio value
            self.portfolio_values.append(
                {"Date": current_date, "Portfolio Value": total_value}
            )

    def analyze_agent_performance(self):
        """
        Analyze and display the backtest performance metrics.
        Formerly 'analyze_performance'
        """
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # Calculate total return
        final_value = self.portfolio["portfolio_value"]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        print(f"Total Return: {total_return * 100:.2f}%")

        # Plot the portfolio value over time
        performance_df["Portfolio Value"].plot(
            title="Portfolio Value Over Time", figsize=(12, 6)
        )
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.show()

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change()

        # Calculate Sharpe Ratio (assuming 252 trading days in a year)
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * (252**0.5)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Calculate Maximum Drawdown
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = performance_df["Portfolio Value"] / rolling_max - 1
        max_drawdown = drawdown.min()
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

        return performance_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--end_date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)",
    )
    args = parser.parse_args()

    # Create an instance of TradingBacktester
    backtester = TradingBacktester(
        trading_agent=run_trading_system,  # updated import
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )

    # Run the backtesting process
    backtester.run_agent_backtest()
    performance_df = backtester.analyze_agent_performance()
