from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src.tools import fetch_price_data
from src.agents import run_trading_system


class TradingBacktester:
    def __init__(self, trading_agent, ticker, start_date, end_date, initial_capital):
        self.trading_agent = trading_agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.account = {"cash": initial_capital, "stock": 0}
        self.account_values = []

    def parse_agent_decision(self, agent_output):
        """Parse the JSON output from the agent's final decision."""
        try:
            import json
            decision = json.loads(agent_output)
            return decision["action"], decision["quantity"]
        except:
            return "hold", 0

    def execute_agent_trade(self, action, quantity, current_price):
        """Validate and execute trades based on account constraints."""
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.account["cash"]:
                self.account["stock"] += quantity
                self.account["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = int(self.account["cash"] // current_price)
                if max_quantity > 0:
                    self.account["stock"] += max_quantity
                    self.account["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.account["stock"])
            if quantity > 0:
                self.account["cash"] += quantity * current_price
                self.account["stock"] -= quantity
                return quantity
            return 0
        return 0

    def run_agent_backtest(self):
        """Run the backtesting loop, simulating trades over a range of dates."""
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        print("\nStarting backtest...")
        print(f"{'Date':<12} {'Ticker':<6} {'Action':<6} {'Quantity':>8} {'Price':>8} "
              f"{'Cash':>12} {'Stock':>8} {'Total Value':>12}")
        print("-" * 70)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            agent_output = self.trading_agent(
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.account
            )

            action, quantity = self.parse_agent_decision(agent_output)
            df = fetch_price_data(self.ticker, lookback_start, current_date_str)
            current_price = df.iloc[-1]["close"]

            # Execute the trade with validation
            executed_quantity = self.execute_agent_trade(action, quantity, current_price)

            # Update total account value
            total_value = self.account["cash"] + self.account["stock"] * current_price
            self.account["portfolio_value"] = total_value

            # Log the current state with executed quantity
            print(
                f"{current_date.strftime('%Y-%m-%d'):<12} {self.ticker:<6} {action:<6} "
                f"{executed_quantity:>8} {current_price:>8.2f} {self.account['cash']:>12.2f} "
                f"{self.account['stock']:>8} {total_value:>12.2f}"
            )

            # Record the account value
            self.account_values.append(
                {"Date": current_date, "Portfolio Value": total_value}
            )

    def analyze_agent_performance(self):
        """Analyze and display the backtest performance metrics."""
        performance_df = pd.DataFrame(self.account_values).set_index("Date")

        # Calculate total return
        total_return = ((self.account["portfolio_value"] - self.initial_capital)
                        / self.initial_capital)
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
        sharpe_ratio = (mean_daily_return / std_daily_return) * (252 ** 0.5)
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
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--initial_capital",
        type=float,
        default=100000,
        help="Initial capital amount (default: 100000)"
    )
    args = parser.parse_args()

    # Create an instance of TradingBacktester
    backtester = TradingBacktester(
        trading_agent=run_trading_system,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )

    # Run the backtesting process
    backtester.run_agent_backtest()
    performance_df = backtester.analyze_agent_performance()
