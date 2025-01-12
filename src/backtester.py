from datetime import datetime, timedelta
from typing import Callable, Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json

from main import run_trading_system
from tools.api import fetch_price_data


class TradingBacktester:
    """
    A backtesting class that simulates trades over a specified date range
    using a multi-agent trading system.

    Attributes:
        trading_agent (Callable): A callable that takes in trading parameters and
            returns a final decision (in JSON format).
        ticker (str): The stock ticker symbol to be tested (e.g., "AAPL").
        start_date (str): The start date for backtesting in "YYYY-MM-DD" format.
        end_date (str): The end date for backtesting in "YYYY-MM-DD" format.
        initial_capital (float): Initial amount of cash available for trading.
        selected_analysts (Optional[List[str]]): If provided, only these analysts will be run.

    Example:
        >>> backtester = TradingBacktester(
        ...     trading_agent=run_trading_system,
        ...     ticker="AAPL",
        ...     start_date="2023-01-01",
        ...     end_date="2023-06-01",
        ...     initial_capital=100000
        ... )
        >>> backtester.run_agent_backtest()
        >>> results_df = backtester.analyze_agent_performance()
    """

    def __init__(
        self,
        trading_agent: Callable[..., str],
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        selected_analysts: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the TradingBacktester.

        Args:
            trading_agent (Callable[..., str]): The callable to run the trading system.
            ticker (str): Stock ticker symbol to test.
            start_date (str): Backtest start date in "YYYY-MM-DD" format.
            end_date (str): Backtest end date in "YYYY-MM-DD" format.
            initial_capital (float): Initial amount of cash to start with.
            selected_analysts (Optional[List[str]]): Names/IDs of analysts to include.
        """
        self.trading_agent = trading_agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.selected_analysts = selected_analysts

        # Portfolio structure
        self.portfolio: Dict[str, Union[float, int]] = {
            "cash": initial_capital,
            "stock": 0
        }
        self.portfolio_values: List[Dict[str, Union[datetime, float]]] = []

    def parse_agent_decision(self, agent_output: str) -> Tuple[str, int]:
        """
        Parse the JSON output from the agent's final decision.

        Args:
            agent_output (str): The JSON string containing the agent's decision.

        Returns:
            Tuple[str, int]: A tuple of (action, quantity).
        """
        try:
            decision = json.loads(agent_output)
            action = decision.get("action", "hold")
            quantity = decision.get("quantity", 0)
            return action, int(quantity)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Log or handle parsing error; revert to hold
            print(f"[WARNING] Error parsing agent decision: {agent_output}\n  {e}")
            return "hold", 0

    def execute_agent_trade(
        self,
        action: str,
        quantity: int,
        current_price: float
    ) -> int:
        """
        Validate and execute trades based on portfolio constraints.

        Args:
            action (str): The intended action ("buy", "sell", or "hold").
            quantity (int): The number of shares requested to buy or sell.
            current_price (float): The current stock price.

        Returns:
            int: The actual number of shares traded (could differ if constraints apply).
        """
        # Basic validation
        if quantity < 0:
            print(f"[WARNING] Negative quantity requested: {quantity}. Forcing to 0.")
            quantity = 0

        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Enough cash to buy full quantity
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Partial buy if insufficient cash
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0

        elif action == "sell" and quantity > 0:
            # Only sell up to how many shares we have
            quantity = min(quantity, int(self.portfolio["stock"]))
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0

        # action == "hold" or no valid trade
        return 0

    def run_agent_backtest(self) -> None:
        """
        Run the backtesting loop, simulating trades over a range of dates.

        For each date in the date range, it calls the trading system, obtains
        an action and quantity, and executes the trade if it meets portfolio constraints.
        """
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        print("\nStarting backtest...")
        print(
            f"{'Date':<12} {'Ticker':<6} {'Action':<6} "
            f"{'Qty':>6} {'Price':>8} {'Cash':>12} {'Stock':>8} {'Total Value':>12}"
        )
        print("-" * 82)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            # Call the trading agent
            try:
                final_state_str = self.trading_agent(
                    ticker=self.ticker,
                    start_date=lookback_start,
                    end_date=current_date_str,
                    portfolio=self.portfolio,
                )
            except Exception as e:
                # If the agent call fails, skip for that date (or handle differently)
                print(f"[ERROR] Agent call failed on {current_date_str}: {e}")
                continue

            # Parse the final decision
            action, quantity = self.parse_agent_decision(final_state_str)

            # Fetch current price data
            df = fetch_price_data(self.ticker, lookback_start, current_date_str)
            if df.empty:
                # No data returned, skip
                continue

            # Current day's price is the last row's close
            current_price = df.iloc[-1]["close"]

            # Execute the trade
            executed_quantity = self.execute_agent_trade(action, quantity, current_price)

            # Update total portfolio value
            total_value = float(self.portfolio["cash"]) + float(self.portfolio["stock"]) * float(current_price)
            self.portfolio["portfolio_value"] = total_value

            # Log the current state
            print(
                f"{current_date_str:<12} {self.ticker:<6} {action:<6} "
                f"{executed_quantity:>6} {current_price:>8.2f} "
                f"{self.portfolio['cash']:>12.2f} {self.portfolio['stock']:>8} {total_value:>12.2f}"
            )

            # Record portfolio value over time
            self.portfolio_values.append(
                {
                    "Date": current_date,
                    "Portfolio Value": total_value,
                    "Action": action,
                    "Quantity": executed_quantity
                }
            )

    def analyze_agent_performance(self) -> pd.DataFrame:
        """
        Analyze and display the backtest performance metrics.

        Returns:
            pd.DataFrame: DataFrame containing daily portfolio values and performance metrics.
        """
        if not self.portfolio_values:
            print("No backtest data to analyze.")
            return pd.DataFrame()

        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # Calculate total return
        final_value = performance_df["Portfolio Value"].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        print(f"\nTotal Return: {total_return * 100:.2f}%")

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
        if std_daily_return == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (mean_daily_return / std_daily_return) * (252**0.5)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Calculate Maximum Drawdown
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = performance_df["Portfolio Value"] / rolling_max - 1
        max_drawdown = drawdown.min()
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

        return performance_df


def main() -> None:
    """
    Command-line entry point for running a backtesting simulation.
    """
    parser = argparse.ArgumentParser(description="Run backtesting simulation")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--start_date", type=str,
                        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--initial_capital", type=float, default=100000,
                        help="Initial capital amount (default: 100000)")
    args = parser.parse_args()

    # Create an instance of TradingBacktester
    backtester = TradingBacktester(
        trading_agent=run_trading_system,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital
    )

    # Run the backtesting process
    backtester.run_agent_backtest()
    backtester.analyze_agent_performance()


if __name__ == "__main__":
    main()
