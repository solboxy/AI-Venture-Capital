import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt

from tools.api import fetch_price_data

# Configure the logger (you can adjust the logging level and format here)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TradingBacktester:
    """
    A backtesting class that simulates trades over a specified date range,
    using a multi-agent trading system.

    Attributes:
        trading_agent (Callable): A callable that takes in parameters and
            returns a final decision (in JSON format).
        ticker (str): The stock ticker symbol to be tested (e.g., "AAPL").
        start_date (str): The start date for backtesting (YYYY-MM-DD).
        end_date (str): The end date for backtesting (YYYY-MM-DD).
        initial_capital (float): Initial amount of cash available.
        date_frequency (str): Frequency for the date range generation (default: "B").
        selected_analysts (Optional[List[str]]): If provided, only these analysts will be run.
    """

    def __init__(
        self,
        trading_agent: Callable[..., str],
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        date_frequency: str = "B",
        selected_analysts: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the TradingBacktester.

        Args:
            trading_agent (Callable[..., str]): The callable to run the trading system.
            ticker (str): Ticker symbol to test (e.g., "AAPL").
            start_date (str): Backtest start date in "YYYY-MM-DD" format.
            end_date (str): Backtest end date in "YYYY-MM-DD" format.
            initial_capital (float): Initial amount of cash to start with.
            date_frequency (str): Pandas-compatible frequency string (e.g., "B" for business days).
            selected_analysts (Optional[List[str]]): Names/IDs of analysts to include.
        """
        self._validate_inputs(ticker, start_date, end_date, initial_capital)

        self.trading_agent = trading_agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.date_frequency = date_frequency
        self.selected_analysts = selected_analysts

        # Portfolio structure
        self.portfolio: Dict[str, Union[float, int]] = {
            "cash": initial_capital,
            "stock": 0
        }

        # Store daily performance
        self.portfolio_values: List[Dict[str, Union[datetime, float]]] = []

    @staticmethod
    def _validate_inputs(
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float
    ) -> None:
        """Validate constructor inputs."""
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Invalid ticker: must be a non-empty string.")

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD'.")

        if start > end:
            raise ValueError("Start date cannot be after end date.")

        if initial_capital < 0:
            raise ValueError("Initial capital must be non-negative.")

    def parse_agent_decision(self, agent_output: str) -> (str, int):
        """
        Parse the JSON output from the agent's final decision.

        Args:
            agent_output (str): JSON string containing the agent's decision.

        Returns:
            (action, quantity) as (str, int).
        """
        try:
            decision = json.loads(agent_output)
            action = decision.get("action", "hold")
            quantity = decision.get("quantity", 0)
            return action, int(quantity)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Failed to parse agent decision. Defaulting to HOLD.\nAgent Output: %s\nError: %s",
                agent_output,
                e
            )
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
            action (str): "buy", "sell", or "hold".
            quantity (int): Number of shares to buy or sell.
            current_price (float): Latest stock price.

        Returns:
            int: Actual number of shares traded.
        """
        if quantity < 0:
            logger.warning("Requested trade quantity was negative (%d). Forcing to 0.", quantity)
            quantity = 0

        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Partial buy if not enough cash
                max_quantity = int(self.portfolio["cash"] // current_price)
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0

        elif action == "sell" and quantity > 0:
            # Sell only up to what we own
            quantity = min(quantity, int(self.portfolio["stock"]))
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0

        # Either "hold" or invalid action
        return 0

    def run_agent_backtest(self) -> None:
        """
        Run the backtesting loop, simulating trades over a range of dates.
        """
        dates = pd.date_range(self.start_date, self.end_date, freq=self.date_frequency)
        logger.info("Starting backtest for ticker %s from %s to %s", self.ticker, self.start_date, self.end_date)

        # Header for daily logging
        logger.info(
            "%-10s  %-6s  %-6s  %-8s  %-8s  %-12s  %-8s  %-12s",
            "Date", "Ticker", "Action", "Quantity", "Price", "Cash", "Stock", "TotalVal"
        )

        for current_date in dates:
            current_str = current_date.strftime("%Y-%m-%d")
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")

            # Call the trading agent
            try:
                final_state_str = self.trading_agent(
                    ticker=self.ticker,
                    start_date=lookback_start,
                    end_date=current_str,
                    portfolio=self.portfolio,
                )
            except Exception as e:
                logger.error("Agent call failed on %s: %s", current_str, e)
                continue

            action, quantity = self.parse_agent_decision(final_state_str)

            # Fetch current price
            df = fetch_price_data(self.ticker, lookback_start, current_str)
            if df.empty:
                # No data for this date, skip
                continue
            current_price = df.iloc[-1]["close"]

            # Execute the trade
            traded_qty = self.execute_agent_trade(action, quantity, current_price)

            # Calculate total portfolio value
            total_value = float(self.portfolio["cash"]) + float(self.portfolio["stock"]) * current_price
            self.portfolio["portfolio_value"] = total_value

            # Log daily stats
            logger.info(
                "%-10s  %-6s  %-6s  %-8d  %-8.2f  %-12.2f  %-8d  %-12.2f",
                current_str, self.ticker, action, traded_qty, current_price,
                self.portfolio["cash"], self.portfolio["stock"], total_value
            )

            # Record performance
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Action": action,
                "Quantity": traded_qty
            })

    def analyze_agent_performance(self) -> pd.DataFrame:
        """
        Analyze and display the backtest performance metrics.

        Returns:
            pd.DataFrame: DataFrame with daily values and performance stats.
        """
        if not self.portfolio_values:
            logger.warning("No backtest data collected. Cannot analyze performance.")
            return pd.DataFrame()

        df_perf = pd.DataFrame(self.portfolio_values).set_index("Date")
        final_val = df_perf["Portfolio Value"].iloc[-1]
        total_return = (final_val - self.initial_capital) / self.initial_capital
        logger.info("Final portfolio value: %.2f (Initial: %.2f)", final_val, self.initial_capital)
        logger.info("Total return: %.2f%%", 100 * total_return)

        # Plot portfolio value over time
        df_perf["Portfolio Value"].plot(
            title=f"Portfolio Value Over Time: {self.ticker}",
            figsize=(10, 5)
        )
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.show()

        # Calculate daily return, Sharpe ratio, max drawdown
        df_perf["Daily Return"] = df_perf["Portfolio Value"].pct_change()
        mean_daily_ret = df_perf["Daily Return"].mean()
        std_daily_ret = df_perf["Daily Return"].std()

        if std_daily_ret == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (mean_daily_ret / std_daily_ret) * (252 ** 0.5)

        logger.info("Mean Daily Return: %.4f  |  StdDev of Daily Return: %.4f", mean_daily_ret, std_daily_ret)
        logger.info("Sharpe Ratio (annualized): %.2f", sharpe_ratio)

        rolling_max = df_perf["Portfolio Value"].cummax()
        drawdown = df_perf["Portfolio Value"] / rolling_max - 1
        max_drawdown = drawdown.min()
        logger.info("Maximum Drawdown: %.2f%%", 100 * max_drawdown)

        return df_perf


def main():
    """
    CLI entry point for running a backtest simulation.
    """
    parser = argparse.ArgumentParser(description="Run backtesting simulation.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--start_date", type=str,
                        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--initial_capital", type=float, default=100000.0,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--date_freq", type=str, default="B",
                        help="Pandas frequency for date range, default is 'B' (business days).")
    args = parser.parse_args()

    # Placeholder for your main trading system
    # You may replace this with your actual multi-agent function
    def example_trading_agent(**kwargs):
        # Example: return a JSON string indicating "hold" or "buy/sell"
        return json.dumps({"action": "hold", "quantity": 0})

    # Create a TradingBacktester instance
    backtester = TradingBacktester(
        trading_agent=example_trading_agent,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        date_frequency=args.date_freq
    )

    # Run the backtest
    backtester.run_agent_backtest()

    # Analyze performance
    backtester.analyze_agent_performance()


if __name__ == "__main__":
    main()
