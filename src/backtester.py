import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt

# 1) Import your multi-agent pipeline from main.py
from main import run_trading_system

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
    using the multi-agent trading system from main.py each day.

    Attributes:
        ticker (str): Ticker symbol to test (e.g., "AAPL").
        start_date (str): Backtest start date in YYYY-MM-DD format.
        end_date (str): Backtest end date in YYYY-MM-DD format.
        initial_capital (float): Initial amount of cash to start with.
        date_frequency (str): Pandas frequency string (e.g. "B" for business days).
        show_reasoning (bool): Whether to show agent reasoning logs in the console.
        portfolio (Dict[str, Union[float, int]]): Represents the current holdings ('cash' and 'stock').
        portfolio_values (List[Dict[str, Union[datetime, float]]]): 
            Stores daily portfolio performance results.
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        date_frequency: str = "B",
        show_reasoning: bool = False,
    ) -> None:
        """
        Initializes the TradingBacktester.

        Args:
            ticker (str): Ticker symbol to test (e.g., "AAPL").
            start_date (str): Backtest start date in "YYYY-MM-DD" format.
            end_date (str): Backtest end date in "YYYY-MM-DD" format.
            initial_capital (float): Initial amount of cash to start with.
            date_frequency (str, optional): Pandas frequency string 
                (e.g., "B" for business days). Defaults to "B".
            show_reasoning (bool, optional): Whether to show agent reasoning logs in the console. 
                Defaults to False.
        """
        self._validate_inputs(ticker, start_date, end_date, initial_capital)

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.date_frequency = date_frequency
        self.show_reasoning = show_reasoning

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
        """
        Validates constructor inputs.

        Args:
            ticker (str): Ticker symbol (e.g., "AAPL").
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
            initial_capital (float): Initial capital to start with.

        Raises:
            ValueError: If the input parameters are invalid.
        """
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
        Parses the JSON output from the multi-agent system's final decision to determine 
        what action and quantity to take.

        Args:
            agent_output (str): JSON string returned by the multi-agent system 
                                (e.g., '{"action":"buy","quantity":10,"confidence":0.9}').

        Returns:
            (str, int): A tuple of (action, quantity). 
                        Action is one of "buy", "sell", or "hold". 
                        Quantity is the integer number of shares.
        """
        try:
            decision = json.loads(agent_output)
            action = decision.get("action", "hold")
            quantity = decision.get("quantity", 0)
            return action.lower(), int(quantity)
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
        Validates and executes trades based on the current portfolio constraints.

        Args:
            action (str): "buy", "sell", or "hold".
            quantity (int): Number of shares to buy or sell.
            current_price (float): Latest stock price from the market data.

        Returns:
            int: Actual number of shares traded (may be less than requested if constraints apply).
        """
        if quantity < 0:
            logger.warning("Requested trade quantity was negative (%d). Forcing to 0.", quantity)
            quantity = 0

        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                # Execute full buy
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

        # "hold" or invalid action => no change
        return 0

    def run_agent_backtest(self) -> None:
        """
        Runs the backtesting loop, simulating trades over the specified date range.
        Each day, this method calls the multi-agent pipeline from main.py to get a decision,
        then executes that decision on the current portfolio.
        """
        dates = pd.date_range(self.start_date, self.end_date, freq=self.date_frequency)
        logger.info("Starting backtest for ticker %s from %s to %s",
                    self.ticker, self.start_date, self.end_date)

        # Header for daily logging
        logger.info(
            "%-10s  %-6s  %-6s  %-8s  %-8s  %-12s  %-8s  %-12s",
            "Date", "Ticker", "Action", "Quantity", "Price", "Cash", "Stock", "TotalVal"
        )

        for current_date in dates:
            current_str = current_date.strftime("%Y-%m-%d")
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")

            # 1) Call the multi-agent pipeline to get the final decision
            try:
                agent_output = run_trading_system(
                    ticker=self.ticker,
                    start_date=lookback_start,
                    end_date=current_str,
                    portfolio=self.portfolio,
                    show_reasoning=self.show_reasoning
                )
            except Exception as e:
                logger.error("Agent call failed on %s: %s", current_str, e)
                continue

            # 2) Parse the decision (action, quantity)
            action, quantity = self.parse_agent_decision(agent_output)

            # 3) Get the current price
            df = fetch_price_data(self.ticker, lookback_start, current_str)
            if df.empty:
                logger.warning("No price data for %s on %s. Skipping trade.", self.ticker, current_str)
                continue

            current_price = df.iloc[-1]["close"]

            # 4) Execute the trade
            traded_qty = self.execute_agent_trade(action, quantity, current_price)

            # 5) Calculate total portfolio value
            total_value = float(self.portfolio["cash"]) + float(self.portfolio["stock"]) * current_price
            self.portfolio["portfolio_value"] = total_value

            # 6) Log daily stats
            logger.info(
                "%-10s  %-6s  %-6s  %-8d  %-8.2f  %-12.2f  %-8d  %-12.2f",
                current_str, self.ticker, action, traded_qty, current_price,
                self.portfolio["cash"], self.portfolio["stock"], total_value
            )

            # 7) Record performance
            self.portfolio_values.append({
                "Date": current_date,
                "Portfolio Value": total_value,
                "Action": action,
                "Quantity": traded_qty
            })

    def analyze_agent_performance(self) -> pd.DataFrame:
        """
        Analyzes and displays the backtest performance metrics, such as total return, 
        Sharpe ratio, and max drawdown. Also plots the portfolio value over time.

        Returns:
            pd.DataFrame: A DataFrame with daily portfolio values and performance stats.
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

        # Calculate daily return, Sharpe ratio, and max drawdown
        df_perf["Daily Return"] = df_perf["Portfolio Value"].pct_change()
        mean_daily_ret = df_perf["Daily Return"].mean()
        std_daily_ret = df_perf["Daily Return"].std()

        if std_daily_ret == 0:
            sharpe_ratio = 0.0
        else:
            # Annualize using ~252 trading days/year
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
    CLI entry point for running a backtest simulation, using the multi-agent pipeline.
    Example usage:
        python backtester.py --ticker AAPL --start_date 2023-01-01 --end_date 2023-03-01
    """
    parser = argparse.ArgumentParser(description="Run backtesting simulation with multi-agent pipeline.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--start_date", type=str,
                        default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                        help="Start date in YYYY-MM-DD format")
    parser.add_argument("--initial_capital", type=float, default=100000.0,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--date_freq", type=str, default="B",
                        help="Pandas frequency for date range (default: 'B' for business days).")
    parser.add_argument("--show_reasoning", action="store_true",
                        help="If set, logs agent reasoning to the console.")
    args = parser.parse_args()

    # Instantiate the backtester
    backtester = TradingBacktester(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        date_frequency=args.date_freq,
        show_reasoning=args.show_reasoning,
    )

    # Run the backtest
    backtester.run_agent_backtest()

    # Analyze performance
    backtester.analyze_agent_performance()


if __name__ == "__main__":
    main()
