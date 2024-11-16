# AI Venture Capital Fund

An AI-powered venture capital fund that leverages multiple intelligent agents to identify, evaluate, and execute investment opportunities. The system utilizes specialized agents working in harmony:

1. **Market Intelligence Agent** - Gathers and preprocesses data on startups, industries, and market trends.
2. **Quantitative Analysis Agent** - Evaluates metrics such as financial health, growth potential, and market fit.
3. **Risk Evaluation Agent** - Assesses investment risks and determines diversification strategies.
4. **Portfolio Management Agent** - Makes final investment decisions and optimizes portfolio allocation.

## Features

- **Multi-agent architecture** for strategic investment decisions.
- **Startup evaluation** using key indicators like growth rates, revenue projections, and market share.
- **Risk management** with scenario analysis and portfolio diversification.
- **Portfolio optimization** with data-driven investment allocation.
- **Backtesting capabilities** to evaluate historical performance.
- **Support for multiple industries and sectors.**

## Prerequisites

- Python 3.9+
- Poetry (recommended) or Docker
- OpenAI API key

## Setup

Clone the repository:

```bash
git clone https://github.com/your-repo/ai-vc-fund.git
cd ai-vc-fund
```

### Using Poetry (Recommended)

1. Install Poetry (if not already installed):
   **MacOS:**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   **Windows:**

   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

2. Install dependencies:

   ```bash
   poetry install
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   **Mac/Linux:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   export FINANCIAL_DATASETS_API_KEY='your-api-key-here'
   ```
   **Windows:**
   ```powershell
   $env:OPENAI_API_KEY='your-api-key-here'
   $env:FINANCIAL_DATASETS_API_KEY='your-api-key-here'
   ```

### Using Docker

1. Build the Docker image:

   ```bash
   docker build -t ai-vc-fund .
   ```

2. Run the Docker container:
   ```bash
   docker run -it ai-vc-fund
   ```

## Usage

### Running the AI VC Fund System

To evaluate investment opportunities and generate recommendations:

**Poetry:**

```bash
poetry run python agents.py --sector "FinTech" --start-date 2024-01-01 --end-date 2024-03-01
```

**Docker:**

```bash
docker run -it ai-vc-fund --sector "FinTech" --start-date 2024-01-01 --end-date 2024-03-01
```

**Example Output:**

```json
{
  "action": "invest",
  "amount": 500000,
  "sector": "FinTech",
  "company": "InnovateX"
}
```

### Running the Hedge Fund

To analyze and act on individual stock tickers:

**Poetry:**

```bash
poetry run python src/agents.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-03-01
```

**Example Output:**

```json
{
  "action": "buy",
  "quantity": 50000
}
```

### Running the Hedge Fund (with Decisions)

This will print the decisions of each agent to the console:

**Poetry:**

```bash
poetry run python src/agents.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-03-01 --show-decisions
```

**Example Output:**

```
=====         Quant Agent          =====
Quant Trading Signal: neutral
Confidence (0-1, higher is better): 0.25
========================================

=====    Risk Management Agent     =====
Max Position Size: 5000.0
Risk Score: 4
========================================

=====  Portfolio Management Agent  =====
{
  "action": "buy",
  "quantity": 5000
}
========================================
```

### Running the Backtester

To evaluate the performance of investment strategies on historical data:

**Poetry:**

```bash
poetry run python src/backtester.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-03-01
```

**Example Output:**

```
Starting backtest...
Date         Ticker Action Quantity    Price         Cash    Stock  Total Value
----------------------------------------------------------------------
2024-01-01   AAPL   buy       519.0   192.53        76.93    519.0    100000.00
2024-01-02   AAPL   hold          0   185.64        76.93    519.0     96424.09
2024-01-03   AAPL   hold          0   184.25        76.93    519.0     95702.68
2024-01-04   AAPL   hold          0   181.91        76.93    519.0     94488.22
2024-01-05   AAPL   hold          0   181.18        76.93    519.0     94109.35
2024-01-08   AAPL   sell        519   185.56     96382.57      0.0     96382.57
2024-01-09   AAPL   buy       520.0   185.14       109.77    520.0     96382.57
```

## Project Structure

```
ai-vc-fund/
├── src/
│   ├── agents.py            # Main agent definitions and workflow
│   ├── backtester.py        # Backtesting functionality
│   ├── tools.py             # Analysis and evaluation tools
├── pyproject.toml           # Poetry configuration
├── .env.example             # Environment variables
├── Dockerfile               # Docker configuration
└── README.md                # Documentation
```

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
