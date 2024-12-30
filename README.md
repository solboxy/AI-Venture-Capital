# AI Venture Fund

An AI-driven venture fund leveraging cutting-edge agents to identify, evaluate, and execute investment opportunities. This system is powered by a collaborative network of specialized agents:

1. **Market Intelligence Agent** - Gathers and analyzes data on startups, industries, and emerging trends.
2. **Quantitative Analysis Agent** - Evaluates financial performance, scalability, and market potential.
3. **Risk Evaluation Agent** - Identifies potential risks and formulates mitigation strategies.
4. **Portfolio Management Agent** - Manages investment decisions and ensures optimal portfolio allocation.
5. **Sentiment Analysis Agent** - Monitors public sentiment and media trends for additional insights.
6. **Fundamentals Evaluation Agent** - Assesses underlying business fundamentals for sound investment decisions.

## Key Features

- **Comprehensive multi-agent system** for intelligent, data-backed investments.
- **In-depth startup evaluation** leveraging key performance indicators like revenue, market growth, and customer acquisition.
- **Advanced sentiment tracking** for a holistic understanding of market dynamics.
- **Risk-adjusted decision-making** with robust scenario planning.
- **Portfolio optimization tools** that blend quantitative and qualitative inputs.
- **Backtesting engine** to assess strategy effectiveness.
- **Support for a diverse range of industries and sectors.**

## Prerequisites

- Python 3.9+
- Poetry or Docker (preferred for environment setup)
- API keys for services like OpenAI and financial data providers

## Setup

Clone the repository:

```bash
git clone https://github.com/your-repo/ai-venture-fund.git
cd ai-venture-fund
```

### Installation Using Poetry

1. Install Poetry (if not already installed):
   **MacOS/Linux:**

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

3. Configure environment variables:

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

### Installation Using Docker

1. Build the Docker image:

   ```bash
   docker build -t ai-venture-fund .
   ```

2. Run the Docker container:

   ```bash
   docker run -it ai-venture-fund
   ```

## Usage

### Running the System

To evaluate investment opportunities:

**Poetry:**

```bash
poetry run python src/agents.py --ticker "AAPL" --start-date 2024-01-01 --end-date 2024-06-30
poetry run python src/main.py --ticker "AAPL" --start-date 2024-01-01 --end-date 2024-06-30
```

**Docker:**

```bash
docker run -it ai-venture-fund --ticker "AAPL" --start-date 2024-01-01 --end-date 2024-06-30
```

**Example Output:**

```json
{
  "action": "invest",
  "amount": 750000,
  "ticker": "AAPL",
  "company": "Apple Inc."
}
```

### Running Backtests

Evaluate historical strategies using backtesting:

**Poetry:**

```bash
poetry run python src/backtester.py --ticker "MSFT" --start-date 2023-01-01 --end-date 2023-12-31
```

**Example Output:**

```
Starting backtest...
Date         Ticker    Action    Amount       Cash       Portfolio Value
----------------------------------------------------------------------
2023-01-01   MSFT      invest    500,000.00   500,000.00    1,000,000.00
2023-06-30   MSFT      hold            0.00   500,000.00    1,050,000.00
2023-12-31   MSFT      divest    550,000.00         0.00    1,050,000.00
```

### Analyzing Agent Decisions

Gain insights into agent-specific reasoning:

```bash
poetry run python src/agents.py --ticker "GOOG" --show-decisions
poetry run python src/main.py --ticker "GOOG" --show-decisions
```

**Example Output:**

```
===== Market Intelligence Agent =====
Trend: Positive
Ticker Sentiment: 0.8
Company Interest: High
=====================================

===== Quantitative Analysis Agent =====
Key Metrics: Positive
Company Score: 85/100
========================================

===== Risk Evaluation Agent =====
Risk Level: Moderate
Max Allocation: $500,000
========================================

===== Portfolio Management Agent =====
Action: Invest
Amount: $450,000
Reasoning: Favorable metrics with manageable risks.
========================================

===== Sentiment Analysis Agent =====
Sentiment Score: Positive (0.78)
Signal: Invest
=====================================

===== Fundamentals Evaluation Agent =====
Business Metrics: Solid
Recommendation: Invest $300,000
========================================
```

## Project Layout

```
ai-venture-fund/
├── src/
│   ├── agents.py           # Core agent logic
│   ├── backtester.py       # Backtesting engine
│   ├── tools.py            # Supporting analysis tools
├── pyproject.toml          # Poetry configuration
├── Dockerfile              # Docker image setup
├── .env.example            # Sample environment variables
├── README.md               # Documentation
```

## Contributing

We welcome contributions! Follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Submit a Pull Request.

## License

This project is distributed under the MIT License. See the LICENSE file for details.
