# AI Venture Capital Fund

An AI-powered venture capital fund leveraging advanced agents to discover, evaluate, and execute investment opportunities. This system comprises multiple specialized agents working collaboratively:

1. **Market Intelligence Agent** - Collects and processes data on startups, industries, and emerging market trends.
2. **Quantitative Analysis Agent** - Assesses financial metrics, growth potential, and product-market fit.
3. **Risk Evaluation Agent** - Examines potential risks and develops diversification strategies.
4. **Portfolio Management Agent** - Finalizes investment decisions and optimizes portfolio allocations.

## Key Features

- **Collaborative multi-agent framework** for data-driven investment strategies.
- **Comprehensive startup evaluation** using indicators like revenue growth, profitability, and market size.
- **Dynamic risk management** with scenario simulations and position sizing.
- **Portfolio optimization** using quantitative and qualitative insights.
- **Backtesting tools** to evaluate past performance.
- **Support for diverse industries and sectors.**

## Prerequisites

- Python 3.9+
- Poetry or Docker (recommended for environment setup)
- API keys for external services (e.g., OpenAI, financial data providers)

## Setup

Clone the repository:

```bash
git clone https://github.com/your-repo/ai-vc-fund.git
cd ai-vc-fund
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
   docker build -t ai-vc-fund .
   ```

2. Run the Docker container:

   ```bash
   docker run -it ai-vc-fund
   ```

## Usage

### Running the System

To analyze investment opportunities:

**Poetry:**

```bash
poetry run python src/agents.py --sector "Healthcare" --start-date 2024-01-01 --end-date 2024-06-30
```

**Docker:**

```bash
docker run -it ai-vc-fund --sector "Healthcare" --start-date 2024-01-01 --end-date 2024-06-30
```

**Example Output:**

```json
{
  "action": "invest",
  "amount": 750000,
  "sector": "Healthcare",
  "company": "MediGen"
}
```

### Running Backtests

Evaluate past strategies with historical data:

**Poetry:**

```bash
poetry run python src/backtester.py --sector "FinTech" --start-date 2023-01-01 --end-date 2023-12-31
```

**Example Output:**

```
Starting backtest...
Date         Sector    Action    Amount       Cash       Portfolio Value
----------------------------------------------------------------------
2023-01-01   FinTech   invest    500,000.00   500,000.00    1,000,000.00
2023-06-30   FinTech   hold            0.00   500,000.00    1,050,000.00
2023-12-31   FinTech   divest    550,000.00         0.00    1,050,000.00
```

### Analyzing Agent Decisions

Gain insights into agent-specific recommendations:

```bash
poetry run python src/agents.py --sector "AI" --show-decisions
```

**Example Output:**

```
===== Market Intelligence Agent =====
Trend: Positive
Sector Sentiment: 0.8
Startup Interest: High
=====================================

===== Quantitative Analysis Agent =====
Key Metrics: Positive
Startup Score: 85/100
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
```

## Project Layout

```
ai-vc-fund/
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
