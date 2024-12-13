# AI Venture Capital Fund

An AI-driven venture capital fund leveraging cutting-edge agents to identify, evaluate, and execute investment opportunities. This system is powered by a collaborative network of specialized agents:

1. **Market Intelligence Agent** - Gathers and analyzes data on startups, industries, and emerging trends.
2. **Quantitative Analysis Agent** - Evaluates financial performance, scalability, and market potential.
3. **Risk Evaluation Agent** - Identifies potential risks and formulates mitigation strategies.
4. **Portfolio Management Agent** - Manages investment decisions and ensures optimal portfolio allocation.
5. **Sentiment Analysis Agent** - Monitors public sentiment and media trends for additional insights.
6. **Fundamentals Evaluation Agent** - Assesses underlying business fundamentals for sound investment decisions.

<img width="1025" alt="Screenshot" src="https://github.com/user-attachments/assets/6e51851c-b4ee-4463-a016-3e9d7b886e7e">

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

To evaluate investment opportunities:

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

Evaluate historical strategies using backtesting:

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

Gain insights into agent-specific reasoning:

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
