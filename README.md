<img width="1060" alt="Screenshot" src="https://raw.githubusercontent.com/solboxy/AI-Venture-Capital/refs/heads/main/imgs/BloomBan.png" />

# Bloom AI

An AI-driven venture fund leveraging cutting-edge agents to identify, evaluate, and execute investment opportunities. This system is powered by a collaborative network of specialized agents:

1. **Quantitative Analysis Agent** - Evaluates financial performance, scalability, and market potential.
2. **Risk Evaluation Agent** - Identifies potential risks and formulates mitigation strategies.
3. **Portfolio Management Agent** - Manages investment decisions and ensures optimal portfolio allocation.
4. **Sentiment Analysis Agent** - Monitors public sentiment and media trends for additional insights.
5. **Fundamentals Evaluation Agent** - Assesses underlying business fundamentals for sound investment decisions.
6. **Valuation Analysis Agent** - Calculates intrinsic stock values, evaluates growth potential, and generates actionable trading signals.

---

<img width="1060" alt="Screenshot" src="https://raw.githubusercontent.com/solboxy/AI-Venture-Capital/refs/heads/main/imgs/githubgraph-01.png" />

## Key Features

- **Comprehensive multi-agent system** for intelligent, data-backed investments.
- **In-depth startup evaluation** leveraging key performance indicators like revenue, market growth, and customer acquisition.
- **Advanced sentiment tracking** for a holistic understanding of market dynamics.
- **Risk-adjusted decision-making** with robust scenario planning.
- **Portfolio optimization tools** that blend quantitative and qualitative inputs.
- **Backtesting engine** to assess strategy effectiveness.
- **Valuation Insights** Delivers precise stock valuations and trading signals through the Valuation Analysis Agent.
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

Set the API keys in the .env file:

```
OPENAI_API_KEY=your-openai-api-key
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
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
poetry run python src/main.py --ticker "AAPL" --start-date 2024-01-01 --end-date 2024-06-30
```

**Docker:**

```bash
docker run -it ai-venture-fund --ticker "AAPL" --start-date 2024-01-01 --end-date 2024-06-30
```

**Example Output:**

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

### Running Backtests

Evaluate historical strategies using backtesting:

**Poetry:**

```bash
poetry run python src/backtester.py --ticker "MSFT" --start-date 2023-01-01 --end-date 2023-12-31
```

**Example Output:**

<img width="941" alt="Screenshot" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />

### Analyzing Agent Decisions

Gain insights into agent-specific reasoning:

```bash
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
|   |
|   ├── graph/
|   |   ├── state.py
│   ├── agents/
│   │   ├── fundamentals_agent.py
│   │   ├── decision_agent.py
|   |   ├── risk_evaluation_agent.py
│   │   ├── sentiment_agent.py
|   |   ├── technical_analysis_agent.py
|   |   ├── valuation_analysis_agent.py
│   ├── tools/
│   │   ├── api.py
│   ├── backtester.py
│   ├── main.py
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
