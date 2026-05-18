"""
core/react_agent.py — Layer 6: Autonomous ReAct Agent for Portfolio Monitoring
Source: week6_capstone
Framework: LangChain ReAct (Reasoning + Acting)
"""
import os
import json
import yfinance as yf
from typing import Optional

from core.guardrails import FinancialGuardrails


# ── Tools ─────────────────────────────────────────────────────────────────────

def get_stock_price(ticker: str) -> str:
    """Get current stock price for a single ticker."""
    try:
        stock = yf.Ticker(ticker.strip().upper())
        hist = stock.history(period="1d")
        if hist.empty:
            return f"No price data available for {ticker}."
        price = hist["Close"].iloc[-1]
        return f"Current price of {ticker.upper()}: ${price:.2f}"
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"


def get_portfolio_value(holdings_json: str) -> str:
    """
    Calculate total portfolio value.
    Input: JSON string mapping tickers to share counts, e.g. '{"AAPL": 100, "MSFT": 50}'
    """
    try:
        holdings = json.loads(holdings_json)
        total = 0.0
        lines = []
        for ticker, shares in holdings.items():
            stock = yf.Ticker(ticker.strip().upper())
            hist = stock.history(period="1d")
            if hist.empty:
                lines.append(f"{ticker}: no data")
                continue
            price = hist["Close"].iloc[-1]
            value = shares * price
            total += value
            lines.append(f"{ticker}: {shares} shares @ ${price:.2f} = ${value:,.2f}")
        lines.append(f"\nTotal Portfolio Value: ${total:,.2f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error calculating portfolio value: {str(e)}"


def check_portfolio_alerts(holdings_json: str) -> str:
    """
    Check for significant price movements (>5%) in the last trading day.
    Input: JSON string mapping tickers to share counts, e.g. '{"AAPL": 100, "MSFT": 50}'
    """
    try:
        holdings = json.loads(holdings_json)
        alerts = []
        clean = []
        threshold = 0.05

        for ticker in holdings:
            stock = yf.Ticker(ticker.strip().upper())
            hist = stock.history(period="5d")
            if len(hist) < 2:
                continue
            current = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            change = (current - prev) / prev
            if abs(change) >= threshold:
                direction = "▲" if change > 0 else "▼"
                alerts.append(
                    f"ALERT: {ticker.upper()} {direction} {change*100:.2f}% "
                    f"(${prev:.2f} → ${current:.2f})"
                )
            else:
                clean.append(f"{ticker.upper()}: {change*100:+.2f}% — within normal range")

        if alerts:
            return "\n".join(alerts) + "\n\n" + "\n".join(clean)
        return "No significant price movements detected.\n" + "\n".join(clean)
    except Exception as e:
        return f"Error checking alerts: {str(e)}"


# ── Agent ─────────────────────────────────────────────────────────────────────

class PortfolioReActAgent:
    """
    Single autonomous ReAct agent for portfolio monitoring.
    Source: week6_capstone
    Framework: LangChain create_react_agent + AgentExecutor
    Tools: GetStockPrice · GetPortfolioValue · CheckPortfolioAlerts
    """

    REACT_PROMPT = """You are a portfolio monitoring AI agent for Meridian Wealth Partners. \
Your job is to analyze portfolios, identify issues, and provide clear recommendations.

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important rules:
- Always use a tool at least once before giving a Final Answer
- Keep Action Input exactly as required by the tool description
- Be specific and cite actual numbers in your Final Answer

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    def __init__(self, model: str = "gpt-4o"):
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain.tools import Tool
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

        tools = [
            Tool(
                name="GetStockPrice",
                func=get_stock_price,
                description=(
                    "Get the current price of a single stock. "
                    "Input must be a ticker symbol only, e.g. AAPL"
                ),
            ),
            Tool(
                name="GetPortfolioValue",
                func=get_portfolio_value,
                description=(
                    "Calculate the total value of a portfolio. "
                    'Input must be a JSON string mapping tickers to share counts, '
                    'e.g. {"AAPL": 100, "MSFT": 50}'
                ),
            ),
            Tool(
                name="CheckPortfolioAlerts",
                func=check_portfolio_alerts,
                description=(
                    "Check if any portfolio positions moved more than 5% in the last trading day. "
                    'Input must be a JSON string mapping tickers to share counts, '
                    'e.g. {"AAPL": 100, "MSFT": 50}'
                ),
            ),
        ]

        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template=self.REACT_PROMPT,
        )

        agent = create_react_agent(llm, tools, prompt)
        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def run(self, task: str) -> dict:
        """
        Run the agent on a task string.
        Returns {"output": str, "steps": list[dict], "iterations": int}
        """
        result = self._executor.invoke({"input": task})
        steps = []
        for action, observation in result.get("intermediate_steps", []):
            steps.append({
                "thought_and_action": action.log.strip(),
                "tool": action.tool,
                "tool_input": action.tool_input,
                "observation": observation,
            })
        return {
            "output": result["output"],
            "steps": steps,
            "iterations": len(steps),
        }


# ── Safe wrapper ──────────────────────────────────────────────────────────────

class SafeAgentExecutor:
    """
    Wraps PortfolioReActAgent with FinancialGuardrails input/output validation.
    Reuses core.guardrails — no new guardrail logic.
    """

    def __init__(self, agent: PortfolioReActAgent):
        self._agent = agent
        self._guardrails = FinancialGuardrails()

    def run(self, user_input: str) -> dict:
        in_check = self._guardrails.validate_input(user_input)
        if not in_check.passed:
            return {
                "status": "blocked",
                "message": f"Input blocked by guardrails: {'; '.join(in_check.violations)}",
                "steps": [],
                "iterations": 0,
            }

        try:
            result = self._agent.run(user_input)
        except Exception as e:
            return {"status": "error", "message": str(e), "steps": [], "iterations": 0}

        out_check = self._guardrails.validate_output(result["output"])
        result["output"] = out_check.modified_content
        result["status"] = "success"
        return result


# ── Evaluator ─────────────────────────────────────────────────────────────────

class AgentEvaluator:
    """
    Runs predefined test cases against the agent and measures accuracy + efficiency.
    Source: week6_capstone AgentEvaluator
    """

    # Fixed test suite — keywords that must appear in a passing answer
    TEST_CASES = [
        {
            "input": "What is the current price of Apple stock (AAPL)?",
            "must_contain": ["aapl", "$"],
            "label": "Single stock price lookup",
        },
        {
            "input": 'Calculate the total value of this portfolio: {"AAPL": 100, "MSFT": 50}',
            "must_contain": ["total", "$"],
            "label": "Portfolio value calculation",
        },
        {
            "input": 'Check if any positions in {"AAPL": 100, "GOOGL": 75} have moved more than 5% today',
            "must_contain": ["aapl", "googl"],
            "label": "Portfolio alert check",
        },
        {
            "input": 'Monitor this portfolio: {"TSLA": 50, "NVDA": 30}. Calculate total value and check for significant price movements.',
            "must_contain": ["total", "$"],
            "label": "Multi-step monitoring task",
        },
    ]

    def evaluate(self, agent: PortfolioReActAgent) -> dict:
        results = []
        for tc in self.TEST_CASES:
            try:
                result = agent.run(tc["input"])
                output_lower = result["output"].lower()
                correct = all(kw in output_lower for kw in tc["must_contain"])
                results.append({
                    "label": tc["label"],
                    "input": tc["input"],
                    "correct": correct,
                    "iterations": result["iterations"],
                    "output": result["output"],
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "label": tc["label"],
                    "input": tc["input"],
                    "correct": False,
                    "iterations": 0,
                    "output": "",
                    "error": str(e),
                })

        passed = sum(r["correct"] for r in results)
        accuracy = passed / len(results) if results else 0.0
        avg_steps = sum(r["iterations"] for r in results) / len(results) if results else 0.0
        return {
            "accuracy": accuracy,
            "avg_steps": avg_steps,
            "passed": passed,
            "total": len(results),
            "results": results,
        }
