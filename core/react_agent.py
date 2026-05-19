"""
core/react_agent.py — Layer 6: Autonomous ReAct Agent for Portfolio Monitoring
Framework: OpenAI tool-calling API (Reasoning + Acting loop)
"""
import os
import json
import yfinance as yf


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


# ── Tool registry ──────────────────────────────────────────────────────────────

_TOOL_FUNCTIONS = {
    "GetStockPrice": get_stock_price,
    "GetPortfolioValue": get_portfolio_value,
    "CheckPortfolioAlerts": check_portfolio_alerts,
}

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "GetStockPrice",
            "description": (
                "Get the current price of a single stock. "
                "Input must be a ticker symbol only, e.g. AAPL"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GetPortfolioValue",
            "description": (
                "Calculate the total value of a portfolio. "
                'Input must be a JSON string mapping tickers to share counts, '
                'e.g. {"AAPL": 100, "MSFT": 50}'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings_json": {
                        "type": "string",
                        "description": 'JSON string mapping tickers to shares, e.g. {"AAPL": 100}',
                    }
                },
                "required": ["holdings_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CheckPortfolioAlerts",
            "description": (
                "Check if any portfolio positions moved more than 5% in the last trading day. "
                'Input must be a JSON string mapping tickers to share counts, '
                'e.g. {"AAPL": 100, "MSFT": 50}'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings_json": {
                        "type": "string",
                        "description": 'JSON string mapping tickers to shares, e.g. {"AAPL": 100}',
                    }
                },
                "required": ["holdings_json"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a portfolio monitoring AI agent for Meridian Wealth Partners. "
    "Your job is to analyze portfolios, identify issues, and provide clear recommendations. "
    "Before each tool call, briefly explain your reasoning in 1-2 sentences (what you are about to do and why). "
    "Always use at least one tool before giving your final answer. "
    "Be specific and cite actual numbers in your response."
)


# ── Agent ─────────────────────────────────────────────────────────────────────

class PortfolioReActAgent:
    """
    Autonomous ReAct agent for portfolio monitoring using OpenAI tool-calling.
    Tools: GetStockPrice · GetPortfolioValue · CheckPortfolioAlerts
    """

    def __init__(self, model: str = "gpt-4o"):
        import openai
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._max_iterations = 10

    # gpt-4o pricing per million tokens
    _COST_PER_M = {"input": 2.50, "output": 10.00}

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self._COST_PER_M["input"] / 1_000_000
            + completion_tokens * self._COST_PER_M["output"] / 1_000_000
        )

    def run(self, task: str) -> dict:
        """
        Run the agent on a task string.
        Returns {"output": str, "steps": list[dict], "iterations": int,
                 "prompt_tokens": int, "completion_tokens": int, "cost_usd": float}
        """
        from core.cost import log_call
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        steps = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for _ in range(self._max_iterations):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            usage = response.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            # No tool calls — agent is done
            if not msg.tool_calls:
                cost_usd = self._compute_cost(total_prompt_tokens, total_completion_tokens)
                log_call(
                    model=self._model,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    cost_usd=cost_usd,
                    technique="react",
                    caller="PortfolioReActAgent",
                )
                return {
                    "output": msg.content or "",
                    "steps": steps,
                    "iterations": len(steps),
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "cost_usd": cost_usd,
                }

            # Append assistant message with tool calls
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]})

            # thought is the model's text content before tool calls
            thought = (msg.content or "").strip()

            # Execute each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                fn = _TOOL_FUNCTIONS.get(fn_name)
                if fn is None:
                    observation = f"Unknown tool: {fn_name}"
                else:
                    # Each tool takes a single positional arg
                    arg_val = next(iter(fn_args.values()), "") if fn_args else ""
                    observation = fn(arg_val)

                steps.append({
                    "thought": thought,
                    "tool": fn_name,
                    "tool_input": fn_args,
                    "observation": observation,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

        # Hit max iterations — log and return whatever we have
        cost_usd = self._compute_cost(total_prompt_tokens, total_completion_tokens)
        log_call(
            model=self._model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost_usd=cost_usd,
            technique="react",
            caller="PortfolioReActAgent",
        )
        last_content = messages[-1].get("content", "Max iterations reached without final answer.")
        return {
            "output": last_content,
            "steps": steps,
            "iterations": len(steps),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "cost_usd": cost_usd,
        }


# ── Safe wrapper ──────────────────────────────────────────────────────────────

class SafeAgentExecutor:
    """
    Wraps PortfolioReActAgent with FinancialGuardrails input/output validation.
    Reuses core.guardrails — no new guardrail logic.
    """

    def __init__(self, agent: PortfolioReActAgent):
        from core.guardrails import FinancialGuardrails
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
    """

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
