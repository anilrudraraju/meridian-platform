"""
core/crew_agents.py — Layer 7: Multi-Agent Portfolio Analysis with CrewAI
Source: week7_capstone
Framework: CrewAI sequential crew — Research → Risk → Portfolio Manager
"""
import os
import time
import yfinance as yf
import pandas as pd
from typing import Optional, Callable


# ── Ticker normalisation ───────────────────────────────────────────────────────
_TICKER_ALIASES = {
    # Crypto — must have -USD suffix on Yahoo Finance
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "DOGE": "DOGE-USD", "ADA": "ADA-USD", "XRP": "XRP-USD",
    # Berkshire variants
    "BRK": "BRK-B", "BRK.B": "BRK-B", "BRK/B": "BRK-B",
    "BRK.A": "BRK-A", "BRK/A": "BRK-A",
}

def _normalise(ticker: str) -> str:
    t = ticker.strip().upper()
    return _TICKER_ALIASES.get(t, t)

def _parse_tickers(portfolio_csv: str) -> list[str]:
    return [_normalise(t) for t in portfolio_csv.split(",") if t.strip()]


# ── Tools ─────────────────────────────────────────────────────────────────────

def _get_portfolio_data(tickers_csv: str) -> str:
    """
    Batch-fetch price, 52-week range, P/E, market cap for all tickers in one call.
    Uses yf.download() for prices (one HTTP request) then yf.Ticker.info per symbol.
    """
    symbols = _parse_tickers(tickers_csv)
    if not symbols:
        return "No valid tickers provided."

    # One batch download for all prices
    try:
        raw = yf.download(symbols if len(symbols) > 1 else symbols[0],
                          period="1y", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw.rename(columns={"Close": symbols[0]}) if "Close" in raw.columns else raw
            close = pd.DataFrame(close)
    except Exception as e:
        return f"Price download failed: {e}"

    lines = []
    for symbol in symbols:
        try:
            # Price from batch download
            col = symbol if symbol in close.columns else None
            if col and not close[col].dropna().empty:
                price = float(close[col].dropna().iloc[-1])
                high52 = float(close[col].dropna().max())
                low52 = float(close[col].dropna().min())
            else:
                price = high52 = low52 = None

            # Fundamentals — one call per ticker but with a small delay
            time.sleep(0.5)
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE", "N/A")
            mcap = info.get("marketCap", 0)
            sector = info.get("sector", "N/A")

            lines.append(
                f"--- {symbol} ---\n"
                f"  Price: {'${:.2f}'.format(price) if price else 'N/A'}\n"
                f"  52-Week High/Low: {'${:.2f}'.format(high52) if high52 else 'N/A'} / "
                f"{'${:.2f}'.format(low52) if low52 else 'N/A'}\n"
                f"  P/E Ratio: {pe}\n"
                f"  Market Cap: ${mcap:,}\n"
                f"  Sector: {sector}"
            )
        except Exception as e:
            lines.append(f"--- {symbol} ---\n  Error: {e}")

    return "\n".join(lines) if lines else "No data retrieved."


def _calculate_portfolio_risk(tickers_csv: str) -> str:
    """
    Calculate annualised volatility and correlation for a comma-separated list of tickers.
    """
    try:
        tickers = [_normalise(t) for t in tickers_csv.split(",") if t.strip()]
        if len(tickers) == 1:
            data = yf.download(tickers[0], period="1y", progress=False)["Close"]
            returns = data.pct_change().dropna()
            vol = float(returns.std()) * (252 ** 0.5)
            return f"Portfolio Risk Analysis:\n{tickers[0]}: {vol:.2%} annual volatility\n(Single asset — no correlation data)"

        data = yf.download(tickers, period="1y", progress=False)
        # handle both flat and multi-level column index
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data

        returns = close.pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)
        corr = returns.corr()

        lines = ["Portfolio Risk Analysis:"]
        for t in tickers:
            if t in volatility:
                lines.append(f"  {t}: {volatility[t]:.2%} annual volatility")

        lines.append("\nCorrelation Matrix:")
        for t1 in tickers:
            for t2 in tickers:
                if t1 < t2 and t1 in corr.columns and t2 in corr.columns:
                    lines.append(f"  {t1} / {t2}: {corr.loc[t1, t2]:.2f}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error calculating risk: {str(e)}"


# ── Crew ──────────────────────────────────────────────────────────────────────

class PortfolioAnalysisCrew:
    """
    Three-agent sequential crew for portfolio analysis.
    Agents: Research Analyst → Risk Specialist → Portfolio Manager
    Tools:  GetStockData (Research) · CalculatePortfolioRisk (Risk)
    """

    def __init__(self, model: str = "gpt-4o"):
        self._model = model

    def run(self, portfolio: str, on_task: Optional[Callable] = None) -> dict:
        """
        Analyse a portfolio string (comma-separated tickers).
        on_task(agent_role, status_msg) is called after each task completes.
        Returns {"output": str, "agent_outputs": list[dict], "portfolio": str}
        """
        from crewai import Agent, Task, Crew, Process, LLM
        from crewai.tools import tool as crewai_tool

        llm = LLM(model=self._model, api_key=os.environ.get("OPENAI_API_KEY"))

        # Wrap plain functions as CrewAI tools
        @crewai_tool("GetPortfolioData")
        def get_portfolio_data(tickers: str) -> str:
            """Fetch price, 52-week range, P/E ratio, and market cap for ALL tickers at once. Input: comma-separated ticker symbols, e.g. AAPL,MSFT,BTC"""
            return _get_portfolio_data(tickers)

        @crewai_tool("CalculatePortfolioRisk")
        def calculate_portfolio_risk(tickers: str) -> str:
            """Calculate annualised volatility and correlation matrix for a portfolio. Input: comma-separated ticker symbols, e.g. AAPL,MSFT,GOOGL"""
            return _calculate_portfolio_risk(tickers)

        # ── Agents ────────────────────────────────────────────────────────────
        research_agent = Agent(
            role="Financial Research Analyst",
            goal="Gather comprehensive current data on every holding in the portfolio",
            backstory=(
                "You are an experienced equity research analyst at Meridian Wealth Partners "
                "with expertise in gathering and synthesising financial data across sectors."
            ),
            tools=[get_portfolio_data],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        risk_agent = Agent(
            role="Portfolio Risk Specialist",
            goal="Assess portfolio risk, identify concentration and volatility concerns",
            backstory=(
                "You are a quantitative risk analyst with deep expertise in portfolio "
                "risk management, volatility analysis, and correlation-based diversification."
            ),
            tools=[calculate_portfolio_risk],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        pm_agent = Agent(
            role="Senior Portfolio Manager",
            goal="Synthesise research and risk analysis into clear, actionable client recommendations",
            backstory=(
                "You are a seasoned portfolio manager with 20+ years managing wealth for "
                "high-net-worth clients. You translate complex analysis into concise, "
                "jargon-free guidance with specific next steps."
            ),
            tools=[],
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        # ── Task callbacks ────────────────────────────────────────────────────
        agent_outputs = []

        def make_callback(role):
            def cb(output):
                raw = output.raw if hasattr(output, "raw") else str(output)
                agent_outputs.append({"agent": role, "output": raw})
                if on_task:
                    on_task(role, raw)
            return cb

        # ── Tasks ─────────────────────────────────────────────────────────────
        # Normalise tickers before passing into task descriptions
        normalised = ", ".join(_parse_tickers(portfolio))

        research_task = Task(
            description=(
                f"Analyse this portfolio: {normalised}.\n"
                f"Call GetPortfolioData ONCE with all tickers together: '{normalised}'\n"
                "Report current price, 52-week range, P/E ratio, market cap, and sector for every holding."
            ),
            agent=research_agent,
            expected_output="A data report covering price, valuation, and sector for every holding.",
            callback=make_callback(research_agent.role),
        )

        risk_task = Task(
            description=(
                f"Using the research findings, assess the risk of this portfolio.\n"
                f"Call CalculatePortfolioRisk ONCE with all tickers together: '{normalised}'\n"
                "Interpret the volatility figures and correlation matrix. Flag concentration or sector risk."
            ),
            agent=risk_agent,
            expected_output="Risk assessment with volatility figures, correlation insights, and key risk flags.",
            callback=make_callback(risk_agent.role),
        )

        synthesis_task = Task(
            description=(
                "Review the research report and risk assessment from your colleagues.\n"
                "Provide a concise executive summary with:\n"
                "1. Portfolio strengths\n"
                "2. Key risks to address\n"
                "3. Specific actionable recommendations (buy / trim / hold / diversify)"
            ),
            agent=pm_agent,
            expected_output="Executive summary with portfolio assessment and 3–5 specific recommendations.",
            callback=make_callback(pm_agent.role),
        )

        # ── Crew ──────────────────────────────────────────────────────────────
        crew = Crew(
            agents=[research_agent, risk_agent, pm_agent],
            tasks=[research_task, risk_task, synthesis_task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff(inputs={"portfolio": portfolio})
        final = result.raw if hasattr(result, "raw") else str(result)

        return {
            "output": final,
            "agent_outputs": agent_outputs,
            "portfolio": portfolio,
        }
