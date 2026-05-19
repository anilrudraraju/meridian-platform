"""
core/crew_agents.py — Layer 7: Multi-Agent Portfolio Analysis with CrewAI
Source: week7_capstone
Framework: CrewAI sequential crew — Research → Risk → Portfolio Manager
"""
import os
import json
import yfinance as yf
import pandas as pd
from typing import Optional, Callable


# ── Tools ─────────────────────────────────────────────────────────────────────

def _get_stock_data(ticker: str) -> str:
    """Fetch current price, 52-week range, P/E ratio, and market cap for one ticker."""
    try:
        stock = yf.Ticker(ticker.strip().upper())
        info = stock.info
        hist = stock.history(period="1y")
        if hist.empty:
            return f"{ticker}: no price data available."
        price = hist["Close"].iloc[-1]
        return (
            f"Stock: {ticker.upper()}\n"
            f"Current Price: ${price:.2f}\n"
            f"52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"52-Week Low:  ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"Market Cap: ${info.get('marketCap', 0):,}\n"
            f"Sector: {info.get('sector', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


def _calculate_portfolio_risk(tickers_csv: str) -> str:
    """
    Calculate annualised volatility and correlation for a comma-separated list of tickers.
    """
    try:
        tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
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
        @crewai_tool("GetStockData")
        def get_stock_data(ticker: str) -> str:
            """Get stock data including current price, 52-week range, P/E, and market cap for a single ticker symbol."""
            return _get_stock_data(ticker)

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
            tools=[get_stock_data],
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
        research_task = Task(
            description=(
                f"Analyse every holding in this portfolio: {portfolio}.\n"
                "For each ticker call GetStockData and report current price, "
                "valuation metrics, sector, and recent 52-week range."
            ),
            agent=research_agent,
            expected_output="A data report covering price, valuation, and sector for every holding.",
            callback=make_callback(research_agent.role),
        )

        risk_task = Task(
            description=(
                f"Using the research findings, assess the risk of this portfolio: {portfolio}.\n"
                "Call CalculatePortfolioRisk with ALL tickers together, then interpret "
                "the volatility and correlation data. Flag any concentration or sector risk."
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
