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
    # Renamed tickers
    "FB": "META",       # Facebook → Meta Platforms (Oct 2021)
    "GOOG": "GOOGL",   # prefer voting shares
    "TWITTER": "X",    # Twitter → X (delisted, but just in case)
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
            mcap = info.get("marketCap", 0)
            sector = info.get("sector", "N/A")

            # P/E: trailing → forward → derive from EPS → explain if negative earnings
            trailing_pe = info.get("trailingPE")
            forward_pe  = info.get("forwardPE")
            trailing_eps = info.get("trailingEps")
            if trailing_pe and trailing_pe > 0:
                pe_str = f"{trailing_pe:.1f}x (trailing)"
            elif forward_pe and forward_pe > 0:
                pe_str = f"{forward_pe:.1f}x (forward estimate)"
            elif trailing_eps is not None and trailing_eps < 0:
                pe_str = f"N/A — negative trailing EPS (${trailing_eps:.2f})"
            elif price and trailing_eps and trailing_eps > 0:
                pe_str = f"{price / trailing_eps:.1f}x (calculated)"
            else:
                pe_str = "N/A"

            lines.append(
                f"--- {symbol} ---\n"
                f"  Price: {'${:.2f}'.format(price) if price else 'N/A'}\n"
                f"  52-Week High/Low: {'${:.2f}'.format(high52) if high52 else 'N/A'} / "
                f"{'${:.2f}'.format(low52) if low52 else 'N/A'}\n"
                f"  P/E Ratio: {pe_str}\n"
                f"  Market Cap: ${mcap:,}\n"
                f"  Sector: {sector}"
            )
        except Exception as e:
            lines.append(f"--- {symbol} ---\n  Error: {e}")

    return "\n".join(lines) if lines else "No data retrieved."


def _calculate_portfolio_risk(holdings_csv: str) -> str:
    """
    Calculate weighted portfolio volatility and per-asset volatility + correlation.
    Input format: 'TICKER:weight,TICKER:weight,...' where weight is a decimal (e.g. AAPL:0.40,MSFT:0.35,GOOGL:0.25).
    """
    try:
        tickers, weights = [], {}
        for part in holdings_csv.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                t, w = part.split(":", 1)
                t = _normalise(t.strip())
                tickers.append(t)
                weights[t] = float(w.strip())
            else:
                t = _normalise(part)
                tickers.append(t)

        if not tickers:
            return "No valid tickers provided."

        # Equal weights if not specified
        if not weights:
            w = 1 / len(tickers)
            weights = {t: w for t in tickers}

        if len(tickers) == 1:
            data = yf.download(tickers[0], period="1y", progress=False)["Close"]
            returns = data.pct_change().dropna()
            vol = float(returns.std()) * (252 ** 0.5)
            return (f"Portfolio Risk Analysis:\n"
                    f"{tickers[0]} ({weights[tickers[0]]:.0%} allocation): {vol:.2%} annual volatility\n"
                    "(Single asset — no diversification benefit)")

        data = yf.download(tickers, period="1y", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data

        returns = close.pct_change().dropna()
        vol_per_asset = returns.std() * (252 ** 0.5)
        corr = returns.corr()
        cov = returns.cov() * 252

        # Weighted portfolio volatility: σ_p = sqrt(w^T Σ w)
        available = [t for t in tickers if t in cov.columns]
        w_vec = pd.Series({t: weights.get(t, 0) for t in available})
        w_vec = w_vec / w_vec.sum()
        port_vol = float((w_vec @ cov.loc[available, available] @ w_vec) ** 0.5)

        lines = [f"Weighted Portfolio Volatility: {port_vol:.2%} annualised\n",
                 "Per-Asset Volatility (with allocation weight):"]
        for t in tickers:
            if t in vol_per_asset:
                lines.append(f"  {t} ({weights.get(t, 0):.0%}): {vol_per_asset[t]:.2%} annual volatility")

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

    def run(self, holdings: dict, on_task: Optional[Callable] = None) -> dict:
        """
        Analyse a portfolio.
        holdings: dict mapping ticker -> decimal weight, e.g. {"AAPL": 0.40, "MSFT": 0.35, "GOOGL": 0.25}
        on_task(agent_role, status_msg) is called after each task completes.
        Returns {"output": str, "agent_outputs": list[dict], "holdings": dict}
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
        def calculate_portfolio_risk(holdings: str) -> str:
            """Calculate weighted portfolio volatility and per-asset risk. Input: comma-separated TICKER:weight pairs (decimal weights), e.g. AAPL:0.40,MSFT:0.35,GOOGL:0.25"""
            return _calculate_portfolio_risk(holdings)

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

        # ── Build strings from holdings dict ──────────────────────────────────
        tickers_csv = ", ".join(holdings.keys())
        # "AAPL (40%), MSFT (35%), GOOGL (25%)" — for agent context
        holdings_summary = ", ".join(f"{t} ({w:.0%})" for t, w in holdings.items())
        # "AAPL:0.40,MSFT:0.35,GOOGL:0.25" — for CalculatePortfolioRisk tool
        risk_input = ",".join(f"{t}:{w}" for t, w in holdings.items())

        # ── Tasks ─────────────────────────────────────────────────────────────
        research_task = Task(
            description=(
                f"Analyse the following portfolio holdings:\n{holdings_summary}\n\n"
                f"Call GetPortfolioData ONCE with all tickers: '{tickers_csv}'\n"
                "For each holding report: current price, 52-week range, P/E ratio, market cap, and sector.\n"
                "Note the allocation percentage next to each ticker in your report."
            ),
            agent=research_agent,
            expected_output="A data report covering price, valuation, sector, and allocation weight for every holding.",
            callback=make_callback(research_agent.role),
        )

        risk_task = Task(
            description=(
                f"Assess the risk of this portfolio:\n{holdings_summary}\n\n"
                f"Call CalculatePortfolioRisk ONCE with: '{risk_input}'\n"
                "Report the weighted portfolio volatility, per-asset volatility, and correlation matrix.\n"
                "Flag any concentration risk (single position > 30%), sector concentration, or high-correlation pairs."
            ),
            agent=risk_agent,
            expected_output="Risk report with weighted portfolio volatility, per-asset figures, correlation insights, and concentration flags.",
            callback=make_callback(risk_agent.role),
        )

        synthesis_task = Task(
            description=(
                f"Review the research and risk reports for this portfolio:\n{holdings_summary}\n\n"
                "Provide a concise executive summary with:\n"
                "1. Portfolio strengths\n"
                "2. Key risks (reference specific allocation percentages)\n"
                "3. Specific actionable recommendations — cite current weights and suggest target weights "
                "(e.g. 'trim AAPL from 40% to 25%, redeploy into ...')"
            ),
            agent=pm_agent,
            expected_output="Executive summary with weight-aware portfolio assessment and 3–5 specific recommendations citing current allocations.",
            callback=make_callback(pm_agent.role),
        )

        # ── Crew ──────────────────────────────────────────────────────────────
        crew = Crew(
            agents=[research_agent, risk_agent, pm_agent],
            tasks=[research_task, risk_task, synthesis_task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff(inputs={"portfolio": holdings_summary})
        final = result.raw if hasattr(result, "raw") else str(result)

        return {
            "output": final,
            "agent_outputs": agent_outputs,
            "holdings": holdings,
        }
