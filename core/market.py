from typing import List, Dict, Tuple


class MarketDataFetcher:
    """Live portfolio data via yfinance — Milestone 3.1"""

    def fetch_portfolio(self, holdings: Dict[str, float]):
        import yfinance as yf
        results, errors, total_value = [], [], 0.0
        for ticker, shares in holdings.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                info = stock.info
                if hist.empty:
                    errors.append(ticker); continue
                price = hist["Close"].iloc[-1]
                has_prev = len(hist) >= 2
                prev = hist["Close"].iloc[-2] if has_prev else price
                day_chg = round((price - prev) / prev * 100, 2) if prev and prev != 0 else 0.0
                if not has_prev:
                    errors.append(f"{ticker}: only 1 day of history — day change shown as 0%")

                hist_1y = stock.history(period="1y")
                first_close = hist_1y["Close"].iloc[0] if len(hist_1y) >= 2 else None
                ytd = (round((hist_1y["Close"].iloc[-1] - first_close) / first_close * 100, 2)
                       if first_close and first_close != 0 else 0.0)

                value = price * shares
                results.append({
                    "Ticker": ticker, "Shares": shares,
                    "Price ($)": round(price, 2),
                    "Day Chg %": day_chg,
                    "Value ($)": round(value, 2),
                    "1Y Return %": ytd,
                    "Sector": info.get("sector") or "N/A",
                    "Beta": round(info.get("beta") or 0, 2),
                })
                total_value += value
            except Exception as e:
                errors.append(f"{ticker}: {str(e)[:40]}")
        return results, total_value, errors
