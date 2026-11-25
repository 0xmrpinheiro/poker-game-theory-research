"""Compare a fixed monthly return-sign rule with buy-and-hold."""

import argparse
import math
from datetime import date
from pathlib import Path

import pandas as pd


def sample_closes():
    """Return a deterministic synthetic path, not observations from a ticker."""
    returns = [0.04, -0.02, 0.08, -0.01, 0.06, 0.03, -0.02, -0.015, 0.07, 0.02, -0.01, 0.05]
    prices = [100.0]
    for change in returns * 4:
        prices.append(prices[-1] * (1 + change))
    dates = pd.period_range("2019-12", periods=len(prices), freq="M")
    return pd.Series(prices, index=dates, name="Close")


def monthly_closes(close, *, daily=False):
    """Validate dated closes and select one observation per consecutive month."""
    if close.empty:
        raise ValueError("No closing prices were returned.")
    dates = pd.to_datetime(close.index, format="ISO8601", errors="coerce")
    if dates.isna().any() or dates.has_duplicates:
        raise ValueError("Closing prices must have valid, unique dates.")
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    close = pd.Series(pd.to_numeric(close, errors="coerce").to_numpy(), index=dates).sort_index()
    if not close.map(math.isfinite).all() or (close <= 0).any():
        raise ValueError("Closing prices must be positive and finite.")

    months = close.index.to_period("M")
    if daily:
        close = close.groupby(months).last()
    else:
        if months.has_duplicates:
            raise ValueError("Supply exactly one closing price per month in the CSV.")
        close.index = months
    if not close.index.equals(pd.period_range(close.index[0], close.index[-1], freq="M")):
        raise ValueError("Monthly closing prices must be consecutive; check for missing months.")
    return close.rename("Close")


def download_closes(ticker, start, end):
    from yfinance import download

    if start.day != 1 or end.day != 1 or start >= end:
        raise ValueError("Use first-of-month start and exclusive end dates, in chronological order.")
    today = date.today()
    if end > today.replace(day=1):
        raise ValueError("The end date must not include an unfinished month.")
    if not ticker or any(c.isspace() or c == "," for c in ticker):
        raise ValueError("Supply a single ticker such as AAPL.")

    prices = download(ticker, start=start, end=end, interval="1d", auto_adjust=True,
                      multi_level_index=False, threads=False, progress=False)
    if prices is None or prices.empty or "Close" not in prices:
        raise ValueError(f"No adjusted daily close data for {ticker}.")
    close = monthly_closes(prices["Close"], daily=True)
    expected = pd.period_range(start, pd.Period(end, freq="M") - 1, freq="M")
    if not close.index.equals(expected):
        raise ValueError("The provider did not return every month in the requested period.")
    return close


def compare(close, initial_capital=10_000.0):
    """Apply last month's signal only to subsequent returns on a shared timeline."""
    if len(close) < 4:
        raise ValueError("At least four monthly closes are needed (two warm-up returns and one scored return).")
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Starting capital must be positive and finite.")

    returns = close.pct_change(fill_method=None)
    previous = returns.shift(1)
    two_ago = returns.shift(2)
    exposed = ((previous > 0) | ((previous <= 0) & (two_ago <= 0))).iloc[3:]
    evaluated = returns.iloc[3:]
    start = pd.DataFrame(
        {"heuristic": [initial_capital], "buy_and_hold": [initial_capital]}, index=close.index[2:3],
    )
    values = pd.DataFrame({
        "heuristic": initial_capital * (1 + evaluated.where(exposed, 0.0)).cumprod(),
        "buy_and_hold": initial_capital * (1 + evaluated).cumprod(),
    })
    return pd.concat([start, values]), exposed


def count_patterns(close):
    """Count observed three-return patterns for description, not signal selection."""
    signs = ["+" if value > 0 else "-" for value in close.pct_change(fill_method=None).dropna()]
    counts = pd.Series(["".join(signs[i:i + 3]) for i in range(len(signs) - 2)], dtype="str")
    return counts.value_counts().sort_index()


def save_plot(path, values, source):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    fig, ax = plt.subplots(figsize=(9.5, 4), layout="constrained")
    dates = values.index.to_timestamp(how="end").normalize()
    ax.plot(dates, values["buy_and_hold"], color="#2455a4", label="Buy and hold", linewidth=2.3)
    ax.plot(dates, values["heuristic"], color="#aa4714", label="Return-sign heuristic", linewidth=2.3)
    ax.set_title(f"Monthly comparison · {source}", loc="left", fontweight="bold")
    ax.set(xlabel="Month end", ylabel="Hypothetical portfolio value")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", action="store_true", help="Use synthetic monthly closes (the default).")
    source.add_argument("--csv", type=Path, help="Read a CSV with Date and Close columns, one row per month.")
    source.add_argument("--ticker", help="Download adjusted daily closes, e.g. AAPL.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 1, 1), help="Exclusive end date.")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--plot", type=Path, help="Save both portfolio curves to a PNG file.")
    args = parser.parse_args(argv)

    try:
        if args.csv:
            close = monthly_closes(pd.read_csv(args.csv, index_col="Date")["Close"])
            label = f"monthly CSV ({args.csv.name})"
        elif args.ticker:
            ticker = args.ticker.strip().upper()
            close = download_closes(ticker, args.start, args.end)
            label = f"{ticker} adjusted closes from Yahoo Finance"
        else:
            close = sample_closes()
            label = "synthetic sample (not market data)"
        values, exposure = compare(close, args.capital)
        if args.plot:
            save_plot(args.plot, values, label)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Source: {label}")
    print(f"Shared starting close: {values.index[0]}; starting capital: {args.capital:,.2f}")
    print(f"Evaluated months: {exposure.index[0]} through {exposure.index[-1]} ({len(exposure)})")
    print(f"Heuristic exposure: {exposure.sum()}/{len(exposure)} months")
    for column, name in [("heuristic", "Return-sign heuristic"), ("buy_and_hold", "Buy and hold")]:
        value = values[column].iloc[-1]
        total_return = value / args.capital - 1
        print(f"{name}: {value:,.2f} ({total_return:+.2%})")
    patterns = count_patterns(close)
    print("Three-month patterns (descriptive only; '-' includes flat months):")
    print("  " + ", ".join(f"{pattern}: {count}" for pattern, count in patterns.items()))
    if args.plot:
        print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()