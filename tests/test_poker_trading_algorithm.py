import subprocess
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from poker_trading_algorithm import compare, count_patterns, download_closes, monthly_closes, sample_closes


ROOT = Path(__file__).resolve().parents[1]


class MonthlyComparisonTests(TestCase):
    def test_starting_value_and_benchmark_dates_are_aligned(self):
        close = pd.Series([100, 90, 81, 100, 120, 90, 135], index=pd.period_range("2020-01", periods=7, freq="M"))
        values, exposure = compare(close, initial_capital=100)
        self.assertEqual(exposure.tolist(), [True, True, True, False])
        pd.testing.assert_index_equal(values.index, close.index[2:])
        self.assertEqual(values.iloc[0].tolist(), [100, 100])
        self.assertAlmostEqual(values["heuristic"].iloc[-1], 100 * 90 / 81)
        self.assertAlmostEqual(values["buy_and_hold"].iloc[-1], 100 * 135 / 81)

    def test_a_months_own_return_cannot_change_its_exposure(self):
        original = sample_closes().iloc[:7]
        changed = original.copy()
        changed.iloc[3] *= 0.5
        _, original_exposure = compare(original)
        _, changed_exposure = compare(changed)
        self.assertEqual(original_exposure.iloc[0], changed_exposure.iloc[0])

    def test_pattern_count_does_not_invent_a_missing_first_return(self):
        close = pd.Series([100, 120, 120, 110, 121])
        self.assertEqual(count_patterns(close).to_dict(), {"+--": 1, "--+": 1})

    def test_csv_requires_one_positive_close_per_consecutive_month(self):
        close = pd.Series([110, 100], index=["2020-02-29", "2020-01-31"])
        self.assertEqual(monthly_closes(close).tolist(), [100, 110])
        missing = pd.Series([100, 110], index=["2020-01-31", "2020-03-31"])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            monthly_closes(missing)
        duplicate = pd.Series([100, 110], index=["2020-01-01", "2020-01-31"])
        with self.assertRaisesRegex(ValueError, "one closing price per month"):
            monthly_closes(duplicate)

    def test_daily_download_uses_last_available_close_of_each_month(self):
        prices = pd.DataFrame({"Close": [99, 100, 120]}, index=pd.to_datetime([
            "2020-01-03", "2020-01-31", "2020-02-28",
        ]))
        start, end = date(2020, 1, 1), date(2020, 3, 1)
        with patch("yfinance.download", return_value=prices) as download:
            close = download_closes("AAPL", start, end)
        self.assertEqual(close.tolist(), [100, 120])
        self.assertEqual(close.index.astype(str).tolist(), ["2020-01", "2020-02"])
        self.assertTrue(download.call_args.kwargs["auto_adjust"])
        self.assertEqual(download.call_args.kwargs["interval"], "1d")

    def test_offline_quickstart_writes_a_headless_plot(self):
        with TemporaryDirectory() as directory:
            plot = Path(directory) / "poker-comparison.png"
            result = self.run_cli("--sample", "--plot", str(plot))
            self.assertTrue(plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("synthetic sample (not market data)", result)
        self.assertIn("Shared starting close: 2020-02", result)
        self.assertIn("through 2023-12 (46)", result)
        self.assertIn("Buy and hold:", result)

    def test_csv_quickstart_reproduces_sample_values(self):
        sample = sample_closes().rename_axis("Date")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "monthly.csv"
            sample.to_csv(path)
            output = self.run_cli("--csv", str(path))
        self.assertIn("monthly CSV (monthly.csv)", output)
        values, _ = compare(sample)
        self.assertIn(f"Buy and hold: {values['buy_and_hold'].iloc[-1]:,.2f}", output)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "poker_trading_algorithm.py"), *args], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout