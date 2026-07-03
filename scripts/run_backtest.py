#!/usr/bin/env python3
"""Weekly backtest runner for CI."""

import sys

from forecasting import MAPE_THRESHOLD, run_backtest_check


def main() -> int:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPCX"
    result = run_backtest_check(ticker=ticker, threshold=MAPE_THRESHOLD)

    m = result["metrics"]
    b = result["baseline"]
    print(f"Ticker: {result['ticker']}")
    print(f"Model MAPE: {m['mape']:.2f}% | Baseline MAPE: {b['mape']:.2f}%")
    print(f"Model RMSE: {m['rmse']:.2f} | Baseline RMSE: {b['rmse']:.2f}")
    print(f"Threshold: {result['threshold']}% | Passed: {result['passed']}")
    print(f"Best params: {result['params']}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
