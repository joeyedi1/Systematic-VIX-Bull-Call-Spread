# scripts/export_review_package.py
import os
from pathlib import Path

files_to_include = [
    "config/settings.py",
    "signals/composite_score.py",
    "backtest/engine.py",
    "strikes/selector.py",
    "features/indicators.py",
    "regime/hmm_classifier.py",
]

output = []
output.append("# VIX Bull Call Spread v1.3 — Full Code for Review\n\n")

# Add backtest results
output.append("## BACKTEST RESULTS (v1.3)\n```")
output.append("""Total Trades:     20
Win Rate:         75.0%
Avg Win:          +$1.48
Avg Loss:         $-0.83
Profit Factor:    5.34
Total P&L:        $18.00
Max Drawdown:     $-4.98
Sharpe Ratio:     1.98
Avg Holding Days: 17

Trade Log:
  #1  2023-12-08→2023-12-20 C12/C16 +$1.45 EXIT_PROFIT_TARGET
  #2  2023-12-11→2023-12-20 C12/C16 +$1.61 EXIT_PROFIT_TARGET
  #3  2023-12-26→2024-01-19 C14/C18 -$0.65 EXIT_TIME_STOP
  #4  2023-12-27→2024-01-22 C13/C17 -$0.39 EXIT_TIME_STOP
  #5  2024-06-07→2024-07-01 C12/C16 +$0.48 EXIT_TIME_STOP [scaled-out]
  #6  2024-06-10→2024-07-04 C12/C16 +$0.43 EXIT_TIME_STOP [scaled-out]
  #7  2024-07-05→2024-07-19 C12/C16 +$1.76 EXIT_PROFIT_TARGET
  #8  2024-07-08→2024-07-19 C12/C16 +$1.84 EXIT_PROFIT_TARGET
  #9  2024-08-15→2024-09-04 C15/C20 +$2.33 EXIT_PROFIT_TARGET
  #10 2024-08-16→2024-09-04 C15/C20 +$2.33 EXIT_PROFIT_TARGET
  #11 2024-12-02→2024-12-18 C14/C18 +$1.82 EXIT_PROFIT_TARGET
  #12 2024-12-03→2024-12-18 C14/C18 +$1.90 EXIT_PROFIT_TARGET
  #13 2025-06-09→2025-06-16 C17/C23 +$1.54 EXIT_REGIME_CHANGE
  #14 2025-06-10→2025-06-16 C17/C22 +$1.82 EXIT_REGIME_CHANGE
  #15 2025-08-08→2025-09-01 C16/C21 -$0.53 EXIT_TIME_STOP
  #16 2025-08-12→2025-09-08 C15/C20 +$0.57 EXIT_TIME_STOP [scaled-out]
  #17 2025-10-27→2025-11-04 C17/C22 +$0.73 EXIT_REGIME_CHANGE
  #18 2025-11-28→2025-12-22 C17/C22 -$1.32 EXIT_TIME_STOP
  #19 2025-12-03→2025-12-29 C17/C22 -$1.26 EXIT_TIME_STOP
  #20 2026-01-09→2026-01-19 C15/C20 +$1.54 EXIT_REGIME_CHANGE""")
output.append("```\n\n")

# Hedge effectiveness
output.append("## HEDGE EFFECTIVENESS\n```")
output.append("""Worst 1% NQ days:  +$0.00 (strategy inactive)
Worst 5% NQ days:  +$3.72
Worst 10% NQ days: +$4.45
All NQ down days:  +$15.56
All NQ up days:    +$2.44
Down-day correlation (active): -0.326
NQ-only max drawdown:  -$11,534K
NQ+Hedge max drawdown: -$11,534K (0% reduction at current sizing)""")
output.append("```\n\n")

# Version history
output.append("## VERSION HISTORY\n```")
output.append("""v1.0: 64 trades, 56.2% WR, 1.41 Sharpe, 2.66 PF, -$14.28 DD (baseline)
v1.1: 38 trades, 63.2% WR, 1.76 Sharpe, 3.31 PF, -$8.22 DD (tuned: max 2 concurrent, 0.70 threshold, 10d cooldown)
v1.2: 19 trades, 57.9% WR, 2.02 Sharpe, 3.62 PF, -$4.77 DD (banned TRANSITION, unconditional regime exit, moneyness strikes, EMA vol, reduced COT to 5%)
v1.3: 20 trades, 75.0% WR, 1.98 Sharpe, 5.34 PF, -$4.98 DD (scale-out exits, normalized spread width, signal-reset cooldown, Sharpe on premium-at-risk)""")
output.append("```\n\n")

# Source code
for fpath in files_to_include:
    p = Path(fpath)
    if p.exists():
        output.append(f"## FILE: {fpath}\n```python\n")
        output.append(p.read_text(encoding="utf-8"))
        output.append("\n```\n\n")
    else:
        output.append(f"## FILE: {fpath} — NOT FOUND\n\n")

Path("v13_review_package.md").write_text("\n".join(output), encoding="utf-8")
print(f"Created v13_review_package.md ({len(''.join(output))} chars)")