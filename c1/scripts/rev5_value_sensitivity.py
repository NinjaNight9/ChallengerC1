#!/usr/bin/env python
"""Run the frozen ChallengerC1 value stage and force non-approval-grade provenance.

The underlying evaluator selects C1-VALUE/C1-CORE10 thresholds on development
only and grades the untouched holdout afterward. TennisExplorer historical
prices are displayed historical averages without immutable prestart quote
timestamps, so this wrapper *always* marks betting_policy_approved false.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def _arg_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    out_dir = _arg_value("--out-dir")
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "scripts" / "evaluate_value_from_artifacts.py"), run_name="__main__")

    if out_dir:
        report_path = Path(out_dir) / "FINAL_VALUE_EVALUATION.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.update({
                "evaluation_grade": "NON_APPROVAL_GRADE_SENSITIVITY",
                "odds_source": "TennisExplorer",
                "odds_price_type": "historical_displayed_average",
                "immutable_prestart_quote_provenance": False,
                "betting_policy_approved": False,
                "approval_blocker": (
                    "Historical TennisExplorer prices do not provide immutable timestamped "
                    "STRICT_PRESTART quote provenance. ROI is sensitivity evidence only."
                ),
            })
            report_path.write_text(json.dumps(report, indent=2, default=str))
            print("\n=== Rev5 provenance-enforced value sensitivity report ===")
            print(json.dumps(report, indent=2, default=str))
