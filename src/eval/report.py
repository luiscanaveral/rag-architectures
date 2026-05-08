import json
import sys
from pathlib import Path

REPORT_DIR = Path(".logs/reports")
logs = sorted(REPORT_DIR.glob("eval.*.log"))

if not logs:
    print("No evaluation reports found.")
    sys.exit(0)

with open(logs[-1]) as f:
    for line in f:
        entry = json.loads(line)
        m = entry["metrics"]
        scores = "  ".join(
            f"{k}: {v['score']:.3f} {'PASS' if v['passed'] else 'FAIL'}"
            for k, v in m.items()
        )
        print(f"{entry['architecture']:15s} {entry['test_case']:20s} {scores}")

print(f"\nReport: {logs[-1]}")
