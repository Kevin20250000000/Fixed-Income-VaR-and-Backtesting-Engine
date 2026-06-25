"""Documentation generator for SR 11-7 compliance."""

import datetime


def generate_sr_11_7_report(model_name: str, assumptions: dict, validation_results: dict, stress_summaries: dict) -> str:
    lines = [
        f"# SR 11-7 Model Validation Report - {model_name}\n",
        f"Generated: {datetime.date.today().isoformat()}\n",
        "## 1. Model Overview\n",
        "This report documents model assumptions, limitations, validation results, stress testing, and governance controls in line with SR 11-7.\n",
        "## 2. Model Assumptions\n",
    ]
    for key, value in assumptions.items():
        lines.append(f"- **{key}**: {value}\n")
    lines.extend([
        "## 3. Limitations\n",
        "- Model-specific limitations should be documented here.\n",
        "## 4. Validation Results\n",
    ])
    for section, result in validation_results.items():
        lines.append(f"### {section}\n")
        lines.append(f"```
{result}
```")
    lines.extend([
        "## 5. Stress Test Summaries\n",
    ])
    for name, summary in stress_summaries.items():
        lines.append(f"### {name}\n- {summary}\n")
    lines.append("## 6. Governance Framework\n")
    lines.append("- Owners, controls, escalation, and change management are maintained in model governance registers.\n")
    return "\n".join(lines)
