"""
Formats the combined resilience + overhead report as text. The data itself
now comes from PostgreSQL (instrumentation/db.py: build_overhead_report(),
get_storms()) instead of being assembled from per-process JSON dumps -- this
module is pure string formatting only.
"""

from __future__ import annotations

from typing import Optional


def format_report(report: dict, resilience: Optional[dict] = None, storms: Optional[list] = None) -> str:
    lines = []
    if resilience is not None:
        lines.append(
            f"Resilience P={resilience['P']:.3f}  absorb={resilience['absorption']:.2f} "
            f"adapt={resilience['adaptation']:.2f} trec={resilience['trec']:.2f} "
            f"recov={resilience['recovery_time']:.0f}s"
        )
        lines.append("")
    if storms and len(storms) > 1:
        lines.append("Evolution across storms:")
        for s in storms:
            r = s["resilience"]
            lines.append(
                f"  storm-{s['storm_index']}  P={r['P']:.3f}  absorb={r['absorption']:.2f}  "
                f"escalation_threshold={s['policy_before'].get('escalation_threshold'):.0f}->"
                f"{s['policy_after'].get('escalation_threshold'):.0f}  "
                f"drop_prob_floor={s['policy_before'].get('drop_prob_floor'):.2f}->"
                f"{s['policy_after'].get('drop_prob_floor'):.2f}"
            )
        lines.append("")
    lines.append("Overhead by channel:")
    for ch, stats in report["by_channel"].items():
        lines.append(
            f"  {ch:4s}  count={stats['count']:4d}  bytes={stats['total_bytes']:7d}  "
            f"mean_lat={stats['mean_latency_s']*1000:7.1f}ms  p95={stats['p95_latency_s']*1000:7.1f}ms  "
            f"fail={stats['failures']}"
        )
    lines.append("")
    lines.append("Per agent/process:")
    for owner, stats in report["per_owner"].items():
        llm = stats.get("llm", {})
        llm_part = f"  llm={llm['count']:3d} (fallback={llm['fallback_count']})" if llm.get("count") else ""
        lines.append(
            f"  {owner:12s}  invocations={stats['invocations']:4d}  "
            f"mcp={stats['mcp']['count']:3d}  a2a={stats['a2a']['count']:3d}{llm_part}"
        )
    llm_summary = report.get("llm_summary", {})
    if llm_summary.get("count"):
        lines.append("")
        lines.append(
            f"LLM usage: {llm_summary['count']} calls "
            f"({llm_summary['fallback_count']} fell back)  "
            f"tokens_in={llm_summary['total_tokens_in']}  tokens_out={llm_summary['total_tokens_out']}  "
            f"mean_latency={llm_summary['mean_latency_s']:.2f}s"
        )
    return "\n".join(lines)
