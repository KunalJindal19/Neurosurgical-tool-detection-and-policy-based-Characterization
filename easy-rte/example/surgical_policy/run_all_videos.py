#!/usr/bin/env python3
"""
Runtime Verification Summary Generator
=======================================
Runs the surgical policy monitor on multiple prediction videos and
generates a summary table for the research paper.

Replicates the EXACT same logic as:
  - test_image.py   (tracking counters)
  - surgical_policy_main.c  (policy violation checks)

Policies (from surgical_policy.erte):
  P1: Suction + Bipolar conflict           → violation if both active
  P2: Dissecting max continuous usage       → violation if CONSEC_Dissecting > 5
  P3: Suction cumulative total              → violation if TOTAL_Suction > 15
  P4: Dissecting cooldown                   → violation if Dissecting active during cooldown
  P5: Cusa requires prior Dissecting usage  → violation if Cusa active without Dissecting_Ever_Used
"""

import pickle
import os

# ─── Policy thresholds (from F_surgical_policy.h) ───
MAX_DISSECTING_CONSEC = 5     # CONST_p2_dissecting_time_MAX_Dissecting
MAX_TOTAL_SUCTION     = 15    # CONST_p3_total_suction_MAX_TOTAL_Suction
COOLDOWN_ON_THRESHOLD = 5     # Frames ON before cooldown triggers
COOLDOWN_OFF_THRESHOLD = 5    # Frames OFF before cooldown resets

# ─── Tool label mapping (from detection model) ───
# Index 0 = Background, 1 = Suction, 2 = Bipolar, 3 = Other, 4 = Dissecting, 5 = Cusa
TOOL_NAMES = ["Background", "Suction", "Bipolar", "Other", "Dissecting", "Cusa"]
# We only track these 4 tools (skip Background=0 and Other=3)
TRACKED_TOOLS = [1, 2, 4, 5]  # indices into presence_vector
TRACKED_NAMES = ["Suction", "Bipolar", "Dissecting", "Cusa"]


def process_video(pickle_path, num_frames):
    """
    Process a single video's prediction file through the full pipeline.
    Replicates test_image.py tracking + surgical_policy_main.c policy checks.
    
    Returns a dict with all collected statistics.
    """
    with open(pickle_path, 'rb') as f:
        last_video, preds_video = pickle.load(f)

    # ─── Persistent tracking state (same as test_image.py) ───
    consec_counts = [0, 0, 0, 0]       # Consecutive frames ON  [Suction, Bipolar, Dissecting, Cusa]
    consec_off    = [0, 0, 0, 0]       # Consecutive frames OFF
    total_active  = [0, 0, 0, 0]       # Cumulative total usage
    prev_state    = [0, 0, 0, 0]       # Previous frame state
    dissecting_ever_used = 0
    in_cooldown_state = 0

    # ─── Statistics collectors ───
    tool_frame_counts = [0, 0, 0, 0]   # Total frames each tool is active
    max_consec_on     = [0, 0, 0, 0]   # Max consecutive ON per tool
    max_consec_off    = [0, 0, 0, 0]   # Max consecutive OFF per tool

    policy_violations = {
        'P1': [],  # list of frame numbers where violated
        'P2': [],
        'P3': [],
        'P4': [],
        'P5': [],
    }

    frames_with_any_violation = set()
    frames_with_no_tools = 0

    for frame_idx in range(num_frames):
        frame_id = frame_idx + 1  # 1-based frame ID

        # ─── Step 1: Build presence vector (same as test_image.py) ───
        presence_vector = [0] * 6

        if frame_id in preds_video:
            for det in preds_video[frame_id]:
                scores = det['scores']
                for i in range(len(scores)):
                    if scores[i] == 1:
                        presence_vector[i] = 1

        # ─── Step 2: Update tracking counters (same as test_image.py) ───
        for k, tool_idx in enumerate(TRACKED_TOOLS):
            if presence_vector[tool_idx] == 1:
                # Tool is ON
                total_active[k] += 1
                consec_counts[k] += 1
                consec_off[k] = 0
            else:
                # Tool is OFF
                if k == 2 and prev_state[k] == 1 and consec_counts[k] >= COOLDOWN_ON_THRESHOLD:
                    # Dissecting transition ON→OFF after 5+ streak: trigger cooldown
                    in_cooldown_state = 1

                consec_off[k] += 1
                consec_counts[k] = 0

            prev_state[k] = presence_vector[tool_idx]

        # Check if Dissecting has ever been used
        if presence_vector[4] == 1:  # index 4 = Dissecting
            dissecting_ever_used = 1

        # Reset cooldown if Dissecting has been OFF for >= 5 frames
        if consec_off[2] >= COOLDOWN_OFF_THRESHOLD:
            in_cooldown_state = 0

        # ─── Step 3: Collect tool usage stats ───
        for k, tool_idx in enumerate(TRACKED_TOOLS):
            if presence_vector[tool_idx] == 1:
                tool_frame_counts[k] += 1
            if consec_counts[k] > max_consec_on[k]:
                max_consec_on[k] = consec_counts[k]
            if consec_off[k] > max_consec_off[k]:
                max_consec_off[k] = consec_off[k]

        any_tool_present = any(presence_vector[t] for t in TRACKED_TOOLS)
        if not any_tool_present:
            frames_with_no_tools += 1

        # ─── Step 4: Evaluate policies (same conditions as surgical_policy_main.c) ───
        # These use the RAW inputs, exactly as the C code does

        # P1: Suction && Bipolar
        suction_on    = presence_vector[1]
        bipolar_on    = presence_vector[2]
        dissecting_on = presence_vector[4]
        cusa_on       = presence_vector[5]

        if suction_on and bipolar_on:
            policy_violations['P1'].append(frame_id)
            frames_with_any_violation.add(frame_id)

        # P2: Dissecting && (CONSEC_Dissecting > MAX_DISSECTING_CONSEC)
        if dissecting_on and consec_counts[2] > MAX_DISSECTING_CONSEC:
            policy_violations['P2'].append(frame_id)
            frames_with_any_violation.add(frame_id)

        # P3: Suction && (TOTAL_Suction > MAX_TOTAL_SUCTION)
        if suction_on and total_active[0] > MAX_TOTAL_SUCTION:
            policy_violations['P3'].append(frame_id)
            frames_with_any_violation.add(frame_id)

        # P4: Dissecting && IN_COOLDOWN_Dissecting
        if dissecting_on and in_cooldown_state:
            policy_violations['P4'].append(frame_id)
            frames_with_any_violation.add(frame_id)

        # P5: Cusa && !Dissecting_Ever_Used
        if cusa_on and not dissecting_ever_used:
            policy_violations['P5'].append(frame_id)
            frames_with_any_violation.add(frame_id)

    # ─── Build results dict ───
    total_violations = sum(len(v) for v in policy_violations.values())
    compliant_frames = num_frames - len(frames_with_any_violation)
    compliance_rate = (compliant_frames / num_frames) * 100 if num_frames > 0 else 0

    first_violation_frame = None
    all_violation_frames = sorted(frames_with_any_violation)
    if all_violation_frames:
        first_violation_frame = all_violation_frames[0]

    return {
        'num_frames': num_frames,
        'tool_frame_counts': tool_frame_counts,
        'max_consec_on': max_consec_on,
        'policy_violations': policy_violations,
        'total_violations': total_violations,
        'frames_with_violations': len(frames_with_any_violation),
        'compliant_frames': compliant_frames,
        'compliance_rate': compliance_rate,
        'first_violation_frame': first_violation_frame,
        'frames_with_no_tools': frames_with_no_tools,
    }


def generate_table(results, output_path):
    """
    Generate a formatted summary table and write to file.
    results: dict mapping video name → process_video() output
    """
    video_names = list(results.keys())
    
    lines = []
    lines.append("=" * 80)
    lines.append("  RUNTIME VERIFICATION SUMMARY — SURGICAL POLICY MONITOR")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Policies:")
    lines.append("  P1: Suction + Bipolar cannot be active simultaneously")
    lines.append("  P2: Dissecting max continuous usage > 5 frames")
    lines.append("  P3: Suction cumulative total > 15 frames")
    lines.append("  P4: Dissecting used during cooldown period")
    lines.append("  P5: Cusa used without prior Dissecting usage")
    lines.append("")

    # ─── Table 1: Overall Summary ───
    lines.append("-" * 80)
    lines.append("TABLE 1: Per-Video Runtime Verification Summary")
    lines.append("-" * 80)
    lines.append("")

    # Header
    col_w = 18  # column width for video data
    metric_w = 42  # metric label width
    header = f"{'Metric':<{metric_w}}"
    for vname in video_names:
        header += f"| {vname:^{col_w}}"
    lines.append(header)
    lines.append("-" * metric_w + ("+" + "-" * (col_w + 1)) * len(video_names))

    def add_row(label, values, fmt="{}"):
        row = f"{label:<{metric_w}}"
        for v in values:
            row += f"| {fmt.format(v):^{col_w}}"
        lines.append(row)

    # General info
    add_row("Total Frames", [r['num_frames'] for r in results.values()])
    lines.append("")

    # Tool presence
    lines.append(f"{'  — Tool Presence —':<{metric_w}}" + ("|" + " " * (col_w + 1)) * len(video_names))
    for k, tool_name in enumerate(TRACKED_NAMES):
        vals = []
        for r in results.values():
            count = r['tool_frame_counts'][k]
            pct = (count / r['num_frames']) * 100
            vals.append(f"{count} ({pct:.1f}%)")
        add_row(f"  {tool_name} (frames / %)", vals)

    lines.append("")

    # Max consecutive ON
    lines.append(f"{'  — Max Consecutive ON —':<{metric_w}}" + ("|" + " " * (col_w + 1)) * len(video_names))
    for k, tool_name in enumerate(TRACKED_NAMES):
        add_row(f"  {tool_name}", [r['max_consec_on'][k] for r in results.values()])

    lines.append("")

    # Policy violations
    lines.append(f"{'  — Policy Violations —':<{metric_w}}" + ("|" + " " * (col_w + 1)) * len(video_names))
    policy_labels = {
        'P1': 'P1: Suction+Bipolar',
        'P2': 'P2: Dissecting Overuse',
        'P3': 'P3: Suction Total',
        'P4': 'P4: Dissecting Cooldown',
        'P5': 'P5: Cusa Precedence',
    }
    for pid, plabel in policy_labels.items():
        add_row(f"  {plabel}", [len(r['policy_violations'][pid]) for r in results.values()])

    lines.append("-" * metric_w + ("+" + "-" * (col_w + 1)) * len(video_names))
    add_row("Total Violation Events", [r['total_violations'] for r in results.values()])
    add_row("Frames with Violation(s)", [r['frames_with_violations'] for r in results.values()])
    add_row("Compliant Frames", [r['compliant_frames'] for r in results.values()])
    add_row("Compliance Rate", [f"{r['compliance_rate']:.1f}%" for r in results.values()])

    first_viol_vals = []
    for r in results.values():
        fv = r['first_violation_frame']
        first_viol_vals.append(str(fv) if fv else "—")
    add_row("First Violation at Frame", first_viol_vals)

    lines.append("")
    lines.append("")

    # ─── Table 2: Violation Frame Details ───
    lines.append("-" * 80)
    lines.append("TABLE 2: Violated Frame Numbers (per policy, per video)")
    lines.append("-" * 80)
    lines.append("")
    for vname, r in results.items():
        lines.append(f"  {vname}:")
        for pid, plabel in policy_labels.items():
            frames = r['policy_violations'][pid]
            if frames:
                lines.append(f"    {plabel}: frames {frames}")
            else:
                lines.append(f"    {plabel}: — (no violations)")
        lines.append("")

    # ─── LaTeX table for the paper ───
    lines.append("")
    lines.append("=" * 80)
    lines.append("  LaTeX TABLE (ready to paste into your paper)")
    lines.append("=" * 80)
    lines.append("")

    n = len(video_names)
    col_spec = "l" + "c" * n
    lines.append(f"\\begin{{table}}[htbp]")
    lines.append(f"\\centering")
    lines.append(f"\\caption{{Runtime Verification Results Across Surgical Videos}}")
    lines.append(f"\\label{{tab:runtime_results}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"\\hline")

    # Header row
    latex_header = "\\textbf{Metric}"
    for vname in video_names:
        latex_header += f" & \\textbf{{{vname}}}"
    latex_header += " \\\\"
    lines.append(latex_header)
    lines.append("\\hline")
    lines.append("\\hline")

    # Total frames
    latex_row = "Total Frames"
    for r in results.values():
        latex_row += f" & {r['num_frames']}"
    latex_row += " \\\\"
    lines.append(latex_row)
    lines.append("\\hline")

    # Tool presence section
    lines.append("\\multicolumn{" + str(n + 1) + "}{l}{\\textit{Tool Presence (frames / \\%)}} \\\\")
    lines.append("\\hline")
    for k, tool_name in enumerate(TRACKED_NAMES):
        latex_row = f"\\quad {tool_name}"
        for r in results.values():
            count = r['tool_frame_counts'][k]
            pct = (count / r['num_frames']) * 100
            latex_row += f" & {count} ({pct:.1f}\\%)"
        latex_row += " \\\\"
        lines.append(latex_row)
    lines.append("\\hline")

    # Max consecutive ON
    lines.append("\\multicolumn{" + str(n + 1) + "}{l}{\\textit{Max Consecutive ON (frames)}} \\\\")
    lines.append("\\hline")
    for k, tool_name in enumerate(TRACKED_NAMES):
        latex_row = f"\\quad {tool_name}"
        for r in results.values():
            latex_row += f" & {r['max_consec_on'][k]}"
        latex_row += " \\\\"
        lines.append(latex_row)
    lines.append("\\hline")

    # Policy violations
    lines.append("\\multicolumn{" + str(n + 1) + "}{l}{\\textit{Policy Violations (frame count)}} \\\\")
    lines.append("\\hline")
    for pid, plabel in policy_labels.items():
        latex_row = f"\\quad {plabel}"
        for r in results.values():
            count = len(r['policy_violations'][pid])
            latex_row += f" & {count}"
        latex_row += " \\\\"
        lines.append(latex_row)
    lines.append("\\hline")

    # Summary metrics
    latex_row = "Total Violation Events"
    for r in results.values():
        latex_row += f" & {r['total_violations']}"
    latex_row += " \\\\"
    lines.append(latex_row)

    latex_row = "Frames with Violation(s)"
    for r in results.values():
        latex_row += f" & {r['frames_with_violations']}"
    latex_row += " \\\\"
    lines.append(latex_row)

    latex_row = "Compliance Rate"
    for r in results.values():
        latex_row += f" & {r['compliance_rate']:.1f}\\%"
    latex_row += " \\\\"
    lines.append(latex_row)

    latex_row = "First Violation Frame"
    for r in results.values():
        fv = r['first_violation_frame']
        latex_row += f" & {fv if fv else '---'}"
    latex_row += " \\\\"
    lines.append(latex_row)

    lines.append("\\hline")
    lines.append(f"\\end{{tabular}}")
    lines.append(f"\\end{{table}}")

    # Write to file
    output_text = "\n".join(lines) + "\n"
    with open(output_path, 'w') as f:
        f.write(output_text)

    # Also print to console
    print(output_text)
    print(f"\n✅ Results written to: {output_path}")


# ─── MAIN ───
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    videos = {
        'Video 1': {'file': 'preds_video1.pckl', 'frames': 72},
        'Video 2': {'file': 'preds_video2.pckl', 'frames': 104},
        'Video 3': {'file': 'preds_video3.pckl', 'frames': 131},
    }

    results = {}
    for vname, vinfo in videos.items():
        fpath = os.path.join(SCRIPT_DIR, vinfo['file'])
        print(f"Processing {vname} ({vinfo['file']}, {vinfo['frames']} frames)...")
        results[vname] = process_video(fpath, vinfo['frames'])
        print(f"  ✓ Done. {results[vname]['total_violations']} total violation events.\n")

    output_path = os.path.join(SCRIPT_DIR, "verification_results.txt")
    generate_table(results, output_path)
