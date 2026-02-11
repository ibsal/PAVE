import argparse
import csv
import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _to_float(value):
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except Exception:
        return math.nan


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _power_metric(row):
    raw_power = _to_float(row.get("raw_power_W"))
    if _finite(raw_power) and raw_power > 0.0:
        return raw_power
    objective = _to_float(row.get("objective"))
    if _finite(objective) and objective > 0.0:
        return objective
    return math.nan


def _rect_from_qc(span, chord, xqc):
    if not (_finite(span) and _finite(chord) and _finite(xqc)):
        return None
    if span <= 0.0 or chord <= 0.0:
        return None
    x_le = xqc - 0.25 * chord
    y_min = -0.5 * span
    return (x_le, y_min, chord, span)


def _load_rows(csv_path, run_id=None):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    available_run_ids = []
    seen = set()
    for row in rows:
        rid = str(row.get("run_id", "")).strip()
        if rid and rid not in seen:
            seen.add(rid)
            available_run_ids.append(rid)

    if run_id is None:
        if available_run_ids:
            run_id = available_run_ids[0]
        else:
            run_id = str(rows[0].get("run_id", "")).strip()

    filtered = [r for r in rows if str(r.get("run_id", "")).strip() == run_id]
    if not filtered:
        raise RuntimeError(
            f"run_id '{run_id}' not found in {csv_path}. Available run_ids: {available_run_ids}"
        )

    filtered.sort(
        key=lambda r: (
            _to_float(r.get("eval_index"))
            if _finite(_to_float(r.get("eval_index")))
            else float("inf")
        )
    )
    return filtered, run_id


def _compute_limits(rows):
    x_min = float("inf")
    x_max = float("-inf")
    y_max_abs = 0.0
    for row in rows:
        wing = _rect_from_qc(
            _to_float(row.get("wingSpan")),
            _to_float(row.get("wingChord")),
            _to_float(row.get("xwqc")),
        )
        tail = _rect_from_qc(
            _to_float(row.get("hSpan")),
            _to_float(row.get("hChord")),
            _to_float(row.get("xhtqc")),
        )
        for rect in (wing, tail):
            if rect is None:
                continue
            x0, y0, w, h = rect
            x1 = x0 + w
            y1 = y0 + h
            x_min = min(x_min, x0)
            x_max = max(x_max, x1)
            y_max_abs = max(y_max_abs, abs(y0), abs(y1))

    if not math.isfinite(x_min) or not math.isfinite(x_max):
        x_min, x_max = -1.0, 2.0
    if y_max_abs <= 0.0:
        y_max_abs = 3.0

    x_pad = 0.15 * max(1e-9, (x_max - x_min))
    y_pad = 0.12 * max(1e-9, y_max_abs)
    return (x_min - x_pad, x_max + x_pad, -(y_max_abs + y_pad), y_max_abs + y_pad)


def _status_color(status):
    st = str(status).strip().lower()
    if st == "valid":
        return "#2ca02c"
    if st == "reject":
        return "#d62728"
    return "#1f77b4"


def build_animation(rows, run_id, interval_ms):
    eval_idx = []
    metric = []
    status = []
    reject_reason = []

    for row in rows:
        ev = _to_float(row.get("eval_index"))
        eval_idx.append(ev if _finite(ev) else float(len(eval_idx) + 1))
        metric.append(_power_metric(row))
        status.append(str(row.get("status", "")).strip().lower())
        reject_reason.append(str(row.get("reject_reason", "")).strip())

    best_valid = []
    running = math.nan
    for m, st in zip(metric, status):
        if st == "valid" and _finite(m):
            if not _finite(running) or m < running:
                running = m
        best_valid.append(running)

    finite_metric = [m for m in metric if _finite(m)]
    if finite_metric:
        y_min = min(finite_metric)
        y_max = max(finite_metric)
    else:
        y_min, y_max = 0.0, 1.0
    y_span = max(1e-9, y_max - y_min)
    y_min -= 0.10 * y_span
    y_max += 0.10 * y_span

    x0, x1, y0, y1 = _compute_limits(rows)
    fig, (ax_plan, ax_obj) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )
    fig.suptitle(f"Optimizer Wing/Tail Search (run_id={run_id})")

    ax_plan.set_xlim(x0, x1)
    ax_plan.set_ylim(y0, y1)
    ax_plan.set_aspect("equal", adjustable="box")
    ax_plan.set_xlabel("x (m)")
    ax_plan.set_ylabel("y (m)")
    ax_plan.grid(True, linestyle="--", alpha=0.35)
    ax_plan.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax_plan.set_title("2D Planform")

    wing_patch = Rectangle((0.0, 0.0), 0.0, 0.0, facecolor="#4c72b0", alpha=0.35, edgecolor="#1f355e", lw=1.5)
    tail_patch = Rectangle((0.0, 0.0), 0.0, 0.0, facecolor="#dd8452", alpha=0.35, edgecolor="#7f4a24", lw=1.5)
    ax_plan.add_patch(wing_patch)
    ax_plan.add_patch(tail_patch)

    wing_qc_pt, = ax_plan.plot([], [], marker="o", markersize=5, color="#1f355e", linestyle="None")
    tail_qc_pt, = ax_plan.plot([], [], marker="o", markersize=5, color="#7f4a24", linestyle="None")
    plan_text = ax_plan.text(
        0.02,
        0.98,
        "",
        transform=ax_plan.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#888888"},
    )

    ax_obj.set_xlim(min(eval_idx), max(eval_idx) if len(eval_idx) > 1 else min(eval_idx) + 1.0)
    ax_obj.set_ylim(y_min, y_max)
    ax_obj.set_xlabel("Evaluation Index")
    ax_obj.set_ylabel("Power Metric (W)")
    ax_obj.grid(True, linestyle="--", alpha=0.35)
    ax_obj.set_title("Objective History")

    hist_line, = ax_obj.plot([], [], color="#4c72b0", lw=1.6, alpha=0.9, label="metric")
    best_line, = ax_obj.plot([], [], color="#2ca02c", lw=1.8, alpha=0.9, label="best valid so far")
    current_pt, = ax_obj.plot([], [], marker="o", markersize=6, linestyle="None", color="#000000")
    ax_obj.legend(loc="upper right")

    def _init():
        hist_line.set_data([], [])
        best_line.set_data([], [])
        current_pt.set_data([], [])
        plan_text.set_text("")
        return (
            wing_patch,
            tail_patch,
            wing_qc_pt,
            tail_qc_pt,
            plan_text,
            hist_line,
            best_line,
            current_pt,
        )

    def _update(i):
        row = rows[i]
        st = status[i]
        color = _status_color(st)

        wing = _rect_from_qc(
            _to_float(row.get("wingSpan")),
            _to_float(row.get("wingChord")),
            _to_float(row.get("xwqc")),
        )
        tail = _rect_from_qc(
            _to_float(row.get("hSpan")),
            _to_float(row.get("hChord")),
            _to_float(row.get("xhtqc")),
        )

        if wing is None:
            wing_patch.set_xy((0.0, 0.0))
            wing_patch.set_width(0.0)
            wing_patch.set_height(0.0)
            wing_qc_pt.set_data([], [])
        else:
            wing_patch.set_xy((wing[0], wing[1]))
            wing_patch.set_width(wing[2])
            wing_patch.set_height(wing[3])
            wing_qc_pt.set_data([_to_float(row.get("xwqc"))], [0.0])

        if tail is None:
            tail_patch.set_xy((0.0, 0.0))
            tail_patch.set_width(0.0)
            tail_patch.set_height(0.0)
            tail_qc_pt.set_data([], [])
        else:
            tail_patch.set_xy((tail[0], tail[1]))
            tail_patch.set_width(tail[2])
            tail_patch.set_height(tail[3])
            tail_qc_pt.set_data([_to_float(row.get("xhtqc"))], [0.0])

        wing_patch.set_facecolor(color)
        tail_patch.set_facecolor(color)

        current_metric = metric[i]
        best_now = best_valid[i]
        reason = reject_reason[i] if reject_reason[i] else "-"
        txt = (
            f"eval={int(eval_idx[i]) if _finite(eval_idx[i]) else i + 1}\n"
            f"status={st}\n"
            f"reason={reason}\n"
            f"metric={current_metric:.2f} W\n" if _finite(current_metric) else
            f"eval={int(eval_idx[i]) if _finite(eval_idx[i]) else i + 1}\n"
            f"status={st}\n"
            f"reason={reason}\n"
            f"metric=nan\n"
        )
        txt += f"best_valid={best_now:.2f} W" if _finite(best_now) else "best_valid=none"
        plan_text.set_text(txt)

        hist_x = []
        hist_y = []
        for j in range(i + 1):
            if _finite(metric[j]):
                hist_x.append(eval_idx[j])
                hist_y.append(metric[j])
        hist_line.set_data(hist_x, hist_y)

        best_x = []
        best_y = []
        for j in range(i + 1):
            if _finite(best_valid[j]):
                best_x.append(eval_idx[j])
                best_y.append(best_valid[j])
        best_line.set_data(best_x, best_y)

        if _finite(current_metric):
            current_pt.set_data([eval_idx[i]], [current_metric])
            current_pt.set_color(color)
        else:
            current_pt.set_data([], [])

        ax_plan.set_title(f"2D Planform | frame {i + 1}/{len(rows)}")
        return (
            wing_patch,
            tail_patch,
            wing_qc_pt,
            tail_qc_pt,
            plan_text,
            hist_line,
            best_line,
            current_pt,
        )

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(rows),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    return fig, anim


def main():
    parser = argparse.ArgumentParser(
        description="Animate optimizer_endurance wing/tail search from CSV log."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("optimizer_endurance_eval_log.csv"),
        help="Path to optimizer evaluation log CSV.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run_id to animate. Default: first run_id in the CSV.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame stride (use >1 to speed up long runs).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap number of frames after stepping (0 means no cap).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=70,
        help="Animation frame interval in milliseconds for interactive display.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (.gif or .mp4). If omitted, show interactively only.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=14,
        help="FPS when saving animation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="DPI when saving animation.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (useful with --output).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    if args.step < 1:
        raise ValueError("--step must be >= 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    rows, selected_run_id = _load_rows(args.csv, run_id=args.run_id)
    if args.step > 1:
        rows = rows[:: args.step]
    if args.max_frames > 0:
        rows = rows[: args.max_frames]
    if not rows:
        raise RuntimeError("No rows left after applying --step / --max-frames.")

    fig, anim = build_animation(rows, selected_run_id, interval_ms=args.interval_ms)

    if args.output is not None:
        suffix = args.output.suffix.lower()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=args.fps)
            anim.save(str(args.output), writer=writer, dpi=args.dpi)
        elif suffix == ".mp4":
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=1800)
            anim.save(str(args.output), writer=writer, dpi=args.dpi)
        else:
            anim.save(str(args.output), dpi=args.dpi)
        print(f"Saved animation: {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
