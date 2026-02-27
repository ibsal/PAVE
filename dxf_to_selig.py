#!/usr/bin/env python3
"""
DXF to Selig .dat Converter for DevFoam
========================================

Converts wing profile DXF files (with B-splines, arcs, and lines) into
Selig-format .dat files suitable for DevFoam CNC foam cutting.

Handles:
  - B-spline airfoil contours (upper + lower surfaces)
  - Circular spar cutouts (ARC entities)
  - Wire entry/exit slots (LINE entities)

Setup:
------
  pip install numpy scipy matplotlib

Usage:
------
  python dxf_to_selig.py Wing_Detail_Design_-_R_Foam.dxf
  python dxf_to_selig.py Wing_Detail_Design_-_R_Foam.dxf -o my_output
  python dxf_to_selig.py Wing_Detail_Design_-_R_Foam.dxf --points 201 --no-plot

Options:
  -o, --output      Output filename base (default: derived from input)
  -n, --points      Points per surface for OML (default: 101, cosine-spaced)
  --lower-points    Points for lower surface with cutouts (default: 401)
  --no-cutouts      Export outer mold line only, ignore arcs/lines
  --no-plot         Skip verification plot
  --no-oml          Skip separate OML-only .dat export
"""

import argparse
import os
import sys
import numpy as np
from scipy.interpolate import BSpline, interp1d


# ─────────────────────────────────────────────────────────
#  DXF PARSING
# ─────────────────────────────────────────────────────────

def parse_dxf_pairs(filepath):
    """Read a DXF file and return list of (group_code, value) pairs."""
    with open(filepath, 'r', errors='replace') as f:
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    lines = content.split('\n')
    pairs = []
    i = 0
    while i + 1 < len(lines):
        try:
            code = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        pairs.append((code, lines[i + 1].strip()))
        i += 2
    return pairs


def extract_entities(pairs):
    """Extract SPLINE, ARC, and LINE entities from DXF pairs."""
    splines, arcs, dxf_lines = [], [], []
    i = 0
    while i < len(pairs):
        code, value = pairs[i]

        if code == 0 and value == 'SPLINE':
            degree, knots, cx, cy = 3, [], [], []
            j = i + 1
            while j < len(pairs) and pairs[j][0] != 0:
                c, v = pairs[j]
                if   c == 71: degree = int(v)
                elif c == 40: knots.append(float(v))
                elif c == 10: cx.append(float(v))
                elif c == 20: cy.append(float(v))
                j += 1
            if cx and knots:
                splines.append({
                    'degree': degree,
                    'knots': np.array(knots),
                    'ctrl_x': np.array(cx),
                    'ctrl_y': np.array(cy),
                })
            i = j

        elif code == 0 and value == 'ARC':
            center_x = center_y = radius = start_ang = end_ang = 0
            j = i + 1
            while j < len(pairs) and pairs[j][0] != 0:
                c, v = pairs[j]
                if   c == 10: center_x = float(v)
                elif c == 20: center_y = float(v)
                elif c == 40: radius = float(v)
                elif c == 50: start_ang = float(v)
                elif c == 51: end_ang = float(v)
                j += 1
            arcs.append({'cx': center_x, 'cy': center_y, 'r': radius,
                         'start': start_ang, 'end': end_ang})
            i = j

        elif code == 0 and value == 'LINE':
            x1 = y1 = x2 = y2 = 0
            j = i + 1
            while j < len(pairs) and pairs[j][0] != 0:
                c, v = pairs[j]
                if   c == 10: x1 = float(v)
                elif c == 20: y1 = float(v)
                elif c == 11: x2 = float(v)
                elif c == 21: y2 = float(v)
                j += 1
            dxf_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            i = j

        else:
            i += 1

    return splines, arcs, dxf_lines


# ─────────────────────────────────────────────────────────
#  B-SPLINE EVALUATION
# ─────────────────────────────────────────────────────────

def eval_bspline(s, n=400):
    """Evaluate a B-spline at n parameter values."""
    k, d = s['knots'], s['degree']
    t = np.linspace(k[d], k[-(d + 1)], n)
    return BSpline(k, s['ctrl_x'], d)(t), BSpline(k, s['ctrl_y'], d)(t)


# ─────────────────────────────────────────────────────────
#  SPLINE CHAINING → UPPER / LOWER SURFACES
# ─────────────────────────────────────────────────────────

def chain_splines(splines, verbose=True):
    """Chain B-splines into upper (TE→LE) and lower (LE→TE) surface arrays."""
    evaluated = [eval_bspline(s, 400) for s in splines]

    if verbose:
        for i, (x, y) in enumerate(evaluated):
            print(f"  Spline {i}: x=[{x[0]:.3f} → {x[-1]:.3f}], "
                  f"y=[{y[0]:.3f} → {y[-1]:.3f}]")

    te_x = max(e[0].max() for e in evaluated)
    le_x = min(e[0].min() for e in evaluated)
    if verbose:
        print(f"  TE x={te_x:.4f}, LE x={le_x:.4f}, chord={te_x - le_x:.4f}")

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def chain_from(start_idx, start_rev, used, stop_le=False):
        pts_x, pts_y = [], []
        idx, rev = start_idx, start_rev
        while idx is not None:
            used.add(idx)
            x, y = evaluated[idx]
            if rev:
                x, y = x[::-1], y[::-1]
            pts_x.extend(x)
            pts_y.extend(y)
            if stop_le and abs(x[-1] - le_x) < 0.05:
                break
            end = (x[-1], y[-1])
            best = (None, False, 0.15)
            for k in range(len(splines)):
                if k in used:
                    continue
                sx, sy = evaluated[k][0][0], evaluated[k][1][0]
                ex, ey = evaluated[k][0][-1], evaluated[k][1][-1]
                ds = dist(end, (sx, sy))
                de = dist(end, (ex, ey))
                if ds < best[2]:
                    best = (k, False, ds)
                if de < best[2]:
                    best = (k, True, de)
            idx, rev = best[0], best[1]
        return np.array(pts_x), np.array(pts_y)

    # Find spline endpoints at trailing edge
    te_cands = []
    for i, (x, y) in enumerate(evaluated):
        if abs(x[0] - te_x) < 0.05:
            te_cands.append((i, False, x, y))
        if abs(x[-1] - te_x) < 0.05:
            te_cands.append((i, True, x[::-1], y[::-1]))

    # Classify upper vs lower by mid-section y
    scored = [(idx, rev, np.mean(y[len(y) // 4:len(y) // 2]))
              for idx, rev, x, y in te_cands]
    scored.sort(key=lambda s: -s[2])

    if len(scored) < 2:
        raise ValueError(
            f"Expected 2 spline branches at TE, found {len(scored)}. "
            "Check DXF geometry.")

    upper_s = (scored[0][0], scored[0][1])
    lower_s = (scored[1][0], scored[1][1])

    if verbose:
        print(f"  Upper starts: Spline {upper_s[0]}, "
              f"Lower starts: Spline {lower_s[0]}")

    # Chain upper surface (TE → LE)
    used = {lower_s[0]}
    upper_x, upper_y = chain_from(upper_s[0], upper_s[1], used, stop_le=True)

    # Chain lower surface (LE → TE) from remaining splines
    upper_used = set(used)
    upper_used.discard(lower_s[0])
    remaining = [i for i in range(len(splines)) if i not in upper_used]

    le_cands = []
    for i in remaining:
        x, y = evaluated[i]
        le_cands.append((i, False, abs(x[0] - le_x)))
        le_cands.append((i, True, abs(x[-1] - le_x)))
    le_cands.sort(key=lambda c: c[2])

    lower_used = set(upper_used)
    lower_x, lower_y = chain_from(le_cands[0][0], le_cands[0][1], lower_used)

    # Ensure correct orientation
    if upper_x[0] < upper_x[-1]:
        upper_x, upper_y = upper_x[::-1], upper_y[::-1]
    if lower_x[0] > lower_x[-1]:
        lower_x, lower_y = lower_x[::-1], lower_y[::-1]

    if verbose:
        print(f"  Upper: {len(upper_x)} pts, "
              f"x=[{upper_x[0]:.3f} → {upper_x[-1]:.3f}]")
        print(f"  Lower: {len(lower_x)} pts, "
              f"x=[{lower_x[0]:.3f} → {lower_x[-1]:.3f}]")

    return upper_x, upper_y, lower_x, lower_y


# ─────────────────────────────────────────────────────────
#  ARC EVALUATION
# ─────────────────────────────────────────────────────────

def eval_arc_ccw(arc, n=100):
    """Evaluate arc counterclockwise from start to end angle."""
    a0, a1 = arc['start'], arc['end']
    if a1 <= a0:
        a1 += 360.0
    angles = np.linspace(a0, a1, n)
    rad = np.radians(angles)
    x = arc['cx'] + arc['r'] * np.cos(rad)
    y = arc['cy'] + arc['r'] * np.sin(rad)
    return x, y


# ─────────────────────────────────────────────────────────
#  BUILD AND INSERT CUTOUTS
# ─────────────────────────────────────────────────────────

def build_cutouts(arcs, dxf_lines, verbose=True):
    """Match each arc with its two wire-entry slot lines."""
    cutouts = []
    for arc in arcs:
        sx = arc['cx'] + arc['r'] * np.cos(np.radians(arc['start']))
        sy = arc['cy'] + arc['r'] * np.sin(np.radians(arc['start']))
        ex = arc['cx'] + arc['r'] * np.cos(np.radians(arc['end']))
        ey = arc['cy'] + arc['r'] * np.sin(np.radians(arc['end']))

        matched = []
        for ln in dxf_lines:
            for lx, ly in [(ln['x1'], ln['y1']), (ln['x2'], ln['y2'])]:
                if np.hypot(lx - sx, ly - sy) < 0.01 or \
                   np.hypot(lx - ex, ly - ey) < 0.01:
                    matched.append(ln)
                    break

        if verbose:
            print(f"  Arc center=({arc['cx']:.3f}, {arc['cy']:.3f}), "
                  f"r={arc['r']:.4f}, matched {len(matched)} slot lines")

        cutouts.append({
            'arc': arc,
            'arc_start_pt': (sx, sy),
            'arc_end_pt': (ex, ey),
            'slot_lines': matched,
            'x_pos': arc['cx'],
        })

    cutouts.sort(key=lambda c: c['x_pos'])
    return cutouts


def insert_cutouts_into_lower(lower_x, lower_y, cutouts, verbose=True):
    """
    Splice cutout wire paths into the lower surface.
    For each cutout: slot down → arc (nearly full circle) → slot back up.
    """
    result_x, result_y = list(lower_x), list(lower_y)

    for cutout in cutouts:
        arc = cutout['arc']
        slots = cutout['slot_lines']
        if len(slots) < 2:
            if verbose:
                print(f"  Warning: cutout at x={cutout['x_pos']:.3f} "
                      "has <2 slot lines, skipping")
            continue

        slots.sort(key=lambda ln: min(ln['x1'], ln['x2']))
        left_slot, right_slot = slots[0], slots[1]

        def slot_endpoints(slot):
            p1 = (slot['x1'], slot['y1'])
            p2 = (slot['x2'], slot['y2'])
            return (p1, p2) if p1[1] < p2[1] else (p2, p1)

        l_surf, l_arc = slot_endpoints(left_slot)
        r_surf, r_arc = slot_endpoints(right_slot)

        arc_start_pt = cutout['arc_start_pt']
        arc_end_pt = cutout['arc_end_pt']

        d_l_start = np.hypot(l_arc[0] - arc_start_pt[0],
                             l_arc[1] - arc_start_pt[1])
        d_l_end = np.hypot(l_arc[0] - arc_end_pt[0],
                           l_arc[1] - arc_end_pt[1])

        arc_x, arc_y = eval_arc_ccw(arc, 100)
        if d_l_start >= d_l_end:
            arc_x, arc_y = arc_x[::-1], arc_y[::-1]

        detour_x = [l_surf[0], l_arc[0]] + list(arc_x) + [r_arc[0], r_surf[0]]
        detour_y = [l_surf[1], l_arc[1]] + list(arc_y) + [r_arc[1], r_surf[1]]

        # Find insertion indices
        rx = np.array(result_x)
        left_x_min = min(l_surf[0], r_surf[0])
        right_x_max = max(l_surf[0], r_surf[0])

        insert_start = insert_end = None
        for k in range(len(rx) - 1):
            if rx[k] <= left_x_min and rx[k + 1] >= left_x_min:
                insert_start = k + 1
            if rx[k] <= right_x_max and rx[k + 1] >= right_x_max:
                insert_end = k + 1
                break

        if insert_start is not None and insert_end is not None:
            result_x = result_x[:insert_start] + detour_x + result_x[insert_end:]
            result_y = result_y[:insert_start] + detour_y + result_y[insert_end:]
            if verbose:
                print(f"  Inserted cutout at x≈{cutout['x_pos']:.3f} "
                      f"({len(detour_x)} pts)")
        elif verbose:
            print(f"  Warning: could not find insertion point for "
                  f"cutout at x={cutout['x_pos']:.3f}")

    return np.array(result_x), np.array(result_y)


# ─────────────────────────────────────────────────────────
#  NORMALIZE + RESAMPLE + WRITE
# ─────────────────────────────────────────────────────────

def normalize(ux, uy, lx, ly):
    """Normalize to unit chord with LE at (0, 0)."""
    all_x = np.concatenate([ux, lx])
    all_y = np.concatenate([uy, ly])
    le_x, te_x = all_x.min(), all_x.max()
    chord = te_x - le_x
    le_y = all_y[np.argmin(all_x)]
    return ((ux - le_x) / chord, (uy - le_y) / chord,
            (lx - le_x) / chord, (ly - le_y) / chord,
            chord, le_x, le_y)


def resample_cosine(x, y, n=101):
    """Resample with cosine spacing for good LE/TE resolution."""
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    s /= s[-1]
    mask = np.concatenate([[True], np.diff(s) > 1e-12])
    s, x, y = s[mask], x[mask], y[mask]
    beta = np.linspace(0, np.pi, n)
    s_new = np.clip(0.5 * (1 - np.cos(beta)), s[0], s[-1])
    return interp1d(s, x, kind='cubic')(s_new), \
           interp1d(s, y, kind='cubic')(s_new)


def resample_linear(x, y, n=401):
    """Linear resample preserving sharp corners (for cutout paths)."""
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    s /= s[-1]
    mask = np.concatenate([[True], np.diff(s) > 1e-12])
    s, x, y = s[mask], x[mask], y[mask]
    s_new = np.linspace(0, 1, n)
    return interp1d(s, x, kind='linear')(s_new), \
           interp1d(s, y, kind='linear')(s_new)


def write_selig(path, name, upper_x, upper_y, lower_x, lower_y,
                n_upper=101, n_lower=None, has_cutouts=False):
    """
    Write Selig-format .dat file.
    Upper surface: cosine-spaced (TE → LE)
    Lower surface: cosine-spaced (OML) or linear (with cutouts)
    """
    ux, uy = resample_cosine(upper_x, upper_y, n_upper)
    if ux[0] < ux[-1]:
        ux, uy = ux[::-1], uy[::-1]
    ux[0], ux[-1] = 1.0, 0.0

    if has_cutouts:
        nl = n_lower or max(401, len(lower_x) // 2)
        lx, ly = resample_linear(lower_x, lower_y, nl)
    else:
        nl = n_lower or n_upper
        lx, ly = resample_cosine(lower_x, lower_y, nl)

    if lx[0] > lx[-1]:
        lx, ly = lx[::-1], ly[::-1]
    lx[0], lx[-1] = 0.0, 1.0

    with open(path, 'w') as f:
        f.write(f"{name}\n")
        for xi, yi in zip(ux, uy):
            f.write(f"  {xi:11.7f}  {yi:11.7f}\n")
        for xi, yi in zip(lx[1:], ly[1:]):
            f.write(f"  {xi:11.7f}  {yi:11.7f}\n")

    total = len(ux) + len(lx) - 1
    return ux, uy, lx, ly, total


# ─────────────────────────────────────────────────────────
#  VERIFICATION PLOT
# ─────────────────────────────────────────────────────────

def make_plot(dat_cut_path, dat_oml_path, plot_path):
    """Generate verification plot comparing cutout and OML profiles."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot")
        return

    def read_dat(path):
        with open(path) as f:
            lines = f.readlines()
        pts = []
        for l in lines[1:]:
            p = l.split()
            if len(p) == 2:
                pts.append((float(p[0]), float(p[1])))
        return np.array(pts)

    pts_cut = read_dat(dat_cut_path)
    has_oml = dat_oml_path and os.path.exists(dat_oml_path)
    pts_oml = read_dat(dat_oml_path) if has_oml else None

    nrows = 1
    # Detect cutouts: lower surface y excursions well above OML
    lower_cut = pts_cut[pts_cut[:, 0] < 0.5]  # rough check
    if pts_oml is not None and np.max(pts_cut[:, 1]) > np.max(pts_oml[:, 1]) * 0.5:
        nrows = 2

    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 * nrows))
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    if has_oml:
        ax.plot(pts_oml[:, 0], pts_oml[:, 1], 'b-', lw=1, alpha=0.5,
                label='OML')
        ax.fill(pts_oml[:, 0], pts_oml[:, 1], alpha=0.08, color='blue')
    ax.plot(pts_cut[:, 0], pts_cut[:, 1], 'r-', lw=1.2,
            label='Cutting path')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.set_title('Airfoil Profile — DevFoam Cutting Path', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if nrows > 1:
        ax2 = axes[1]
        ax2.plot(pts_cut[:, 0], pts_cut[:, 1], 'r-', lw=1.5,
                 label='Wire path')
        if has_oml:
            ax2.plot(pts_oml[:, 0], pts_oml[:, 1], 'b--', lw=1, alpha=0.5,
                     label='OML')
        ax2.set_xlabel('x/c')
        ax2.set_ylabel('y/c')
        ax2.set_title('Lower Surface Detail — Cutouts & Slots')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # Auto-zoom to lower surface region
        ly_all = pts_cut[:, 1]
        ax2.set_ylim(ly_all.min() - 0.01, max(ly_all.max(), 0.1) + 0.01)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Verification plot: {plot_path}")


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert wing DXF to Selig .dat for DevFoam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('dxf', help='Input DXF file path')
    parser.add_argument('-o', '--output', default=None,
                        help='Output filename base (default: from input name)')
    parser.add_argument('-n', '--points', type=int, default=101,
                        help='Points per surface for upper/OML (default: 101)')
    parser.add_argument('--lower-points', type=int, default=401,
                        help='Points for lower surface with cutouts (default: 401)')
    parser.add_argument('--no-cutouts', action='store_true',
                        help='Export outer mold line only')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip verification plot')
    parser.add_argument('--no-oml', action='store_true',
                        help='Skip separate OML-only .dat')
    args = parser.parse_args()

    if not os.path.isfile(args.dxf):
        print(f"Error: file not found: {args.dxf}")
        sys.exit(1)

    base = args.output or os.path.splitext(os.path.basename(args.dxf))[0]
    base = base.replace(' ', '_').replace('-', '_')
    # Output to current working directory by default
    outdir = '.'

    print("=" * 60)
    print("DXF → Selig .dat Converter for DevFoam")
    print("=" * 60)

    # ── Parse ──
    print(f"\n[1] Parsing: {args.dxf}")
    pairs = parse_dxf_pairs(args.dxf)
    splines, arcs, dxf_lines = extract_entities(pairs)
    print(f"    {len(splines)} splines, {len(arcs)} arcs, {len(dxf_lines)} lines")

    if not splines:
        print("Error: no SPLINE entities found in DXF.")
        sys.exit(1)

    # ── Chain splines ──
    print("\n[2] Chaining splines → airfoil contour...")
    upper_x, upper_y, lower_x, lower_y = chain_splines(splines)

    # ── Normalize (reference from outer contour) ──
    print("\n[3] Normalizing to unit chord...")
    ux, uy, lx, ly, chord, le_x_abs, le_y_abs = normalize(
        upper_x, upper_y, lower_x, lower_y)
    print(f"    Chord: {chord:.4f} DXF units")
    print(f"    Thickness: ~{(np.max(uy) - np.min(ly)):.4f}c")

    # ── Cutouts ──
    dat_oml_path = None
    if not args.no_cutouts and arcs and dxf_lines:
        print(f"\n[4] Building {len(arcs)} cutout(s)...")
        cutouts = build_cutouts(arcs, dxf_lines)

        print("\n[5] Inserting cutouts into lower surface...")
        lower_cut_x, lower_cut_y = insert_cutouts_into_lower(
            lower_x, lower_y, cutouts)
        lx_cut = (lower_cut_x - le_x_abs) / chord
        ly_cut = (lower_cut_y - le_y_abs) / chord

        # Write cutout version
        dat_cut = os.path.join(outdir, f"{base}_cutouts.dat")
        ux_o, uy_o, lx_o, ly_o, n_tot = write_selig(
            dat_cut, f"{base}_cutouts", ux, uy, lx_cut, ly_cut,
            n_upper=args.points, n_lower=args.lower_points,
            has_cutouts=True)
        print(f"\n[6] Written: {dat_cut} ({n_tot} pts)")

        # OML version
        if not args.no_oml:
            dat_oml_path = os.path.join(outdir, f"{base}_OML.dat")
            n_oml = write_selig(
                dat_oml_path, f"{base}_OML", ux, uy, lx, ly,
                n_upper=args.points, has_cutouts=False)[-1]
            print(f"    Written: {dat_oml_path} ({n_oml} pts)")

        dat_primary = dat_cut
    else:
        if not args.no_cutouts:
            print("\n    No arcs/lines found — exporting OML only")
        dat_primary = os.path.join(outdir, f"{base}.dat")
        ux_o, uy_o, lx_o, ly_o, n_tot = write_selig(
            dat_primary, base, ux, uy, lx, ly,
            n_upper=args.points, has_cutouts=False)
        print(f"\n[4] Written: {dat_primary} ({n_tot} pts)")

    # ── Quality ──
    step = 7 if not args.no_cutouts and arcs else 5
    print(f"\n[{step}] Quality:")
    te_gap = abs(uy_o[0] - ly_o[-1])
    print(f"    TE gap:  {te_gap:.6f}c "
          f"{'(blunt)' if te_gap > 0.001 else '(sharp)'}")
    print(f"    TE upper: (1.0, {uy_o[0]:.7f})")
    print(f"    LE:       (0.0, {uy_o[-1]:.7f})")
    print(f"    TE lower: (1.0, {ly_o[-1]:.7f})")

    # ── Plot ──
    if not args.no_plot:
        plot_path = os.path.join(outdir, f"{base}_verify.png")
        make_plot(dat_primary, dat_oml_path, plot_path)

    print(f"\n{'=' * 60}")
    print(f"Done! Primary output: {dat_primary}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
