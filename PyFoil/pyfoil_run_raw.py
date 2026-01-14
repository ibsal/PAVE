# batch_xfoil_adaptive.py
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict


# -----------------------------
# Parsing / utilities
# -----------------------------
def _fmt_hms(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{ss:02d}"
    return f"{m:d}:{ss:02d}"


def _parse_aseq_args(raw: List[str]) -> List[Tuple[float, float, float]]:
    segs: List[Tuple[float, float, float]] = []
    for triplet in raw:
        parts = triplet.strip().split()
        if len(parts) != 3:
            raise ValueError(f'--aseq expects 3 numbers like "--aseq -4 10 0.5", got: {triplet!r}')
        segs.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return segs


def _looks_numeric_row(parts: List[str]) -> bool:
    if len(parts) < 3:
        return False
    try:
        float(parts[0]); float(parts[1]); float(parts[2])
        return True
    except Exception:
        return False


def read_xfoil_polar(path: Path) -> Tuple[List[str], List[str], List[float], List[float]]:
    """
    Returns:
      header_lines (all lines up to first numeric row, inclusive of column header if present)
      numeric_lines (raw numeric lines)
      alpha_list
      cl_list

    Works with XFOIL .pol files.
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: List[str] = []
    nums: List[str] = []
    alpha: List[float] = []
    cl: List[float] = []

    in_numeric = False
    for ln in txt:
        s = ln.rstrip("\n")
        parts = s.strip().split()
        if (not in_numeric) and _looks_numeric_row(parts):
            in_numeric = True
        if not in_numeric:
            header.append(s)
            continue

        # numeric row
        if _looks_numeric_row(parts):
            nums.append(s)
            try:
                a = float(parts[0])
                c = float(parts[1])
                alpha.append(a)
                cl.append(c)
            except Exception:
                pass
        else:
            # if there's junk mid-file, keep it with header-ish
            header.append(s)

    return header, nums, alpha, cl


def merge_polars(base_path: Path, ext_path: Optional[Path], out_path: Path) -> None:
    """
    Merge two polar files by alpha. Keeps base header.
    For duplicate alphas, keeps the line from the file that has the larger alpha range (ext tends to win).
    """
    header_b, nums_b, _, _ = read_xfoil_polar(base_path)

    # alpha->line mapping using rounded alpha to avoid float drift in text
    def _to_map(lines: List[str]) -> Dict[float, str]:
        m: Dict[float, str] = {}
        for ln in lines:
            parts = ln.strip().split()
            if not _looks_numeric_row(parts):
                continue
            a = float(parts[0])
            key = round(a, 4)
            m[key] = ln.strip()
        return m

    m = _to_map(nums_b)

    if ext_path is not None and ext_path.exists() and ext_path.stat().st_size > 200:
        _, nums_e, _, _ = read_xfoil_polar(ext_path)
        me = _to_map(nums_e)
        # let ext override duplicates (it’s usually later and/or better converged)
        m.update(me)

    # sort by alpha key
    keys = sorted(m.keys())
    out_lines: List[str] = []
    if header_b:
        out_lines.extend(header_b)
    for k in keys:
        out_lines.append(m[k])

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def should_extend_to_stall(alpha: List[float], cl: List[float], base_end: float) -> Tuple[bool, float, float]:
    """
    Heuristic: extend if we likely haven't reached stall by base_end.
    Returns (extend?, last_alpha, cl_slope_end)
    """
    if len(alpha) < 8 or len(cl) < 8:
        return False, float("nan"), float("nan")

    # keep finite pairs and sort
    pairs = [(a, c) for a, c in zip(alpha, cl) if (a == a and c == c)]
    pairs.sort(key=lambda x: x[0])
    a = [p[0] for p in pairs]
    c = [p[1] for p in pairs]

    last_a = a[-1]
    if last_a < base_end - 0.25:
        # didn't even reach planned base end — extension might help, but may also be convergence-limited.
        # We'll extend only if trend is still increasing.
        pass

    # slope near end using last 3 points (more stable than last 2)
    a1, a2, a3 = a[-3], a[-2], a[-1]
    c1, c2, c3 = c[-3], c[-2], c[-1]
    da = (a3 - a1) if (a3 - a1) != 0 else 1e-9
    slope = (c3 - c1) / da  # dCL/dalpha [per deg]

    # Determine if CLmax is at (or very near) the end
    cmax = max(c)
    imax = c.index(cmax)
    # If max occurs within last 2 points AND slope is still positive, we're probably still pre-stall
    max_near_end = (len(c) - 1 - imax) <= 2

    # If we already see a clear drop after max, stall is likely reached
    # "clear drop" threshold: drop > 0.03 absolute (tunable)
    if imax < len(c) - 3:
        drop = cmax - c[-1]
        if drop > 0.03:
            return False, last_a, slope

    # Extend if:
    # - We haven't obviously stalled AND
    # - we're still climbing near end (slope > small threshold) AND
    # - CLmax near end
    if max_near_end and slope > 0.01:
        return True, last_a, slope

    return False, last_a, slope


# -----------------------------
# XFOIL scripting / execution
# -----------------------------
def make_xfoil_script(
    airfoil_filename: str,
    polar_filename: str,
    reynolds: float,
    mach: float,
    iters: int,
    aseq_segments: List[Tuple[float, float, float]],
    ncrit: Optional[float],
) -> str:
    """
    IMPORTANT for your XFOIL v6.99:
      - DO NOT insert blank lines inside OPER except where XFOIL is prompting (PACC dump file, exiting VPAR, exiting OPER).
    """
    L: List[str] = []
    L += [f"LOAD {airfoil_filename}", ""]
    L += ["PANE"]
    L += ["OPER"]

    if ncrit is not None:
        L += ["VPAR", f"N {ncrit}", ""]  # blank exits VPAR back to OPER

    L += [f"VISC {int(reynolds)}"]
    L += [f"MACH {mach:.4f}"]
    L += [f"ITER {int(iters)}"]

    # Start polar accumulation
    L += ["PACC", polar_filename, ""]  # blank = no dump file

    for a0, a1, da in aseq_segments:
        L += [f"ASEQ {a0:.3f} {a1:.3f} {da:.3f}"]

    # Stop polar accumulation
    L += ["PACC", ""]  # blank stops

    # Exit OPER and quit
    L += [""]  # blank exits OPER
    L += ["QUIT", ""]
    return "\n".join(L) + "\n"


def run_xfoil_popen(
    xfoil_exe: Path,
    workdir: Path,
    run_in_path: Path,
    timeout_s: float,
) -> Tuple[int, str]:
    """
    Run XFOIL with stdin from run_in_path. Kill if exceeds timeout.
    Returns (return_code, combined_output_text). return_code=124 indicates timeout killed.
    """
    with run_in_path.open("r", encoding="utf-8", errors="ignore") as fin:
        p = subprocess.Popen(
            [str(xfoil_exe)],
            cwd=str(workdir),
            stdin=fin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            out, _ = p.communicate(timeout=timeout_s)
            return int(p.returncode or 0), out or ""
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                out, _ = p.communicate(timeout=1.0)
            except Exception:
                out = ""
            return 124, (out or "") + "\n[TIMEOUT]\n"


def build_extension_segments(last_alpha: float, alpha_max: float) -> List[Tuple[float, float, float]]:
    """
    Conservative extension to reduce hangs:
      - extend to 18 by 0.25
      - then to alpha_max by 0.5 (if needed)
    """
    segs: List[Tuple[float, float, float]] = []
    # avoid duplicating last point
    start = last_alpha + 0.25
    if start >= alpha_max:
        return segs

    mid = min(18.0, alpha_max)
    if start < mid:
        segs.append((start, mid, 0.25))
        start = mid + 0.25

    if start < alpha_max:
        segs.append((start, alpha_max, 0.5))

    return segs


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--xfoil", required=True, help="Path to xfoil executable (or 'xfoil' if on PATH)")
    ap.add_argument("--airfoils", required=True, help="Folder containing .dat airfoil files")
    ap.add_argument("--out", required=True, help="Output folder for polar files")

    # Your 1 ft chord, <40 kt envelope:
    ap.add_argument("--res", nargs="+", type=float, default=[150000, 200000, 300000, 400000, 500000, 600000])
    ap.add_argument("--mach", type=float, default=0.0)
    ap.add_argument("--iter", type=int, default=150, help="Lower = faster; raise (200-400) if too many failures.")
    ap.add_argument("--ncrit", type=float, default=9.0)

    # Adaptive stall targeting
    ap.add_argument("--base-end", type=float, default=16.0, help="Base sweep upper alpha (deg).")
    ap.add_argument("--alpha-max", type=float, default=22.0, help="Maximum alpha to attempt (deg).")

    # Timeouts (fast)
    ap.add_argument("--timeout-base", type=float, default=6.0, help="Seconds per base run before kill.")
    ap.add_argument("--timeout-ext", type=float, default=8.0, help="Seconds per extension run before kill.")

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")

    # Base sweep (fast + likely to reach CLmax for many low-Re airfoils)
    ap.add_argument(
        "--aseq",
        action="append",
        default=["-4 10 0.5", "10 16 0.25"],
        help='Base AoA segments: repeatable flag, e.g. --aseq "-4 10 0.5" --aseq "10 16 0.25"',
    )

    args = ap.parse_args()
    base_segments = _parse_aseq_args(args.aseq)

    # Resolve xfoil executable (allow "xfoil" from PATH)
    xfoil_exe = Path(args.xfoil)
    if not xfoil_exe.exists():
        import shutil as _sh
        resolved = _sh.which(args.xfoil)
        if resolved is None:
            raise SystemExit(
                f"XFOIL executable not found: {args.xfoil}\n"
                f"Tip: pass full path like C:\\XFOIL\\xfoil.exe or ensure it's on PATH."
            )
        xfoil_exe = Path(resolved)

    airfoil_dir = Path(args.airfoils)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    airfoils = sorted(airfoil_dir.glob("*.dat"))
    if args.limit is not None:
        airfoils = airfoils[: args.limit]
    if not airfoils:
        raise SystemExit(f"No .dat files found in {airfoil_dir}")

    workdir = out_dir / "_xfoil_work"
    workdir.mkdir(parents=True, exist_ok=True)

    total = len(airfoils) * len(args.res)
    done = 0
    t0 = time.time()

    for af in airfoils:
        for Re in args.res:
            done += 1
            tag = f"{af.stem}_Re{int(Re/1000)}k"
            final_pol = out_dir / f"{tag}.pol"
            base_pol = out_dir / f"{tag}.base.pol"
            ext_pol = out_dir / f"{tag}.ext.pol"
            log_base = out_dir / f"{tag}.base.log"
            log_ext = out_dir / f"{tag}.ext.log"

            if final_pol.exists() and not args.overwrite and final_pol.stat().st_size > 300:
                elapsed = time.time() - t0
                avg = elapsed / max(1, done)
                eta = avg * (total - done)
                print(f"[{done}/{total}] SKIP {final_pol.name} | ETA { _fmt_hms(eta) }")
                continue

            # Clean workdir
            for p in workdir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass

            # Copy airfoil to workdir as simple filename
            local_af = workdir / "airfoil.dat"
            shutil.copyfile(af, local_af)

            # ---------------- base run ----------------
            script_base = make_xfoil_script(
                airfoil_filename=local_af.name,
                polar_filename=base_pol.name,  # written into workdir
                reynolds=Re,
                mach=args.mach,
                iters=args.iter,
                aseq_segments=base_segments,
                ncrit=args.ncrit,
            )
            run_in = workdir / "run.in"
            run_in.write_text(script_base, encoding="utf-8")

            print(f"[{done}/{total}] RUN  {af.name} Re={int(Re)}", end="")

            rc_b, out_b = run_xfoil_popen(xfoil_exe, workdir, run_in, timeout_s=float(args.timeout_base))
            log_base.write_text(out_b, encoding="utf-8", errors="ignore")

            work_base = workdir / base_pol.name
            if work_base.exists():
                shutil.move(str(work_base), str(base_pol))
            else:
                # ensure old file doesn't linger
                base_pol.unlink(missing_ok=True)

            if rc_b == 124:
                # killed on timeout
                print(" -> BASE TIMEOUT (killed)", end="")
                # no polar; no extension
                final_pol.unlink(missing_ok=True)
                ext_pol.unlink(missing_ok=True)
                # ETA
                elapsed = time.time() - t0
                avg = elapsed / max(1, done)
                eta = avg * (total - done)
                print(f" | ETA { _fmt_hms(eta) }")
                continue

            if not base_pol.exists() or base_pol.stat().st_size < 300:
                print(" -> BASE FAIL (no polar)", end="")
                final_pol.unlink(missing_ok=True)
                ext_pol.unlink(missing_ok=True)
                elapsed = time.time() - t0
                avg = elapsed / max(1, done)
                eta = avg * (total - done)
                print(f" | ETA { _fmt_hms(eta) }")
                continue

            # Decide whether to extend
            _, _, a_list, cl_list = read_xfoil_polar(base_pol)
            extend, last_a, slope = should_extend_to_stall(a_list, cl_list, base_end=float(args.base_end))

            # ---------------- extension run (optional) ----------------
            did_extend = False
            if extend and (last_a == last_a) and (last_a < float(args.alpha_max) - 0.2):
                ext_segments = build_extension_segments(last_alpha=float(last_a), alpha_max=float(args.alpha_max))
                if ext_segments:
                    did_extend = True

                    # Clean workdir files except airfoil
                    for p in workdir.glob("*"):
                        if p.name.lower() != "airfoil.dat":
                            try:
                                p.unlink()
                            except Exception:
                                pass

                    script_ext = make_xfoil_script(
                        airfoil_filename=local_af.name,
                        polar_filename=ext_pol.name,  # written into workdir
                        reynolds=Re,
                        mach=args.mach,
                        iters=max(100, args.iter),  # keep iters reasonable
                        aseq_segments=ext_segments,
                        ncrit=args.ncrit,
                    )
                    run_in.write_text(script_ext, encoding="utf-8")

                    rc_e, out_e = run_xfoil_popen(xfoil_exe, workdir, run_in, timeout_s=float(args.timeout_ext))
                    log_ext.write_text(out_e, encoding="utf-8", errors="ignore")

                    work_ext = workdir / ext_pol.name
                    if work_ext.exists():
                        shutil.move(str(work_ext), str(ext_pol))
                    else:
                        ext_pol.unlink(missing_ok=True)

                    if rc_e == 124:
                        # extension hung; ignore ext
                        ext_pol.unlink(missing_ok=True)
                        did_extend = False

                    if ext_pol.exists() and ext_pol.stat().st_size < 300:
                        ext_pol.unlink(missing_ok=True)
                        did_extend = False

            # ---------------- merge to final ----------------
            merge_polars(base_pol, ext_pol if did_extend else None, final_pol)

            # If final is tiny for some reason, flag it
            if not final_pol.exists() or final_pol.stat().st_size < 300:
                print(" -> FAIL (merge tiny)", end="")
            else:
                if did_extend:
                    print(f" -> OK +EXT (lastα={last_a:.2f}, slope={slope:.3f})", end="")
                else:
                    print(" -> OK", end="")

            # ETA
            elapsed = time.time() - t0
            avg = elapsed / max(1, done)
            eta = avg * (total - done)
            print(f" | ETA { _fmt_hms(eta) }")

    print("Done.")


if __name__ == "__main__":
    main()
