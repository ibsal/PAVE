"""
airfoil_polars.py
=================

A small Python library for loading XFOIL polar files (and similar whitespace tables)
and querying aerodynamic coefficients with interpolation in:
  - angle of attack (alpha, deg)
  - Reynolds number (optional, when multiple polars provided)

Features
--------
- Robust parser for XFOIL .pol tables (alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr if present)
- Polar(alpha)->CL/CD/CM/etc via 1D interpolation
- PolarSet(alpha, Re)->CL/CD/CM/etc via 2D interpolation (alpha then Re)
- Best L/D, CLmax, lift-curve slope estimate

No heavy dependencies (only numpy + stdlib)

Typical workflow
----------------
1) Run batch XFOIL to generate polars named like:
      naca2412_Re150k.pol
      naca2412_Re200k.pol
      ...
2) Load them:

    from airfoil_polars import PolarSet
    ps = PolarSet.from_folder("./polars", airfoil="naca2412")

3) Query:
    cl = ps.cl(alpha_deg=4.2, reynolds=350_000)
    cd = ps.cd(alpha_deg=4.2, reynolds=350_000)
    ld = ps.ld(alpha_deg=4.2, reynolds=350_000)

4) Single polar:
    from airfoil_polars import Polar
    p = Polar.from_file("naca2412_Re400k.pol", reynolds=400_000)
    print(p.best_ld())

Notes
-----
- Interpolation defaults to clamping at the ends (safe for most sizing problems).
- If your .pol files are the "merged" output from the adaptive script, that's fine;
  this parser ignores headers and reads numeric lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re
import numpy as np


# ---------------------------
# Utility helpers
# ---------------------------
_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_number(s: str) -> bool:
    return bool(_NUM_RE.match(s.strip()))


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(np.asarray(arrs[0], dtype=float), dtype=bool)
    for a in arrs:
        a = np.asarray(a, dtype=float)
        m &= np.isfinite(a)
    return m


def _interp1(
    x: np.ndarray,
    y: np.ndarray,
    xq: float,
    *,
    clamp: bool = True,
) -> float:
    """
    1D linear interpolation (sorted internally).
    If clamp=True, clamps queries to endpoints (no extrapolation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot interpolate: empty data.")
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if clamp:
        if xq <= x[0]:
            return float(y[0])
        if xq >= x[-1]:
            return float(y[-1])
        return float(np.interp(xq, x, y))

    # extrapolate linearly at ends
    if xq <= x[0]:
        if x.size < 2:
            return float(y[0])
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] + slope * (xq - x[0]))
    if xq >= x[-1]:
        if x.size < 2:
            return float(y[-1])
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(y[-1] + slope * (xq - x[-1]))
    return float(np.interp(xq, x, y))


def _infer_re_from_name(path: Union[str, Path]) -> Optional[float]:
    """
    Try to infer Reynolds from a filename like 'ag10_Re400k.pol' or 'ag10_Re400000.pol'.
    Returns Re as float or None.
    """
    s = Path(path).stem
    # Re400k, Re150K, Re400000
    m = re.search(r"[ _-]Re(\d+(?:\.\d+)?)([kKmM]?)", s)
    if not m:
        return None
    val = float(m.group(1))
    suf = m.group(2).lower()
    if suf == "k":
        val *= 1e3
    elif suf == "m":
        val *= 1e6
    return float(val)


# ---------------------------
# Polar representation
# ---------------------------
@dataclass(frozen=True)
class Polar:
    """
    A single polar at a fixed Reynolds/Mach/Ncrit (if known).
    Required columns: alpha_deg, cl, cd
    Optional columns: cdp, cm, xtr_top, xtr_bot
    """
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cdp: Optional[np.ndarray] = None
    cm: Optional[np.ndarray] = None
    xtr_top: Optional[np.ndarray] = None
    xtr_bot: Optional[np.ndarray] = None

    reynolds: Optional[float] = None
    mach: Optional[float] = None
    ncrit: Optional[float] = None
    source_path: Optional[str] = None

    @staticmethod
    def from_file(
        path: Union[str, Path],
        *,
        reynolds: Optional[float] = None,
        mach: Optional[float] = None,
        ncrit: Optional[float] = None,
    ) -> "Polar":
        path = Path(path)
        txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()

        # collect numeric rows
        rows: List[List[float]] = []
        for ln in txt:
            parts = ln.strip().split()
            if len(parts) < 3:
                continue
            # must start with 3 numeric fields: alpha cl cd
            if not (_is_number(parts[0]) and _is_number(parts[1]) and _is_number(parts[2])):
                continue
            try:
                rows.append([float(p) for p in parts])
            except Exception:
                continue

        if len(rows) < 5:
            raise ValueError(f"Not enough numeric rows parsed from polar file: {path}")

        arr = np.asarray(rows, dtype=float)

        alpha = arr[:, 0]
        cl = arr[:, 1]
        cd = arr[:, 2]
        cdp = arr[:, 3] if arr.shape[1] > 3 else None
        cm = arr[:, 4] if arr.shape[1] > 4 else None
        xtr_top = arr[:, 5] if arr.shape[1] > 5 else None
        xtr_bot = arr[:, 6] if arr.shape[1] > 6 else None

        m = _finite_mask(alpha, cl, cd)
        alpha, cl, cd = alpha[m], cl[m], cd[m]
        if cdp is not None:
            cdp = cdp[m]
        if cm is not None:
            cm = cm[m]
        if xtr_top is not None:
            xtr_top = xtr_top[m]
        if xtr_bot is not None:
            xtr_bot = xtr_bot[m]

        if reynolds is None:
            reynolds = _infer_re_from_name(path)

        return Polar(
            alpha_deg=alpha,
            cl=cl,
            cd=cd,
            cdp=cdp,
            cm=cm,
            xtr_top=xtr_top,
            xtr_bot=xtr_bot,
            reynolds=reynolds,
            mach=mach,
            ncrit=ncrit,
            source_path=str(path),
        )

    # ---- coefficient queries (1D in alpha) ----
    def coeff(self, name: str, alpha_deg: float, *, clamp: bool = True) -> float:
        name = name.strip().lower()
        if name in ("a", "alpha", "alpha_deg"):
            return float(alpha_deg)
        if name == "cl":
            return _interp1(self.alpha_deg, self.cl, alpha_deg, clamp=clamp)
        if name == "cd":
            return _interp1(self.alpha_deg, self.cd, alpha_deg, clamp=clamp)
        if name == "cdp":
            if self.cdp is None:
                raise KeyError("This polar has no CDp column.")
            return _interp1(self.alpha_deg, self.cdp, alpha_deg, clamp=clamp)
        if name == "cm":
            if self.cm is None:
                raise KeyError("This polar has no Cm column.")
            return _interp1(self.alpha_deg, self.cm, alpha_deg, clamp=clamp)
        if name in ("xtr_top", "top_xtr", "topxtr"):
            if self.xtr_top is None:
                raise KeyError("This polar has no Top Xtr column.")
            return _interp1(self.alpha_deg, self.xtr_top, alpha_deg, clamp=clamp)
        if name in ("xtr_bot", "bot_xtr", "botxtr", "btm_xtr"):
            if self.xtr_bot is None:
                raise KeyError("This polar has no Bot Xtr column.")
            return _interp1(self.alpha_deg, self.xtr_bot, alpha_deg, clamp=clamp)
        raise KeyError(f"Unknown coefficient name: {name}")

    def cl_at(self, alpha_deg: float, *, clamp: bool = True) -> float:
        return self.coeff("cl", alpha_deg, clamp=clamp)

    def cd_at(self, alpha_deg: float, *, clamp: bool = True) -> float:
        return self.coeff("cd", alpha_deg, clamp=clamp)

    def ld_at(self, alpha_deg: float, *, clamp: bool = True) -> float:
        cl = self.cl_at(alpha_deg, clamp=clamp)
        cd = self.cd_at(alpha_deg, clamp=clamp)
        return float(cl / cd) if cd != 0 else float("inf")

    # ---- summary metrics ----
    def best_ld(self) -> Tuple[float, float, float]:
        cd = np.asarray(self.cd, dtype=float)
        cl = np.asarray(self.cl, dtype=float)
        m = _finite_mask(cd, cl) & (cd > 0)
        if not np.any(m):
            raise ValueError("No valid points with cd > 0 to compute L/D.")
        ld = cl[m] / cd[m]
        i = int(np.argmax(ld))
        alpha = float(self.alpha_deg[m][i])
        return alpha, float(cl[m][i]), float(ld[i])

    def cl_max(self) -> Tuple[float, float]:
        i = int(np.nanargmax(self.cl))
        return float(self.alpha_deg[i]), float(self.cl[i])

    def lift_curve_slope(self, alpha_min: float = -2.0, alpha_max: float = 6.0) -> float:
        a = np.asarray(self.alpha_deg, dtype=float)
        cl = np.asarray(self.cl, dtype=float)
        m = _finite_mask(a, cl) & (a >= alpha_min) & (a <= alpha_max)
        if np.sum(m) < 3:
            raise ValueError("Not enough points in the requested alpha window.")
        A = np.column_stack([a[m], np.ones(np.sum(m))])
        coef, _, _, _ = np.linalg.lstsq(A, cl[m], rcond=None)
        return float(coef[0])  # per degree

    def cdo_and_lift_slope(self, alpha_min: float = -2.0, alpha_max: float = 6.0) -> Tuple[float, float]:
        a = np.asarray(self.alpha_deg, dtype=float)
        cl = np.asarray(self.cl, dtype=float)
        cd = np.asarray(self.cd, dtype=float)
        m = _finite_mask(a, cl, cd) & (a >= alpha_min) & (a <= alpha_max)
        if np.sum(m) < 3:
            raise ValueError("Not enough points in the requested alpha window.")

        a_m = a[m]
        cl_m = cl[m]
        cd_m = cd[m]

        a_mean = a_m.mean()
        cl_mean = cl_m.mean()
        da = a_m - a_mean
        denom = float(np.dot(da, da))
        if denom == 0.0:
            raise ValueError("Degenerate alpha window for lift slope.")
        lift_slope = float(np.dot(da, cl_m - cl_mean) / denom)

        x = cl_m * cl_m
        x_mean = x.mean()
        cd_mean = cd_m.mean()
        dx = x - x_mean
        denom_cd = float(np.dot(dx, dx))
        if denom_cd == 0.0:
            raise ValueError("Degenerate CL^2 window for Cdo fit.")
        k = float(np.dot(dx, cd_m - cd_mean) / denom_cd)
        cdo = float(cd_mean - k * x_mean)

        return cdo, lift_slope


# ---------------------------
# PolarSet (multiple Reynolds)
# ---------------------------
@dataclass
class PolarSet:
    """
    A collection of Polars at different Reynolds numbers for a single airfoil.
    Provides interpolation in alpha + Reynolds.
    """
    polars: List[Polar]
    name: str = "unknown_airfoil"

    def __post_init__(self) -> None:
        # Ensure all have Reynolds
        for p in self.polars:
            if p.reynolds is None:
                raise ValueError("All polars in PolarSet must have reynolds set (infer from filename or pass explicitly).")
        # Sort by Reynolds
        self.polars.sort(key=lambda p: float(p.reynolds))

    @property
    def reynolds_list(self) -> np.ndarray:
        return np.asarray([float(p.reynolds) for p in self.polars], dtype=float)

    @staticmethod
    def from_files(paths: Sequence[Union[str, Path]], *, name: str = "unknown_airfoil") -> "PolarSet":
        polars = [Polar.from_file(p) for p in paths]
        return PolarSet(polars=polars, name=name)

    @staticmethod
    def from_folder(
        folder: Union[str, Path],
        *,
        airfoil: Optional[str] = None,
        pattern: str = "*.pol",
        name: Optional[str] = None,
    ) -> "PolarSet":
        """
        Loads all polar files in a folder.
        If airfoil is provided, filters filenames starting with that stem (case-insensitive).
        """
        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        if airfoil is not None:
            af = airfoil.lower()
            files = [f for f in files if f.stem.lower().startswith(af)]
        if not files:
            raise FileNotFoundError(f"No polar files found in {folder} matching pattern={pattern} airfoil={airfoil}")

        polars = [Polar.from_file(f) for f in files]
        if name is None:
            name = airfoil or "unknown_airfoil"
        return PolarSet(polars=polars, name=name)

    # ---- 2D interpolation: alpha then Reynolds ----
    def _interp_re(
        self,
        values: Sequence[float],
        reynolds: float,
        *,
        clamp: bool = True,
    ) -> float:
        re = self.reynolds_list
        vals = np.asarray(values, dtype=float)
        return _interp1(re, vals, float(reynolds), clamp=clamp)

    def coeff(
        self,
        name: str,
        *,
        alpha_deg: float,
        reynolds: float,
        clamp_alpha: bool = True,
        clamp_re: bool = True,
    ) -> float:
        """
        Coefficient interpolation:
          1) Interpolate each Polar at alpha
          2) Interpolate those values across Reynolds
        """
        vals = [p.coeff(name, alpha_deg, clamp=clamp_alpha) for p in self.polars]
        return self._interp_re(vals, reynolds, clamp=clamp_re)

    def cl(self, *, alpha_deg: float, reynolds: float, clamp_alpha: bool = True, clamp_re: bool = True) -> float:
        return self.coeff("cl", alpha_deg=alpha_deg, reynolds=reynolds, clamp_alpha=clamp_alpha, clamp_re=clamp_re)

    def cd(self, *, alpha_deg: float, reynolds: float, clamp_alpha: bool = True, clamp_re: bool = True) -> float:
        return self.coeff("cd", alpha_deg=alpha_deg, reynolds=reynolds, clamp_alpha=clamp_alpha, clamp_re=clamp_re)

    def cm(self, *, alpha_deg: float, reynolds: float, clamp_alpha: bool = True, clamp_re: bool = True) -> float:
        return self.coeff("cm", alpha_deg=alpha_deg, reynolds=reynolds, clamp_alpha=clamp_alpha, clamp_re=clamp_re)

    def cdp(self, *, alpha_deg: float, reynolds: float, clamp_alpha: bool = True, clamp_re: bool = True) -> float:
        return self.coeff("cdp", alpha_deg=alpha_deg, reynolds=reynolds, clamp_alpha=clamp_alpha, clamp_re=clamp_re)

    def ld(self, *, alpha_deg: float, reynolds: float, clamp_alpha: bool = True, clamp_re: bool = True) -> float:
        cl = self.cl(alpha_deg=alpha_deg, reynolds=reynolds, clamp_alpha=clamp_alpha, clamp_re=clamp_re)
        cd = self.cd(alpha_deg=alpha_deg, reynolds=reynolds, clamp_alpha=clamp_alpha, clamp_re=clamp_re)
        return float(cl / cd) if cd != 0 else float("inf")

    # ---- Useful aggregated metrics ----
    def best_ld(self, *, reynolds: float, clamp_re: bool = True) -> Tuple[float, float, float]:
        """
        Approximate best L/D at a given Re:
          - take union of alpha samples across polars
          - interpolate CL/CD at each alpha and maximize L/D
        """
        alphas = np.unique(np.concatenate([p.alpha_deg for p in self.polars]).astype(float))
        cls = np.array([self.cl(alpha_deg=float(a), reynolds=reynolds, clamp_re=clamp_re) for a in alphas], dtype=float)
        cds = np.array([self.cd(alpha_deg=float(a), reynolds=reynolds, clamp_re=clamp_re) for a in alphas], dtype=float)

        m = _finite_mask(cls, cds) & (cds > 0)
        if not np.any(m):
            raise ValueError("No valid points with cd > 0 for L/D.")
        ld = cls[m] / cds[m]
        i = int(np.argmax(ld))
        return float(alphas[m][i]), float(cls[m][i]), float(ld[i])

    def cl_max(self, *, reynolds: float, clamp_re: bool = True) -> Tuple[float, float]:
        """
        Approximate CLmax at a given Re by sampling at the union alpha grid.
        """
        alphas = np.unique(np.concatenate([p.alpha_deg for p in self.polars]).astype(float))
        cls = np.array([self.cl(alpha_deg=float(a), reynolds=reynolds, clamp_re=clamp_re) for a in alphas], dtype=float)
        i = int(np.nanargmax(cls))
        return float(alphas[i]), float(cls[i])
    

    def cdo_and_lift_slope(
        self,
        *,
        reynolds: float,
        alpha_min: float = -2.0,
        alpha_max: float = 6.0,
        clamp_re: bool = True,
    ) -> Tuple[float, float]:
        cdo_vals = []
        slope_vals = []
        for p in self.polars:
            cdo, slope = p.cdo_and_lift_slope(alpha_min=alpha_min, alpha_max=alpha_max)
            cdo_vals.append(cdo)
            slope_vals.append(slope)

        cdo_i = self._interp_re(cdo_vals, reynolds, clamp=clamp_re)
        slope_i = self._interp_re(slope_vals, reynolds, clamp=clamp_re)
        return float(cdo_i), float(slope_i)


# ---------------------------
# Convenience loader
# ---------------------------
def load_polar_or_set(path: Union[str, Path]) -> Union[Polar, PolarSet]:
    """
    If path is a file -> Polar
    If path is a folder -> PolarSet (loads *.pol)
    """
    p = Path(path)
    if p.is_dir():
        return PolarSet.from_folder(p)
    return Polar.from_file(p)


# ---------------------------
# Quick demo
# ---------------------------
if __name__ == "__main__":
    # Example:
    #   python airfoil_polars.py ./polars naca2412 400000 4.0
    import sys

    if len(sys.argv) >= 5:
        folder = sys.argv[1]
        airfoil = sys.argv[2]
        Re = float(sys.argv[3])
        alpha = float(sys.argv[4])

        ps = PolarSet.from_folder(folder, airfoil=airfoil)
        print(f"Loaded {len(ps.polars)} polars for {airfoil}. Re list: {ps.reynolds_list.tolist()}")

        cl = ps.cl(alpha_deg=alpha, reynolds=Re)
        cd = ps.cd(alpha_deg=alpha, reynolds=Re)
        cm = None
        try:
            cm = ps.cm(alpha_deg=alpha, reynolds=Re)
        except Exception:
            pass

        print(f"alpha={alpha:.2f} deg, Re={Re:.0f}: CL={cl:.4f}, CD={cd:.5f}, L/D={cl/cd:.1f}" + (f", CM={cm:.4f}" if cm is not None else ""))

        a_best, cl_best, ld_best = ps.best_ld(reynolds=Re)
        print(f"Best L/D at Re={Re:.0f}: alpha={a_best:.2f} deg, CL={cl_best:.4f}, L/D={ld_best:.1f}")
    else:
        print("airfoil_polars.py: import Polar / PolarSet for programmatic use.")
        print("Demo: python airfoil_polars.py <folder> <airfoil_stem> <Re> <alpha>")

