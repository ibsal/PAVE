from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

EPS = 1e-12


def _poly_props(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if x[0] == x[-1] and y[0] == y[-1]:
        x = x[:-1]
        y = y[:-1]
    if x.size < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    x1 = xc[1:]
    y1 = yc[1:]
    x0 = xc[:-1]
    y0 = yc[:-1]
    cross = x0 * y1 - x1 * y0
    a_s = 0.5 * np.sum(cross)
    if abs(a_s) < EPS:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if a_s < 0.0:
        return _poly_props(x[::-1], y[::-1])

    area = a_s
    cx = np.sum((x0 + x1) * cross) / (6.0 * area)
    cy = np.sum((y0 + y1) * cross) / (6.0 * area)
    ixx0 = np.sum((y0 * y0 + y0 * y1 + y1 * y1) * cross) / 12.0
    iyy0 = np.sum((x0 * x0 + x0 * x1 + x1 * x1) * cross) / 12.0
    return float(area), float(cx), float(cy), float(ixx0), float(iyy0)


def _offset_polygon(x: np.ndarray, y: np.ndarray, d: float):
    if d <= 0.0:
        return x.copy(), y.copy()

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if n < 3:
        return np.empty(0), np.empty(0)

    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    ex = x2 - x
    ey = y2 - y
    L = np.hypot(ex, ey)
    if np.any(L < EPS):
        return np.empty(0), np.empty(0)

    ux = ex / L
    uy = ey / L
    nx = -uy
    ny = ux
    ox = x + d * nx
    oy = y + d * ny

    xo = np.empty(n, dtype=float)
    yo = np.empty(n, dtype=float)
    for i in range(n):
        ip = (i - 1) % n
        ax, ay = ox[ip], oy[ip]
        bx, by = ox[i], oy[i]
        ux1, uy1 = ux[ip], uy[ip]
        ux2, uy2 = ux[i], uy[i]
        den = ux1 * uy2 - uy1 * ux2
        if abs(den) < 1e-14:
            nnx = nx[ip] + nx[i]
            nny = ny[ip] + ny[i]
            nn = np.hypot(nnx, nny)
            if nn < EPS:
                nnx, nny, nn = nx[i], ny[i], max(np.hypot(nx[i], ny[i]), EPS)
            xo[i] = x[i] + d * nnx / nn
            yo[i] = y[i] + d * nny / nn
        else:
            t = ((bx - ax) * uy2 - (by - ay) * ux2) / den
            xo[i] = ax + t * ux1
            yo[i] = ay + t * uy1

    if _poly_props(xo, yo)[0] <= EPS:
        return np.empty(0), np.empty(0)
    return xo, yo


def _read_airfoil_xy(
    dat_path: Optional[Union[str, Path]],
    x_coords: Optional[Sequence[float]],
    y_coords: Optional[Sequence[float]],
):
    if dat_path is not None:
        pts = []
        for line in Path(dat_path).read_text().splitlines():
            s = line.strip().split()
            if len(s) >= 2:
                try:
                    pts.append((float(s[0]), float(s[1])))
                except ValueError:
                    pass
        xy = np.asarray(pts, dtype=float)
        x = xy[:, 0]
        y = xy[:, 1]
    else:
        if x_coords is None or y_coords is None:
            raise ValueError("Provide dat_path or both x_coords and y_coords.")
        x = np.asarray(x_coords, dtype=float).ravel()
        y = np.asarray(y_coords, dtype=float).ravel()

    if x.size < 3 or y.size < 3 or x.size != y.size:
        raise ValueError("Invalid airfoil coordinates.")
    if x[0] == x[-1] and y[0] == y[-1]:
        x = x[:-1]
        y = y[:-1]
    a_signed = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if a_signed < 0.0:
        x = x[::-1]
        y = y[::-1]
    a0 = _poly_props(x, y)[0]
    if a0 <= EPS:
        raise ValueError("Airfoil polygon area is zero.")
    return x, y


def _tube_table(tubes: Iterable[Dict[str, float]]):
    t = list(tubes or [])
    if not t:
        z = np.zeros(0, dtype=float)
        return {"x": z, "y": z, "ro": z, "A": z, "I": z, "E": z, "sigma_y": z + np.inf, "length": z + np.inf, "nu": z + np.nan}

    x = np.asarray([ti["x"] for ti in t], dtype=float)
    y = np.asarray([ti["y"] for ti in t], dtype=float)
    od = np.asarray([ti["od"] for ti in t], dtype=float)
    ro = 0.5 * np.maximum(od, 0.0)
    id_raw = np.asarray([ti.get("id", np.nan) for ti in t], dtype=float)
    wall = np.asarray([ti.get("wall_t", ti.get("wall", np.nan)) for ti in t], dtype=float)
    ri = np.where(np.isfinite(id_raw), 0.5 * np.maximum(id_raw, 0.0), np.where(np.isfinite(wall), np.maximum(ro - wall, 0.0), 0.0))
    ri = np.minimum(ri, ro)

    A = np.pi * np.maximum(ro * ro - ri * ri, 0.0)
    I = 0.25 * np.pi * np.maximum(ro**4 - ri**4, 0.0)
    E = np.asarray([ti["E"] for ti in t], dtype=float)
    sigma_y = np.asarray([ti.get("sigma_y", np.inf) for ti in t], dtype=float)
    length = np.asarray([ti.get("length_from_root", ti.get("length", np.inf)) for ti in t], dtype=float)
    nu = np.asarray([ti.get("nu", np.nan) for ti in t], dtype=float)
    return {"x": x, "y": y, "ro": ro, "ri": ri, "A": A, "I": I, "E": E, "sigma_y": sigma_y, "length": length, "nu": nu}


def _y_bounds_at_x(x: np.ndarray, y: np.ndarray, xq: float):
    x0 = x
    y0 = y
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    ys = []
    for i in range(x0.size):
        xa, xb = x0[i], x1[i]
        ya, yb = y0[i], y1[i]
        lo, hi = (xa, xb) if xa <= xb else (xb, xa)
        if xq < lo - EPS or xq > hi + EPS:
            continue
        dx = xb - xa
        if abs(dx) < EPS:
            ys.extend([ya, yb])
        else:
            t = (xq - xa) / dx
            if -EPS <= t <= 1.0 + EPS:
                ys.append(ya + t * (yb - ya))
    if not ys:
        j = int(np.argmin(np.abs(x - xq)))
        return float(y[j]), float(y[j])
    return float(max(ys)), float(min(ys))


def _roving_table(
    rovings: Optional[Dict[str, Dict[str, float]]],
    tube_x: np.ndarray,
    xn: np.ndarray,
    yn: np.ndarray,
    include_middle_rovings: bool = True,
):
    r = rovings or {}
    tx = np.sort(np.asarray(tube_x, dtype=float).ravel())
    tx = np.clip(tx, 0.0, 1.0)
    tx = np.unique(tx)
    if tx.size >= 1:
        bounds = np.concatenate((np.array([0.0], dtype=float), tx, np.array([1.0], dtype=float)))
        lane_x_all = 0.5 * (bounds[:-1] + bounds[1:])
        if include_middle_rovings or lane_x_all.size <= 2:
            lane_x = lane_x_all
            lane_ids = np.arange(1, lane_x_all.size + 1, dtype=int)
        else:
            lane_x = np.array([lane_x_all[0], lane_x_all[-1]], dtype=float)
            lane_ids = np.array([1, lane_x_all.size], dtype=int)
    else:
        lane_x = np.array([0.25, 0.75], dtype=float)
        lane_ids = np.array([1, 2], dtype=int)

    names_list = []
    xr_list = []
    y_top_list = []
    y_bot_list = []
    cfg_list = []
    legacy_t = ["qc_tension", "mid_tension", "tq_tension"]
    legacy_c = ["qc_compression", "mid_compression", "tq_compression"]
    for i, x_lane in enumerate(lane_x):
        y_t, y_b = _y_bounds_at_x(xn, yn, float(x_lane))
        lane_id = int(lane_ids[i])
        key_t = f"lane{lane_id}_tension"
        key_c = f"lane{lane_id}_compression"
        cfg_t = r.get(key_t, None)
        cfg_c = r.get(key_c, None)
        if cfg_t is None and i < len(legacy_t):
            cfg_t = r.get(legacy_t[i], None)
        if cfg_c is None and i < len(legacy_c):
            cfg_c = r.get(legacy_c[i], None)
        cfg_t = cfg_t if cfg_t is not None else {}
        cfg_c = cfg_c if cfg_c is not None else {}

        names_list.extend([key_t, key_c])
        xr_list.extend([float(x_lane), float(x_lane)])
        y_top_list.extend([y_t, y_t])
        y_bot_list.extend([y_b, y_b])
        cfg_list.extend([cfg_t, cfg_c])

    names = np.asarray(names_list, dtype=object)
    xr = np.asarray(xr_list, dtype=float)
    y_top = np.asarray(y_top_list, dtype=float)
    y_bot = np.asarray(y_bot_list, dtype=float)
    cnt = np.array([cfg.get("count", 0.0) for cfg in cfg_list], dtype=float)
    a1 = np.array([cfg.get("area", 0.0) for cfg in cfg_list], dtype=float)
    E_t = np.array([cfg.get("E_tension", cfg.get("E", 0.0)) for cfg in cfg_list], dtype=float)
    E_c = np.array([cfg.get("E_compression", cfg.get("E", 0.0)) for cfg in cfg_list], dtype=float)
    sy_t = np.array([cfg.get("sigma_tension", cfg.get("sigma_y", np.inf)) for cfg in cfg_list], dtype=float)
    sy_c = np.array([cfg.get("sigma_compression", cfg.get("sigma_y", np.inf)) for cfg in cfg_list], dtype=float)
    A1 = np.maximum(a1, 0.0)
    return {
        "names": names,
        "x": xr,
        "y_top": y_top,
        "y_bottom": y_bot,
        "A1": A1,
        "count_default": np.maximum(cnt, 0.0),
        "E_tension": E_t,
        "E_compression": E_c,
        "sigma_tension": sy_t,
        "sigma_compression": sy_c,
    }


def _piecewise_count_from_spec(spec, y_span: np.ndarray, default_count: float):
    y = np.asarray(y_span, dtype=float).ravel()
    if y.size == 0:
        return np.zeros(0, dtype=float)

    def _equal_segments(vals):
        vals = np.asarray(vals, dtype=float).ravel()
        if vals.size == 0:
            return np.zeros_like(y)
        edges = np.linspace(0.0, float(y[-1]), vals.size + 1)
        idx = np.searchsorted(edges[1:], y, side="right")
        idx = np.clip(idx, 0, vals.size - 1)
        return np.maximum(vals[idx], 0.0)

    if spec is None:
        return np.full_like(y, max(float(default_count), 0.0))
    if np.isscalar(spec):
        return np.full_like(y, max(float(spec), 0.0))
    if isinstance(spec, (list, tuple, np.ndarray)):
        return _equal_segments(spec)
    if isinstance(spec, dict):
        if "count_segments" in spec:
            return _equal_segments(spec["count_segments"])
        b = spec.get("count_span_breaks_in", spec.get("span_breaks_in"))
        v = spec.get("count_span_values", spec.get("count_by_span"))
        if b is not None and v is not None:
            br = np.asarray(b, dtype=float).ravel()
            vv = np.asarray(v, dtype=float).ravel()
            if br.size == vv.size:
                br = np.append(br, y[-1])
            if br.size != vv.size + 1:
                raise ValueError("Roving count profile requires len(breaks)=len(values)+1.")
            if np.any(np.diff(br) < -EPS):
                raise ValueError("Roving count breaks must be nondecreasing.")
            idx = np.searchsorted(br[1:], y, side="right")
            idx = np.clip(idx, 0, vv.size - 1)
            return np.maximum(vv[idx], 0.0)
        if "count" in spec and np.isscalar(spec["count"]):
            return np.full_like(y, max(float(spec["count"]), 0.0))
    return np.full_like(y, max(float(default_count), 0.0))


def _roving_count_matrix(rovings: Optional[Dict[str, Dict[str, float]]], rov: Dict[str, np.ndarray], y_span: np.ndarray):
    r = rovings or {}
    n = y_span.size
    m = rov["names"].size
    c = np.zeros((n, m), dtype=float)
    for j, nm in enumerate(rov["names"]):
        cfg = r.get(str(nm), {})
        c[:, j] = _piecewise_count_from_spec(cfg, y_span, float(rov["count_default"][j]))
    return c


def schrenk_half_wing_load(
    *,
    span: float,
    root_chord: float,
    tip_chord: float,
    lift_half: float,
    n_span: int = 201,
):
    n = int(max(n_span, 1))
    b2 = 0.5 * float(span)
    if n == 1:
        y = np.array([0.0], dtype=float)
        c = np.array([float(root_chord)], dtype=float)
        z = np.zeros(1, dtype=float)
        return {"y": y, "chord": c, "q": z, "V": z, "M": z, "c_planform": c, "c_elliptic": c, "c_schrenk": c}

    y = np.linspace(0.0, b2, n, dtype=float)
    eta = np.clip(y / max(b2, EPS), 0.0, 1.0)
    c_plan = float(root_chord) + (float(tip_chord) - float(root_chord)) * eta
    S = 0.5 * (float(root_chord) + float(tip_chord)) * float(span)
    c_ell = (4.0 * S / (np.pi * max(float(span), EPS))) * np.sqrt(np.maximum(0.0, 1.0 - eta * eta))
    c_sch = 0.5 * (c_plan + c_ell)

    den = float(np.sum(0.5 * (c_sch[:-1] + c_sch[1:]) * np.diff(y)))
    q = float(lift_half) * c_sch / max(den, EPS)
    dy = np.diff(y)
    dV = 0.5 * (q[:-1] + q[1:]) * dy
    V = np.concatenate((np.cumsum(dV[::-1])[::-1], np.array([0.0])))
    dM = 0.5 * (V[:-1] + V[1:]) * dy
    M = np.concatenate((np.cumsum(dM[::-1])[::-1], np.array([0.0])))
    return {"y": y, "chord": c_plan, "q": q, "V": V, "M": M, "c_planform": c_plan, "c_elliptic": c_ell, "c_schrenk": c_sch}


def wing_span_analysis(
    *,
    span: float,
    root_chord: float,
    tip_chord: float,
    lift_half: float,
    skin_thickness: float,
    Eface: float,
    Ecore: float,
    Gcore: float,
    k_wr: float,
    tubes: Sequence[Dict[str, float]] = (),
    rovings: Optional[Dict[str, Dict[str, float]]] = None,
    include_middle_rovings: bool = True,
    sigma_y_face: float = np.inf,
    sigma_y_core: float = np.inf,
    n_span: int = 201,
    Mx: Optional[Sequence[float]] = None,
    My: Union[float, Sequence[float]] = 0.0,
    x_ac_frac: float = 0.25,
    x_sc_frac: Optional[float] = None,
    torsion_dist: Union[float, Sequence[float]] = 0.0,
    nu_face: float = 0.30,
    nu_tube: float = 0.30,
    dat_path: Optional[Union[str, Path]] = None,
    x_coords: Optional[Sequence[float]] = None,
    y_coords: Optional[Sequence[float]] = None,
):
    xn, yn = _read_airfoil_xy(dat_path, x_coords, y_coords)
    a_n, cx_n, cy_n, ixx0_n, iyy0_n = _poly_props(xn, yn)
    if a_n <= EPS:
        raise ValueError("Airfoil area must be positive.")

    load = schrenk_half_wing_load(span=span, root_chord=root_chord, tip_chord=tip_chord, lift_half=lift_half, n_span=n_span)
    y_span = load["y"]
    chord = load["chord"]
    n = y_span.size

    if Mx is None:
        mx = load["M"].copy()
    else:
        mx = np.asarray(Mx, dtype=float).ravel()
        if mx.size != n:
            raise ValueError("Mx size must match span discretization.")

    if np.isscalar(My):
        my = np.full(n, float(My), dtype=float)
    else:
        my = np.asarray(My, dtype=float).ravel()
        if my.size != n:
            raise ValueError("My size must match span discretization.")

    tube = _tube_table(tubes)
    nt = tube["x"].size
    rov = _roving_table(rovings, tube["x"], xn, yn, include_middle_rovings=include_middle_rovings)
    nr = rov["x"].size
    rov_count = _roving_count_matrix(rovings, rov, y_span) if nr else np.zeros((n, 0), dtype=float)
    A_r = rov_count * rov["A1"][None, :] if nr else np.zeros((n, 0), dtype=float)
    xt_all = chord[:, None] * tube["x"][None, :] if nt else np.zeros((n, 0))
    yt_all = chord[:, None] * tube["y"][None, :] if nt else np.zeros((n, 0))
    x_ref_all = chord[:, None] * rov["x"][None, :] if nr else np.zeros((n, 0))
    y_top_all = chord[:, None] * rov["y_top"][None, :] if nr else np.zeros((n, 0))
    y_bot_all = chord[:, None] * rov["y_bottom"][None, :] if nr else np.zeros((n, 0))
    active = (y_span[:, None] <= tube["length"][None, :]) if nt else np.zeros((n, 0), dtype=bool)

    A_skin = np.zeros(n, dtype=float)
    cx_skin = np.zeros(n, dtype=float)
    cy_skin = np.zeros(n, dtype=float)
    Ixxc_skin = np.zeros(n, dtype=float)
    Iyyc_skin = np.zeros(n, dtype=float)

    A_core = np.zeros(n, dtype=float)
    cx_core = np.zeros(n, dtype=float)
    cy_core = np.zeros(n, dtype=float)
    Ixxc_core = np.zeros(n, dtype=float)
    Iyyc_core = np.zeros(n, dtype=float)
    A_mid = np.zeros(n, dtype=float)
    p_mid = np.zeros(n, dtype=float)

    inner_xy = [None] * n
    A_hole = np.pi * tube["ro"] ** 2 if nt else np.zeros(0, dtype=float)
    I_hole_c = 0.25 * np.pi * tube["ro"] ** 4 if nt else np.zeros(0, dtype=float)

    for i in range(n):
        c = chord[i]
        c2 = c * c
        c4 = c2 * c2
        Ao = a_n * c2
        cxo = cx_n * c
        cyo = cy_n * c
        Ixx0_o = ixx0_n * c4
        Iyy0_o = iyy0_n * c4

        xi_n, yi_n = _offset_polygon(xn, yn, skin_thickness / max(c, EPS))
        if xi_n.size:
            ai_n, cxi_n, cyi_n, ixx0i_n, iyy0i_n = _poly_props(xi_n, yi_n)
            Ai = ai_n * c2
            cxi = cxi_n * c
            cyi = cyi_n * c
            Ixx0_i = ixx0i_n * c4
            Iyy0_i = iyy0i_n * c4
            inner_xy[i] = (xi_n * c, yi_n * c)
            xm = xi_n * c
            ym = yi_n * c
            A_mid[i] = Ai
            p_mid[i] = float(np.sum(np.hypot(np.diff(np.append(xm, xm[0])), np.diff(np.append(ym, ym[0])))))
        else:
            Ai = cxi = cyi = Ixx0_i = Iyy0_i = 0.0
            inner_xy[i] = (np.empty(0), np.empty(0))

        if Ai > EPS:
            A_skin[i] = max(Ao - Ai, 0.0)
            if A_skin[i] > EPS:
                cx_skin[i] = (Ao * cxo - Ai * cxi) / A_skin[i]
                cy_skin[i] = (Ao * cyo - Ai * cyi) / A_skin[i]
                Ixx0_s = Ixx0_o - Ixx0_i
                Iyy0_s = Iyy0_o - Iyy0_i
                Ixxc_skin[i] = max(Ixx0_s - A_skin[i] * cy_skin[i] ** 2, 0.0)
                Iyyc_skin[i] = max(Iyy0_s - A_skin[i] * cx_skin[i] ** 2, 0.0)
            else:
                A_skin[i] = 0.0
        else:
            A_skin[i] = Ao
            cx_skin[i] = cxo
            cy_skin[i] = cyo
            Ixxc_skin[i] = max(Ixx0_o - Ao * cyo * cyo, 0.0)
            Iyyc_skin[i] = max(Iyy0_o - Ao * cxo * cxo, 0.0)

        if Ai > EPS:
            xh = xt_all[i] if nt else np.zeros(0, dtype=float)
            yh = yt_all[i] if nt else np.zeros(0, dtype=float)
            Ah = A_hole
            Ah_sum = float(np.sum(Ah))
            Aci = Ai - Ah_sum
            if Aci > EPS:
                xAh = float(np.sum(Ah * xh))
                yAh = float(np.sum(Ah * yh))
                Ixx0_h = float(np.sum(I_hole_c + Ah * yh * yh))
                Iyy0_h = float(np.sum(I_hole_c + Ah * xh * xh))
                cx_core[i] = (Ai * cxi - xAh) / Aci
                cy_core[i] = (Ai * cyi - yAh) / Aci
                Ixx0_c = Ixx0_i - Ixx0_h
                Iyy0_c = Iyy0_i - Iyy0_h
                A_core[i] = Aci
                Ixxc_core[i] = max(Ixx0_c - A_core[i] * cy_core[i] ** 2, 0.0)
                Iyyc_core[i] = max(Iyy0_c - A_core[i] * cx_core[i] ** 2, 0.0)

    EA_t = active * (tube["E"][None, :] * tube["A"][None, :]) if nt else np.zeros((n, 0))
    EA_base = Eface * A_skin + Ecore * A_core + np.sum(EA_t, axis=1)
    EA_base_safe = np.where(EA_base > EPS, EA_base, np.nan)
    x_na_base = (Eface * A_skin * cx_skin + Ecore * A_core * cx_core + np.sum(EA_t * xt_all, axis=1)) / EA_base_safe
    y_na_base = (Eface * A_skin * cy_skin + Ecore * A_core * cy_core + np.sum(EA_t * yt_all, axis=1)) / EA_base_safe
    EIxx_t_base = np.sum(active * (tube["E"][None, :] * (tube["I"][None, :] + tube["A"][None, :] * (yt_all - y_na_base[:, None]) ** 2)), axis=1) if nt else 0.0
    EIyy_t_base = np.sum(active * (tube["E"][None, :] * (tube["I"][None, :] + tube["A"][None, :] * (xt_all - x_na_base[:, None]) ** 2)), axis=1) if nt else 0.0
    EIxx_base = Eface * (Ixxc_skin + A_skin * (cy_skin - y_na_base) ** 2) + Ecore * (Ixxc_core + A_core * (cy_core - y_na_base) ** 2) + EIxx_t_base
    EIyy_base = Eface * (Iyyc_skin + A_skin * (cx_skin - x_na_base) ** 2) + Ecore * (Iyyc_core + A_core * (cx_core - x_na_base) ** 2) + EIyy_t_base
    kx_base = np.divide(mx, EIxx_base, out=np.zeros_like(mx), where=EIxx_base > EPS)
    ky_base = np.divide(my, EIyy_base, out=np.zeros_like(my), where=EIyy_base > EPS)

    if nr:
        strain_top = kx_base[:, None] * (y_top_all - y_na_base[:, None]) + ky_base[:, None] * (x_ref_all - x_na_base[:, None])
        strain_bot = kx_base[:, None] * (y_bot_all - y_na_base[:, None]) + ky_base[:, None] * (x_ref_all - x_na_base[:, None])
        lane_is_tension = np.array([("tension" in str(nm)) for nm in rov["names"]], dtype=bool)[None, :]
        choose_top = np.where(lane_is_tension, strain_top >= strain_bot, strain_top <= strain_bot)
        xr_all = x_ref_all
        yr_all = np.where(choose_top, y_top_all, y_bot_all)
        strain_sel = np.where(choose_top, strain_top, strain_bot)
        E_r_eff = np.where(strain_sel >= 0.0, rov["E_tension"][None, :], rov["E_compression"][None, :])
    else:
        xr_all = np.zeros((n, 0), dtype=float)
        yr_all = np.zeros((n, 0), dtype=float)
        E_r_eff = np.zeros((n, 0), dtype=float)

    EA_r = E_r_eff * A_r if nr else np.zeros((n, 0))
    EA = EA_base + np.sum(EA_r, axis=1)
    EA_safe = np.where(EA > EPS, EA, np.nan)

    x_na = (Eface * A_skin * cx_skin + Ecore * A_core * cx_core + np.sum(EA_t * xt_all, axis=1) + np.sum(EA_r * xr_all, axis=1)) / EA_safe
    y_na = (Eface * A_skin * cy_skin + Ecore * A_core * cy_core + np.sum(EA_t * yt_all, axis=1) + np.sum(EA_r * yr_all, axis=1)) / EA_safe

    EIxx_t = np.sum(active * (tube["E"][None, :] * (tube["I"][None, :] + tube["A"][None, :] * (yt_all - y_na[:, None]) ** 2)), axis=1) if nt else 0.0
    EIyy_t = np.sum(active * (tube["E"][None, :] * (tube["I"][None, :] + tube["A"][None, :] * (xt_all - x_na[:, None]) ** 2)), axis=1) if nt else 0.0
    EIxx_r = np.sum(E_r_eff * A_r * (yr_all - y_na[:, None]) ** 2, axis=1) if nr else 0.0
    EIyy_r = np.sum(E_r_eff * A_r * (xr_all - x_na[:, None]) ** 2, axis=1) if nr else 0.0

    EIxx = Eface * (Ixxc_skin + A_skin * (cy_skin - y_na) ** 2) + Ecore * (Ixxc_core + A_core * (cy_core - y_na) ** 2) + EIxx_t + EIxx_r
    EIyy = Eface * (Iyyc_skin + A_skin * (cx_skin - x_na) ** 2) + Ecore * (Iyyc_core + A_core * (cx_core - x_na) ** 2) + EIyy_t + EIyy_r

    kx = np.divide(mx, EIxx, out=np.zeros_like(mx), where=EIxx > EPS)
    ky = np.divide(my, EIyy, out=np.zeros_like(my), where=EIyy > EPS)

    tube_sigma = np.where(
        active,
        tube["E"][None, :] * (kx[:, None] * (yt_all - y_na[:, None]) + ky[:, None] * (xt_all - x_na[:, None])),
        0.0,
    ) if nt else np.zeros((n, 0))
    tube_yield = np.where(active, np.abs(tube_sigma) / np.maximum(tube["sigma_y"][None, :], EPS), 0.0) if nt else np.zeros((n, 0))
    tube_yield_max = np.max(tube_yield, axis=1) if nt else np.zeros(n, dtype=float)
    rov_sigma = E_r_eff * (kx[:, None] * (yr_all - y_na[:, None]) + ky[:, None] * (xr_all - x_na[:, None])) if nr else np.zeros((n, 0))
    rov_allow = np.where(rov_sigma >= 0.0, rov["sigma_tension"][None, :], rov["sigma_compression"][None, :]) if nr else np.zeros((n, 0))
    rov_yield = np.abs(rov_sigma) / np.maximum(rov_allow, EPS) if nr else np.zeros((n, 0))
    rov_yield_max = np.max(rov_yield, axis=1) if nr else np.zeros(n, dtype=float)

    skin_sig_min = np.zeros(n, dtype=float)
    skin_sig_max = np.zeros(n, dtype=float)
    skin_comp_x = np.zeros(n, dtype=float)
    skin_comp_y = np.zeros(n, dtype=float)
    skin_abs_x = np.zeros(n, dtype=float)
    skin_abs_y = np.zeros(n, dtype=float)
    core_sig_min = np.zeros(n, dtype=float)
    core_sig_max = np.zeros(n, dtype=float)
    core_sig_abs = np.zeros(n, dtype=float)
    core_abs_x = np.full(n, np.nan, dtype=float)
    core_abs_y = np.full(n, np.nan, dtype=float)
    for i in range(n):
        xo = xn * chord[i]
        yo = yn * chord[i]
        s_out = Eface * (kx[i] * (yo - y_na[i]) + ky[i] * (xo - x_na[i]))
        xs = xo
        ys = yo
        ss = s_out
        xi, yi = inner_xy[i]
        if xi.size:
            s_in = Eface * (kx[i] * (yi - y_na[i]) + ky[i] * (xi - x_na[i]))
            xs = np.concatenate((xs, xi))
            ys = np.concatenate((ys, yi))
            ss = np.concatenate((ss, s_in))
        imin = int(np.argmin(ss))
        iabs = int(np.argmax(np.abs(ss)))
        smin = float(ss[imin])
        smax = float(np.max(ss))
        skin_comp_x[i] = float(xs[imin])
        skin_comp_y[i] = float(ys[imin])
        skin_abs_x[i] = float(xs[iabs])
        skin_abs_y[i] = float(ys[iabs])
        skin_sig_min[i] = smin
        skin_sig_max[i] = smax

        if A_core[i] > EPS:
            cands_v = []
            cands_x = []
            cands_y = []
            if xi.size:
                c_in = Ecore * (kx[i] * (yi - y_na[i]) + ky[i] * (xi - x_na[i]))
                cmin = float(np.min(c_in))
                cmax = float(np.max(c_in))
                cands_v.append(c_in)
                cands_x.append(xi)
                cands_y.append(yi)
            else:
                cmin = cmax = 0.0
            if nt:
                g = np.hypot(ky[i], kx[i])
                c_ctr = Ecore * (kx[i] * (yt_all[i] - y_na[i]) + ky[i] * (xt_all[i] - x_na[i]))
                dc = Ecore * tube["ro"] * g
                cmin = min(cmin, float(np.min(c_ctr - dc)))
                cmax = max(cmax, float(np.max(c_ctr + dc)))
                if g > EPS:
                    dx = ky[i] / g
                    dy_g = kx[i] / g
                    xh = xt_all[i]
                    yh = yt_all[i]
                    rh = tube["ro"]
                    cands_v.extend([c_ctr + dc, c_ctr - dc])
                    cands_x.extend([xh + rh * dx, xh - rh * dx])
                    cands_y.extend([yh + rh * dy_g, yh - rh * dy_g])
            core_sig_min[i] = cmin
            core_sig_max[i] = cmax
            core_sig_abs[i] = max(abs(cmin), abs(cmax))
            if cands_v:
                vv = np.concatenate([np.asarray(v, dtype=float).ravel() for v in cands_v])
                xx = np.concatenate([np.asarray(v, dtype=float).ravel() for v in cands_x])
                yy = np.concatenate([np.asarray(v, dtype=float).ravel() for v in cands_y])
                j = int(np.argmax(np.abs(vv)))
                core_abs_x[i] = float(xx[j])
                core_abs_y[i] = float(yy[j])

    dy = np.diff(y_span)
    if dy.size:
        theta = np.concatenate(([0.0], np.cumsum(0.5 * (kx[:-1] + kx[1:]) * dy)))
        defl = np.concatenate(([0.0], np.cumsum(0.5 * (theta[:-1] + theta[1:]) * dy)))
    else:
        theta = np.zeros_like(y_span)
        defl = np.zeros_like(y_span)

    if np.isscalar(torsion_dist):
        m_add = np.full(n, float(torsion_dist), dtype=float)
    else:
        m_add = np.asarray(torsion_dist, dtype=float).ravel()
        if m_add.size != n:
            raise ValueError("torsion_dist size must match span discretization.")
    x_ac = float(x_ac_frac) * chord
    if x_sc_frac is None:
        x_sc = (float(np.min(tube["x"])) if nt else 0.25) * chord
    else:
        x_sc = float(x_sc_frac) * chord
    m_t = load["q"] * (x_ac - x_sc) + m_add
    if dy.size:
        dT = 0.5 * (m_t[:-1] + m_t[1:]) * dy
        torque = np.concatenate((np.cumsum(dT[::-1])[::-1], np.array([0.0])))
    else:
        torque = np.zeros_like(y_span)

    G_face = float(Eface) / (2.0 * (1.0 + float(nu_face)))
    GJ_skin = np.where((A_mid > EPS) & (p_mid > EPS) & (skin_thickness > EPS), 4.0 * A_mid * A_mid * G_face * skin_thickness / p_mid, 0.0)
    J_core = Ixxc_core + Iyyc_core
    GJ_core = float(Gcore) * J_core
    if nt:
        g_raw = np.asarray([t.get("G", np.nan) for t in tubes], dtype=float)
        nu_t = np.where(np.isfinite(tube["nu"]), tube["nu"], float(nu_tube))
        G_tube = np.where(np.isfinite(g_raw), g_raw, tube["E"] / (2.0 * (1.0 + nu_t)))
        J_tube = 2.0 * tube["I"]
        GJ_tube = np.sum(active * (G_tube[None, :] * J_tube[None, :]), axis=1)
    else:
        GJ_tube = np.zeros(n, dtype=float)
    GJ = GJ_skin + GJ_core + GJ_tube
    twist_rate = np.divide(torque, GJ, out=np.zeros_like(torque), where=GJ > EPS)
    if dy.size:
        twist = np.concatenate(([0.0], np.cumsum(0.5 * (twist_rate[:-1] + twist_rate[1:]) * dy)))
    else:
        twist = np.zeros_like(y_span)
    twist_deg = np.degrees(twist)

    skin_sig_abs = np.maximum(np.abs(skin_sig_min), np.abs(skin_sig_max))
    skin_yield = skin_sig_abs / max(float(sigma_y_face), EPS)
    core_yield = core_sig_abs / max(float(sigma_y_core), EPS)
    yield_max = np.maximum.reduce([skin_yield, core_yield, tube_yield_max, rov_yield_max])

    sigma_wr = float(k_wr) * (float(Eface) * float(Ecore) * float(Gcore)) ** (1.0 / 3.0)
    skin_sigma_comp = np.maximum(0.0, -skin_sig_min)
    wrinkling_index = skin_sigma_comp / max(sigma_wr, EPS)
    fos_yield = np.divide(1.0, yield_max, out=np.full_like(yield_max, np.inf), where=yield_max > EPS)
    fos_wrinkling = np.divide(1.0, wrinkling_index, out=np.full_like(wrinkling_index, np.inf), where=wrinkling_index > EPS)
    fos_governing = np.minimum(fos_yield, fos_wrinkling)

    i_y = int(np.nanargmax(yield_max))
    i_w = int(np.nanargmax(wrinkling_index))
    i_fg = int(np.nanargmin(fos_governing))
    y_terms = np.array([skin_yield[i_y], core_yield[i_y], tube_yield_max[i_y], rov_yield_max[i_y]], dtype=float)
    i_mode_y = int(np.nanargmax(y_terms))
    if i_mode_y == 0:
        y_mode = "skin"
        x_yield = float(skin_abs_x[i_y])
        y_yield = float(skin_abs_y[i_y])
    elif i_mode_y == 1:
        y_mode = "core"
        x_yield = float(core_abs_x[i_y]) if np.isfinite(core_abs_x[i_y]) else float("nan")
        y_yield = float(core_abs_y[i_y]) if np.isfinite(core_abs_y[i_y]) else float("nan")
    elif i_mode_y == 2 and nt:
        j = int(np.argmax(tube_yield[i_y]))
        y_mode = f"tube_{j+1}"
        x_yield = float(xt_all[i_y, j])
        y_yield = float(yt_all[i_y, j])
    elif i_mode_y == 3 and nr:
        j = int(np.argmax(rov_yield[i_y]))
        y_mode = str(rov["names"][j])
        x_yield = float(xr_all[i_y, j])
        y_yield = float(yr_all[i_y, j])
    else:
        y_mode = "unknown"
        x_yield = float("nan")
        y_yield = float("nan")

    if fos_wrinkling[i_fg] <= fos_yield[i_fg]:
        gov_mode = "wrinkling"
        x_gov = float(skin_comp_x[i_fg])
        y_gov = float(skin_comp_y[i_fg])
    else:
        gov_mode = y_mode if i_fg == i_y else "yield"
        x_gov = float(x_yield) if i_fg == i_y else float("nan")
        y_gov = float(y_yield) if i_fg == i_y else float("nan")
    return {
        "span_y": y_span,
        "chord": chord,
        "load": load,
        "Mx": mx,
        "My": my,
        "neutral_axis": {"x": x_na, "y": y_na},
        "EI": {"xx": EIxx, "yy": EIyy},
        "curvature": {"kx": kx, "ky": ky},
        "beam": {
            "shear": load["V"],
            "bending": mx,
            "slope": theta,
            "deflection": defl,
            "torque": torque,
            "GJ": GJ,
            "twist_rate_rad_per_in": twist_rate,
            "twist_rad": twist,
            "twist_deg": twist_deg,
        },
        "stress": {
            "skin_min": skin_sig_min,
            "skin_max": skin_sig_max,
            "core_min": core_sig_min,
            "core_max": core_sig_max,
            "tube": tube_sigma,
            "roving": rov_sigma,
            "roving_count": rov_count,
        },
        "yield_index": {
            "skin": skin_yield,
            "core": core_yield,
            "tube": tube_yield,
            "roving": rov_yield,
            "roving_names": rov["names"],
            "max": yield_max,
            "max_value": float(yield_max[i_y]),
            "max_y": float(y_span[i_y]),
            "max_x": x_yield,
            "max_z": y_yield,
            "max_mode": y_mode,
        },
        "wrinkling": {
            "sigma_wr": sigma_wr,
            "sigma_comp_skin": skin_sigma_comp,
            "index": wrinkling_index,
            "max_value": float(wrinkling_index[i_w]),
            "max_y": float(y_span[i_w]),
            "max_x": float(skin_comp_x[i_w]),
            "max_z": float(skin_comp_y[i_w]),
        },
        "factor_of_safety": {
            "yield": fos_yield,
            "wrinkling": fos_wrinkling,
            "governing": fos_governing,
            "governing_min": float(fos_governing[i_fg]),
            "governing_min_y": float(y_span[i_fg]),
            "governing_min_x": x_gov,
            "governing_min_z": y_gov,
            "governing_mode": gov_mode,
        },
        "pass": {
            "yield": bool(np.nanmax(yield_max) <= 1.0),
            "wrinkling": bool(np.nanmax(wrinkling_index) <= 1.0),
        },
    }


def wing_beam_failure(
    *,
    skin_thickness: float,
    tube1: Optional[Dict[str, float]] = None,
    tube2: Optional[Dict[str, float]] = None,
    tubes: Sequence[Dict[str, float]] = (),
    rovings: Optional[Dict[str, Dict[str, float]]] = None,
    include_middle_rovings: bool = True,
    Eface: float,
    Ecore: float,
    Gcore: float,
    k_wr: float,
    Mx: float = 0.0,
    My: float = 0.0,
    chord: float = 1.0,
    sigma_y_face: float = np.inf,
    sigma_y_core: float = np.inf,
    dat_path: Optional[Union[str, Path]] = None,
    x_coords: Optional[Sequence[float]] = None,
    y_coords: Optional[Sequence[float]] = None,
):
    t = list(tubes or [])
    if tube1 is not None:
        t.append(tube1)
    if tube2 is not None:
        t.append(tube2)

    out = wing_span_analysis(
        span=2.0,
        root_chord=chord,
        tip_chord=chord,
        lift_half=0.0,
        skin_thickness=skin_thickness,
        Eface=Eface,
        Ecore=Ecore,
        Gcore=Gcore,
        k_wr=k_wr,
        tubes=t,
        rovings=rovings,
        include_middle_rovings=include_middle_rovings,
        sigma_y_face=sigma_y_face,
        sigma_y_core=sigma_y_core,
        n_span=1,
        Mx=[Mx],
        My=[My],
        dat_path=dat_path,
        x_coords=x_coords,
        y_coords=y_coords,
    )
    return {
        "neutral_axis": {"x": float(out["neutral_axis"]["x"][0]), "y": float(out["neutral_axis"]["y"][0])},
        "EI": {"xx": float(out["EI"]["xx"][0]), "yy": float(out["EI"]["yy"][0])},
        "tube_stress": out["stress"]["tube"][0].copy(),
        "roving_stress": out["stress"]["roving"][0].copy(),
        "yield_index": {
            "skin": float(out["yield_index"]["skin"][0]),
            "core": float(out["yield_index"]["core"][0]),
            "tube": out["yield_index"]["tube"][0].copy(),
            "roving": out["yield_index"]["roving"][0].copy(),
            "roving_names": out["yield_index"]["roving_names"].copy(),
            "max": float(out["yield_index"]["max"][0]),
            "max_x": float(out["yield_index"]["max_x"]),
            "max_mode": str(out["yield_index"]["max_mode"]),
        },
        "wrinkling": {
            "sigma_comp": float(out["wrinkling"]["sigma_comp_skin"][0]),
            "sigma_wr": float(out["wrinkling"]["sigma_wr"]),
            "index": float(out["wrinkling"]["index"][0]),
            "max_x": float(out["wrinkling"]["max_x"]),
        },
        "factor_of_safety": {
            "yield": float(out["factor_of_safety"]["yield"][0]),
            "wrinkling": float(out["factor_of_safety"]["wrinkling"][0]),
            "governing": float(out["factor_of_safety"]["governing"][0]),
            "governing_x": float(out["factor_of_safety"]["governing_min_x"]),
            "governing_mode": str(out["factor_of_safety"]["governing_mode"]),
        },
        "pass": {"yield": bool(out["yield_index"]["max"][0] <= 1.0), "wrinkling": bool(out["wrinkling"]["index"][0] <= 1.0)},
    }


def plot_section_layout(
    *,
    chord: float,
    skin_thickness: float,
    tubes: Sequence[Dict[str, float]],
    rovings: Optional[Dict[str, Dict[str, float]]] = None,
    include_middle_rovings: bool = True,
    show_rovings: bool = True,
    span_station: float = 0.0,
    dat_path: Optional[Union[str, Path]] = None,
    x_coords: Optional[Sequence[float]] = None,
    y_coords: Optional[Sequence[float]] = None,
    ax=None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting.") from e

    xn, yn = _read_airfoil_xy(dat_path, x_coords, y_coords)
    xo = xn * chord
    yo = yn * chord
    xi_n, yi_n = _offset_polygon(xn, yn, skin_thickness / max(chord, EPS))
    xi = xi_n * chord if xi_n.size else np.empty(0)
    yi = yi_n * chord if yi_n.size else np.empty(0)
    tube = _tube_table(tubes)
    rov = _roving_table(rovings, tube["x"], xn, yn, include_middle_rovings=include_middle_rovings) if show_rovings else None

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    ax.set_aspect("equal", adjustable="box")

    ax.fill(xo, yo, color="#80b1d3", alpha=0.28, zorder=1, label="Skin")
    if xi.size:
        ax.fill(xi, yi, color="#a6d96a", alpha=0.60, zorder=2, label="Core")

    th = np.linspace(0.0, 2.0 * np.pi, 140)
    for j in range(tube["x"].size):
        xc = chord * tube["x"][j]
        yc = chord * tube["y"][j]
        ro = tube["ro"][j]
        ri = tube["ri"][j]
        active = span_station <= tube["length"][j]
        xco = xc + ro * np.cos(th)
        yco = yc + ro * np.sin(th)
        ax.fill(xco, yco, color="white", alpha=1.0, zorder=5)  # persistent core cutout
        if active and ro > ri + EPS:
            xci = xc + ri * np.cos(th)
            yci = yc + ri * np.sin(th)
            ax.fill(xco, yco, color="#2b2b2b", alpha=0.95, zorder=6)
            ax.fill(xci, yci, color="white", alpha=1.0, zorder=7)
            ax.plot(xco, yco, color="#111111", lw=1.1, zorder=8)
            ax.plot(xci, yci, color="#111111", lw=1.0, zorder=8)
        else:
            ax.plot(xco, yco, color="#666666", lw=1.0, ls="--", zorder=6)

    if rov is not None and rov["x"].size:
        x_lane = chord * rov["x"][::2]
        y_lane_top = chord * rov["y_top"][::2]
        y_lane_bot = chord * rov["y_bottom"][::2]
        lane_names = np.asarray(rov["names"][::2], dtype=object)
        ax.scatter(x_lane, y_lane_top, s=26, color="#ff8c00", edgecolors="#5f2f00", linewidths=0.5, zorder=9)
        ax.scatter(x_lane, y_lane_bot, s=26, color="#ff8c00", edgecolors="#5f2f00", linewidths=0.5, zorder=9)
        y_off = 0.010 * chord
        for xi_, yi_, nm in zip(x_lane, y_lane_top, lane_names):
            lbl = str(nm).split("_", 1)[0].replace("lane", "L")
            ax.text(xi_, yi_ + y_off, lbl, fontsize=7, ha="center", va="bottom", color="#5f2f00", zorder=10)

    ax.plot(np.append(xo, xo[0]), np.append(yo, yo[0]), color="#1f4e79", lw=1.3)
    if xi.size:
        ax.plot(np.append(xi, xi[0]), np.append(yi, yi[0]), color="#2f7f4f", lw=1.0)
    ax.set_xlabel("x [in]")
    ax.set_ylabel("y [in]")
    ax.set_title(f"Section @ y={span_station:.2f} in")
    return ax


def plot_span_results(out, mirror_full_span: bool = True):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting.") from e

    y = np.asarray(out["span_y"], dtype=float)
    if mirror_full_span and y.size > 1:
        y_plot = np.concatenate((-y[1:][::-1], y))

        def _m1(v: np.ndarray, odd: bool = False):
            v = np.asarray(v, dtype=float).ravel()
            left = v[1:][::-1]
            if odd:
                left = -left
            return np.concatenate((left, v))

        def _m2(a: np.ndarray):
            a = np.asarray(a, dtype=float)
            return np.vstack((a[1:][::-1, :], a))

        shear = _m1(out["beam"]["shear"], odd=True)
        bending = _m1(out["beam"]["bending"], odd=False)
        deflection = _m1(out["beam"]["deflection"], odd=False)
        skin_max = _m1(out["stress"]["skin_max"], odd=False)
        skin_min = _m1(out["stress"]["skin_min"], odd=False)
        core_max = _m1(out["stress"]["core_max"], odd=False)
        core_min = _m1(out["stress"]["core_min"], odd=False)
        tube = _m2(out["stress"]["tube"]) if np.asarray(out["stress"]["tube"]).ndim == 2 else out["stress"]["tube"]
        rov = _m2(out["stress"]["roving"]) if np.asarray(out["stress"]["roving"]).ndim == 2 else out["stress"]["roving"]
    else:
        y_plot = y
        shear = out["beam"]["shear"]
        bending = out["beam"]["bending"]
        deflection = out["beam"]["deflection"]
        skin_max = out["stress"]["skin_max"]
        skin_min = out["stress"]["skin_min"]
        core_max = out["stress"]["core_max"]
        core_min = out["stress"]["core_min"]
        tube = out["stress"]["tube"]
        rov = out["stress"]["roving"]

    twist_deg = _m1(out["beam"]["twist_deg"], odd=False) if mirror_full_span and y.size > 1 else out["beam"]["twist_deg"]
    fig, axs = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)

    axs[0, 0].plot(y_plot, shear, lw=1.8)
    axs[0, 0].set_title("Shear V(y)")
    axs[0, 0].set_xlabel("y from centerline [in]")
    axs[0, 0].set_ylabel("V [lbf]")
    axs[0, 0].grid(alpha=0.25)

    axs[0, 1].plot(y_plot, bending, lw=1.8)
    axs[0, 1].set_title("Bending M(y)")
    axs[0, 1].set_xlabel("y from centerline [in]")
    axs[0, 1].set_ylabel("M [lbf*in]")
    axs[0, 1].grid(alpha=0.25)

    axs[1, 0].plot(y_plot, deflection, lw=1.8)
    axs[1, 0].set_title("Deflection w(y)")
    axs[1, 0].set_xlabel("y from centerline [in]")
    axs[1, 0].set_ylabel("w [in]")
    axs[1, 0].grid(alpha=0.25)

    axs[1, 1].plot(y_plot, skin_max, label="Skin max", lw=1.6)
    axs[1, 1].plot(y_plot, skin_min, label="Skin min", lw=1.6)
    axs[1, 1].set_title("Skin Stress")
    axs[1, 1].set_xlabel("y from centerline [in]")
    axs[1, 1].set_ylabel("Stress [psi]")
    axs[1, 1].grid(alpha=0.25)
    axs[1, 1].legend(loc="best", fontsize=8)

    axs[2, 0].plot(y_plot, core_max, label="Foam max", lw=1.6)
    axs[2, 0].plot(y_plot, core_min, label="Foam min", lw=1.6)
    axs[2, 0].set_title("Foam/Core Stress")
    axs[2, 0].set_xlabel("y from centerline [in]")
    axs[2, 0].set_ylabel("Stress [psi]")
    axs[2, 0].grid(alpha=0.25)
    axs[2, 0].legend(loc="best", fontsize=8)

    if tube.ndim == 2 and tube.shape[1] > 0:
        for j in range(tube.shape[1]):
            axs[2, 1].plot(y_plot, tube[:, j], lw=1.4, label=f"Tube {j+1}")
    if rov.ndim == 2 and rov.shape[1] > 0:
        names = out["yield_index"].get("roving_names", np.array([f"R{j+1}" for j in range(rov.shape[1])], dtype=object))
        for j in range(rov.shape[1]):
            axs[2, 1].plot(y_plot, rov[:, j], "--", lw=1.2, label=str(names[j]))
    axs[2, 1].set_title("Tube/Roving Stress")
    axs[2, 1].set_xlabel("y from centerline [in]")
    axs[2, 1].set_ylabel("Stress [psi]")
    axs[2, 1].grid(alpha=0.25)
    axs[2, 1].legend(loc="best", fontsize=8)

    axs[3, 0].plot(y_plot, twist_deg, lw=1.8)
    axs[3, 0].set_title("Twist")
    axs[3, 0].set_xlabel("y from centerline [in]")
    axs[3, 0].set_ylabel("theta [deg]")
    axs[3, 0].grid(alpha=0.25)

    if "roving_count" in out["stress"]:
        rc = out["stress"]["roving_count"]
        rc_plot = _m2(rc) if (mirror_full_span and y.size > 1 and np.asarray(rc).ndim == 2) else np.asarray(rc)
        if rc_plot.ndim == 2 and rc_plot.shape[1] > 0:
            for j in range(rc_plot.shape[1]):
                axs[3, 1].plot(y_plot, rc_plot[:, j], lw=1.4, label=str(out["yield_index"]["roving_names"][j]))
            axs[3, 1].legend(loc="best", fontsize=8)
    axs[3, 1].set_title("Roving Count")
    axs[3, 1].set_xlabel("y from centerline [in]")
    axs[3, 1].set_ylabel("count [-]")
    axs[3, 1].grid(alpha=0.25)

    return fig, axs


def roving_mass_from_output(
    out: Dict[str, Any],
    *,
    linear_density_g_per_m: float,
    both_wings: bool = True,
):
    y = np.asarray(out.get("span_y", []), dtype=float).ravel()
    rc = np.asarray(out.get("stress", {}).get("roving_count", []), dtype=float)
    if y.size < 2 or rc.ndim != 2 or rc.shape[0] != y.size or rc.shape[1] == 0:
        mass_half_g = 0.0
    else:
        cnt_sum = np.sum(np.maximum(rc, 0.0), axis=1)
        cnt_len_in = float(np.sum(0.5 * (cnt_sum[:-1] + cnt_sum[1:]) * np.diff(y)))
        cnt_len_m = cnt_len_in * 0.0254
        mass_half_g = float(linear_density_g_per_m) * cnt_len_m
    mass_total_g = mass_half_g * (2.0 if both_wings else 1.0)
    return {
        "half_wing_g": mass_half_g,
        "total_g": mass_total_g,
        "half_wing_lb": mass_half_g / 453.59237,
        "total_lb": mass_total_g / 453.59237,
    }


def _get_roving_profile(cfg: Dict[str, Any]) -> Tuple[str, np.ndarray, np.ndarray]:
    if "count_segments" in cfg:
        vals = np.asarray(cfg["count_segments"], dtype=float).ravel()
        w = np.ones_like(vals)
        return "count_segments", vals, w
    if "count_span_values" in cfg:
        vals = np.asarray(cfg["count_span_values"], dtype=float).ravel()
        br = cfg.get("count_span_breaks_in", cfg.get("span_breaks_in"))
        if br is not None:
            br = np.asarray(br, dtype=float).ravel()
            if br.size == vals.size + 1:
                w = np.maximum(np.diff(br), 0.0)
                return "count_span_values", vals, w
        return "count_span_values", vals, np.ones_like(vals)
    if "count_by_span" in cfg:
        vals = np.asarray(cfg["count_by_span"], dtype=float).ravel()
        br = cfg.get("count_span_breaks_in", cfg.get("span_breaks_in"))
        if br is not None:
            br = np.asarray(br, dtype=float).ravel()
            if br.size == vals.size + 1:
                w = np.maximum(np.diff(br), 0.0)
                return "count_by_span", vals, w
        return "count_by_span", vals, np.ones_like(vals)
    v = float(cfg.get("count", 0.0))
    return "count", np.array([v], dtype=float), np.array([1.0], dtype=float)


def optimize_roving_counts(
    *,
    analysis_kwargs: Dict[str, Any],
    rovings: Dict[str, Dict[str, Any]],
    min_fos: float = 1.0,
    step: int = 1,
    max_iter: int = 1000,
    weight_by_span: bool = False,
    objective_mode: str = "peak_then_total",
    optimize_tube_lengths: bool = False,
    tube_length_step: float = 1.0,
    tube_length_min: Union[float, Sequence[float]] = 0.0,
    tube_length_max: Optional[Union[float, Sequence[float]]] = None,
    allow_increase_to_feasible: bool = True,
    max_feas_iter: int = 1000,
    verbose: bool = False,
):
    if step <= 0:
        raise ValueError("step must be > 0")
    if tube_length_step <= 0:
        raise ValueError("tube_length_step must be > 0")
    base = deepcopy(rovings)
    analysis_base = dict(analysis_kwargs)
    base_tubes = deepcopy(analysis_base.pop("tubes", []))
    nt = len(base_tubes)
    half_span = 0.5 * float(analysis_base.get("span", 0.0))
    len_cur = np.asarray([float(t.get("length_from_root", t.get("length", half_span))) for t in base_tubes], dtype=float)
    if np.isscalar(tube_length_min):
        len_min = np.full(nt, max(float(tube_length_min), 0.0), dtype=float)
    else:
        len_min = np.asarray(tube_length_min, dtype=float).ravel()
        if len_min.size != nt:
            raise ValueError("tube_length_min length must match number of tubes.")
        len_min = np.maximum(len_min, 0.0)
    if tube_length_max is None:
        len_max = np.full(nt, max(half_span, 0.0), dtype=float)
    elif np.isscalar(tube_length_max):
        len_max = np.full(nt, max(float(tube_length_max), 0.0), dtype=float)
    else:
        len_max = np.asarray(tube_length_max, dtype=float).ravel()
        if len_max.size != nt:
            raise ValueError("tube_length_max length must match number of tubes.")
        len_max = np.maximum(len_max, 0.0)
    if nt:
        len_max = np.maximum(len_max, len_min)
        len_cur = np.clip(len_cur, len_min, len_max)
    keys = sorted(base.keys())
    mode: Dict[str, str] = {}
    count: Dict[str, np.ndarray] = {}
    wgt: Dict[str, np.ndarray] = {}
    for k in keys:
        m, c, w = _get_roving_profile(base[k])
        mode[k] = m
        count[k] = np.maximum(np.rint(c).astype(int), 0)
        wgt[k] = np.asarray(w, dtype=float).ravel()
        if wgt[k].size != count[k].size:
            wgt[k] = np.ones_like(count[k], dtype=float)

    def _build_rovings() -> Dict[str, Dict[str, Any]]:
        out = deepcopy(base)
        for k in keys:
            arr = count[k].astype(float)
            cfg = out.get(k, {})
            m = mode[k]
            if m == "count":
                cfg["count"] = float(arr[0])
            elif m == "count_segments":
                cfg["count_segments"] = arr.tolist()
            elif m == "count_span_values":
                cfg["count_span_values"] = arr.tolist()
            else:
                cfg["count_by_span"] = arr.tolist()
            out[k] = cfg
        return out

    def _build_tubes() -> Sequence[Dict[str, Any]]:
        t = deepcopy(base_tubes)
        for i in range(len(t)):
            t[i]["length_from_root"] = float(len_cur[i]) if nt else float(t[i].get("length_from_root", t[i].get("length", 0.0)))
        return t

    def _objective() -> Tuple[float, float, float]:
        all_counts = np.concatenate([count[k].astype(float).ravel() for k in keys]) if keys else np.zeros(0, dtype=float)
        peak = float(np.max(all_counts)) if all_counts.size else 0.0
        if weight_by_span:
            total = float(sum(np.sum(count[k] * wgt[k]) for k in keys))
        else:
            total = float(sum(np.sum(count[k]) for k in keys))
        tube_len_total = float(np.sum(len_cur)) if nt else 0.0
        if objective_mode == "total":
            return (0.0, total, tube_len_total)
        if objective_mode == "peak_then_total":
            return (peak, total, tube_len_total)
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    def _evaluate():
        r = _build_rovings()
        t = _build_tubes()
        out = wing_span_analysis(rovings=r, tubes=t, **analysis_base)
        fos = float(out["factor_of_safety"]["governing_min"])
        return r, t, out, fos

    cur_rov, cur_tubes, cur_out, cur_fos = _evaluate()
    history = [{"iter": 0, "objective": _objective(), "fos": cur_fos, "phase": "start"}]
    if cur_fos < min_fos:
        if not allow_increase_to_feasible:
            raise ValueError(f"Starting design is already below min_fos ({cur_fos:.4g} < {min_fos:.4g}).")
        for itf in range(1, int(max_feas_iter) + 1):
            best = None
            best_fos = cur_fos
            best_obj = (float("inf"), float("inf"), float("inf"))
            best_counts = None
            best_lens = None
            best_out = None
            best_rov = None
            best_tubes = None
            for k in keys:
                for j in range(count[k].size):
                    count[k][j] += step
                    cand_rov, cand_tubes, cand_out, cand_fos = _evaluate()
                    cand_obj = _objective()
                    improved = cand_fos > best_fos + 1e-12 or (abs(cand_fos - best_fos) <= 1e-12 and cand_obj < best_obj)
                    if improved:
                        best = (k, j)
                        best_fos = cand_fos
                        best_obj = cand_obj
                        best_counts = {kk: vv.copy() for kk, vv in count.items()}
                        best_lens = len_cur.copy()
                        best_out = cand_out
                        best_rov = cand_rov
                        best_tubes = cand_tubes
                    count[k][j] -= step
            if optimize_tube_lengths and nt:
                for j in range(nt):
                    if len_cur[j] + tube_length_step <= len_max[j] + EPS:
                        len_cur[j] += tube_length_step
                        cand_rov, cand_tubes, cand_out, cand_fos = _evaluate()
                        cand_obj = _objective()
                        improved = cand_fos > best_fos + 1e-12 or (abs(cand_fos - best_fos) <= 1e-12 and cand_obj < best_obj)
                        if improved:
                            best = ("tube_len", j)
                            best_fos = cand_fos
                            best_obj = cand_obj
                            best_counts = {kk: vv.copy() for kk, vv in count.items()}
                            best_lens = len_cur.copy()
                            best_out = cand_out
                            best_rov = cand_rov
                            best_tubes = cand_tubes
                        len_cur[j] -= tube_length_step
            if best is None or best_fos <= cur_fos + 1e-12:
                break
            for kk in keys:
                count[kk] = best_counts[kk]
            if best_lens is not None:
                len_cur[:] = best_lens
            cur_out = best_out
            cur_rov = best_rov
            cur_tubes = best_tubes
            cur_fos = best_fos
            history.append({"iter": itf, "objective": best_obj, "fos": cur_fos, "last_change": best, "phase": "feasibility"})
            if verbose:
                print(f"[opt-feas] iter={itf} obj={best_obj} fos={cur_fos:.4f} changed={best}")
            if cur_fos >= min_fos:
                break
        if cur_fos < min_fos:
            raise ValueError(f"Could not reach min_fos={min_fos:.4g}; best FOS={cur_fos:.4g}.")

    for it in range(1, int(max_iter) + 1):
        cur_obj = _objective()
        best = None
        best_obj = cur_obj
        best_fos = cur_fos
        best_counts = None
        best_lens = None
        best_out = None
        best_rov = None
        best_tubes = None

        for k in keys:
            for j in range(count[k].size):
                if count[k][j] < step:
                    continue
                count[k][j] -= step
                cand_rov, cand_tubes, cand_out, cand_fos = _evaluate()
                cand_obj = _objective()
                feasible = cand_fos >= min_fos
                better = feasible and (cand_obj < best_obj or (cand_obj == best_obj and cand_fos > best_fos))
                if better:
                    best = (k, j)
                    best_obj = cand_obj
                    best_fos = cand_fos
                    best_counts = {kk: vv.copy() for kk, vv in count.items()}
                    best_lens = len_cur.copy()
                    best_out = cand_out
                    best_rov = cand_rov
                    best_tubes = cand_tubes
                count[k][j] += step
        if optimize_tube_lengths and nt:
            for j in range(nt):
                if len_cur[j] - tube_length_step < len_min[j] - EPS:
                    continue
                len_cur[j] -= tube_length_step
                cand_rov, cand_tubes, cand_out, cand_fos = _evaluate()
                cand_obj = _objective()
                feasible = cand_fos >= min_fos
                better = feasible and (cand_obj < best_obj or (cand_obj == best_obj and cand_fos > best_fos))
                if better:
                    best = ("tube_len", j)
                    best_obj = cand_obj
                    best_fos = cand_fos
                    best_counts = {kk: vv.copy() for kk, vv in count.items()}
                    best_lens = len_cur.copy()
                    best_out = cand_out
                    best_rov = cand_rov
                    best_tubes = cand_tubes
                len_cur[j] += tube_length_step

        if best is None:
            break
        for kk in keys:
            count[kk] = best_counts[kk]
        if best_lens is not None:
            len_cur[:] = best_lens
        cur_out = best_out
        cur_rov = best_rov
        cur_tubes = best_tubes
        cur_fos = best_fos
        history.append({"iter": it, "objective": best_obj, "fos": cur_fos, "last_change": best, "phase": "minimize"})
        if verbose:
            print(f"[opt] iter={it} objective={best_obj} fos={cur_fos:.4f} changed={best}")

    cur_out["optimized_tubes"] = deepcopy(cur_tubes)
    cur_out["optimized_rovings"] = deepcopy(cur_rov)
    return cur_rov, cur_out, history


def run_dae21_example(plot: bool = True):
    # ---------- Replace these first ----------
    CHORD_IN = 10.25
    SPAN_IN = 15.0 * 12.0
    ACC =-3
    TOTAL_LIFT_LBF = 50*ACC*1.5
    LIFT_HALF_LBF = 0.5 * TOTAL_LIFT_LBF
    AIRFOIL_DAT = Path(__file__).with_name("coord_seligFmt") / "psu94097.dat"

    SKIN_THICKNESS_IN = 0.02
    E_FACE_PSI = 2.70e6
    SIGMA_Y_FACE_PSI = 7.0e4

    E_CORE_PSI = 200
    G_CORE_PSI = 300
    SIGMA_Y_CORE_PSI = 2.7e1
    K_WR = 0.76

    E_TUBE1_PSI = 17e6
    E_TUBE2_PSI = 17e6
    SIGMA_Y_TUBE1_PSI = 200e6
    SIGMA_Y_TUBE2_PSI = 200e6
    TUBE_LENGTH_IN = 36

    X_QC = 0.25
    X_AFT = 2.0 / 3.0
    X_MID = 0.5 * (X_QC + X_AFT)
    X_AC_FRAC = 0.25
    X_SC_FRAC = 0.30  # quick estimate; set to X_QC for near-zero torsional moment
    TUBE1_OD_IN = 0.878000
    TUBE1_ID_IN = 0.625
    TUBE2_OD_IN = 0.441000
    TUBE2_ID_IN = 0.313000
    NU_FACE = 0.30
    NU_TUBE1 = 0.30
    NU_TUBE2 = 0.30
    TORSION_DIST_LBF = 0.0  # additional distributed torsion [lbf]

    ROVING_AREA_ONE_IN2 = 70e-5 * 25/12
    ROVING_E_TENSION_PSI = 33.5e6
    ROVING_E_COMPRESSION_PSI = (33.5e6)/1.7
    ROVING_SIGMA_TENSION_PSI = 545e3
    ROVING_SIGMA_COMPRESSION_PSI = 300e3
    ROVING_LINEAR_DENSITY_G_PER_M = 1.654
    ROVING_COUNT_SEGMENTS_L1_TENSION = [21.0, 5.0, 0.0]  # lane1: LE->tube1
    ROVING_COUNT_SEGMENTS_L1_COMPRESSION = [21.0, 6.0, 0.0]
    ROVING_COUNT_SEGMENTS_L2_TENSION = [16.0, 0.0, 0.0]  # lane2: tube1->tube2
    ROVING_COUNT_SEGMENTS_L2_COMPRESSION = [6.0, 0.0, 0.0]
    ROVING_COUNT_SEGMENTS_L3_TENSION = [6.0, 5.0, 0.0]   # lane3: tube2->tube3
    ROVING_COUNT_SEGMENTS_L3_COMPRESSION = [6.0, 5.0, 0.0]
    ROVING_COUNT_SEGMENTS_L4_TENSION = [6.0, 5.0, 0.0]   # lane4: tube3->TE
    ROVING_COUNT_SEGMENTS_L4_COMPRESSION = [6.0, 5.0, 0.0]
    INCLUDE_MIDDLE_ROVINGS = True  # True -> one lane in every tube/edge gap
    RUN_ROVING_OPTIMIZER = True
    OPTIMIZER_MIN_FOS = 1.01
    OPTIMIZER_STEP = 1
    OPTIMIZER_MAX_ITER = 2000
    OPTIMIZER_N_SPAN = 121  # use lower resolution for faster optimization search
    OPTIMIZER_OBJECTIVE_MODE = "peak_then_total"  # "peak_then_total" or "total"
    OPTIMIZE_TUBE_LENGTHS = True
    OPTIMIZER_TUBE_LENGTH_STEP_IN = 0.5
    OPTIMIZER_TUBE_LENGTH_MIN_IN = 0.0
    OPTIMIZER_TUBE_LENGTH_MAX_IN = (72.0 -1)/2
    # Optional explicit breaks format (more flexible):
    # ROVING_BREAKS_IN = [0.0, SPAN_IN / 6.0, SPAN_IN / 3.0, SPAN_IN / 2.0]
    # Use with: "count_span_breaks_in": ROVING_BREAKS_IN, "count_span_values": [60, 40, 20]
    # ---------------------------------------

    xaf, yaf = _read_airfoil_xy(AIRFOIL_DAT, None, None)
    yq_t, yq_b = _y_bounds_at_x(xaf, yaf, X_QC)
    ym_t, ym_b = _y_bounds_at_x(xaf, yaf, X_MID)
    ya_t, ya_b = _y_bounds_at_x(xaf, yaf, X_AFT)
    tubes = [
        {"x": X_QC, "y": 0.5 * (yq_t + yq_b), "od": TUBE1_OD_IN, "id": TUBE1_ID_IN, "E": E_TUBE1_PSI, "sigma_y": SIGMA_Y_TUBE1_PSI, "nu": NU_TUBE1, "length_from_root": TUBE_LENGTH_IN},
        {"x": X_MID, "y": 0.5 * (ym_t + ym_b), "od": TUBE1_OD_IN, "id": TUBE1_ID_IN, "E": E_TUBE1_PSI, "sigma_y": SIGMA_Y_TUBE1_PSI, "nu": NU_TUBE1, "length_from_root": TUBE_LENGTH_IN},
        {"x": X_AFT, "y": 0.5 * (ya_t + ya_b), "od": TUBE2_OD_IN, "id": TUBE2_ID_IN, "E": E_TUBE2_PSI, "sigma_y": SIGMA_Y_TUBE2_PSI, "nu": NU_TUBE2, "length_from_root": TUBE_LENGTH_IN},
    ]
    rovings = {
        "lane1_tension": {
            "count_segments": ROVING_COUNT_SEGMENTS_L1_TENSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        },
        "lane1_compression": {
            "count_segments": ROVING_COUNT_SEGMENTS_L1_COMPRESSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        },
        "lane2_tension": {
            "count_segments": ROVING_COUNT_SEGMENTS_L2_TENSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        },
        "lane2_compression": {
            "count_segments": ROVING_COUNT_SEGMENTS_L2_COMPRESSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        },
        "lane3_tension": {
            "count_segments": ROVING_COUNT_SEGMENTS_L3_TENSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        },
        "lane3_compression": {
            "count_segments": ROVING_COUNT_SEGMENTS_L3_COMPRESSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        },
    }
    if INCLUDE_MIDDLE_ROVINGS:
        rovings["lane4_tension"] = {
            "count_segments": ROVING_COUNT_SEGMENTS_L4_TENSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        }
        rovings["lane4_compression"] = {
            "count_segments": ROVING_COUNT_SEGMENTS_L4_COMPRESSION,
            "area": ROVING_AREA_ONE_IN2,
            "E_tension": ROVING_E_TENSION_PSI,
            "E_compression": ROVING_E_COMPRESSION_PSI,
            "sigma_tension": ROVING_SIGMA_TENSION_PSI,
            "sigma_compression": ROVING_SIGMA_COMPRESSION_PSI,
        }

    analysis_kwargs = dict(
        span=SPAN_IN,
        root_chord=CHORD_IN,
        tip_chord=CHORD_IN,
        lift_half=LIFT_HALF_LBF,
        skin_thickness=SKIN_THICKNESS_IN,
        Eface=E_FACE_PSI,
        Ecore=E_CORE_PSI,
        Gcore=G_CORE_PSI,
        k_wr=K_WR,
        tubes=tubes,
        include_middle_rovings=INCLUDE_MIDDLE_ROVINGS,
        sigma_y_face=SIGMA_Y_FACE_PSI,
        sigma_y_core=SIGMA_Y_CORE_PSI,
        x_ac_frac=X_AC_FRAC,
        x_sc_frac=X_SC_FRAC,
        torsion_dist=TORSION_DIST_LBF,
        nu_face=NU_FACE,
        nu_tube=NU_TUBE1,
        n_span=241,
        dat_path=AIRFOIL_DAT,
    )
    if RUN_ROVING_OPTIMIZER:
        try:
            analysis_kwargs_opt = dict(analysis_kwargs)
            analysis_kwargs_opt["n_span"] = int(min(max(21, OPTIMIZER_N_SPAN), analysis_kwargs["n_span"]))
            rovings, out, opt_hist = optimize_roving_counts(
                analysis_kwargs=analysis_kwargs_opt,
                rovings=rovings,
                min_fos=OPTIMIZER_MIN_FOS,
                step=OPTIMIZER_STEP,
                max_iter=OPTIMIZER_MAX_ITER,
                weight_by_span=False,
                objective_mode=OPTIMIZER_OBJECTIVE_MODE,
                optimize_tube_lengths=OPTIMIZE_TUBE_LENGTHS,
                tube_length_step=OPTIMIZER_TUBE_LENGTH_STEP_IN,
                tube_length_min=OPTIMIZER_TUBE_LENGTH_MIN_IN,
                tube_length_max=OPTIMIZER_TUBE_LENGTH_MAX_IN,
                allow_increase_to_feasible=True,
                max_feas_iter=OPTIMIZER_MAX_ITER,
                verbose=False,
            )
            opt_tubes = out.get("optimized_tubes", analysis_kwargs["tubes"])
            out = wing_span_analysis(rovings=rovings, tubes=opt_tubes, **{k: v for k, v in analysis_kwargs.items() if k != "tubes"})
            if float(out["factor_of_safety"]["governing_min"]) < OPTIMIZER_MIN_FOS:
                rovings, out, _ = optimize_roving_counts(
                    analysis_kwargs={**analysis_kwargs, "tubes": opt_tubes},
                    rovings=rovings,
                    min_fos=OPTIMIZER_MIN_FOS,
                    step=OPTIMIZER_STEP,
                    max_iter=0,
                    weight_by_span=False,
                    objective_mode=OPTIMIZER_OBJECTIVE_MODE,
                    optimize_tube_lengths=OPTIMIZE_TUBE_LENGTHS,
                    tube_length_step=OPTIMIZER_TUBE_LENGTH_STEP_IN,
                    tube_length_min=OPTIMIZER_TUBE_LENGTH_MIN_IN,
                    tube_length_max=OPTIMIZER_TUBE_LENGTH_MAX_IN,
                    allow_increase_to_feasible=True,
                    max_feas_iter=OPTIMIZER_MAX_ITER,
                    verbose=False,
                )
                opt_tubes = out.get("optimized_tubes", opt_tubes)
                out = wing_span_analysis(rovings=rovings, tubes=opt_tubes, **{k: v for k, v in analysis_kwargs.items() if k != "tubes"})
            print(f"Roving optimizer iterations: {len(opt_hist) - 1}")
            lane_names = [str(nm) for nm in np.asarray(out.get("yield_index", {}).get("roving_names", []), dtype=object).ravel()]
            if not lane_names:
                lane_names = sorted(rovings.keys())
            for nm in lane_names:
                cfg = rovings.get(nm, {})
                if "count_segments" in cfg:
                    print(f"  {nm} count_segments = {cfg['count_segments']}")
                elif "count_span_values" in cfg:
                    print(f"  {nm} count_span_values = {cfg['count_span_values']}")
                else:
                    print(f"  {nm} count = {cfg.get('count', 0)}")
            if OPTIMIZE_TUBE_LENGTHS:
                lens = [float(t.get("length_from_root", t.get("length", 0.0))) for t in opt_tubes]
                print(f"  tube lengths from root [in] = {lens}")
        except ValueError as e:
            print(f"Roving optimizer skipped: {e}")
            out = wing_span_analysis(rovings=rovings, **analysis_kwargs)
    else:
        out = wing_span_analysis(rovings=rovings, **analysis_kwargs)

    print(f"Root moment Mx = {out['Mx'][0]:.6g} lbf*in")
    print(f"Max yield index = {out['yield_index']['max_value']:.6g} at y = {out['yield_index']['max_y']:.3f} in, x = {out['yield_index']['max_x']:.3f} in ({out['yield_index']['max_mode']})")
    print(f"Max wrinkling index = {out['wrinkling']['max_value']:.6g} at y = {out['wrinkling']['max_y']:.3f} in, x = {out['wrinkling']['max_x']:.3f} in")
    print(f"FOS(yield) min = {np.nanmin(out['factor_of_safety']['yield']):.6g}")
    print(f"FOS(wrinkling) min = {np.nanmin(out['factor_of_safety']['wrinkling']):.6g}")
    print(f"FOS(governing) min = {out['factor_of_safety']['governing_min']:.6g} at y = {out['factor_of_safety']['governing_min_y']:.3f} in, x = {out['factor_of_safety']['governing_min_x']:.3f} in ({out['factor_of_safety']['governing_mode']})")
    print(f"Tip twist = {out['beam']['twist_deg'][-1]:.6g} deg")
    rm = roving_mass_from_output(out, linear_density_g_per_m=ROVING_LINEAR_DENSITY_G_PER_M, both_wings=True)
    print(f"Roving mass total = {rm['total_g']:.6g} g ({rm['total_lb']:.6g} lb)")
    print(f"Pass(yield)={out['pass']['yield']} | Pass(wrinkling)={out['pass']['wrinkling']}")

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        plot_section_layout(
            chord=CHORD_IN,
            skin_thickness=SKIN_THICKNESS_IN,
            tubes=tubes,
            rovings=rovings,
            include_middle_rovings=INCLUDE_MIDDLE_ROVINGS,
            span_station=0.0,
            dat_path=AIRFOIL_DAT,
            ax=axs[0],
        )
        plot_section_layout(
            chord=CHORD_IN,
            skin_thickness=SKIN_THICKNESS_IN,
            tubes=tubes,
            rovings=rovings,
            include_middle_rovings=INCLUDE_MIDDLE_ROVINGS,
            span_station=min(0.5 * SPAN_IN, TUBE_LENGTH_IN + 1.0),
            dat_path=AIRFOIL_DAT,
            ax=axs[1],
        )
        axs[0].set_title("Section @ Root (y=0 in)")
        axs[1].set_title("Section @ Outboard Station")
        plt.tight_layout()
        plot_span_results(out)
        plt.show()

    return out


if __name__ == "__main__":
    run_dae21_example(plot=True)
