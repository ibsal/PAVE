"""
stl_to_vsp.py (v5 - loop-based slicer + selectable end caps)
------------------------------------------------------------
Fixes:
  - Slice boundary is built from intersection SEGMENTS, stitched into a loop (not angle-sorted points)
  - Optional flat end(s) instead of forced point caps
  - PCHIP interpolation to avoid spline overshoot ripples

Dependencies:
  pip install numpy scipy

Usage:
  python stl_to_vsp.py fuselage.stl output.vsp3 --axis X --slices 35 --tail_cap flat
"""

import argparse
import math
import random
import string
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator


# ── ID generator (10 uppercase alpha chars, matches VSP3 format) ──────────────

def _uid() -> str:
    return ''.join(random.choices(string.ascii_uppercase, k=10))


def _sci(v: float) -> str:
    return f"{v:.18e}"


# ── STL loader ────────────────────────────────────────────────────────────────

def load_stl(path: str) -> np.ndarray:
    """
    Returns triangles as (N, 3, 3) float64 array.
    Supports binary STL and ASCII STL.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STL not found: {path}")

    is_binary = False
    with open(path, "rb") as f:
        f.read(80)
        ntri_bytes = f.read(4)
        if len(ntri_bytes) == 4:
            n_tri = struct.unpack("<I", ntri_bytes)[0]
            expected = 80 + 4 + n_tri * 50
            is_binary = abs(path.stat().st_size - expected) < 10

    if is_binary:
        tris = np.empty((n_tri, 3, 3), dtype=np.float64)
        with open(path, "rb") as f:
            f.seek(84)
            for i in range(n_tri):
                f.read(12)  # normal
                v0 = struct.unpack("<fff", f.read(12))
                v1 = struct.unpack("<fff", f.read(12))
                v2 = struct.unpack("<fff", f.read(12))
                tris[i, 0, :] = v0
                tris[i, 1, :] = v1
                tris[i, 2, :] = v2
                f.read(2)   # attr
        print(f"  Loaded binary STL: {n_tri:,} triangles")
        return tris

    tris_list, cur = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                p = line.split()
                cur.append([float(p[1]), float(p[2]), float(p[3])])
                if len(cur) == 3:
                    tris_list.append(cur)
                    cur = []
    print(f"  Loaded ASCII STL: {len(tris_list):,} triangles")
    return np.array(tris_list, dtype=np.float64)


# ── Robust slice boundary: segments → loop(s) → largest loop → resample ───────

def _quantize(p: np.ndarray, eps: float) -> tuple[int, int]:
    return (int(round(p[0] / eps)), int(round(p[1] / eps)))


def slice_at(tris: np.ndarray, pos: float, axis: int = 0,
             n_resample: int = 256,
             eps_scale: float = 1e-6) -> np.ndarray:
    """
    Slice triangles at pos along axis.
    Returns ordered resampled boundary points (N,2).

    Key: build intersection segments and stitch into closed loop(s).
    """
    ax_b = (axis + 1) % 3
    ax_c = (axis + 2) % 3

    # Estimate scale for quantization eps
    verts = tris.reshape(-1, 3)
    span_b = float(verts[:, ax_b].max() - verts[:, ax_b].min())
    span_c = float(verts[:, ax_c].max() - verts[:, ax_c].min())
    scale = max(span_b, span_c, 1.0)
    eps = eps_scale * scale

    segments = []

    for tri in tris:
        vals = tri[:, axis]
        vmin = vals.min()
        vmax = vals.max()
        if pos < vmin or pos > vmax:
            continue

        ip = []
        for i in range(3):
            j = (i + 1) % 3
            a, b = vals[i], vals[j]
            if (a < pos and b > pos) or (a > pos and b < pos):
                t = (pos - a) / (b - a)
                p = tri[i] + t * (tri[j] - tri[i])
                ip.append(np.array([p[ax_b], p[ax_c]], dtype=np.float64))

        if len(ip) == 2:
            segments.append((ip[0], ip[1]))

    if len(segments) < 8:
        return np.array([])

    # Build node map via quantization
    node_coords = []
    node_index = {}
    edges = []

    def get_node(pt: np.ndarray) -> int:
        key = _quantize(pt, eps)
        if key in node_index:
            return node_index[key]
        idx = len(node_coords)
        node_index[key] = idx
        node_coords.append(pt)
        return idx

    for a, b in segments:
        ia = get_node(a)
        ib = get_node(b)
        if ia != ib:
            edges.append((ia, ib))

    if len(edges) < 8:
        return np.array([])

    # adjacency
    adj = [[] for _ in range(len(node_coords))]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Walk loops: start from unused edge and follow until closed
    used = set()

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    loops = []

    for u, v in edges:
        ek = edge_key(u, v)
        if ek in used:
            continue

        # start chain
        chain = [u, v]
        used.add(ek)

        # extend forward
        while True:
            cur = chain[-1]
            prev = chain[-2]
            nxts = [w for w in adj[cur] if w != prev]
            if not nxts:
                break

            # choose next that continues loop; if multiple, pick the one that makes smallest turn
            if len(nxts) == 1:
                nxt = nxts[0]
            else:
                p_prev = node_coords[prev]
                p_cur = node_coords[cur]
                v0 = p_cur - p_prev
                v0n = v0 / (np.linalg.norm(v0) + 1e-12)
                best = None
                best_score = -1e9
                for w in nxts:
                    pw = node_coords[w]
                    v1 = pw - p_cur
                    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
                    score = float(np.dot(v0n, v1n))  # prefer straighter continuation
                    if score > best_score:
                        best_score = score
                        best = w
                nxt = best

            ek2 = edge_key(cur, nxt)
            if ek2 in used:
                # if we closed the loop, stop
                if nxt == chain[0]:
                    chain.append(nxt)
                break

            chain.append(nxt)
            used.add(ek2)

            # closed?
            if chain[-1] == chain[0]:
                break

        if len(chain) >= 6 and chain[0] == chain[-1]:
            loops.append(chain[:-1])  # drop duplicate close node

    if not loops:
        return np.array([])

    # Pick largest-area loop (outer boundary)
    def poly_area(idx_list):
        pts = np.array([node_coords[i] for i in idx_list], dtype=np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    areas = [abs(poly_area(L)) for L in loops]
    loop = loops[int(np.argmax(areas))]

    boundary = np.array([node_coords[i] for i in loop], dtype=np.float64)
    if len(boundary) < 8:
        return np.array([])

    # Ensure consistent order (CCW)
    if poly_area(loop) < 0:
        boundary = boundary[::-1]

    # Resample by arc length
    b = np.vstack([boundary, boundary[0]])
    seg = np.linalg.norm(np.diff(b, axis=0), axis=1)
    L = float(np.sum(seg))
    if L < 1e-12:
        return boundary

    s = np.concatenate([[0.0], np.cumsum(seg)])
    t_even = np.linspace(0.0, L, n_resample, endpoint=False)
    y = np.interp(t_even, s, b[:, 0])
    z = np.interp(t_even, s, b[:, 1])
    return np.column_stack([y, z])


# ── Rounded rectangle fitting ─────────────────────────────────────────────────

def fit_rounded_rect(pts: np.ndarray):
    """
    Fit a rounded rectangle to boundary points.
    Returns (W_full, H_full, r_full, yc, zc).
    """
    if len(pts) < 12:
        return 0.02, 0.02, 0.002, 0.0, 0.0

    yc = float(np.mean(pts[:, 0]))
    zc = float(np.mean(pts[:, 1]))
    W0 = max(float(np.max(pts[:, 0]) - np.min(pts[:, 0])), 1e-6)
    H0 = max(float(np.max(pts[:, 1]) - np.min(pts[:, 1])), 1e-6)
    r0 = min(W0, H0) * 0.2

    def sdf_rr(y, z, W, H, r):
        r = min(r, 0.5 * W - 1e-9, 0.5 * H - 1e-9)
        qy = np.abs(y) - 0.5 * W + r
        qz = np.abs(z) - 0.5 * H + r
        return (np.sqrt(np.maximum(qy, 0)**2 + np.maximum(qz, 0)**2)
                + np.minimum(np.maximum(qy, qz), 0) - r)

    def residual(p):
        W, H, r, y0, z0 = p
        if W <= 0 or H <= 0 or r <= 0:
            return 1e18
        r = min(r, 0.5 * W - 1e-9, 0.5 * H - 1e-9)
        y = pts[:, 0] - y0
        z = pts[:, 1] - z0
        d = sdf_rr(y, z, W, H, r)
        return float(np.sum(d * d))

    res = minimize(
        residual,
        [W0, H0, r0, yc, zc],
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None), (1e-6, None), (None, None), (None, None)],
        options={"maxiter": 800, "ftol": 1e-14},
    )

    W, H, r, y0, z0 = res.x
    r = min(float(r), 0.5 * float(W) - 1e-9, 0.5 * float(H) - 1e-9)
    return float(W), float(H), float(r), float(y0), float(z0)


# ── VSP3 writer bits (kept close to your structure) ───────────────────────────

# Tangency/continuity defaults: disable constraint flags so VSP lofting does not
# force blended transitions between neighboring stations.
_XSEC_ANGLE_PARMS = [
    "AftCluster",
    "FwdCluster",
    "SectTess_U",
    "RLSym",
    "TBSym",
    "AllSym",
    "ContinuityBottom",
    "ContinuityTop",
    "ContinuityLeft",
    "ContinuityRight",
    "TopLAngle",
    "TopLAngleSet",
    "TopRAngle",
    "TopRAngleSet",
    "BottomLAngle",
    "BottomLAngleSet",
    "BottomRAngle",
    "BottomRAngleSet",
    "TopLSlew",
    "TopLSlewSet",
    "TopRSlew",
    "TopRSlewSet",
    "BottomLSlew",
    "BottomLSlewSet",
    "BottomRSlew",
    "BottomRSlewSet",
    "TopLStrength",
    "TopLStrengthSet",
    "TopRStrength",
    "TopRStrengthSet",
    "BottomLStrength",
    "BottomLStrengthSet",
    "BottomRStrength",
    "BottomRStrengthSet",
    "TopLRAngleEq",
    "TopLRSlewEq",
    "TopLRStrengthEq",
    "BottomLRAngleEq",
    "BottomLRSlewEq",
    "BottomLRStrengthEq",
    "LeftLRAngleEq",
    "LeftLRSlewEq",
    "LeftLRStrengthEq",
    "RightLRAngleEq",
    "RightLRSlewEq",
    "RightLRStrengthEq",
]

_XSEC_DEFAULTS = {
    "AftCluster": 1.0,
    "FwdCluster": 1.0,
    "SectTess_U": 12.0,
    "RLSym": 1.0,
    "TBSym": 1.0,
    "AllSym": 0.0,
    "ContinuityBottom": 0.0,
    "ContinuityTop": 0.0,
    "ContinuityLeft": 0.0,
    "ContinuityRight": 0.0,
}

_CAP_TRIM_CHEVRON = """                  <Cap>
                    <LE_Cap_Length Value="{one}" ID="{id1}"/>
                    <LE_Cap_Offset Value="{zero}" ID="{id2}"/>
                    <LE_Cap_Strength Value="5.000000000000000000e-01" ID="{id3}"/>
                    <LE_Cap_Type Value="{one}" ID="{id4}"/>
                    <TE_Cap_Length Value="{one}" ID="{id5}"/>
                    <TE_Cap_Offset Value="{zero}" ID="{id6}"/>
                    <TE_Cap_Strength Value="5.000000000000000000e-01" ID="{id7}"/>
                    <TE_Cap_Type Value="{one}" ID="{id8}"/>
                  </Cap>
                  <Chevron>
                    <AllSym Value="{one}" ID="{id9}"/>
                    <BottomAmplitude Value="{one}" ID="{id10}"/>
                    <Bottom_Angle Value="{zero}" ID="{id11}"/>
                    <Bottom_Slew Value="{zero}" ID="{id12}"/>
                    <Chevron_Type Value="{zero}" ID="{id13}"/>
                    <LeftAmplitude Value="{one}" ID="{id14}"/>
                    <Left_Angle Value="{zero}" ID="{id15}"/>
                    <Left_Slew Value="{zero}" ID="{id16}"/>
                    <Number Value="8.000000000000000000e+00" ID="{id17}"/>
                    <Off_Duty Value="{zero}" ID="{id18}"/>
                    <On_Duty Value="{zero}" ID="{id19}"/>
                    <Peak_Radius Value="{zero}" ID="{id20}"/>
                    <RLSym Value="{one}" ID="{id21}"/>
                    <RightAmplitude Value="{one}" ID="{id22}"/>
                    <Right_Angle Value="{zero}" ID="{id23}"/>
                    <Right_Slew Value="{zero}" ID="{id24}"/>
                    <TBSym Value="{one}" ID="{id25}"/>
                    <TopAmplitude Value="{one}" ID="{id26}"/>
                    <Top_Angle Value="{zero}" ID="{id27}"/>
                    <Top_Slew Value="{zero}" ID="{id28}"/>
                    <Valley_Radius Value="{zero}" ID="{id29}"/>
                    <W01_Center Value="2.500000000000000000e-01" ID="{id30}"/>
                    <W01_Center_Guide Value="5.000000000000000000e+00" ID="{id31}"/>
                    <W01_End Value="5.000000000000000000e-01" ID="{id32}"/>
                    <W01_End_Guide Value="5.000000000000000000e+00" ID="{id33}"/>
                    <W01_Mode Value="{zero}" ID="{id34}"/>
                    <W01_Start Value="{zero}" ID="{id35}"/>
                    <W01_Start_Guide Value="5.000000000000000000e+00" ID="{id36}"/>
                    <W01_Width Value="5.000000000000000000e-01" ID="{id37}"/>
                  </Chevron>
                  <Close>
                    <LE_Close_AbsRel Value="{zero}" ID="{id38}"/>
                    <LE_Close_Thick Value="{zero}" ID="{id39}"/>
                    <LE_Close_Thick_Chord Value="{zero}" ID="{id40}"/>
                    <LE_Close_Type Value="{zero}" ID="{id41}"/>
                    <TE_Close_AbsRel Value="{zero}" ID="{id42}"/>
                    <TE_Close_Thick Value="{zero}" ID="{id43}"/>
                    <TE_Close_Thick_Chord Value="{zero}" ID="{id44}"/>
                    <TE_Close_Type Value="{zero}" ID="{id45}"/>
                  </Close>
                  <Trim>
                    <LE_Trim_AbsRel Value="{zero}" ID="{id46}"/>
                    <LE_Trim_Thick Value="{zero}" ID="{id47}"/>
                    <LE_Trim_Thick_Chord Value="{zero}" ID="{id48}"/>
                    <LE_Trim_Type Value="{zero}" ID="{id49}"/>
                    <LE_Trim_X Value="{zero}" ID="{id50}"/>
                    <LE_Trim_X_Chord Value="{zero}" ID="{id51}"/>
                    <TE_Trim_AbsRel Value="{zero}" ID="{id52}"/>
                    <TE_Trim_Thick Value="{zero}" ID="{id53}"/>
                    <TE_Trim_Thick_Chord Value="{zero}" ID="{id54}"/>
                    <TE_Trim_Type Value="{zero}" ID="{id55}"/>
                    <TE_Trim_X Value="{zero}" ID="{id56}"/>
                    <TE_Trim_X_Chord Value="{zero}" ID="{id57}"/>
                  </Trim>"""


def _cap_trim_chevron() -> str:
    ids = {f"id{i}": _uid() for i in range(1, 58)}
    return _CAP_TRIM_CHEVRON.format(
        zero="0.000000000000000000e+00",
        one="1.000000000000000000e+00",
        **ids
    )


def _xsec_background() -> str:
    return f"""                  <XSecCurve_Background>
                    <XSecFlipImageFlag Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecImageFlag Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecImageH Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecImagePreserveAR Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecImageW Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecImageXOffset Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecImageYOffset Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <XSecLockImageFlag Value="0.000000000000000000e+00" ID="{_uid()}"/>
                  </XSecCurve_Background>"""


def build_xsec(x_pct: float, y_pct: float, z_pct: float,
               w_half: float, h_half: float, ref_length: float,
               is_point: bool = False, radius_full: float = 0.0) -> str:
    """
    Type 1 = point
    Type 4 = rounded rect
    """
    xsec_pc_id = _uid()
    curve_pc_id = _uid()

    angle_lines = [
        f'                <XLocPercent Value="{_sci(x_pct)}" ID="{_uid()}"/>',
        f'                <XRotate Value="0.000000000000000000e+00" ID="{_uid()}"/>',
        f'                <YLocPercent Value="{_sci(y_pct)}" ID="{_uid()}"/>',
        f'                <YRotate Value="0.000000000000000000e+00" ID="{_uid()}"/>',
        f'                <ZLocPercent Value="{_sci(z_pct)}" ID="{_uid()}"/>',
        f'                <ZRotate Value="0.000000000000000000e+00" ID="{_uid()}"/>',
        f'                <RefLength Value="{_sci(ref_length)}" ID="{_uid()}"/>',
    ]

    for parm in _XSEC_ANGLE_PARMS:
        if parm.endswith("Set") or parm.endswith("Eq"):
            val = 0.0
        else:
            val = _XSEC_DEFAULTS.get(parm, 0.0)
        angle_lines.append(
            f'                <{parm} Value="{_sci(val)}" ID="{_uid()}"/>'
        )

    angle_block = "\n".join(angle_lines)

    if is_point:
        curve_type = 1
        curve_params = f"""                  <XSecCurve>
                    <Area Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <DeltaX Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <DeltaY Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <HWRatio Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <Scale Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <ShiftLE Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <Theta Value="0.000000000000000000e+00" ID="{_uid()}"/>
                  </XSecCurve>"""
    else:
        curve_type = 4
        W = max(2.0 * w_half, 1e-9)
        H = max(2.0 * h_half, 1e-9)

        r = max(float(radius_full), 1e-9)
        r = min(r, 0.5 * W - 1e-9, 0.5 * H - 1e-9)

        hw = H / W if W > 0 else 1.0
        area = W * H - (4.0 - math.pi) * r * r

        curve_params = f"""                  <XSecCurve>
                    <Area Value="{_sci(area)}" ID="{_uid()}"/>
                    <DeltaX Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <DeltaY Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <HWRatio Value="{_sci(hw)}" ID="{_uid()}"/>
                    <RoundRectXSec_KeyCorner Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <RoundRectXSec_Radius Value="{_sci(r)}" ID="{_uid()}"/>
                    <RoundRectXSec_RadiusBL Value="{_sci(r)}" ID="{_uid()}"/>
                    <RoundRectXSec_RadiusBR Value="{_sci(r)}" ID="{_uid()}"/>
                    <RoundRectXSec_RadiusTL Value="{_sci(r)}" ID="{_uid()}"/>
                    <RoundRect_Keystone Value="5.000000000000000000e-01" ID="{_uid()}"/>
                    <RoundRect_Skew Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <RoundRect_VSkew Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <RoundedRect_Height Value="{_sci(H)}" ID="{_uid()}"/>
                    <RoundedRect_RadiusSymmetryType Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <RoundedRect_Width Value="{_sci(W)}" ID="{_uid()}"/>
                    <Scale Value="1.000000000000000000e+00" ID="{_uid()}"/>
                    <ShiftLE Value="0.000000000000000000e+00" ID="{_uid()}"/>
                    <Theta Value="0.000000000000000000e+00" ID="{_uid()}"/>
                  </XSecCurve>"""

    cap_trim = _cap_trim_chevron()
    bg = _xsec_background()

    return f"""          <XSec>
            <ParmContainer>
              <ID>{xsec_pc_id}</ID>
              <n>Default</n>
              <XSec>
{angle_block}
              </XSec>
            </ParmContainer>
            <XSec>
              <Type>0</Type>
              <GroupName>XSec</GroupName>
              <XSecCurve>
                <ParmContainer>
                  <ID>{curve_pc_id}</ID>
                  <n>Default</n>
{cap_trim}
{curve_params}
{bg}
                </ParmContainer>
                <XSecCurve>
                  <Type>{curve_type}</Type>
                  <XSecCurveDriverGroup>
                    <NumVar>4</NumVar>
                    <NumChoices>2</NumChoices>
                    <ChoiceVec>0, 2, </ChoiceVec>
                  </XSecCurveDriverGroup>
                </XSecCurve>
              </XSecCurve>
            </XSec>
          </XSec>"""


_POINT_CAP_BLEND = (
    (0.00, 0.02),
    (0.03, 0.12),
    (0.10, 0.26),
    (0.24, 0.46),
    (0.45, 0.68),
    (0.72, 0.86),
    (0.90, 0.96),
)


def _build_point_cap_xsecs(anchor_sec: dict, body_length: float,
                           at_start: bool) -> list[str]:
    if not anchor_sec or body_length <= 0.0:
        return []

    y_pct = float(anchor_sec["yc"]) / body_length
    z_pct = float(anchor_sec["zc"]) / body_length
    anchor_x = float(anchor_sec["x"])
    gap = max(anchor_x, 0.0) if at_start else max(body_length - anchor_x, 0.0)

    blend_specs = []
    if gap > 1e-9:
        for x_frac, size_frac in _POINT_CAP_BLEND:
            if at_start:
                x = gap * x_frac
            else:
                x = body_length - gap * x_frac

            w_half = max(float(anchor_sec["w_half"]) * size_frac, 1e-9)
            h_half = max(float(anchor_sec["h_half"]) * size_frac, 1e-9)
            r_full = max(float(anchor_sec["r_full"]) * size_frac, 1e-9)
            r_full = min(r_full, w_half - 1e-9, h_half - 1e-9)

            blend_specs.append((x, w_half, h_half, max(r_full, 1e-9)))

    blend_specs.sort(key=lambda item: item[0])

    blocks = []
    for x, w_half, h_half, r_full in blend_specs:
        blocks.append(build_xsec(
            x / body_length, y_pct, z_pct,
            w_half, h_half, body_length,
            is_point=False, radius_full=r_full
        ))

    return blocks


def write_vsp3(sections: list, body_length: float, out_path: str,
               name: str = "Fuselage",
               nose_cap: str = "point",
               tail_cap: str = "point"):
    """
    nose_cap / tail_cap: 'point' or 'flat'
    For 'flat', we DO NOT add a point XSec at that end.
    """
    geom_id = _uid()
    fuse_id = _uid()
    nose_end_cap_option = 0.0 if nose_cap == "point" else 1.0
    tail_end_cap_option = 0.0 if tail_cap == "point" else 1.0

    xsec_blocks = []

    if nose_cap == "point" and sections:
        xsec_blocks.extend(_build_point_cap_xsecs(sections[0], body_length, at_start=True))

    for sec in sections:
        x_pct = sec["x"] / body_length
        y_pct = sec["yc"] / body_length
        z_pct = sec["zc"] / body_length
        xsec_blocks.append(build_xsec(
            x_pct, y_pct, z_pct,
            sec["w_half"], sec["h_half"], body_length,
            is_point=False, radius_full=sec["r_full"]
        ))

    if tail_cap == "point" and sections:
        xsec_blocks.extend(_build_point_cap_xsecs(sections[-1], body_length, at_start=False))

    xsecs_xml = "\n".join(xsec_blocks)

    # Disable VSP's automatic point-like cap when we are building our own tapered
    # nose/tail sections. Otherwise it adds another closure and pinches the tip.
    vsp3 = f"""<?xml version="1.0"?>
<Vsp_Geometry>
  <Version>5</Version>
  <Vehicle>
    <ParmContainer>
      <ID>{_uid()}</ID>
      <n>Vehicle</n>
      <AdjustView>
        <CORX Value="0.000000000000000000e+00" ID="{_uid()}"/>
        <CORY Value="0.000000000000000000e+00" ID="{_uid()}"/>
        <CORZ Value="0.000000000000000000e+00" ID="{_uid()}"/>
        <PanX Value="0.000000000000000000e+00" ID="{_uid()}"/>
        <PanY Value="0.000000000000000000e+00" ID="{_uid()}"/>
        <RotationX Value="2.000000000000000000e+01" ID="{_uid()}"/>
        <RotationY Value="-3.000000000000000000e+01" ID="{_uid()}"/>
        <RotationZ Value="0.000000000000000000e+00" ID="{_uid()}"/>
        <Zoom Value="5.000000000000000000e-02" ID="{_uid()}"/>
      </AdjustView>
    </ParmContainer>
    <Measure>
      <Num_of_Rulers>0</Num_of_Rulers>
      <Num_of_Probes>0</Num_of_Probes>
      <Num_of_RSTprobes>0</Num_of_RSTprobes>
    </Measure>
    <Geom>
      <ParmContainer>
        <ID>{geom_id}</ID>
        <n>FuselageGeom</n>
        <Design>
          <Length Value="{_sci(body_length)}" ID="{_uid()}"/>
          <OrderPolicy Value="0.000000000000000000e+00" ID="{_uid()}"/>
        </Design>
        <EndCap>
          <CapUMaxLength Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <CapUMaxOffset Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <CapUMaxOption Value="{_sci(tail_end_cap_option)}" ID="{_uid()}"/>
          <CapUMaxStrength Value="5.000000000000000000e-01" ID="{_uid()}"/>
          <CapUMaxSweepFlag Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <CapUMinLength Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <CapUMinOffset Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <CapUMinOption Value="{_sci(nose_end_cap_option)}" ID="{_uid()}"/>
          <CapUMinStrength Value="5.000000000000000000e-01" ID="{_uid()}"/>
          <CapUMinSweepFlag Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <CapUMinTess Value="3.000000000000000000e+00" ID="{_uid()}"/>
        </EndCap>
        <Shape>
          <Tess_U Value="2.400000000000000000e+01" ID="{_uid()}"/>
          <Tess_W Value="2.600000000000000000e+01" ID="{_uid()}"/>
          <Wake Value="0.000000000000000000e+00" ID="{_uid()}"/>
        </Shape>
        <Sym>
          <Sym_Ancestor Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <Sym_Ancestor_Origin_Flag Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <Sym_Axial_Flag Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Sym_Planar_Flag Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Sym_Rot_N Value="2.000000000000000000e+00" ID="{_uid()}"/>
        </Sym>
        <XForm>
          <Abs_Or_Relitive_flag Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <Last_Scale Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <Origin Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Scale Value="1.000000000000000000e+00" ID="{_uid()}"/>
          <X_Location Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <X_Rel_Location Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <X_Rel_Rotation Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <X_Rotation Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Y_Location Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Y_Rel_Location Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Y_Rel_Rotation Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Y_Rotation Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Z_Location Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Z_Rel_Location Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Z_Rel_Rotation Value="0.000000000000000000e+00" ID="{_uid()}"/>
          <Z_Rotation Value="0.000000000000000000e+00" ID="{_uid()}"/>
        </XForm>
      </ParmContainer>
      <GeomBase>
        <TypeName>Fuselage</TypeName>
        <TypeID>4</TypeID>
        <TypeFixed>0</TypeFixed>
        <ParentID>NONE</ParentID>
        <Child_List/>
      </GeomBase>
      <Material>
        <n>Default</n>
      </Material>
      <Textures>
        <Num_of_Tex>0</Num_of_Tex>
      </Textures>
      <Geom>
        <Set_List>1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, </Set_List>
        <SubSurfaces/>
        <FeaStructures/>
      </Geom>
      <FuselageGeom>
        <ParmContainer>
          <ID>{fuse_id}</ID>
          <n>Default</n>
        </ParmContainer>
        <XSecSurf>
{xsecs_xml}
        </XSecSurf>
      </FuselageGeom>
    </Geom>
  </Vehicle>
</Vsp_Geometry>
"""
    with open(out_path, "w") as f:
        f.write(vsp3)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def convert(stl_path: str, out_path: str, n_slices: int = 30,
            axis: str = "X", name: str = "Fuselage",
            fit_stations: int | None = None,
            slice_margin: float = 0.01,
            out_margin: float = 0.005,
            nose_cap: str = "point",
            tail_cap: str = "point"):

    axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
    print(f"\n=== STL → OpenVSP Fuselage Converter (v5) ===\n")

    tris = load_stl(stl_path)
    verts = tris.reshape(-1, 3)
    lo = float(verts[:, axis_idx].min())
    hi = float(verts[:, axis_idx].max())
    length = hi - lo

    print(f"  Axis: {axis}  |  range: {lo:.4f} → {hi:.4f}  |  length: {length:.4f}")
    if length <= 0:
        print("ERROR: zero length along axis.")
        sys.exit(1)

    if fit_stations is None:
        n_fit = max(n_slices * 4, 90)
    else:
        n_fit = max(int(fit_stations), 30)

    m = length * slice_margin
    stations = np.linspace(lo + m, hi - m, n_fit)

    raw_x, raw_w, raw_h, raw_r, raw_yc, raw_zc = [], [], [], [], [], []

    for pos in stations:
        pts = slice_at(tris, pos, axis_idx, n_resample=256)
        if len(pts) < 12:
            continue
        W, H, r, yc, zc = fit_rounded_rect(pts)
        raw_x.append(pos - lo)
        raw_w.append(0.5 * W)   # half width
        raw_h.append(0.5 * H)   # half height
        raw_r.append(r)         # full-space radius
        raw_yc.append(yc)
        raw_zc.append(zc)

    if len(raw_x) < 8:
        print("ERROR: not enough valid slices. Try another axis or reduce margins.")
        sys.exit(1)

    raw_x = np.array(raw_x)
    order = np.argsort(raw_x)
    raw_x = raw_x[order]
    raw_w = np.array(raw_w)[order]
    raw_h = np.array(raw_h)[order]
    raw_r = np.array(raw_r)[order]
    raw_yc = np.array(raw_yc)[order]
    raw_zc = np.array(raw_zc)[order]

    # PCHIP interpolation (shape-preserving, no ringing)
    spl_w = PchipInterpolator(raw_x, raw_w)
    spl_h = PchipInterpolator(raw_x, raw_h)
    spl_r = PchipInterpolator(raw_x, raw_r)
    spl_yc = PchipInterpolator(raw_x, raw_yc)
    spl_zc = PchipInterpolator(raw_x, raw_zc)

    # Output stations: if end is flat, include station very close to the end.
    om = length * out_margin
    x0 = 0.0 + (om if nose_cap == "point" else 1e-6)
    x1 = length - (om if tail_cap == "point" else 1e-6)
    out_stations = np.linspace(x0, x1, n_slices)

    sections = []
    for x in out_stations:
        w = max(float(spl_w(x)), 1e-6)
        h = max(float(spl_h(x)), 1e-6)
        r = max(float(spl_r(x)), 1e-6)

        # radius is full-space, must be <= min(W,H)/2 = min(w,h)
        r = min(r, max(min(w, h) - 1e-9, 1e-6))

        yc = float(spl_yc(x))
        zc = float(spl_zc(x))

        sections.append({"x": x, "w_half": w, "h_half": h, "r_full": r, "yc": yc, "zc": zc})

    write_vsp3(sections, length, out_path, name=name, nose_cap=nose_cap, tail_cap=tail_cap)

    print(f"\n  ✓ Written: {out_path}")
    print(f"  ✓ Sections: {len(sections)} | nose_cap={nose_cap} tail_cap={tail_cap}\n")
    print("  Tip: If tail end is flat in STL, run with --tail_cap flat")
    print("       If nose is also flat, run with --nose_cap flat\n")


def main():
    parser = argparse.ArgumentParser(description="Convert fuselage STL to OpenVSP .vsp3")
    parser.add_argument("stl", help="Input STL")
    parser.add_argument("output", help="Output .vsp3")
    parser.add_argument("--slices", type=int, default=35)
    parser.add_argument("--axis", choices=["X", "Y", "Z"], default="X")
    parser.add_argument("--name", default="Fuselage")
    parser.add_argument("--fit_stations", type=int, default=None)
    parser.add_argument("--slice_margin", type=float, default=0.01)
    parser.add_argument("--out_margin", type=float, default=0.005)
    parser.add_argument("--nose_cap", choices=["point", "flat"], default="point")
    parser.add_argument("--tail_cap", choices=["point", "flat"], default="point")
    args = parser.parse_args()

    convert(
        args.stl, args.output,
        n_slices=args.slices,
        axis=args.axis,
        name=args.name,
        fit_stations=args.fit_stations,
        slice_margin=args.slice_margin,
        out_margin=args.out_margin,
        nose_cap=args.nose_cap,
        tail_cap=args.tail_cap,
    )


if __name__ == "__main__":
    main()
