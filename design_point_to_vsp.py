"""
Build a combined OpenVSP model from:
  - a fuselage STL (converted through stl_to_vsp.py)
  - wing/tail geometry cloned from a reference .vsp3 template

Edit the geometry values in the CONFIG block below directly.

Usage:
  python design_point_to_vsp.py fuselage.stl output.vsp3
"""

import argparse
import contextlib
import copy
import io
import math
import random
import string
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import stl_to_vsp


# ----------------------------- CONFIG ---------------------------------
# Enter your design dimensions in `design_units`. They will be converted into
# `stl_units` automatically before the wing/tail geometry is written.
CONFIG = {
    "template_path": str(Path.home() / "Downloads" / "Tico 5 OML.vsp3"),
    "design_units": "m",
    "stl_units": "mm",
    "fuselage_axis": "Y",
    "fuselage_slices": 5,
    "fuselage_nose_cap": "point",
    "fuselage_tail_cap": "flat",
    "fuselage_name": "MyFuselage",
    # Shift the STL-derived fuselage so its forward-most XSec sits at X = 0.
    "fuselage_align_nose_to_x_zero": True,
    # Additional fuselage X translation after nose alignment, in design_units.
    "fuselage_x_offset": 0.0,
    # With fuselage_axis="Y", many STLs end up rolled 90 deg in VSP because
    # the converter maps the remaining axes cyclically. This rotates it back.
    "fuselage_x_rotation_deg": 90.0,
    "fuselage_y_rotation_deg": 0.0,
    "fuselage_z_rotation_deg": 0.0,

    "main_wing_span": 4.5719554,
    "main_wing_chord": 0.3127835,
    "main_wing_xqc": 0.6278262272,
    "main_wing_incidence_deg": 1.83076884,
    "include_main_wing": True,
    "main_wing_y": 0.0,
    "main_wing_z": 0.0,
    "main_wing_sweep_deg": 0.0,
    "main_wing_dihedral_deg": 0.0,
    "lifting_surface_tess_u": 16.0,
    "lifting_surface_tess_w": 33.0,
    "lifting_surface_sect_tess_u": 6.0,
    # Optional Selig/Lednicer airfoil file (.dat/.af). Leave blank to keep the
    # airfoil already stored in the template geom.
    "main_wing_airfoil_dat": str(Path("PyFoil") / "coord_seligFmt" / "psu94097.dat"),

    "include_horizontal_tail": True,
    "horizontal_tail_span": 0.78371218,
    "horizontal_tail_chord": 0.1929609,
    "horizontal_tail_xqc": 1.72966979,
    "horizontal_tail_incidence_deg": 0.65283898,
    "horizontal_tail_y": 0.0,
    "horizontal_tail_z": 0.0,
    "horizontal_tail_sweep_deg": 0.0,
    "horizontal_tail_dihedral_deg": 0.0,
    "horizontal_tail_airfoil_dat": str(Path("PyFoil") / "coord_seligFmt" / "s9033.dat"),

    # "single" -> one center fin
    # "h_tail" -> two fins placed at the ends of the horizontal tail
    "include_vertical_tail": True,
    "vertical_tail_layout": "h_tail",
    "h_tail_split_verticals": True,
    "vertical_tail_height": 0.266,
    "vertical_tail_chord": 0.266,
    "vertical_tail_tip_chord": 0.266,
    "vertical_tail_xqc": 1.72966979,
    "vertical_tail_y": 0.0,
    "vertical_tail_z": 0.0,
    "vertical_tail_sweep_deg": 10,
    "h_tail_upper_height": 0.266,
    "h_tail_upper_root_chord": 0.266,
    "h_tail_upper_tip_chord": 0.1,
    "h_tail_upper_sweep_deg": None,
    "h_tail_lower_height": 0.266,
    "h_tail_lower_root_chord": 0.266,
    "h_tail_lower_tip_chord": 0.266,
    "h_tail_lower_sweep_deg": None,
    "vertical_tail_airfoil_dat": str(Path("PyFoil") / "coord_seligFmt" / "s9033.dat"),

    # H-tail fin placement tweaks (applied on top of horizontal tail tip positions)
    "h_tail_fin_inset": 0.0,
    "h_tail_fin_z_offset": 0.0,

    # Optional straight twin booms. They span automatically from the main wing
    # quarter-chord X position to the horizontal tail quarter-chord X position.
    "include_tail_booms": True,
    "tail_boom_diameter": 0.0254,
    # Center-to-center spacing between the two booms.
    "tail_boom_spacing": 0.500,
    # Added to the horizontal tail Z location (or body Z=0 if no h-tail).
    "tail_boom_z": 0.0,
}


TEMPLATE_MAIN_WING = "WingGeom"
TEMPLATE_HTAIL = "HorStabilizerGeom"
TEMPLATE_VTAIL = "TopVertStablizerGeom"

UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
}


def _uid() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=10))


def _sci(value: float) -> str:
    return f"{float(value):.18e}"


def _unit_scale_factor(design_units: str, stl_units: str) -> float:
    design_key = str(design_units).strip().lower()
    stl_key = str(stl_units).strip().lower()
    if design_key not in UNIT_TO_METERS:
        raise ValueError(f"Unsupported design_units value: {design_units}")
    if stl_key not in UNIT_TO_METERS:
        raise ValueError(f"Unsupported stl_units value: {stl_units}")
    return UNIT_TO_METERS[design_key] / UNIT_TO_METERS[stl_key]


def _converted_cfg() -> dict:
    cfg = dict(CONFIG)
    scale = _unit_scale_factor(cfg["design_units"], cfg["stl_units"])
    for key in (
        "fuselage_x_offset",
        "main_wing_span",
        "main_wing_chord",
        "main_wing_xqc",
        "main_wing_y",
        "main_wing_z",
        "horizontal_tail_span",
        "horizontal_tail_chord",
        "horizontal_tail_xqc",
        "horizontal_tail_y",
        "horizontal_tail_z",
        "vertical_tail_height",
        "vertical_tail_chord",
        "vertical_tail_tip_chord",
        "vertical_tail_xqc",
        "vertical_tail_y",
        "vertical_tail_z",
        "h_tail_upper_height",
        "h_tail_upper_root_chord",
        "h_tail_upper_tip_chord",
        "h_tail_lower_height",
        "h_tail_lower_root_chord",
        "h_tail_lower_tip_chord",
        "h_tail_fin_inset",
        "h_tail_fin_z_offset",
        "tail_boom_diameter",
        "tail_boom_spacing",
        "tail_boom_z",
    ):
        cfg[key] = float(cfg[key]) * scale
    return cfg


def _first_direct_child(parent: ET.Element, tag: str) -> ET.Element | None:
    for child in list(parent):
        if child.tag == tag:
            return child
    return None


def _remove_direct_children(parent: ET.Element, tag: str) -> None:
    for child in list(parent):
        if child.tag == tag:
            parent.remove(child)


def _geom_name(geom: ET.Element) -> str:
    parm = geom.find("./ParmContainer")
    if parm is None:
        return ""
    return (parm.findtext("Name") or parm.findtext("n") or "").strip()


def _find_template_geom(root: ET.Element, type_name: str, geom_name: str) -> ET.Element:
    vehicle = root.find("./Vehicle")
    if vehicle is None:
        raise ValueError("Template file is missing a Vehicle node")

    for geom in vehicle.findall("./Geom"):
        geom_base = geom.find("./GeomBase")
        if geom_base is None:
            continue
        if (geom_base.findtext("TypeName") or "").strip() != type_name:
            continue
        if _geom_name(geom) == geom_name:
            return geom

    raise ValueError(f"Could not find {type_name} geometry '{geom_name}' in template")


def _set_text(parent: ET.Element | None, tag: str, value: str) -> None:
    if parent is None:
        return
    node = parent.find(tag)
    if node is not None:
        node.text = value


def _set_parm_value(parent: ET.Element | None, tag: str, value: float) -> None:
    if parent is None:
        raise ValueError(f"Missing parent when setting '{tag}'")
    node = parent.find(tag)
    if node is None:
        raise ValueError(f"Missing parameter '{tag}'")
    node.set("Value", _sci(value))


def _get_parm_value(parent: ET.Element | None, tag: str, default: float = 0.0) -> float:
    if parent is None:
        return float(default)
    node = parent.find(tag)
    if node is None:
        return float(default)
    return float(node.attrib.get("Value", default))


def _copy_matching_values(source_parent: ET.Element | None, dest_parent: ET.Element | None) -> None:
    if source_parent is None or dest_parent is None:
        return
    for source_child in list(source_parent):
        if "Value" not in source_child.attrib:
            continue
        dest_child = dest_parent.find(source_child.tag)
        if dest_child is not None and "Value" in dest_child.attrib:
            dest_child.set("Value", source_child.attrib["Value"])


def _optional_path(value: str | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _set_flat_end_caps(geom: ET.Element) -> None:
    end_cap = geom.find("./ParmContainer/EndCap")
    if end_cap is None:
        return
    flat_values = {
        "CapUMinOption": 1.0,
        "CapUMaxOption": 1.0,
        "CapUMinStrength": 0.5,
        "CapUMaxStrength": 0.5,
    }
    for tag, value in flat_values.items():
        if end_cap.find(tag) is not None:
            _set_parm_value(end_cap, tag, value)


def _set_lifting_surface_mesh(geom: ET.Element, cfg: dict) -> None:
    shape = geom.find("./ParmContainer/Shape")
    if shape is not None:
        if shape.find("Tess_U") is not None:
            _set_parm_value(shape, "Tess_U", float(cfg["lifting_surface_tess_u"]))
        if shape.find("Tess_W") is not None:
            _set_parm_value(shape, "Tess_W", float(cfg["lifting_surface_tess_w"]))

    for sec in geom.findall("./WingGeom/XSecSurf/XSec/ParmContainer/XSec"):
        if sec.find("SectTess_U") is not None:
            _set_parm_value(sec, "SectTess_U", float(cfg["lifting_surface_sect_tess_u"]))


def _apply_fuselage_rotation(geom: ET.Element, cfg: dict) -> None:
    xform = geom.find("./ParmContainer/XForm")
    if xform is None:
        return

    rotations = {
        "X_Rotation": float(cfg["fuselage_x_rotation_deg"]),
        "X_Rel_Rotation": float(cfg["fuselage_x_rotation_deg"]),
        "Y_Rotation": float(cfg["fuselage_y_rotation_deg"]),
        "Y_Rel_Rotation": float(cfg["fuselage_y_rotation_deg"]),
        "Z_Rotation": float(cfg["fuselage_z_rotation_deg"]),
        "Z_Rel_Rotation": float(cfg["fuselage_z_rotation_deg"]),
    }
    for tag, value in rotations.items():
        if xform.find(tag) is not None:
            _set_parm_value(xform, tag, value)


def _fuselage_nose_x(geom: ET.Element) -> float:
    values = []
    for xsec in geom.findall("./FuselageGeom/XSecSurf/XSec"):
        sec = xsec.find("./ParmContainer/XSec")
        if sec is None:
            continue
        x_pct_node = sec.find("XLocPercent")
        ref_len_node = sec.find("RefLength")
        if x_pct_node is None or ref_len_node is None:
            continue
        x_pct = float(x_pct_node.attrib.get("Value", "0.0"))
        ref_len = float(ref_len_node.attrib.get("Value", "0.0"))
        values.append(x_pct * ref_len)
    if not values:
        return 0.0
    return min(values)


def _apply_fuselage_position(geom: ET.Element, cfg: dict) -> None:
    xform = geom.find("./ParmContainer/XForm")
    if xform is None:
        return

    current_x = _get_parm_value(xform, "X_Location", 0.0)
    extra_offset = float(cfg["fuselage_x_offset"])
    if bool(cfg["fuselage_align_nose_to_x_zero"]):
        target_x = extra_offset - _fuselage_nose_x(geom)
    else:
        target_x = current_x + extra_offset

    for tag in ("X_Location", "X_Rel_Location"):
        if xform.find(tag) is not None:
            _set_parm_value(xform, tag, target_x)


def _has_fuselage_rotation(cfg: dict) -> bool:
    return any(
        abs(float(cfg[key])) > 1e-9
        for key in (
            "fuselage_x_rotation_deg",
            "fuselage_y_rotation_deg",
            "fuselage_z_rotation_deg",
        )
    )


def _needs_fuselage_postprocess(cfg: dict) -> bool:
    return (
        _has_fuselage_rotation(cfg)
        or bool(cfg["fuselage_align_nose_to_x_zero"])
        or abs(float(cfg["fuselage_x_offset"])) > 1e-9
    )


def _airfoil_assignments(cfg: dict) -> dict[str, Path]:
    assignments: dict[str, Path] = {}

    main_path = _optional_path(cfg.get("main_wing_airfoil_dat"))
    if bool(cfg["include_main_wing"]) and main_path is not None:
        assignments["MainWingGeom"] = main_path

    htail_path = _optional_path(cfg.get("horizontal_tail_airfoil_dat"))
    if bool(cfg["include_horizontal_tail"]) and htail_path is not None:
        assignments["HorizontalTailGeom"] = htail_path

    vtail_path = _optional_path(cfg.get("vertical_tail_airfoil_dat"))
    if bool(cfg["include_vertical_tail"]) and vtail_path is not None:
        layout = str(cfg["vertical_tail_layout"]).strip().lower()
        if layout == "h_tail":
            if bool(cfg.get("h_tail_split_verticals")):
                for geom_name in (
                    "LeftUpperVerticalTailGeom",
                    "LeftLowerVerticalTailGeom",
                    "RightUpperVerticalTailGeom",
                    "RightLowerVerticalTailGeom",
                ):
                    assignments[geom_name] = vtail_path
            else:
                assignments["LeftVerticalTailGeom"] = vtail_path
                assignments["RightVerticalTailGeom"] = vtail_path
        else:
            assignments["VerticalTailGeom"] = vtail_path

    return assignments


def _dedupe_sorted_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    deduped: list[tuple[float, float]] = []
    last_x = None
    y_accum = 0.0
    count = 0
    for x_val, y_val in points:
        if last_x is None or abs(x_val - last_x) > 1e-9:
            if last_x is not None:
                deduped.append((last_x, y_accum / max(count, 1)))
            last_x = x_val
            y_accum = y_val
            count = 1
        else:
            y_accum += y_val
            count += 1
    deduped.append((last_x, y_accum / max(count, 1)))
    return deduped


def _interp_points(points: list[tuple[float, float]], x_query: float) -> float:
    if not points:
        return 0.0
    if x_query <= points[0][0]:
        return points[0][1]
    if x_query >= points[-1][0]:
        return points[-1][1]
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        if x_query <= x1:
            dx = max(x1 - x0, 1e-12)
            frac = (x_query - x0) / dx
            return y0 + frac * (y1 - y0)
    return points[-1][1]


def _fit_four_series_from_dat(airfoil_path: Path) -> dict[str, float]:
    points: list[tuple[float, float]] = []
    for line in airfoil_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
        except ValueError:
            continue
        points.append((x_val, y_val))

    if len(points) < 8:
        raise ValueError(f"Not enough coordinate points in airfoil file: {airfoil_path}")

    x_min = min(x for x, _ in points)
    x_max = max(x for x, _ in points)
    chord = max(x_max - x_min, 1e-9)
    norm_points = [((x - x_min) / chord, y / chord) for x, y in points]

    le_idx = min(range(len(norm_points)), key=lambda idx: norm_points[idx][0])
    upper = norm_points[: le_idx + 1]
    lower = norm_points[le_idx:]

    upper = _dedupe_sorted_points(sorted(upper, key=lambda item: item[0]))
    lower = _dedupe_sorted_points(sorted(lower, key=lambda item: item[0]))
    if len(upper) < 2 or len(lower) < 2:
        raise ValueError(f"Could not split upper/lower surfaces for {airfoil_path}")

    sample_count = 101
    x_grid = [idx / (sample_count - 1) for idx in range(sample_count)]
    upper_y = [_interp_points(upper, x_val) for x_val in x_grid]
    lower_y = [_interp_points(lower, x_val) for x_val in x_grid]

    thickness = [yu - yl for yu, yl in zip(upper_y, lower_y)]
    camber_line = [0.5 * (yu + yl) for yu, yl in zip(upper_y, lower_y)]

    thick_idx = max(range(sample_count), key=lambda idx: thickness[idx])
    camber_idx = max(range(sample_count), key=lambda idx: abs(camber_line[idx]))

    max_thickness = max(thickness[thick_idx], 0.0)
    max_camber = camber_line[camber_idx]
    camber_loc = min(max(x_grid[camber_idx], 0.05), 0.95)
    trailing_edge_gap = abs(upper_y[-1] - lower_y[-1])

    area_norm = 0.0
    for idx in range(1, sample_count):
        dx = x_grid[idx] - x_grid[idx - 1]
        area_norm += 0.5 * (thickness[idx] + thickness[idx - 1]) * dx

    return {
        "camber": max_camber,
        "camber_loc": camber_loc,
        "thick_chord": max_thickness,
        "hw_ratio": max_thickness,
        "area_norm": max(area_norm, 1e-6),
        "ideal_cl": max(min(max_camber * 12.0, 1.5), -1.5),
        "sharp_te_flag": 1.0 if trailing_edge_gap <= 1e-3 else 0.0,
    }


def _apply_four_series_fit_to_geom(geom: ET.Element, fit: dict[str, float]) -> bool:
    applied = False
    for xsec in geom.findall("./WingGeom/XSecSurf/XSec"):
        curve_parent = xsec.find("./XSec/XSecCurve")
        if curve_parent is None:
            continue
        curve_type = curve_parent.find("./XSecCurve/Type")
        if curve_type is None:
            continue

        # Use the template's four-series section schema and just overwrite the
        # shape parameters. This is a safe XML fallback when the OpenVSP API is
        # unavailable.
        curve_type.text = "7"

        curve_parms = curve_parent.find("./ParmContainer/XSecCurve")
        if curve_parms is None:
            continue
        if curve_parms.find("Camber") is None or curve_parms.find("ThickChord") is None:
            continue

        if curve_parms.find("CamberInputFlag") is not None:
            _set_parm_value(curve_parms, "CamberInputFlag", 1.0)
        _set_parm_value(curve_parms, "Camber", fit["camber"])
        if curve_parms.find("CamberLoc") is not None:
            _set_parm_value(curve_parms, "CamberLoc", fit["camber_loc"])
        _set_parm_value(curve_parms, "ThickChord", fit["thick_chord"])
        if curve_parms.find("HWRatio") is not None:
            _set_parm_value(curve_parms, "HWRatio", fit["hw_ratio"])
        if curve_parms.find("IdealCl") is not None:
            _set_parm_value(curve_parms, "IdealCl", fit["ideal_cl"])
        if curve_parms.find("SharpTEFlag") is not None:
            _set_parm_value(curve_parms, "SharpTEFlag", fit["sharp_te_flag"])
        if curve_parms.find("Area") is not None:
            curve_chord = max(_get_parm_value(curve_parms, "Chord", 1.0), 1e-6)
            _set_parm_value(curve_parms, "Area", fit["area_norm"] * curve_chord * curve_chord)
        applied = True
    return applied


def _apply_airfoil_files_xml_fallback(output_path: Path, cfg: dict) -> list[str]:
    assignments = _airfoil_assignments(cfg)
    if not assignments:
        return []

    missing = [str(path) for path in assignments.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Airfoil file not found: " + ", ".join(sorted(set(missing))))

    tree = ET.parse(output_path)
    root = tree.getroot()
    vehicle = root.find("./Vehicle")
    if vehicle is None:
        return []

    geoms = {
        _geom_name(geom): geom
        for geom in vehicle.findall("./Geom")
    }

    applied: list[str] = []
    for geom_name, airfoil_path in assignments.items():
        geom = geoms.get(geom_name)
        if geom is None:
            print(f"WARNING: Could not find geometry '{geom_name}' for XML airfoil fallback.")
            continue

        fit = _fit_four_series_from_dat(airfoil_path)
        if _apply_four_series_fit_to_geom(geom, fit):
            applied.append(f"{geom_name} ~= {airfoil_path} (XML 4-series fit)")

    if applied:
        if hasattr(ET, "indent"):
            ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return applied


def _build_simple_boom_xsec_surf(length: float, diameter: float) -> ET.Element:
    length = max(float(length), 1e-6)
    diameter = max(float(diameter), 1e-6)
    half_size = 0.5 * diameter
    station_fracs = (0.0, 0.05, 0.95, 1.0)
    xsec_nodes: list[ET.Element] = []

    for frac in station_fracs:
        xsec_node = ET.fromstring(
            stl_to_vsp.build_xsec(
                frac,
                0.0,
                0.0,
                half_size,
                half_size,
                length,
                is_point=False,
                radius_full=diameter,
            )
        )

        curve_type = xsec_node.find("./XSec/XSecCurve/XSecCurve/Type")
        if curve_type is not None:
            # In the VSP3 XML used here, Type 1 is the circle XSec and Type 2
            # is ellipse.
            curve_type.text = "1"

        curve_parms = xsec_node.find("./XSec/XSecCurve/ParmContainer/XSecCurve")
        if curve_parms is not None:
            area = math.pi * (0.5 * diameter) ** 2
            circle_diameter = curve_parms.find("Circle_Diameter")
            if circle_diameter is None:
                circle_diameter = ET.Element(
                    "Circle_Diameter",
                    {"Value": _sci(diameter), "ID": _uid()},
                )
                curve_parms.append(circle_diameter)
            else:
                circle_diameter.set("Value", _sci(diameter))
            if curve_parms.find("Area") is not None:
                _set_parm_value(curve_parms, "Area", area)
            if curve_parms.find("HWRatio") is not None:
                _set_parm_value(curve_parms, "HWRatio", 1.0)

            # Remove rounded-rectangle-only parms so the node is closer to a
            # native circle XSec and avoids carrying contradictory shape data.
            for child in list(curve_parms):
                if child.tag.startswith("RoundRect") or child.tag.startswith("RoundedRect"):
                    curve_parms.remove(child)

        xsec_nodes.append(xsec_node)

    xsec_surf = ET.Element("XSecSurf")
    for node in xsec_nodes:
        xsec_surf.append(node)
    return xsec_surf


def _apply_airfoil_files_with_openvsp(output_path: Path, cfg: dict) -> list[str]:
    assignments = _airfoil_assignments(cfg)
    if not assignments:
        return []

    missing = [str(path) for path in assignments.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Airfoil file not found: " + ", ".join(sorted(set(missing))))

    try:
        import openvsp as _openvsp_pkg
    except ModuleNotFoundError:
        print("WARNING: OpenVSP Python API is not installed; using XML airfoil fit fallback.")
        return _apply_airfoil_files_xml_fallback(output_path, cfg)

    vsp = getattr(_openvsp_pkg, "vsp", _openvsp_pkg)
    if not all(
        hasattr(vsp, name)
        for name in (
            "ClearVSPModel",
            "ReadVSPFile",
            "FindGeom",
            "GetXSecSurf",
            "GetNumXSec",
            "ChangeXSecShape",
            "GetXSec",
            "ReadFileAirfoil",
            "Update",
            "WriteVSPFile",
            "XS_FILE_AIRFOIL",
            "SET_ALL",
        )
    ):
        print("WARNING: OpenVSP Python API is missing required functions; keeping template airfoils.")
        return []

    applied: list[str] = []
    vsp.ClearVSPModel()
    try:
        vsp.ReadVSPFile(str(output_path))

        for geom_name, airfoil_path in assignments.items():
            geom_id = vsp.FindGeom(geom_name, 0)
            if not geom_id:
                print(f"WARNING: Could not find geometry '{geom_name}' for airfoil import.")
                continue

            xsec_surf_id = vsp.GetXSecSurf(geom_id, 0)
            num_xsec = int(vsp.GetNumXSec(xsec_surf_id))
            for idx in range(num_xsec):
                vsp.ChangeXSecShape(xsec_surf_id, idx, vsp.XS_FILE_AIRFOIL)
                xsec_id = vsp.GetXSec(xsec_surf_id, idx)
                vsp.ReadFileAirfoil(xsec_id, str(airfoil_path))
            applied.append(f"{geom_name} <- {airfoil_path}")

        if applied:
            vsp.Update()
            vsp.WriteVSPFile(str(output_path), vsp.SET_ALL)
    except Exception as exc:
        print(f"WARNING: OpenVSP airfoil import failed; using XML airfoil fit fallback. ({exc})")
        return _apply_airfoil_files_xml_fallback(output_path, cfg)
    finally:
        vsp.ClearVSPModel()

    return applied


def _find_first_geom_by_type(root: ET.Element, type_name: str) -> ET.Element | None:
    vehicle = root.find("./Vehicle")
    if vehicle is None:
        return None

    for geom in vehicle.findall("./Geom"):
        geom_base = geom.find("./GeomBase")
        if geom_base is None:
            continue
        if (geom_base.findtext("TypeName") or "").strip() == type_name:
            return geom
    return None


def _clear_geom_details(geom: ET.Element) -> None:
    subsurfaces = geom.find("./Geom/SubSurfaces")
    if subsurfaces is not None:
        subsurfaces.clear()
    fea_structures = geom.find("./Geom/FeaStructures")
    if fea_structures is not None:
        fea_structures.clear()


def _prune_cloned_wing_geom(geom: ET.Element) -> None:
    allowed = {"ParmContainer", "GeomBase", "Material", "Textures", "Geom", "WingGeom"}
    for child in list(geom):
        if child.tag not in allowed:
            geom.remove(child)


def _collect_owned_ids(node: ET.Element) -> set[str]:
    owned = set()
    for elem in node.iter():
        if elem.tag == "ID" and elem.text:
            owned.add(elem.text.strip())
        attr_id = elem.attrib.get("ID")
        if attr_id:
            owned.add(attr_id)
    return {value for value in owned if value and value != "NONE"}


def _remap_ids(node: ET.Element) -> None:
    old_ids = _collect_owned_ids(node)
    if not old_ids:
        return

    mapping = {old: _uid() for old in old_ids}

    for elem in node.iter():
        if elem.tag == "ID" and elem.text in mapping:
            elem.text = mapping[elem.text]

        if elem.text:
            stripped = elem.text.strip()
            if stripped in mapping:
                elem.text = mapping[stripped]

        for attr_name, attr_value in list(elem.attrib.items()):
            if attr_value in mapping:
                elem.set(attr_name, mapping[attr_value])


def _set_wing_xform(
    geom: ET.Element,
    x_location: float,
    y_location: float,
    z_location: float,
    y_rotation: float,
    x_rotation: float = 0.0,
    z_rotation: float = 0.0,
) -> None:
    xform = geom.find("./ParmContainer/XForm")
    if xform is None:
        raise ValueError("Wing template is missing an XForm block")

    pairs = {
        "X_Location": x_location,
        "X_Rel_Location": x_location,
        "Y_Location": y_location,
        "Y_Rel_Location": y_location,
        "Z_Location": z_location,
        "Z_Rel_Location": z_location,
        "X_Rotation": x_rotation,
        "X_Rel_Rotation": x_rotation,
        "Y_Rotation": y_rotation,
        "Y_Rel_Rotation": y_rotation,
        "Z_Rotation": z_rotation,
        "Z_Rel_Rotation": z_rotation,
    }
    for tag, value in pairs.items():
        _set_parm_value(xform, tag, value)


def _set_planar_symmetry(geom: ET.Element, planar_flag: float) -> None:
    sym = geom.find("./ParmContainer/Sym")
    if sym is not None and sym.find("Sym_Planar_Flag") is not None:
        _set_parm_value(sym, "Sym_Planar_Flag", planar_flag)


def _set_geom_xform(
    geom: ET.Element,
    *,
    x_location: float,
    y_location: float,
    z_location: float,
    x_rotation: float = 0.0,
    y_rotation: float = 0.0,
    z_rotation: float = 0.0,
) -> None:
    xform = geom.find("./ParmContainer/XForm")
    if xform is None:
        return

    values = {
        "X_Location": x_location,
        "X_Rel_Location": x_location,
        "Y_Location": y_location,
        "Y_Rel_Location": y_location,
        "Z_Location": z_location,
        "Z_Rel_Location": z_location,
        "X_Rotation": x_rotation,
        "X_Rel_Rotation": x_rotation,
        "Y_Rotation": y_rotation,
        "Y_Rel_Rotation": y_rotation,
        "Z_Rotation": z_rotation,
        "Z_Rel_Rotation": z_rotation,
    }
    for tag, value in values.items():
        if xform.find(tag) is not None:
            _set_parm_value(xform, tag, value)


def _split_vtail_sweep(cfg: dict, key: str) -> float:
    value = cfg.get(key)
    if value is None:
        return float(cfg["vertical_tail_sweep_deg"])
    return float(value)


def _retune_wing_sections(
    geom: ET.Element,
    total_span: float,
    chord: float,
    sweep_deg: float = 0.0,
    dihedral_deg: float = 0.0,
) -> None:
    xsecs = geom.findall("./WingGeom/XSecSurf/XSec")
    section_nodes = []
    template_spans = []
    template_roots = []
    template_tips = []
    template_projected = []
    template_sec_sweep = []

    for xsec in xsecs:
        sec = xsec.find("./ParmContainer/XSec")
        if sec is None:
            continue
        span_node = sec.find("Span")
        root_node = sec.find("Root_Chord")
        tip_node = sec.find("Tip_Chord")
        if span_node is None or root_node is None or tip_node is None:
            continue
        section_nodes.append(sec)
        template_spans.append(float(span_node.attrib.get("Value", "0.0")))
        template_roots.append(float(root_node.attrib.get("Value", "0.0")))
        template_tips.append(float(tip_node.attrib.get("Value", "0.0")))
        proj_node = sec.find("ProjectedSpan")
        template_projected.append(
            float(proj_node.attrib.get("Value", "0.0")) if proj_node is not None else 0.0
        )
        sec_sweep_node = sec.find("Sec_Sweep")
        template_sec_sweep.append(
            float(sec_sweep_node.attrib.get("Value", "0.0")) if sec_sweep_node is not None else 0.0
        )

    if not section_nodes:
        raise ValueError("Wing template has no editable section spans")

    half_span = max(float(total_span) * 0.5, 1e-6)
    cap_mask = [abs(value) < 1e-12 for value in template_projected]
    fixed_cap_span = sum(
        template_spans[idx] for idx, is_cap in enumerate(cap_mask) if is_cap
    )
    span_budget = max(half_span - fixed_cap_span, 1e-6)
    template_total = sum(
        max(template_spans[idx], 0.0) for idx, is_cap in enumerate(cap_mask) if not is_cap
    )
    if template_total <= 0.0:
        span_values = [template_spans[idx] if cap_mask[idx] else span_budget / max(len(section_nodes), 1) for idx in range(len(section_nodes))]
    else:
        span_values = []
        for idx, value in enumerate(template_spans):
            if cap_mask[idx]:
                span_values.append(value)
            else:
                span_values.append(span_budget * max(value, 0.0) / template_total)

    for idx, (sec, span_value) in enumerate(zip(section_nodes, span_values)):
        if cap_mask[idx]:
            # Keep the cap section's shape, but make it terminate at the
            # requested design chord so it blends into the real wing.
            cap_scale = float(chord) / max(template_tips[idx], 1e-6)
            root_chord = max(template_roots[idx] * cap_scale, 1e-6)
            tip_chord = max(template_tips[idx] * cap_scale, 1e-6)
        else:
            root_chord = max(float(chord), 1e-6)
            tip_chord = max(float(chord), 1e-6)
        area = 0.5 * (root_chord + tip_chord) * span_value
        avg_chord = area / max(span_value, 1e-9)
        taper = tip_chord / max(root_chord, 1e-9)
        aspect = (span_value * span_value) / max(area, 1e-9)
        if abs(template_projected[idx]) < 1e-12:
            projected_span = 0.0
            sec_sweep = template_sec_sweep[idx]
        else:
            projected_span = span_value * max(math.cos(math.radians(dihedral_deg)), 0.0)
            sec_sweep = sweep_deg

        _set_parm_value(sec, "Root_Chord", root_chord)
        _set_parm_value(sec, "Tip_Chord", tip_chord)
        _set_parm_value(sec, "Span", span_value)
        if sec.find("Area") is not None:
            _set_parm_value(sec, "Area", area)
        if sec.find("Avg_Chord") is not None:
            _set_parm_value(sec, "Avg_Chord", avg_chord)
        if sec.find("Taper") is not None:
            _set_parm_value(sec, "Taper", taper)
        if sec.find("Aspect") is not None:
            _set_parm_value(sec, "Aspect", aspect)
        if sec.find("ProjectedSpan") is not None:
            _set_parm_value(sec, "ProjectedSpan", projected_span)
        if sec.find("Sweep") is not None:
            _set_parm_value(sec, "Sweep", sweep_deg)
        if sec.find("Sec_Sweep") is not None:
            _set_parm_value(sec, "Sec_Sweep", sec_sweep)
        if sec.find("Dihedral") is not None:
            _set_parm_value(sec, "Dihedral", dihedral_deg)
        if sec.find("Twist") is not None:
            _set_parm_value(sec, "Twist", 0.0)
        if sec.find("Sweep_Location") is not None:
            _set_parm_value(sec, "Sweep_Location", 0.25)
        if sec.find("Twist_Location") is not None:
            _set_parm_value(sec, "Twist_Location", 0.25)

    wing_group = geom.find("./ParmContainer/WingGeom")
    if wing_group is not None:
        total_area = float(total_span) * float(chord)
        total_ar = (float(total_span) ** 2) / max(total_area, 1e-9)
        for tag, value in (
            ("TotalSpan", total_span),
            ("TotalProjectedSpan", total_span),
            ("TotalChord", chord),
            ("MAC", chord),
            ("TotalArea", total_area),
            ("CurvedArea", total_area),
            ("TotalAR", total_ar),
        ):
            if wing_group.find(tag) is not None:
                _set_parm_value(wing_group, tag, value)


def _retune_vtail_sections(
    geom: ET.Element,
    height: float,
    root_chord: float,
    tip_chord: float,
    sweep_deg: float = 0.0,
) -> None:
    xsecs = geom.findall("./WingGeom/XSecSurf/XSec")
    section_nodes = []
    template_spans = []
    template_roots = []
    template_tips = []
    template_projected = []
    template_sec_sweep = []

    for xsec in xsecs:
        sec = xsec.find("./ParmContainer/XSec")
        if sec is None:
            continue
        span_node = sec.find("Span")
        root_node = sec.find("Root_Chord")
        tip_node = sec.find("Tip_Chord")
        if span_node is None or root_node is None or tip_node is None:
            continue
        section_nodes.append(sec)
        template_spans.append(float(span_node.attrib.get("Value", "0.0")))
        template_roots.append(float(root_node.attrib.get("Value", "0.0")))
        template_tips.append(float(tip_node.attrib.get("Value", "0.0")))
        proj_node = sec.find("ProjectedSpan")
        template_projected.append(
            float(proj_node.attrib.get("Value", "0.0")) if proj_node is not None else 0.0
        )
        sec_sweep_node = sec.find("Sec_Sweep")
        template_sec_sweep.append(
            float(sec_sweep_node.attrib.get("Value", "0.0")) if sec_sweep_node is not None else 0.0
        )

    if not section_nodes:
        raise ValueError("Vertical tail template has no editable section spans")

    height = max(float(height), 1e-6)
    cap_mask = [abs(value) < 1e-12 for value in template_projected]
    fixed_cap_span = sum(
        template_spans[idx] for idx, is_cap in enumerate(cap_mask) if is_cap
    )
    span_budget = max(height - fixed_cap_span, 1e-6)
    template_total = sum(
        max(template_spans[idx], 0.0) for idx, is_cap in enumerate(cap_mask) if not is_cap
    )
    if template_total <= 0.0:
        span_values = [template_spans[idx] if cap_mask[idx] else span_budget / max(len(section_nodes), 1) for idx in range(len(section_nodes))]
    else:
        span_values = []
        for idx, value in enumerate(template_spans):
            if cap_mask[idx]:
                span_values.append(value)
            else:
                span_values.append(span_budget * max(value, 0.0) / template_total)

    def _edge_sweeps(sec_root_chord: float, sec_tip_chord: float, sec_span: float) -> tuple[float, float]:
        qc_dx = sec_span * math.tan(math.radians(sweep_deg))
        chord_delta = sec_root_chord - sec_tip_chord
        le_dx = qc_dx + 0.25 * chord_delta
        te_dx = qc_dx - 0.75 * chord_delta
        le_sweep = math.degrees(math.atan2(le_dx, max(sec_span, 1e-9)))
        te_sweep = math.degrees(math.atan2(te_dx, max(sec_span, 1e-9)))
        return le_sweep, te_sweep

    total_noncap_span = sum(
        span_values[idx] for idx, is_cap in enumerate(cap_mask) if not is_cap
    )
    first_noncap_idx = next((idx for idx, is_cap in enumerate(cap_mask) if not is_cap), None)
    cap_le_sweep = 0.0
    cap_te_sweep = 0.0
    if first_noncap_idx is not None:
        first_span = span_values[first_noncap_idx]
        first_root = max(float(root_chord), 1e-6)
        if total_noncap_span > 0.0:
            first_end_frac = first_span / max(total_noncap_span, 1e-9)
        else:
            first_end_frac = 1.0
        first_tip = max(
            float(root_chord) + (float(tip_chord) - float(root_chord)) * first_end_frac,
            1e-6,
        )
        cap_le_sweep, cap_te_sweep = _edge_sweeps(first_root, first_tip, max(first_span, 1e-9))

    span_cursor = 0.0

    for idx, (sec, span_value) in enumerate(zip(section_nodes, span_values)):
        if cap_mask[idx]:
            cap_scale = float(root_chord) / max(template_tips[idx], 1e-6)
            sec_root_chord = max(template_roots[idx] * cap_scale, 1e-6)
            sec_tip_chord = max(template_tips[idx] * cap_scale, 1e-6)
            le_sweep = cap_le_sweep
            te_sweep = cap_te_sweep
        else:
            start_frac = span_cursor / max(total_noncap_span, 1e-9)
            end_frac = (span_cursor + span_value) / max(total_noncap_span, 1e-9)
            sec_root_chord = max(
                float(root_chord) + (float(tip_chord) - float(root_chord)) * start_frac,
                1e-6,
            )
            sec_tip_chord = max(
                float(root_chord) + (float(tip_chord) - float(root_chord)) * end_frac,
                1e-6,
            )
            span_cursor += span_value
            le_sweep, te_sweep = _edge_sweeps(sec_root_chord, sec_tip_chord, max(span_value, 1e-9))
        area = 0.5 * (sec_root_chord + sec_tip_chord) * span_value
        avg_chord = area / max(span_value, 1e-9)
        taper = sec_tip_chord / max(sec_root_chord, 1e-9)
        aspect = (span_value * span_value) / max(area, 1e-9)
        if abs(template_projected[idx]) < 1e-12:
            projected_span = 0.0
            sec_sweep = 89.0 if first_noncap_idx is not None else template_sec_sweep[idx]
        else:
            projected_span = span_value
            sec_sweep = te_sweep

        _set_parm_value(sec, "Root_Chord", sec_root_chord)
        _set_parm_value(sec, "Tip_Chord", sec_tip_chord)
        _set_parm_value(sec, "Span", span_value)
        if sec.find("Area") is not None:
            _set_parm_value(sec, "Area", area)
        if sec.find("Avg_Chord") is not None:
            _set_parm_value(sec, "Avg_Chord", avg_chord)
        if sec.find("Taper") is not None:
            _set_parm_value(sec, "Taper", taper)
        if sec.find("Aspect") is not None:
            _set_parm_value(sec, "Aspect", aspect)
        if sec.find("ProjectedSpan") is not None:
            _set_parm_value(sec, "ProjectedSpan", projected_span)
        if sec.find("Sweep") is not None:
            _set_parm_value(sec, "Sweep", sweep_deg)
        if sec.find("Sec_Sweep") is not None:
            _set_parm_value(sec, "Sec_Sweep", sec_sweep)
        for tag, value in (
            ("InLESweep", le_sweep),
            ("OutLESweep", le_sweep),
            ("InTESweep", te_sweep),
            ("OutTESweep", te_sweep),
        ):
            if sec.find(tag) is not None:
                _set_parm_value(sec, tag, value)
        if sec.find("Dihedral") is not None:
            _set_parm_value(sec, "Dihedral", 0.0)
        if sec.find("Twist") is not None:
            _set_parm_value(sec, "Twist", 0.0)
        if sec.find("Sweep_Location") is not None:
            _set_parm_value(sec, "Sweep_Location", 0.25)

    wing_group = geom.find("./ParmContainer/WingGeom")
    if wing_group is not None:
        mean_chord = 0.5 * (float(root_chord) + float(tip_chord))
        area = float(height) * mean_chord
        total_ar = (float(height) ** 2) / max(area, 1e-9)
        for tag, value in (
            ("TotalSpan", height),
            ("TotalProjectedSpan", height),
            ("TotalChord", mean_chord),
            ("MAC", mean_chord),
            ("TotalArea", area),
            ("CurvedArea", area),
            ("TotalAR", total_ar),
        ):
            if wing_group.find(tag) is not None:
                _set_parm_value(wing_group, tag, value)


def _extract_fuselage_root(
    stl_path: Path,
    axis: str,
    slices: int,
    nose_cap: str,
    tail_cap: str,
    name: str,
) -> ET.ElementTree:
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_out = Path(tmp_dir) / "fuselage_only.vsp3"
        with contextlib.redirect_stdout(io.StringIO()):
            stl_to_vsp.convert(
                str(stl_path),
                str(temp_out),
                n_slices=int(slices),
                axis=axis,
                name=name,
                nose_cap=nose_cap,
                tail_cap=tail_cap,
            )
        return ET.parse(temp_out)


def _build_main_wing_geom(template_root: ET.Element, cfg: dict) -> ET.Element:
    geom = copy.deepcopy(_find_template_geom(template_root, "Wing", TEMPLATE_MAIN_WING))
    _remap_ids(geom)
    _prune_cloned_wing_geom(geom)
    _clear_geom_details(geom)
    _set_flat_end_caps(geom)
    _set_text(geom.find("./ParmContainer"), "Name", "MainWingGeom")
    _set_text(geom.find("./WingGeom/ParmContainer"), "Name", "MainWing")

    chord = float(cfg["main_wing_chord"])
    x_le = float(cfg["main_wing_xqc"]) - 0.25 * chord
    _set_wing_xform(
        geom,
        x_location=x_le,
        y_location=float(cfg["main_wing_y"]),
        z_location=float(cfg["main_wing_z"]),
        y_rotation=float(cfg["main_wing_incidence_deg"]),
    )
    _set_planar_symmetry(geom, 2.0)
    _retune_wing_sections(
        geom,
        total_span=float(cfg["main_wing_span"]),
        chord=chord,
        sweep_deg=float(cfg["main_wing_sweep_deg"]),
        dihedral_deg=float(cfg["main_wing_dihedral_deg"]),
    )
    _set_lifting_surface_mesh(geom, cfg)
    return geom


def _build_fuselage_geom(template_root: ET.Element, generated_fuselage_geom: ET.Element) -> ET.Element:
    geom = copy.deepcopy(_find_template_geom(template_root, "Fuselage", "FuselageGeom"))
    _remap_ids(geom)

    template_pc = geom.find("./ParmContainer")
    generated_pc = generated_fuselage_geom.find("./ParmContainer")
    if template_pc is not None and generated_pc is not None:
        for block_name in ("Design", "EndCap", "Shape", "Sym", "XForm"):
            _copy_matching_values(generated_pc.find(block_name), template_pc.find(block_name))

    template_fg = geom.find("./FuselageGeom")
    generated_fg = generated_fuselage_geom.find("./FuselageGeom")
    if template_fg is None or generated_fg is None:
        raise ValueError("Fuselage geometry is missing a FuselageGeom block")

    generated_xsec_surf = generated_fg.find("./XSecSurf")
    if generated_xsec_surf is None:
        raise ValueError("Generated fuselage is missing XSecSurf")

    existing_xsec_surf = template_fg.find("./XSecSurf")
    if existing_xsec_surf is not None:
        template_fg.remove(existing_xsec_surf)

    transplanted_xsec_surf = copy.deepcopy(generated_xsec_surf)
    _remap_ids(transplanted_xsec_surf)
    template_fg.append(transplanted_xsec_surf)

    return geom


def _build_horizontal_tail_geom(template_root: ET.Element, cfg: dict) -> ET.Element:
    geom = copy.deepcopy(_find_template_geom(template_root, "Wing", TEMPLATE_HTAIL))
    _remap_ids(geom)
    _prune_cloned_wing_geom(geom)
    _clear_geom_details(geom)
    _set_flat_end_caps(geom)
    _set_text(geom.find("./ParmContainer"), "Name", "HorizontalTailGeom")
    _set_text(geom.find("./WingGeom/ParmContainer"), "Name", "HorizontalTail")

    chord = float(cfg["horizontal_tail_chord"])
    x_le = float(cfg["horizontal_tail_xqc"]) - 0.25 * chord
    _set_wing_xform(
        geom,
        x_location=x_le,
        y_location=float(cfg["horizontal_tail_y"]),
        z_location=float(cfg["horizontal_tail_z"]),
        y_rotation=float(cfg["horizontal_tail_incidence_deg"]),
    )
    _set_planar_symmetry(geom, 2.0)
    _retune_wing_sections(
        geom,
        total_span=float(cfg["horizontal_tail_span"]),
        chord=chord,
        sweep_deg=float(cfg["horizontal_tail_sweep_deg"]),
        dihedral_deg=float(cfg["horizontal_tail_dihedral_deg"]),
    )
    _set_lifting_surface_mesh(geom, cfg)
    return geom


def _build_vertical_tail_geom(
    template_root: ET.Element,
    cfg: dict,
    *,
    name: str,
    y_location: float,
    z_location: float,
    xqc: float,
    height: float,
    root_chord: float,
    tip_chord: float,
    sweep_deg: float,
    x_rotation: float = 90.0,
) -> ET.Element:
    geom = copy.deepcopy(_find_template_geom(template_root, "Wing", TEMPLATE_VTAIL))
    _remap_ids(geom)
    _prune_cloned_wing_geom(geom)
    _clear_geom_details(geom)
    _set_flat_end_caps(geom)
    _set_text(geom.find("./ParmContainer"), "Name", f"{name}Geom")
    _set_text(geom.find("./WingGeom/ParmContainer"), "Name", name)

    x_le = float(xqc) - 0.25 * float(root_chord)
    _set_wing_xform(
        geom,
        x_location=x_le,
        y_location=y_location,
        z_location=z_location,
        y_rotation=0.0,
        x_rotation=x_rotation,
    )
    _set_planar_symmetry(geom, 0.0)
    _retune_vtail_sections(
        geom,
        height=float(height),
        root_chord=float(root_chord),
        tip_chord=float(tip_chord),
        sweep_deg=float(sweep_deg),
    )
    _set_lifting_surface_mesh(geom, cfg)
    return geom


def _tail_boom_geometry(cfg: dict) -> tuple[float, float, float, float, float]:
    if not bool(cfg["include_main_wing"]):
        raise ValueError("Tail booms require include_main_wing = True")
    if not bool(cfg["include_horizontal_tail"]):
        raise ValueError("Tail booms require include_horizontal_tail = True")

    main_xqc = float(cfg["main_wing_xqc"])
    tail_xqc = float(cfg["horizontal_tail_xqc"])
    x_front = min(main_xqc, tail_xqc)
    x_back = max(main_xqc, tail_xqc)
    length = max(x_back - x_front, 1e-6)

    center_y = float(cfg["horizontal_tail_y"])
    half_spacing = 0.5 * abs(float(cfg["tail_boom_spacing"]))
    left_y = center_y - half_spacing
    right_y = center_y + half_spacing
    z_location = float(cfg["horizontal_tail_z"]) + float(cfg["tail_boom_z"])

    return x_front, length, left_y, right_y, z_location


def _build_tail_boom_geoms(source_fuselage_geom: ET.Element, cfg: dict) -> list[ET.Element]:
    if not bool(cfg["include_tail_booms"]):
        return []
    x_location, length, left_y, right_y, z_location = _tail_boom_geometry(cfg)
    diameter = float(cfg["tail_boom_diameter"])

    boom_geoms: list[ET.Element] = []
    for name, y_location in (
        ("LeftTailBoomGeom", left_y),
        ("RightTailBoomGeom", right_y),
    ):
        geom = copy.deepcopy(source_fuselage_geom)
        _remap_ids(geom)
        _set_text(geom.find("./ParmContainer"), "n", name)
        _set_text(geom.find("./FuselageGeom/ParmContainer"), "n", name)
        _set_planar_symmetry(geom, 0.0)
        _set_geom_xform(
            geom,
            x_location=x_location,
            y_location=y_location,
            z_location=z_location,
        )

        design = geom.find("./ParmContainer/Design")
        if design is not None and design.find("Length") is not None:
            _set_parm_value(design, "Length", length)

        fuselage_geom = geom.find("./FuselageGeom")
        if fuselage_geom is None:
            continue
        existing_xsec_surf = fuselage_geom.find("./XSecSurf")
        if existing_xsec_surf is not None:
            fuselage_geom.remove(existing_xsec_surf)
        fuselage_geom.append(_build_simple_boom_xsec_surf(length, diameter))
        boom_geoms.append(geom)

    return boom_geoms


def _build_vertical_tail_geoms(template_root: ET.Element, cfg: dict) -> list[ET.Element]:
    layout = str(cfg["vertical_tail_layout"]).strip().lower()

    if layout == "single":
        return [
            _build_vertical_tail_geom(
                template_root,
                cfg,
                name="VerticalTail",
                y_location=float(cfg["vertical_tail_y"]),
                z_location=float(cfg["vertical_tail_z"]),
                xqc=float(cfg["vertical_tail_xqc"]),
                height=float(cfg["vertical_tail_height"]),
                root_chord=float(cfg["vertical_tail_chord"]),
                tip_chord=float(cfg["vertical_tail_tip_chord"]),
                sweep_deg=float(cfg["vertical_tail_sweep_deg"]),
            )
        ]

    if layout == "h_tail":
        if not bool(cfg["include_horizontal_tail"]):
            raise ValueError("H-tail layout requires include_horizontal_tail = True")

        half_span = 0.5 * float(cfg["horizontal_tail_span"])
        inset = float(cfg["h_tail_fin_inset"])
        center_y = float(cfg["horizontal_tail_y"])
        z_location = float(cfg["horizontal_tail_z"]) + float(cfg["h_tail_fin_z_offset"])

        left_y = center_y - max(half_span - inset, 0.0)
        right_y = center_y + max(half_span - inset, 0.0)

        if bool(cfg.get("h_tail_split_verticals")):
            geoms: list[ET.Element] = []
            for side_name, y_location in (
                ("Left", left_y),
                ("Right", right_y),
            ):
                geoms.append(
                    _build_vertical_tail_geom(
                        template_root,
                        cfg,
                        name=f"{side_name}UpperVerticalTail",
                        y_location=y_location,
                        z_location=z_location,
                        xqc=float(cfg["vertical_tail_xqc"]),
                        height=float(cfg["h_tail_upper_height"]),
                        root_chord=float(cfg["h_tail_upper_root_chord"]),
                        tip_chord=float(cfg["h_tail_upper_tip_chord"]),
                        sweep_deg=_split_vtail_sweep(cfg, "h_tail_upper_sweep_deg"),
                        x_rotation=90.0,
                    )
                )
                geoms.append(
                    _build_vertical_tail_geom(
                        template_root,
                        cfg,
                        name=f"{side_name}LowerVerticalTail",
                        y_location=y_location,
                        z_location=z_location,
                        xqc=float(cfg["vertical_tail_xqc"]),
                        height=float(cfg["h_tail_lower_height"]),
                        root_chord=float(cfg["h_tail_lower_root_chord"]),
                        tip_chord=float(cfg["h_tail_lower_tip_chord"]),
                        sweep_deg=_split_vtail_sweep(cfg, "h_tail_lower_sweep_deg"),
                        x_rotation=-90.0,
                    )
                )
            return geoms

        return [
            _build_vertical_tail_geom(
                template_root,
                cfg,
                name="LeftVerticalTail",
                y_location=left_y,
                z_location=z_location,
                xqc=float(cfg["vertical_tail_xqc"]),
                height=float(cfg["vertical_tail_height"]),
                root_chord=float(cfg["vertical_tail_chord"]),
                tip_chord=float(cfg["vertical_tail_tip_chord"]),
                sweep_deg=float(cfg["vertical_tail_sweep_deg"]),
            ),
            _build_vertical_tail_geom(
                template_root,
                cfg,
                name="RightVerticalTail",
                y_location=right_y,
                z_location=z_location,
                xqc=float(cfg["vertical_tail_xqc"]),
                height=float(cfg["vertical_tail_height"]),
                root_chord=float(cfg["vertical_tail_chord"]),
                tip_chord=float(cfg["vertical_tail_tip_chord"]),
                sweep_deg=float(cfg["vertical_tail_sweep_deg"]),
            ),
        ]

    raise ValueError("vertical_tail_layout must be 'single' or 'h_tail'")


def build_design_point_vsp(
    fuselage_stl: Path,
    output_path: Path,
    template_path: Path,
    axis: str,
    slices: int,
    nose_cap: str,
    tail_cap: str,
) -> None:
    if not fuselage_stl.exists():
        raise FileNotFoundError(f"Fuselage STL not found: {fuselage_stl}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template .vsp3 not found: {template_path}")

    cfg = _converted_cfg()
    include_any_surface = (
        bool(cfg["include_main_wing"])
        or bool(cfg["include_horizontal_tail"])
        or bool(cfg["include_vertical_tail"])
    )
    include_any_extra_geom = include_any_surface or bool(cfg["include_tail_booms"])

    if not include_any_extra_geom and not _needs_fuselage_postprocess(cfg):
        with contextlib.redirect_stdout(io.StringIO()):
            stl_to_vsp.convert(
                str(fuselage_stl),
                str(output_path),
                n_slices=int(slices),
                axis=axis,
                name=str(CONFIG["fuselage_name"]),
                nose_cap=nose_cap,
                tail_cap=tail_cap,
            )
        print(f"Written: {output_path}")
        print("Geometry summary:")
        print("  Main wing: omitted")
        print("  Horizontal tail: omitted")
        print("  Vertical tail: omitted")
        print(
            "  Unit conversion: "
            f"{CONFIG['design_units']} -> {CONFIG['stl_units']} "
            f"(x{_unit_scale_factor(CONFIG['design_units'], CONFIG['stl_units']):.6f})"
        )
        return

    fuselage_tree = _extract_fuselage_root(
        fuselage_stl,
        axis=axis,
        slices=slices,
        nose_cap=nose_cap,
        tail_cap=tail_cap,
        name=str(CONFIG["fuselage_name"]),
    )
    out_root = fuselage_tree.getroot()
    vehicle = out_root.find("./Vehicle")
    if vehicle is None:
        raise ValueError("Generated fuselage file is missing a Vehicle node")

    fuselage_geom = _find_first_geom_by_type(out_root, "Fuselage")
    if fuselage_geom is not None:
        _apply_fuselage_position(fuselage_geom, cfg)
        _apply_fuselage_rotation(fuselage_geom, cfg)

    if include_any_surface:
        template_tree = ET.parse(template_path)
        template_root = template_tree.getroot()

        if bool(cfg["include_main_wing"]):
            vehicle.append(_build_main_wing_geom(template_root, cfg))

        if bool(cfg["include_horizontal_tail"]):
            vehicle.append(_build_horizontal_tail_geom(template_root, cfg))

        if bool(cfg["include_vertical_tail"]):
            for geom in _build_vertical_tail_geoms(template_root, cfg):
                vehicle.append(geom)

    if fuselage_geom is not None and bool(cfg["include_tail_booms"]):
        for geom in _build_tail_boom_geoms(fuselage_geom, cfg):
            vehicle.append(geom)

    if hasattr(ET, "indent"):
        ET.indent(fuselage_tree, space="  ")
    fuselage_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    applied_airfoils = _apply_airfoil_files_with_openvsp(output_path, cfg)

    print(f"Written: {output_path}")
    print("Geometry summary:")
    print(
        "  Main wing: "
        + (
            f"{cfg['main_wing_span']:.4f} span / {cfg['main_wing_chord']:.4f} chord"
            if bool(cfg["include_main_wing"])
            else "omitted"
        )
    )
    if bool(cfg["include_horizontal_tail"]):
        print(
            "  Horizontal tail span/chord: "
            f"{cfg['horizontal_tail_span']:.4f} / {cfg['horizontal_tail_chord']:.4f}"
        )
    else:
        print("  Horizontal tail: omitted")
    print(
        "  Vertical tail: "
        + (
            (
                f"h_tail split "
                f"(upper {cfg['h_tail_upper_height']:.4f} high, "
                f"{cfg['h_tail_upper_root_chord']:.4f}->{cfg['h_tail_upper_tip_chord']:.4f}; "
                f"lower {cfg['h_tail_lower_height']:.4f} high, "
                f"{cfg['h_tail_lower_root_chord']:.4f}->{cfg['h_tail_lower_tip_chord']:.4f})"
                if str(cfg["vertical_tail_layout"]).strip().lower() == "h_tail"
                and bool(cfg.get("h_tail_split_verticals"))
                else f"{cfg['vertical_tail_layout']} "
                f"({cfg['vertical_tail_height']:.4f} high, "
                f"{cfg['vertical_tail_chord']:.4f}->{cfg['vertical_tail_tip_chord']:.4f} chord)"
            )
            if bool(cfg["include_vertical_tail"])
            else "omitted"
        )
    )
    if bool(cfg["include_tail_booms"]):
        boom_x, boom_len, boom_left_y, boom_right_y, _boom_z = _tail_boom_geometry(cfg)
        print(
            "  Tail booms: "
            f"2 x {boom_len:.4f} long / {cfg['tail_boom_diameter']:.4f} dia"
        )
        print(
            "  Tail boom spanwise centers: "
            f"{boom_left_y:.4f}, {boom_right_y:.4f}  (front X = {boom_x:.4f})"
        )
    else:
        print("  Tail booms: omitted")
    print(
        "  Lifting surface mesh: "
        f"Tess_U {cfg['lifting_surface_tess_u']:.0f}, "
        f"Tess_W {cfg['lifting_surface_tess_w']:.0f}, "
        f"SectTess_U {cfg['lifting_surface_sect_tess_u']:.0f}"
    )
    print(
        "  Unit conversion: "
        f"{CONFIG['design_units']} -> {CONFIG['stl_units']} "
        f"(x{_unit_scale_factor(CONFIG['design_units'], CONFIG['stl_units']):.6f})"
    )
    if applied_airfoils:
        print("  Imported airfoils:")
        for item in applied_airfoils:
            print(f"    {item}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a combined OpenVSP model from a fuselage STL and template wing/tail geoms"
    )
    parser.add_argument("fuselage_stl", help="Input fuselage STL path")
    parser.add_argument("output", help="Output .vsp3 path")
    parser.add_argument("--template", default=None, help="Override the template .vsp3 path")
    parser.add_argument("--axis", default=str(CONFIG["fuselage_axis"]), choices=("X", "Y", "Z"))
    parser.add_argument("--slices", type=int, default=int(CONFIG["fuselage_slices"]))
    parser.add_argument("--nose-cap", default=str(CONFIG["fuselage_nose_cap"]), choices=("point", "flat"))
    parser.add_argument("--tail-cap", default=str(CONFIG["fuselage_tail_cap"]), choices=("point", "flat"))
    parser.add_argument(
        "--main-airfoil",
        default=None,
        help="Optional Selig/Lednicer airfoil file for the main wing (.dat/.af)",
    )
    parser.add_argument(
        "--htail-airfoil",
        default=None,
        help="Optional Selig/Lednicer airfoil file for the horizontal tail (.dat/.af)",
    )
    parser.add_argument(
        "--vtail-airfoil",
        default=None,
        help="Optional Selig/Lednicer airfoil file for the vertical tail(s) (.dat/.af)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.main_airfoil is not None:
        CONFIG["main_wing_airfoil_dat"] = args.main_airfoil
    if args.htail_airfoil is not None:
        CONFIG["horizontal_tail_airfoil_dat"] = args.htail_airfoil
    if args.vtail_airfoil is not None:
        CONFIG["vertical_tail_airfoil_dat"] = args.vtail_airfoil

    template_path = Path(args.template) if args.template else Path(CONFIG["template_path"])

    build_design_point_vsp(
        fuselage_stl=Path(args.fuselage_stl),
        output_path=Path(args.output),
        template_path=template_path,
        axis=args.axis,
        slices=int(args.slices),
        nose_cap=args.nose_cap,
        tail_cap=args.tail_cap,
    )


if __name__ == "__main__":
    main()
