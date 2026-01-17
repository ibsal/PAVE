from dataclasses import dataclass
import math
from ambiance import Atmosphere


@dataclass
class MissionState:
    t: float
    x: float
    h: float
    v: float
    gamma: float
    weight: float
    energy_j: float
    phase: str


@dataclass
class MissionProfile:
    dt: float = 0.2
    takeoff_mu: float = 0.02
    takeoff_aoa: float = 0.0
    v_rot: float | None = None
    takeoff_dt: float | None = None
    takeoff_elevator_start_speed: float | None = None
    takeoff_elevator_target_deg: float = 10.0
    takeoff_elevator_rate_deg_s: float = 30.0
    takeoff_aero_stride: int = 1
    takeoff_stall_margin: float = 1.20
    takeoff_climb_margin: float = 1.25
    level_flight_stall_margin: float = 1.25
    landing_stall_margin: float = 1.30
    max_time: float = 4000000.0
    trim_update_dt: float = 5.0
    trim_max_iter: int = 50
    trim_gamma_iters: int = 2
    trim_max_iter_opt: int = 50
    segments: list | None = None
    log_interval: float = 2.0
    speed_opt_points: int = 10
    speed_opt_mode: str = "grid"
    n_span: int | None = None
    power_margin_frac: float = 0.2
    elevator_limit_deg: float = 30.0
    power_derate_frac: float | None = None
    energy_reserve_frac: float | None = None
    max_aoa_deg: float | None = None
    wind_mps: float = 0.0
    loiter_wind_mode: str = "head_tail_avg"


@dataclass
class MissionSegment:
    kind: str
    target_alt: float | None = None
    speed: float | None = None
    speed_min: float | None = None
    speed_max: float | None = None
    time: float | None = None
    distance: float | None = None
    bank_deg: float = 0.0
    dt: float | None = None
    mode: str | None = None
    elevator_deg: float | None = None
    thrust_scale: float | None = None
    wind_mps: float | None = None


class AircraftModel:
    def __init__(self, config):
        self.config = config

    def thrust_available(self, v, rho, power_limit_w=None):
        prop = self.config["propulsion"]
        max_power = prop["max_power_w"]
        if power_limit_w is not None:
            max_power = min(max_power, power_limit_w)
        prop_eff = prop["prop_eff"]
        max_thrust = prop.get("max_thrust_n")
        if v <= 0.1:
            thrust = max_power * prop_eff / 0.1
        else:
            thrust = max_power * prop_eff / v
        if max_thrust is not None:
            thrust = min(thrust, max_thrust)
        return thrust

    def non_lifting_drag_stack(self, rho, mu, v):
        cfg = self.config
        fuselage = cfg["fuselage"]
        boom = cfg["boom"]
        wing = cfg["wing"]
        skin_friction_cf = cfg["aero_funcs"]["skin_friction_cf"]

        components = []
        ff_list = []
        q_list = []
        cfc_list = []
        sratio_list = []
        re_list = []

        re_fuse = rho * v * fuselage["length"] / mu
        re_boom = rho * v * boom["length"] / mu

        components.append("Fuselage")
        fuse_ld = fuselage["length"] / fuselage["diameter"]
        ff_list.append(1 + 60/(fuse_ld**3) + fuse_ld/400)
        q_list.append(1.0)
        cfc_list.append(skin_friction_cf(re_fuse, cfg["roughness"]["k"], fuselage["length"], laminar_frac=cfg["roughness"]["fuselage_lam"], re_crit_per_m=cfg["roughness"]["re_crit_per_m"]))
        fuse_term = max(1.0 - 2.0 / max(fuse_ld, 1e-6), 0.0)
        sratio_list.append((math.pi * fuselage["diameter"] * fuselage["length"] * math.pow(fuse_term, 2.0/3.0) * (1 + 1/(fuse_ld**2))) / wing["area"])
        re_list.append(re_fuse)

        components.append("Boom")
        boom_ld = boom["length"] / boom["diameter"]
        boom_ff = 1 + 60/(boom_ld**3) + boom_ld/400
        ff_list.append(boom_ff)
        q_list.append(1.0)
        cfc_list.append(skin_friction_cf(re_boom, cfg["roughness"]["k"], boom["length"], laminar_frac=cfg["roughness"]["boom_lam"], re_crit_per_m=cfg["roughness"]["re_crit_per_m"]))
        boom_term = max(1.0 - 2.0 / max(boom_ld, 1e-6), 0.0)
        boom_sratio = (math.pi * boom["diameter"] * boom["length"] * math.pow(boom_term, 2.0/3.0) * (1 + 1/(boom_ld**2))) / wing["area"]
        sratio_list.append(boom_sratio * boom["count"])
        re_list.append(re_boom)

        cd_non_lifting = 0.0
        non_lifting_cdo = []
        for ff, q, cfc, sw in zip(ff_list, q_list, cfc_list, sratio_list):
            cdo = ff * q * cfc * sw
            cd_non_lifting += cdo
            non_lifting_cdo.append(cdo)

        return cd_non_lifting, non_lifting_cdo, components, cfc_list, re_list

    def aero(self, flight_aoa, elevator_deflection, v, rho, mu):
        cfg = self.config
        wing = cfg["wing"]
        htail = cfg["htail"]
        vtail = cfg["vtail"]
        integrate_profile_drag = cfg["aero_funcs"]["integrate_profile_drag"]
        sweep_correction_factor = cfg["aero_funcs"]["sweep_correction_factor"]
        integrate_pitching_moment_about_cg = cfg["aero_funcs"]["integrate_pitching_moment_about_cg"]

        alpha = flight_aoa + wing["incidence"]

        cd_profile, cl, _ = integrate_profile_drag(
            wing["foil"],
            wing["root_chord"],
            wing["tip_chord"],
            wing["span"],
            wing["area"],
            alpha,
            rho,
            mu,
            v,
            n=cfg["n_span"],
            symmetric=True,
            chord_breaks=wing["chord_breaks"],
            sweep_breaks=wing["sweep_breaks"],
        )

        e_oswald_w = 1.78*(1.0 - 0.045*(wing["ar"]**0.68)) - 0.64
        e_oswald_w = min(max(e_oswald_w, 0.3), 0.95)
        ar_eff = wing["ar"] * sweep_correction_factor(wing["span"], cfg["n_span"], True, wing["sweep_breaks"])
        cd_w_induced = cl**2/(math.pi * ar_eff * e_oswald_w)

        cd_vtail, _, _ = integrate_profile_drag(
            vtail["foil"],
            vtail["root_chord"],
            vtail["tip_chord"],
            vtail["span"],
            wing["area"],
            vtail["incidence"],
            rho,
            mu,
            v,
            n=cfg["n_span"],
            count=vtail["count"],
            symmetric=False,
            chord_breaks=vtail["chord_breaks"],
            sweep_breaks=vtail["sweep_breaks"],
        )

        downwash = alpha * cfg["downwash_factor"]
        htail_aoa = flight_aoa + htail["incidence"] - downwash + cfg["elevator_tau"] * elevator_deflection
        cd_htail, cl_htail, _ = integrate_profile_drag(
            htail["foil"],
            htail["root_chord"],
            htail["tip_chord"],
            htail["span"],
            wing["area"],
            htail_aoa,
            rho,
            mu,
            v,
            n=cfg["n_span"],
            symmetric=True,
            chord_breaks=htail["chord_breaks"],
            sweep_breaks=htail["sweep_breaks"],
        )
        cd_htail *= cfg["tail_efficiency"]
        cl_htail *= cfg["tail_efficiency"]

        e_oswald_h = 1.78*(1.0 - 0.045*(htail["ar"]**0.68)) - 0.64
        e_oswald_h = min(max(e_oswald_h, 0.3), 0.95)
        har_eff = htail["ar"] * sweep_correction_factor(htail["span"], cfg["n_span"], True, htail["sweep_breaks"])
        cd_htail_induced = (cl_htail**2) * (wing["area"] / htail["area"]) / (math.pi * har_eff * e_oswald_h)

        cd_non_lifting, _, _, _, _ = self.non_lifting_drag_stack(rho, mu, v)

        cd0 = cfg["cd_misc"] + cd_profile + cd_non_lifting + cd_vtail + cd_htail
        cd_induced = cd_w_induced + cd_htail_induced
        cd_total = cd0 + cd_induced

        q_dyn = 0.5 * rho * (v ** 2)
        lift = q_dyn * wing["area"] * (cl + cl_htail)
        drag = q_dyn * wing["area"] * cd_total

        wing_moment, wing_cm_moment = integrate_pitching_moment_about_cg(
            wing["foil"],
            wing["root_chord"],
            wing["tip_chord"],
            wing["span"],
            alpha,
            rho,
            mu,
            v,
            q_dyn,
            cfg["positions"]["x_wqc"],
            cfg["positions"]["x_cg"],
            n=cfg["n_span"],
            symmetric=True,
            chord_breaks=wing["chord_breaks"],
            sweep_breaks=wing["sweep_breaks"],
        )
        htail_moment, htail_cm_moment = integrate_pitching_moment_about_cg(
            htail["foil"],
            htail["root_chord"],
            htail["tip_chord"],
            htail["span"],
            htail_aoa,
            rho,
            mu,
            v,
            q_dyn,
            cfg["positions"]["x_htqc"],
            cfg["positions"]["x_cg"],
            n=cfg["n_span"],
            symmetric=True,
            chord_breaks=htail["chord_breaks"],
            sweep_breaks=htail["sweep_breaks"],
        )
        htail_moment *= cfg["tail_efficiency"]
        htail_cm_moment *= cfg["tail_efficiency"]
        total_moment = wing_moment + wing_cm_moment + htail_moment + htail_cm_moment

        return {
            "Lift": lift,
            "Drag": drag,
            "TotalPitchMoment": total_moment,
            "Cl": cl,
            "ClTail": cl_htail,
            "ClTotal": cl + cl_htail,
            "Cd": cd_total,
            "Cd0": cd0,
            "CdInduced": cd_induced,
            "HTailAOA": htail_aoa,
        }

    def trim(self, v, rho, mu, weight, aoa_guess, elev_guess, max_iter=20, tol_force=1e-3, tol_moment=1e-3, elevator_limit_deg=None, aoa_min_deg=None, aoa_max_deg=None):
        aoa = aoa_guess
        elev = elev_guess
        aoa_clamped = False
        elev_clamped = False
        if elevator_limit_deg is not None:
            elev_new = max(min(elev, elevator_limit_deg), -elevator_limit_deg)
            elev_clamped = elev_clamped or (elev_new != elev)
            elev = elev_new
        if aoa_min_deg is not None:
            aoa_new = max(aoa, aoa_min_deg)
            aoa_clamped = aoa_clamped or (aoa_new != aoa)
            aoa = aoa_new
        if aoa_max_deg is not None:
            aoa_new = min(aoa, aoa_max_deg)
            aoa_clamped = aoa_clamped or (aoa_new != aoa)
            aoa = aoa_new
        converged = False
        last_result = None
        force_tol = max(tol_force, 1e-3 * max(weight, 0.0))
        moment_scale = max(weight, 0.0) * self.config["wing"]["mac"]
        moment_tol = max(tol_moment, 1e-3 * moment_scale)
        last_f1 = None
        last_f2 = None
        for _ in range(max_iter):
            result = self.aero(aoa, elev, v, rho, mu)
            result = dict(result)
            result["AOA"] = aoa
            result["ElevatorDeg"] = elev
            last_result = result
            f1 = result["Lift"] - weight
            f2 = result["TotalPitchMoment"]
            last_f1 = f1
            last_f2 = f2
            if abs(f1) < force_tol and abs(f2) < moment_tol:
                converged = True
                break
            d_aoa = 0.25
            d_elev = 0.5
            result_aoa = self.aero(aoa + d_aoa, elev, v, rho, mu)
            result_elev = self.aero(aoa, elev + d_elev, v, rho, mu)
            f1_aoa = result_aoa["Lift"] - weight
            f2_aoa = result_aoa["TotalPitchMoment"]
            f1_elev = result_elev["Lift"] - weight
            f2_elev = result_elev["TotalPitchMoment"]
            j11 = (f1_aoa - f1) / d_aoa
            j12 = (f1_elev - f1) / d_elev
            j21 = (f2_aoa - f2) / d_aoa
            j22 = (f2_elev - f2) / d_elev
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-9:
                break
            delta_aoa = (-f1 * j22 + f2 * j12) / det
            delta_elev = (-j11 * f2 + j21 * f1) / det
            aoa += delta_aoa
            elev += delta_elev
            if elevator_limit_deg is not None:
                elev_new = max(min(elev, elevator_limit_deg), -elevator_limit_deg)
                elev_clamped = elev_clamped or (elev_new != elev)
                elev = elev_new
            if aoa_min_deg is not None:
                aoa_new = max(aoa, aoa_min_deg)
                aoa_clamped = aoa_clamped or (aoa_new != aoa)
                aoa = aoa_new
            if aoa_max_deg is not None:
                aoa_new = min(aoa, aoa_max_deg)
                aoa_clamped = aoa_clamped or (aoa_new != aoa)
                aoa = aoa_new
        if last_result is not None:
            last_result["Converged"] = converged
            last_result["AoaClamped"] = aoa_clamped
            last_result["ElevatorClamped"] = elev_clamped
            last_result["LiftResidual"] = last_f1 if last_f1 is not None else 0.0
            last_result["MomentResidual"] = last_f2 if last_f2 is not None else 0.0
        return aoa, elev, converged, last_result

    def trim_with_fixed_elevator(self, v, rho, mu, weight, elevator_deg, aoa_guess=2.0, max_iter=8, tol_force=1e-3, aoa_min_deg=None, aoa_max_deg=None):
        aoa = aoa_guess
        aoa_clamped = False
        if aoa_min_deg is not None:
            aoa_new = max(aoa, aoa_min_deg)
            aoa_clamped = aoa_clamped or (aoa_new != aoa)
            aoa = aoa_new
        if aoa_max_deg is not None:
            aoa_new = min(aoa, aoa_max_deg)
            aoa_clamped = aoa_clamped or (aoa_new != aoa)
            aoa = aoa_new
        last_result = None
        force_tol = max(tol_force, 1e-3 * max(weight, 0.0))
        last_f1 = None
        for _ in range(max_iter):
            result = self.aero(aoa, elevator_deg, v, rho, mu)
            result = dict(result)
            result["AOA"] = aoa
            result["ElevatorDeg"] = elevator_deg
            last_result = result
            f1 = result["Lift"] - weight
            last_f1 = f1
            if abs(f1) < force_tol:
                result["Converged"] = True
                return aoa, True, last_result
            d_aoa = 0.25
            result_aoa = self.aero(aoa + d_aoa, elevator_deg, v, rho, mu)
            f1_aoa = result_aoa["Lift"] - weight
            j11 = (f1_aoa - f1) / d_aoa
            if abs(j11) < 1e-9:
                break
            aoa -= f1 / j11
            if aoa_min_deg is not None:
                aoa_new = max(aoa, aoa_min_deg)
                aoa_clamped = aoa_clamped or (aoa_new != aoa)
                aoa = aoa_new
            if aoa_max_deg is not None:
                aoa_new = min(aoa, aoa_max_deg)
                aoa_clamped = aoa_clamped or (aoa_new != aoa)
                aoa = aoa_new
        if last_result is not None:
            last_result["Converged"] = False
            last_result["AoaClamped"] = aoa_clamped
            last_result["ElevatorClamped"] = False
            last_result["LiftResidual"] = last_f1 if last_f1 is not None else 0.0
            last_result["MomentResidual"] = 0.0
        return aoa, False, last_result


def simulate_mission(model, profile, initial_alt, return_summary=False):
    cfg = model.config
    orig_n_span = cfg.get("n_span")
    if profile.n_span is not None:
        cfg["n_span"] = profile.n_span
    state = MissionState(
        t=0.0,
        x=0.0,
        h=initial_alt,
        v=0.0,
        gamma=0.0,
        weight=cfg["weight"],
        energy_j=cfg["propulsion"]["battery_energy_j"],
        phase="takeoff",
    )
    history = [state]
    orbit_time = 0.0
    segment_index = 0
    segment_time = 0.0
    segment_distance = 0.0
    segments = profile.segments or []
    def next_non_takeoff_segment(start_index):
        idx = start_index
        while idx < len(segments) and segments[idx].kind == "takeoff":
            idx += 1
        return idx

    def segment_wind_mps(seg):
        if seg is None:
            return 0.0
        wind = seg.wind_mps if seg.wind_mps is not None else profile.wind_mps
        if seg.kind == "loiter" and profile.loiter_wind_mode == "head_tail_avg":
            return 0.0
        return wind
    next_log_time = 0.0
    last_aero = None
    takeoff_elevator = 0.0
    last_gamma = 0.0
    takeoff_step = 0
    last_takeoff_aero = None
    last_takeoff_v = 0.1
    segment_time_limit = {}
    segment_elevator_lock = {}
    segment_summaries = []
    loiter_accum = {}
    segment_start_energy = None
    segment_start_x = None
    segment_start_t = None
    segment_label = "takeoff"
    current_segment_index = None
    segment_kind = "takeoff"
    last_thrust_used = 0.0
    cruise_trim_segment_index = None
    cruise_trim_v = None
    cruise_trim_result = None
    cruise_trim_gamma = 0.0
    cruise_trim_thrust = 0.0

    def foil_alpha_limits(foil):
        if hasattr(foil, "polars"):
            mins = [float(min(p.alpha_deg)) for p in foil.polars]
            maxs = [float(max(p.alpha_deg)) for p in foil.polars]
            return min(mins), max(maxs)
        if hasattr(foil, "alpha_deg"):
            return float(min(foil.alpha_deg)), float(max(foil.alpha_deg))
        return None, None

    aoa_min_deg, aoa_max_deg = foil_alpha_limits(cfg["wing"]["foil"])
    wing_incidence = cfg["wing"]["incidence"]
    if aoa_min_deg is not None:
        aoa_min_deg -= wing_incidence
    if aoa_max_deg is not None:
        aoa_max_deg -= wing_incidence
    if profile.max_aoa_deg is not None:
        aoa_max_deg = profile.max_aoa_deg

    power_derate = profile.power_derate_frac
    energy_reserve = profile.energy_reserve_frac
    if power_derate is None and energy_reserve is None:
        power_derate = 0.0
        energy_reserve = profile.power_margin_frac
    if power_derate is None:
        power_derate = 0.0
    if energy_reserve is None:
        energy_reserve = profile.power_margin_frac
    power_derate = max(0.0, min(power_derate, 0.9))
    energy_reserve = max(0.0, min(energy_reserve, 0.9))
    power_limit = cfg["propulsion"]["max_power_w"] * (1.0 - power_derate)
    mission_systems_power = cfg["propulsion"].get("mission_systems_power", 0.0)
    if mission_systems_power is None:
        mission_systems_power = 0.0
    mission_systems_power = max(mission_systems_power, 0.0)

    def available_thrust(v, rho):
        return model.thrust_available(v, rho, power_limit_w=power_limit)

    def solve_trim(v, rho, mu, weight_eff, max_iter):
        _, _, _, trim_result = model.trim(
            v,
            rho,
            mu,
            weight_eff,
            2.0,
            0.0,
            max_iter=max_iter,
            elevator_limit_deg=profile.elevator_limit_deg,
            aoa_min_deg=aoa_min_deg,
            aoa_max_deg=aoa_max_deg,
        )
        return trim_result

    def solve_trim_fixed_elevator(v, rho, mu, weight_eff, elevator_deg, max_iter):
        _, _, trim_result = model.trim_with_fixed_elevator(
            v,
            rho,
            mu,
            weight_eff,
            elevator_deg,
            max_iter=max_iter,
            aoa_min_deg=aoa_min_deg,
            aoa_max_deg=aoa_max_deg,
        )
        return trim_result

    def trim_with_gamma(v, rho, mu, weight, bank_deg, thrust_scale):
        gamma = last_gamma
        trim_result = None
        for _ in range(profile.trim_gamma_iters):
            load_factor = 1.0 / max(math.cos(bank_deg), 0.1)
            weight_eff = weight * load_factor * math.cos(gamma)
            trim_result = solve_trim(v, rho, mu, weight_eff, profile.trim_max_iter)
            thrust = available_thrust(v, rho) * thrust_scale
            drag = trim_result["Drag"]
            gamma = math.asin(max(min((thrust - drag) / weight, 0.3), -0.3))
        return gamma, trim_result

    def trim_with_gamma_fixed_elevator(v, rho, mu, weight, bank_deg, thrust_scale, elevator_deg):
        gamma = last_gamma
        trim_result = None
        for _ in range(profile.trim_gamma_iters):
            load_factor = 1.0 / max(math.cos(bank_deg), 0.1)
            weight_eff = weight * load_factor * math.cos(gamma)
            trim_result = solve_trim_fixed_elevator(
                v,
                rho,
                mu,
                weight_eff,
                elevator_deg,
                profile.trim_max_iter_opt,
            )
            thrust = available_thrust(v, rho) * thrust_scale
            drag = trim_result["Drag"]
            gamma = math.asin(max(min((thrust - drag) / weight, 0.3), -0.3))
        return gamma, trim_result

    def optimize_speed(v_min, v_max, mode, weight_eff, thrust_scale, elevator_deg=None, return_trim=False):
        if v_min is None or v_max is None or v_max <= v_min:
            best_v = v_min if v_min is not None else max(state.v, 0.1)
            if not return_trim:
                return best_v
            if elevator_deg is None:
                trim_result = solve_trim(best_v, rho, mu, weight_eff, profile.trim_max_iter_opt)
            else:
                trim_result = solve_trim_fixed_elevator(
                    best_v,
                    rho,
                    mu,
                    weight_eff,
                    elevator_deg,
                    profile.trim_max_iter_opt,
                )
            return best_v, trim_result
        best_v = v_min
        best_metric = None
        best_trim = None
        points = max(profile.speed_opt_points, 3)
        if profile.speed_opt_mode == "cheap":
            samples = [v_min, 0.5 * (v_min + v_max), v_max]
        else:
            samples = [v_min + (v_max - v_min) * i / (points - 1) for i in range(points)]
        for v in samples:
            if elevator_deg is None:
                trim_result = solve_trim(v, rho, mu, weight_eff, profile.trim_max_iter_opt)
            else:
                trim_result = solve_trim_fixed_elevator(
                    v,
                    rho,
                    mu,
                    weight_eff,
                    elevator_deg,
                    profile.trim_max_iter_opt,
                )
            drag = trim_result["Drag"]
            thrust = available_thrust(v, rho) * thrust_scale
            if mode == "max_roc":
                metric = (thrust - drag) * v
            elif mode == "min_energy":
                metric = -drag
            elif mode == "max_endurance":
                metric = -drag * v
            elif mode == "min_time":
                gamma = math.asin(max(min((thrust - drag) / state.weight, 0.3), -0.3))
                metric = -v * math.sin(gamma)
            else:
                metric = (thrust - drag) * v
            if best_metric is None or metric > best_metric:
                best_metric = metric
                best_v = v
                best_trim = trim_result
        if return_trim:
            return best_v, best_trim
        return best_v

    def estimate_cd0_total(rho, mu, v):
        wing = cfg["wing"]
        htail = cfg["htail"]
        vtail = cfg["vtail"]
        integrate_profile_drag = cfg["aero_funcs"]["integrate_profile_drag"]

        flight_aoa = -wing["incidence"]
        alpha = flight_aoa + wing["incidence"]
        cd_profile, _, _ = integrate_profile_drag(
            wing["foil"],
            wing["root_chord"],
            wing["tip_chord"],
            wing["span"],
            wing["area"],
            alpha,
            rho,
            mu,
            v,
            n=cfg["n_span"],
            symmetric=True,
            chord_breaks=wing["chord_breaks"],
            sweep_breaks=wing["sweep_breaks"],
        )

        cd_vtail, _, _ = integrate_profile_drag(
            vtail["foil"],
            vtail["root_chord"],
            vtail["tip_chord"],
            vtail["span"],
            wing["area"],
            vtail["incidence"],
            rho,
            mu,
            v,
            n=cfg["n_span"],
            count=vtail["count"],
            symmetric=False,
            chord_breaks=vtail["chord_breaks"],
            sweep_breaks=vtail["sweep_breaks"],
        )

        downwash = alpha * cfg["downwash_factor"]
        htail_aoa = flight_aoa + htail["incidence"] - downwash
        cd_htail, _, _ = integrate_profile_drag(
            htail["foil"],
            htail["root_chord"],
            htail["tip_chord"],
            htail["span"],
            wing["area"],
            htail_aoa,
            rho,
            mu,
            v,
            n=cfg["n_span"],
            symmetric=True,
            chord_breaks=htail["chord_breaks"],
            sweep_breaks=htail["sweep_breaks"],
        )
        cd_htail *= cfg["tail_efficiency"]

        cd_non_lifting, _, _, _, _ = model.non_lifting_drag_stack(rho, mu, v)
        return cd_non_lifting + cd_profile + cd_vtail + cd_htail + cfg["cd_misc"]

    def endurance_speed_from_eq(weight, rho, v_seed, load_factor=1.0):
        wing = cfg["wing"]
        weight_eff = weight * load_factor
        ar = wing["ar"]
        e_oswald = 1.78 * (1.0 - 0.045 * (ar ** 0.68)) - 0.64
        e_oswald = min(max(e_oswald, 0.3), 0.95)
        k = 1.0 / (math.pi * ar * e_oswald)
        v = v_seed
        for _ in range(2):
            cd0 = estimate_cd0_total(rho, mu, v)
            v = math.sqrt((2.0 * weight_eff / (rho * wing["area"])) * math.sqrt(k / (3.0 * cd0)))
        return v

    def update_loiter_metrics(seg_index, dt, trim_result, v, rho):
        if trim_result is None:
            return
        acc = loiter_accum.setdefault(seg_index, {
            "time_s": 0.0,
            "cl_sum": 0.0,
            "cd_sum": 0.0,
            "cdo_sum": 0.0,
            "ld_sum": 0.0,
            "lift_sum": 0.0,
            "drag_sum": 0.0,
            "aoa_sum": 0.0,
            "elev_sum": 0.0,
            "samples": 0,
        })
        q_dyn = 0.5 * rho * (v ** 2)
        cl_total = trim_result.get("ClTotal")
        if cl_total is None:
            cl_total = trim_result["Lift"] / (q_dyn * cfg["wing"]["area"])
        cd_total = trim_result.get("Cd")
        if cd_total is None:
            cd_total = trim_result["Drag"] / (q_dyn * cfg["wing"]["area"])
        cdo = trim_result.get("Cd0", 0.0)
        lift = trim_result.get("Lift", 0.0)
        drag = trim_result.get("Drag", 0.0)
        ld = (lift / drag) if drag > 0.0 else 0.0
        aoa = trim_result.get("AOA", 0.0)
        elev = trim_result.get("ElevatorDeg", 0.0)
        acc["time_s"] += dt
        acc["cl_sum"] += cl_total * dt
        acc["cd_sum"] += cd_total * dt
        acc["cdo_sum"] += cdo * dt
        acc["ld_sum"] += ld * dt
        acc["lift_sum"] += lift * dt
        acc["drag_sum"] += drag * dt
        acc["aoa_sum"] += aoa * dt
        acc["elev_sum"] += elev * dt
        acc["samples"] += 1

    def loiter_summary(seg_index):
        acc = loiter_accum.get(seg_index)
        if not acc or acc["time_s"] <= 0.0:
            return {}
        t = acc["time_s"]
        return {
            "aoa_deg": acc["aoa_sum"] / t,
            "elev_deg": acc["elev_sum"] / t,
            "cl_avg": acc["cl_sum"] / t,
            "cd_avg": acc["cd_sum"] / t,
            "cdo_avg": acc["cdo_sum"] / t,
            "ld_avg": acc["ld_sum"] / t,
            "lift_avg_n": acc["lift_sum"] / t,
            "drag_avg_n": acc["drag_sum"] / t,
        }

    try:
        segment_start_energy = state.energy_j
        segment_start_x = state.x
        segment_start_t = state.t
        while state.t < profile.max_time:
            atm = Atmosphere(cfg["ground_level"] + state.h)
            rho = float(atm.density)
            mu = float(atm.dynamic_viscosity)
            prev_phase = state.phase
            prev_segment_index = segment_index

            step_dt = profile.dt
            thrust_used = 0.0
            if state.phase == "takeoff":
                takeoff_step += 1
                if profile.takeoff_dt is not None:
                    step_dt = profile.takeoff_dt
                elev_start = profile.takeoff_elevator_start_speed
                if elev_start is None:
                    elev_start = profile.v_rot if profile.v_rot is not None else 0.0
                if state.v >= elev_start:
                    takeoff_elevator = min(
                        profile.takeoff_elevator_target_deg,
                        takeoff_elevator + profile.takeoff_elevator_rate_deg_s * step_dt,
                    )
                eval_aero = (profile.takeoff_aero_stride <= 1) or (takeoff_step % profile.takeoff_aero_stride == 0) or (last_takeoff_aero is None)
                if eval_aero:
                    aero = model.aero(profile.takeoff_aoa, takeoff_elevator, max(state.v, 0.1), rho, mu)
                    last_takeoff_aero = aero
                    last_takeoff_v = max(state.v, 0.1)
                    lift = aero["Lift"]
                    drag = aero["Drag"]
                else:
                    v_scale = (max(state.v, 0.1) / last_takeoff_v) ** 2
                    lift = last_takeoff_aero["Lift"] * v_scale
                    drag = last_takeoff_aero["Drag"] * v_scale
                    aero = last_takeoff_aero
                last_aero = dict(aero)
                last_aero["AOA"] = profile.takeoff_aoa
                last_aero["ElevatorDeg"] = takeoff_elevator
                thrust = available_thrust(max(state.v, 0.1), rho)
                mass = state.weight / 9.81
                normal = max(state.weight - lift, 0.0)
                accel = (thrust - drag - profile.takeoff_mu * normal) / mass
                v_next = max(state.v + accel * step_dt, 0.0)
                x_next = state.x + 0.5 * (state.v + v_next) * step_dt
                if eval_aero or v_next >= v_min_takeoff:
                    aero_next = model.aero(profile.takeoff_aoa, takeoff_elevator, max(v_next, 0.1), rho, mu)
                    lift_next = aero_next["Lift"]
                else:
                    v_scale_next = (max(v_next, 0.1) / last_takeoff_v) ** 2
                    lift_next = last_takeoff_aero["Lift"] * v_scale_next
                cl_max = cfg["wing"]["cl_max_takeoff"]
                v_stall = math.sqrt(2.0 * state.weight / (rho * cfg["wing"]["area"] * cl_max))
                v2 = profile.takeoff_stall_margin * v_stall
                v_climb = profile.takeoff_climb_margin * v_stall
                v_rot = profile.v_rot if profile.v_rot is not None else v2
                v_min_takeoff = max(v_rot, v2, v_climb)
                if v_next >= v_min_takeoff and lift_next >= state.weight:
                    last_aero = dict(aero_next)
                    last_aero["AOA"] = profile.takeoff_aoa
                    last_aero["ElevatorDeg"] = takeoff_elevator
                    if segments:
                        segment_index = next_non_takeoff_segment(segment_index)
                        phase = segments[segment_index].kind if segment_index < len(segments) else "landed"
                    else:
                        phase = "climb"
                else:
                    phase = "takeoff"
                h_next = state.h
                gamma_next = 0.0
                thrust_used = available_thrust(max(v_next, 0.1), rho)
            else:
                if segments:
                    if segment_index >= len(segments):
                        break
                    seg = segments[segment_index]
                    if seg.kind == "takeoff":
                        segment_index = next_non_takeoff_segment(segment_index)
                        seg = segments[segment_index] if segment_index < len(segments) else None
                        if seg is None:
                            break
                    if seg.dt is not None:
                        step_dt = seg.dt
                    v_next = seg.speed if seg.speed is not None else max(state.v, 0.1)
                    bank_rad = math.radians(seg.bank_deg)
                    load_factor = 1.0 / max(math.cos(bank_rad), 0.1)
                    weight_eff = state.weight * load_factor
                    stall_weight = weight_eff * max(math.cos(last_gamma), 0.1)
    
                    if seg.kind == "climb_to":
                        cl_max = cfg["wing"]["cl_max_cruise"]
                        v_stall = math.sqrt(2.0 * stall_weight / (rho * cfg["wing"]["area"] * cl_max))
                        v_min = profile.takeoff_climb_margin * v_stall
                        v_max = seg.speed_max if seg.speed_max is not None else max(v_min * 2.0, v_min + 1.0)
                        elevator_lock = seg.elevator_deg if seg.elevator_deg is not None else segment_elevator_lock.get(segment_index)
                        if seg.mode is not None and seg.speed is None:
                            v_next = optimize_speed(v_min, v_max, seg.mode, weight_eff, 1.0, elevator_deg=elevator_lock)
                        elif v_next < v_min:
                            v_next = v_min
                        thrust_available = available_thrust(max(v_next, 0.1), rho)
                        if seg.elevator_deg is not None:
                            gamma_next, trim_result = trim_with_gamma_fixed_elevator(
                                v_next,
                                rho,
                                mu,
                                state.weight,
                                bank_rad,
                                1.0,
                                seg.elevator_deg,
                            )
                            last_aero = trim_result
                            thrust_used = thrust_available
                        elif elevator_lock is not None:
                            gamma_next, trim_result = trim_with_gamma_fixed_elevator(
                                v_next,
                                rho,
                                mu,
                                state.weight,
                                bank_rad,
                                1.0,
                                elevator_lock,
                            )
                            last_aero = trim_result
                            thrust_used = thrust_available
                        else:
                            gamma_next, trim_result = trim_with_gamma(v_next, rho, mu, state.weight, bank_rad, 1.0)
                            last_aero = trim_result
                            thrust_used = thrust_available
                            locked_elev = trim_result.get("ElevatorDeg") if trim_result else None
                            if locked_elev is not None:
                                segment_elevator_lock[segment_index] = locked_elev
                        if seg.target_alt is not None and gamma_next <= 1e-6:
                            raise ValueError(
                                f"Climb segment {segment_index} cannot climb: gamma={math.degrees(gamma_next):.2f} deg"
                            )
                        if seg.dt is None and seg.target_alt is not None:
                            vertical_speed = v_next * math.sin(gamma_next)
                            if vertical_speed > 1e-6:
                                remaining_alt = seg.target_alt - state.h
                                if remaining_alt > 0.0:
                                    step_dt = remaining_alt / vertical_speed
                        wind_mps = segment_wind_mps(seg)
                        ground_speed = v_next * math.cos(gamma_next) + wind_mps
                        h_next = state.h + v_next * math.sin(gamma_next) * step_dt
                        x_next = state.x + ground_speed * step_dt
                        if seg.dt is None and seg.target_alt is not None and h_next >= seg.target_alt:
                            h_next = seg.target_alt
                            phase = "segment_done"
                        else:
                            phase = seg.kind if h_next < seg.target_alt else "segment_done"
                    elif seg.kind == "descent_to":
                        cl_max = cfg["wing"]["cl_max_cruise"]
                        v_stall = math.sqrt(2.0 * stall_weight / (rho * cfg["wing"]["area"] * cl_max))
                        v_min = profile.level_flight_stall_margin * v_stall
                        v_max = seg.speed_max if seg.speed_max is not None else max(v_min * 2.0, v_min + 1.0)
                        elevator_lock = seg.elevator_deg if seg.elevator_deg is not None else segment_elevator_lock.get(segment_index)
                        if seg.mode is not None and seg.speed is None:
                            v_next = optimize_speed(v_min, v_max, seg.mode, weight_eff, 0.2, elevator_deg=elevator_lock)
                        elif v_next < v_min:
                            v_next = v_min
                        thrust_available = available_thrust(max(v_next, 0.1), rho)
                        thrust_scale = seg.thrust_scale if seg.thrust_scale is not None else 0.2
                        if seg.elevator_deg is not None:
                            gamma_next, trim_result = trim_with_gamma_fixed_elevator(
                                v_next,
                                rho,
                                mu,
                                state.weight,
                                bank_rad,
                                thrust_scale,
                                seg.elevator_deg,
                            )
                            last_aero = trim_result
                            thrust_used = thrust_available * thrust_scale
                        elif elevator_lock is not None:
                            gamma_next, trim_result = trim_with_gamma_fixed_elevator(
                                v_next,
                                rho,
                                mu,
                                state.weight,
                                bank_rad,
                                thrust_scale,
                                elevator_lock,
                            )
                            last_aero = trim_result
                            thrust_used = thrust_available * thrust_scale
                        else:
                            gamma_next, trim_result = trim_with_gamma(v_next, rho, mu, state.weight, bank_rad, thrust_scale)
                            last_aero = trim_result
                            thrust_used = thrust_available * thrust_scale
                            locked_elev = trim_result.get("ElevatorDeg") if trim_result else None
                            if locked_elev is not None:
                                segment_elevator_lock[segment_index] = locked_elev
                        if seg.target_alt is not None and gamma_next >= -1e-6:
                            raise ValueError(
                                f"Descent segment {segment_index} cannot descend: gamma={math.degrees(gamma_next):.2f} deg"
                            )
                        if seg.dt is None and seg.target_alt is not None:
                            vertical_speed = v_next * math.sin(gamma_next)
                            if vertical_speed < -1e-6:
                                remaining_alt = state.h - seg.target_alt
                                if remaining_alt > 0.0:
                                    step_dt = remaining_alt / (-vertical_speed)
                        wind_mps = segment_wind_mps(seg)
                        ground_speed = v_next * math.cos(gamma_next) + wind_mps
                        h_next = max(state.h + v_next * math.sin(gamma_next) * step_dt, seg.target_alt)
                        x_next = state.x + ground_speed * step_dt
                        phase = "segment_done" if h_next <= seg.target_alt else seg.kind
                    elif seg.kind in ("cruise", "loiter"):
                        if cruise_trim_segment_index != segment_index:
                            cl_max = cfg["wing"]["cl_max_cruise"]
                            v_stall = math.sqrt(2.0 * weight_eff / (rho * cfg["wing"]["area"] * cl_max))
                            v_min = profile.level_flight_stall_margin * v_stall
                            v_max = seg.speed_max if seg.speed_max is not None else max(v_min * 2.0, v_min + 1.0)
                            trim_result = None
                            if seg.mode is not None and seg.speed is None:
                                if seg.mode == "max_endurance_eq":
                                    v_next = endurance_speed_from_eq(state.weight, rho, v_min, load_factor)
                                elif seg.mode == "max_endurance":
                                    v_next, trim_result = optimize_speed(
                                        v_min,
                                        v_max,
                                        seg.mode,
                                        weight_eff,
                                        1.0,
                                        return_trim=True,
                                    )
                                else:
                                    v_next = optimize_speed(v_min, v_max, seg.mode, weight_eff, 1.0)
                            if v_next < v_min:
                                v_next = v_min
                            if v_next > v_max:
                                v_next = v_max
                            if trim_result is None:
                                if seg.elevator_deg is not None:
                                    trim_result = solve_trim_fixed_elevator(
                                        v_next,
                                        rho,
                                        mu,
                                        weight_eff,
                                        seg.elevator_deg,
                                        profile.trim_max_iter_opt,
                                    )
                                else:
                                    trim_result = solve_trim(v_next, rho, mu, weight_eff, profile.trim_max_iter_opt)
                            if seg.speed is None:
                                lift_tol = 1e-3 * max(weight_eff, 1.0)
                                for _ in range(10):
                                    aoa = trim_result.get("AOA")
                                    converged = trim_result.get("Converged", True)
                                    lift = trim_result.get("Lift", 0.0)
                                    aoa_violation = aoa_max_deg is not None and (aoa is None or aoa > aoa_max_deg)
                                    if converged and abs(lift - weight_eff) <= lift_tol and not aoa_violation:
                                        break
                                    if lift > 0.0:
                                        v_new = v_next * math.sqrt(weight_eff / lift)
                                    else:
                                        v_new = v_next + 1.0
                                    if aoa_violation or not converged:
                                        v_new = max(v_new, v_next + 0.5)
                                    v_new = max(min(v_new, v_max), v_min)
                                    if abs(v_new - v_next) < 0.05:
                                        break
                                    v_next = v_new
                                    if seg.elevator_deg is not None:
                                        trim_result = solve_trim_fixed_elevator(
                                            v_next,
                                            rho,
                                            mu,
                                            weight_eff,
                                            seg.elevator_deg,
                                            profile.trim_max_iter_opt,
                                        )
                                    else:
                                        trim_result = solve_trim(v_next, rho, mu, weight_eff, profile.trim_max_iter_opt)
                            thrust_available = available_thrust(max(v_next, 0.1), rho)
                            drag = trim_result["Drag"]
                            if drag > thrust_available:
                                if seg.elevator_deg is not None:
                                    gamma_next, trim_result = trim_with_gamma_fixed_elevator(
                                        v_next,
                                        rho,
                                        mu,
                                        state.weight,
                                        bank_rad,
                                        1.0,
                                        seg.elevator_deg,
                                    )
                                else:
                                    gamma_next, trim_result = trim_with_gamma(
                                        v_next,
                                        rho,
                                        mu,
                                        state.weight,
                                        bank_rad,
                                        1.0,
                                    )
                                thrust_used = thrust_available
                            else:
                                gamma_next = 0.0
                                thrust_used = max(drag, 0.0)
                            if seg.target_alt is None and abs(gamma_next) > 1e-3:
                                raise ValueError(
                                    f"Cruise/loiter segment {segment_index} cannot hold altitude: gamma={math.degrees(gamma_next):.2f} deg"
                                )
                            cruise_trim_segment_index = segment_index
                            cruise_trim_v = v_next
                            cruise_trim_result = trim_result
                            cruise_trim_gamma = gamma_next
                            cruise_trim_thrust = thrust_used
                        else:
                            v_next = cruise_trim_v
                            trim_result = cruise_trim_result
                            gamma_next = cruise_trim_gamma
                            thrust_used = cruise_trim_thrust
                        last_aero = trim_result
                        time_limit = seg.time
                        if seg.kind == "loiter" and seg.mode in ("max_endurance_eq", "max_endurance") and seg.time is None and seg.distance is None:
                            if segment_index not in segment_time_limit:
                                energy_reserve_j = cfg["propulsion"]["battery_energy_j"] * energy_reserve
                                available_energy = max(state.energy_j - energy_reserve_j, 0.0)
                                power_required = max(trim_result["Drag"], 0.0) * v_next / cfg["propulsion"]["prop_eff"]
                                prop_power_used = max(min(power_required, power_limit), 0.0)
                                power_used = max(prop_power_used + mission_systems_power, 1e-6)
                                segment_time_limit[segment_index] = available_energy / power_used
                            time_limit = segment_time_limit[segment_index]
                        if seg.dt is None:
                            fast_dt = None
                            wind_mps = segment_wind_mps(seg)
                            ground_speed = v_next * math.cos(gamma_next) + wind_mps
                            if seg.distance is not None and ground_speed > 1e-6:
                                remaining = seg.distance - segment_distance
                                if remaining > 0.0:
                                    fast_dt = remaining / ground_speed
                            elif time_limit is not None:
                                remaining = time_limit - segment_time
                                if remaining > 0.0:
                                    fast_dt = remaining
                            if fast_dt is not None:
                                step_dt = fast_dt
                        wind_mps = segment_wind_mps(seg)
                        ground_speed = v_next * math.cos(gamma_next) + wind_mps
                        h_next = state.h + v_next * math.sin(gamma_next) * step_dt if seg.target_alt is None else seg.target_alt
                        x_next = state.x + ground_speed * step_dt
                        segment_time += step_dt
                        segment_distance += ground_speed * step_dt
                        if seg.distance is not None:
                            phase = "segment_done" if segment_distance >= seg.distance else seg.kind
                        else:
                            phase = "segment_done" if time_limit is not None and segment_time >= time_limit else seg.kind
                        if seg.kind == "loiter":
                            update_loiter_metrics(segment_index, step_dt, trim_result, v_next, rho)
                    elif seg.kind == "landing":
                        cl_max = cfg["wing"]["cl_max_landing"]
                        v_stall = math.sqrt(2.0 * weight_eff / (rho * cfg["wing"]["area"] * cl_max))
                        v_min = profile.landing_stall_margin * v_stall
                        if v_next < v_min:
                            v_next = v_min
                        elevator_lock = seg.elevator_deg if seg.elevator_deg is not None else segment_elevator_lock.get(segment_index)
                        if elevator_lock is not None:
                            trim_result = solve_trim_fixed_elevator(
                                v_next,
                                rho,
                                mu,
                                weight_eff,
                                elevator_lock,
                                profile.trim_max_iter_opt,
                            )
                        else:
                            trim_result = solve_trim(v_next, rho, mu, weight_eff, profile.trim_max_iter_opt)
                            locked_elev = trim_result.get("ElevatorDeg") if trim_result else None
                            if locked_elev is not None:
                                segment_elevator_lock[segment_index] = locked_elev
                        last_aero = trim_result
                        thrust_used = 0.0
                        drag = trim_result["Drag"]
                        gamma_next = math.asin(max(min((thrust_used - drag) / state.weight, 0.0), -0.3))
                        if seg.dt is None:
                            vertical_speed = v_next * math.sin(gamma_next)
                            if vertical_speed < -1e-6 and state.h > 0.0:
                                step_dt = state.h / (-vertical_speed)
                        wind_mps = segment_wind_mps(seg)
                        ground_speed = v_next * math.cos(gamma_next) + wind_mps
                        h_next = max(state.h + v_next * math.sin(gamma_next) * step_dt, 0.0)
                        x_next = state.x + ground_speed * step_dt
                        phase = "landed" if h_next <= 0.0 else seg.kind
                    else:
                        break
    
                else:
                    break

                if phase == "segment_done":
                    segment_index += 1
                    segment_time = 0.0
                    segment_distance = 0.0
                    segment_index = next_non_takeoff_segment(segment_index)
                    phase = segments[segment_index].kind if segment_index < len(segments) else "landed"

            if segments and state.phase != "takeoff":
                state_phase = phase
            else:
                state_phase = phase
    
            thrust_cap = available_thrust(max(v_next, 0.1), rho)
            thrust_used = min(max(thrust_used, 0.0), thrust_cap)
            if thrust_cap > 0.0:
                throttle = thrust_used / thrust_cap
            else:
                throttle = 0.0
            prop_power_used = power_limit * min(max(throttle, 0.0), 1.0)
            total_power_used = prop_power_used + mission_systems_power
            energy_next = max(state.energy_j - total_power_used * step_dt, 0.0)
        
            state = MissionState(
                t=state.t + step_dt,
                x=x_next,
                h=h_next,
                v=v_next,
                gamma=gamma_next,
                weight=state.weight,
                energy_j=energy_next,
                phase=state_phase,
            )
            history.append(state)
            last_gamma = gamma_next
            last_thrust_used = thrust_used
        
            if profile.log_interval > 0.0 and state.t >= next_log_time:
                lift = last_aero["Lift"] if last_aero else 0.0
                drag = last_aero["Drag"] if last_aero else 0.0
                aoa = last_aero.get("AOA", 0.0) if last_aero else 0.0
                elev = last_aero.get("ElevatorDeg", 0.0) if last_aero else 0.0
                thrust = last_thrust_used
                if last_aero:
                    if last_aero.get("Converged", False):
                        trim_status = "ok"
                    elif last_aero.get("AoaClamped", False) or last_aero.get("ElevatorClamped", False):
                        trim_status = "lim"
                    else:
                        trim_status = "nc"
                else:
                    trim_status = "na"
                residual_str = ""
                if last_aero and trim_status == "nc":
                    res_lift = last_aero.get("LiftResidual", 0.0)
                    res_moment = last_aero.get("MomentResidual", 0.0)
                    residual_str = f"  resL={res_lift:8.2f} N  resM={res_moment:8.2f} N*m"
                print(f"[{state.t:7.1f}s] phase={state.phase:<10} h={state.h:7.1f} m  v={state.v:6.2f} m/s  x={state.x:8.1f} m  L={lift:8.1f} N  D={drag:7.1f} N  T={thrust:7.1f} N  aoa={aoa:6.2f} deg  elev={elev:6.2f} deg  trim={trim_status:>3}  E={state.energy_j:9.0f} J{residual_str}")
                next_log_time = state.t + profile.log_interval
    
            if prev_phase == "takeoff" and state.phase != "takeoff":
                summary = {
                    "segment": segment_label,
                    "kind": segment_kind,
                    "time_s": state.t - segment_start_t,
                    "distance_m": state.x - segment_start_x,
                    "energy_used_j": segment_start_energy - state.energy_j,
                }
                if segment_kind == "loiter" and current_segment_index is not None:
                    summary.update(loiter_summary(current_segment_index))
                segment_summaries.append(summary)
                segment_start_energy = state.energy_j
                segment_start_x = state.x
                segment_start_t = state.t
                if segments and segment_index < len(segments):
                    current_segment_index = segment_index
                    segment_label = f"{segments[current_segment_index].kind}:{current_segment_index}"
                    segment_kind = segments[current_segment_index].kind
                else:
                    current_segment_index = None
                    segment_label = "mission_end"
                    segment_kind = "mission_end"
            elif current_segment_index is not None and segment_index != prev_segment_index:
                summary = {
                    "segment": segment_label,
                    "kind": segment_kind,
                    "time_s": state.t - segment_start_t,
                    "distance_m": state.x - segment_start_x,
                    "energy_used_j": segment_start_energy - state.energy_j,
                }
                if segment_kind == "loiter":
                    summary.update(loiter_summary(current_segment_index))
                segment_summaries.append(summary)
                segment_start_energy = state.energy_j
                segment_start_x = state.x
                segment_start_t = state.t
                if segment_index < len(segments):
                    current_segment_index = segment_index
                    segment_label = f"{segments[current_segment_index].kind}:{current_segment_index}"
                    segment_kind = segments[current_segment_index].kind
                else:
                    current_segment_index = None
                    segment_label = "mission_end"
                    segment_kind = "mission_end"
    
            if state.phase == "landed":
                if segment_start_energy is not None and segment_label != "mission_end":
                    summary = {
                        "segment": segment_label,
                        "kind": segment_kind,
                        "time_s": state.t - segment_start_t,
                        "distance_m": state.x - segment_start_x,
                        "energy_used_j": segment_start_energy - state.energy_j,
                    }
                    if segment_kind == "loiter" and current_segment_index is not None:
                        summary.update(loiter_summary(current_segment_index))
                    segment_summaries.append(summary)
                break
    
        if return_summary:
            return history, segment_summaries
        return history
    finally:
        cfg["n_span"] = orig_n_span
