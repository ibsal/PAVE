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
    trim_max_iter: int = 8
    trim_gamma_iters: int = 2
    trim_max_iter_opt: int = 4
    segments: list | None = None
    log_interval: float = 2.0
    speed_opt_points: int = 10
    speed_opt_mode: str = "grid"
    n_span: int | None = None
    foil_cache: dict | None = None
    power_margin_frac: float = 0.2
    elevator_limit_deg: float = 30.0


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
        ff_list.append(1 + 60/((fuselage["length"]/fuselage["diameter"])**3) + (fuselage["length"]/fuselage["diameter"])/400)
        q_list.append(1.0)
        cfc_list.append(skin_friction_cf(re_fuse, cfg["roughness"]["k"], fuselage["length"], laminar_frac=cfg["roughness"]["fuselage_lam"], re_crit_per_m=cfg["roughness"]["re_crit_per_m"]))
        sratio_list.append((math.pi * fuselage["diameter"] * fuselage["length"] * math.pow((1 - 2/(fuselage["length"]/fuselage["diameter"])), 2.0/3.0) * (1 + 1/((fuselage["length"]/fuselage["diameter"])**2)))/wing["area"])
        re_list.append(re_fuse)

        components.append("Boom")
        ff_list.append(boom["count"] * (1 + 60/((boom["length"]/boom["diameter"])**3) + (boom["length"]/boom["diameter"])/400))
        q_list.append(1.0)
        cfc_list.append(skin_friction_cf(re_boom, cfg["roughness"]["k"], boom["length"], laminar_frac=cfg["roughness"]["boom_lam"], re_crit_per_m=cfg["roughness"]["re_crit_per_m"]))
        sratio_list.append((math.pi * boom["diameter"] * boom["length"] * math.pow((1 - 2/(boom["length"]/boom["diameter"])), 2.0/3.0) * (1 + 1/((boom["length"]/boom["diameter"])**2)))/wing["area"])
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
        foil_cache = cfg.get("foil_cache")

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
            foil_cache=foil_cache,
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
            foil_cache=foil_cache,
        )
        cd_vtail *= cfg["tail_efficiency"]

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
            foil_cache=foil_cache,
        )
        cd_htail *= cfg["tail_efficiency"]
        cl_htail *= cfg["tail_efficiency"]

        e_oswald_h = 1.78*(1.0 - 0.045*(htail["ar"]**0.68)) - 0.64
        e_oswald_h = min(max(e_oswald_h, 0.3), 0.95)
        har_eff = htail["ar"] * sweep_correction_factor(htail["span"], cfg["n_span"], True, htail["sweep_breaks"])
        cd_htail_induced = (cl_htail**2) * (wing["area"] / htail["area"]) / (math.pi * har_eff * e_oswald_h * cfg["tail_efficiency"])

        cd_non_lifting, _, _, _, _ = self.non_lifting_drag_stack(rho, mu, v)

        cd_total = cd_w_induced + cfg["cd_misc"] + cd_profile + cd_non_lifting + cd_vtail + cd_htail + cd_htail_induced

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
            foil_cache=foil_cache,
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
            foil_cache=foil_cache,
        )
        htail_moment *= cfg["tail_efficiency"]
        htail_cm_moment *= cfg["tail_efficiency"]
        total_moment = wing_moment + wing_cm_moment + htail_moment + htail_cm_moment

        return {
            "Lift": lift,
            "Drag": drag,
            "TotalPitchMoment": total_moment,
            "Cl": cl,
            "Cd": cd_total,
            "HTailAOA": htail_aoa,
        }

    def trim(self, v, rho, mu, weight, aoa_guess, elev_guess, max_iter=20, tol_force=1e-3, tol_moment=1e-3, elevator_limit_deg=None):
        aoa = aoa_guess
        elev = elev_guess
        if elevator_limit_deg is not None:
            elev = max(min(elev, elevator_limit_deg), -elevator_limit_deg)
        converged = False
        last_result = None
        for _ in range(max_iter):
            result = self.aero(aoa, elev, v, rho, mu)
            result = dict(result)
            result["AOA"] = aoa
            result["ElevatorDeg"] = elev
            last_result = result
            f1 = result["Lift"] - weight
            f2 = result["TotalPitchMoment"]
            if abs(f1) < tol_force and abs(f2) < tol_moment:
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
                elev = max(min(elev, elevator_limit_deg), -elevator_limit_deg)
        return aoa, elev, converged, last_result


def simulate_mission(model, profile, initial_alt, return_summary=False):
    cfg = model.config
    orig_n_span = cfg.get("n_span")
    orig_foil_cache = cfg.get("foil_cache")
    if profile.n_span is not None:
        cfg["n_span"] = profile.n_span
    if profile.foil_cache is not None:
        cfg["foil_cache"] = profile.foil_cache
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
    last_trim = None
    last_trim_time = -1.0
    last_trim_phase = None
    last_trim_v = None
    segment_index = 0
    segment_time = 0.0
    segment_distance = 0.0
    segments = profile.segments or []
    def next_non_takeoff_segment(start_index):
        idx = start_index
        while idx < len(segments) and segments[idx].kind == "takeoff":
            idx += 1
        return idx
    next_log_time = 0.0
    last_aero = None
    takeoff_elevator = 0.0
    last_gamma = 0.0
    takeoff_step = 0
    last_takeoff_aero = None
    last_takeoff_v = 0.1
    trim_cache = {}
    opt_speed_cache = {}
    segment_time_limit = {}
    segment_summaries = []
    segment_start_energy = None
    segment_start_x = None
    segment_start_t = None
    segment_label = "takeoff"
    current_segment_index = None

    power_margin = max(0.0, min(profile.power_margin_frac, 0.9))
    power_limit = cfg["propulsion"]["max_power_w"] * (1.0 - power_margin)

    def available_thrust(v, rho):
        return model.thrust_available(v, rho, power_limit_w=power_limit)

    def trim_key(v, weight_eff, bank_deg):
        return (round(v, 2), round(weight_eff, 1), round(math.degrees(bank_deg), 1))

    def cached_trim(v, rho, mu, weight_eff, bank_deg, max_iter):
        key = trim_key(v, weight_eff, bank_deg)
        cached = trim_cache.get(key)
        if cached is not None:
            return cached
        _, _, _, trim_result = model.trim(
            v,
            rho,
            mu,
            weight_eff,
            2.0,
            0.0,
            max_iter=max_iter,
            elevator_limit_deg=profile.elevator_limit_deg,
        )
        trim_cache[key] = trim_result
        return trim_result

    def trim_with_gamma(v, rho, mu, weight, bank_deg, thrust_scale):
        gamma = last_gamma
        trim_result = None
        for _ in range(profile.trim_gamma_iters):
            load_factor = 1.0 / max(math.cos(bank_deg), 0.1)
            weight_eff = weight * load_factor * math.cos(gamma)
            trim_result = cached_trim(v, rho, mu, weight_eff, bank_deg, profile.trim_max_iter)
            thrust = available_thrust(v, rho) * thrust_scale
            drag = trim_result["Drag"]
            gamma = math.asin(max(min((thrust - drag) / weight, 0.3), -0.3))
        return gamma, trim_result

    def optimize_speed(v_min, v_max, mode, weight_eff, thrust_scale):
        if v_min is None or v_max is None or v_max <= v_min:
            return v_min if v_min is not None else max(state.v, 0.1)
        best_v = v_min
        best_metric = None
        points = max(profile.speed_opt_points, 3)
        if profile.speed_opt_mode == "cheap":
            samples = [v_min, 0.5 * (v_min + v_max), v_max]
        else:
            samples = [v_min + (v_max - v_min) * i / (points - 1) for i in range(points)]
        for v in samples:
            trim_result = cached_trim(v, rho, mu, weight_eff, 0.0, profile.trim_max_iter_opt)
            drag = trim_result["Drag"]
            thrust = available_thrust(v, rho) * thrust_scale
            if mode == "max_roc":
                metric = (thrust - drag) * v
            elif mode == "min_energy":
                metric = -drag * v
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
        return best_v

    def estimate_cd0_total(rho, mu, v):
        wing = cfg["wing"]
        htail = cfg["htail"]
        vtail = cfg["vtail"]
        integrate_profile_drag = cfg["aero_funcs"]["integrate_profile_drag"]
        foil_cache = cfg.get("foil_cache")

        alpha = wing["incidence"]
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
            foil_cache=foil_cache,
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
            foil_cache=foil_cache,
        )
        cd_vtail *= cfg["tail_efficiency"]

        downwash = alpha * cfg["downwash_factor"]
        htail_aoa = htail["incidence"] - downwash
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
            foil_cache=foil_cache,
        )
        cd_htail *= cfg["tail_efficiency"]

        cd_non_lifting, _, _, _, _ = model.non_lifting_drag_stack(rho, mu, v)
        return cd_non_lifting + cd_profile + cd_vtail + cd_htail + cfg["cd_misc"]

    def endurance_speed_from_eq(weight_eff, rho, v_seed):
        wing = cfg["wing"]
        ar = wing["ar"]
        e_oswald = 1.78 * (1.0 - 0.045 * (ar ** 0.68)) - 0.64
        e_oswald = min(max(e_oswald, 0.3), 0.95)
        k = 1.0 / (math.pi * ar * e_oswald)
        v = v_seed
        for _ in range(2):
            cd0 = estimate_cd0_total(rho, mu, v)
            v = math.sqrt((2.0 * weight_eff / (rho * wing["area"])) * math.sqrt(k / (3.0 * cd0)))
        return v

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
                accel = (max(thrust - drag - profile.takeoff_mu * normal,0)) / mass
                v_next = max(state.v + accel * step_dt, 0.0)
                x_next = state.x + v_next * step_dt
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
    
                    trim_key_id = (seg.kind, v_next, seg.bank_deg)
                    if seg.kind in ("cruise", "loiter"):
                        trim_needed = (
                            last_trim is None
                            or last_trim_phase != trim_key_id
                            or last_trim_v is None
                            or abs(last_trim_v - v_next) > 0.1
                        )
                    else:
                        trim_needed = (
                            last_trim is None
                            or last_trim_phase != trim_key_id
                            or last_trim_v is None
                            or abs(last_trim_v - v_next) > 0.1
                            or (state.t - last_trim_time) > profile.trim_update_dt
                        )
                    if seg.elevator_deg is not None:
                        trim_needed = False
                        trim_result = model.aero(0.0, seg.elevator_deg, v_next, rho, mu)
                        trim_result = dict(trim_result)
                        trim_result["AOA"] = 0.0
                        trim_result["ElevatorDeg"] = seg.elevator_deg
                    else:
                        if trim_needed:
                            trim_result = cached_trim(v_next, rho, mu, weight_eff, bank_rad, profile.trim_max_iter)
                            last_trim = trim_result
                            last_trim_time = state.t
                            last_trim_phase = trim_key_id
                            last_trim_v = v_next
                        else:
                            trim_result = last_trim
                    last_aero = trim_result
    
                    if seg.kind == "climb_to":
                        cl_max = cfg["wing"]["cl_max_cruise"]
                        v_stall = math.sqrt(2.0 * weight_eff / (rho * cfg["wing"]["area"] * cl_max))
                        v_min = profile.takeoff_climb_margin * v_stall
                        v_max = seg.speed_max if seg.speed_max is not None else max(v_min * 2.0, v_min + 1.0)
                        if seg.mode is not None and seg.speed is None:
                            if segment_index in opt_speed_cache:
                                v_next = opt_speed_cache[segment_index]
                            else:
                                v_next = optimize_speed(v_min, v_max, seg.mode, weight_eff, 1.0)
                                opt_speed_cache[segment_index] = v_next
                        elif v_next < v_min:
                            v_next = v_min
                        if seg.elevator_deg is not None:
                            thrust = available_thrust(v_next, rho)
                            drag = trim_result["Drag"]
                            gamma_next = math.asin(max(min((thrust - drag) / state.weight, 0.3), -0.3))
                        elif trim_needed:
                            gamma_next, trim_result = trim_with_gamma(v_next, rho, mu, state.weight, bank_rad, 1.0)
                            last_trim = trim_result
                        else:
                            thrust = available_thrust(v_next, rho)
                            drag = trim_result["Drag"]
                            gamma_next = math.asin(max(min((thrust - drag) / state.weight, 0.3), -0.3))
                        h_next = state.h + v_next * math.sin(gamma_next) * step_dt
                        x_next = state.x + v_next * math.cos(gamma_next) * step_dt
                        phase = seg.kind if h_next < seg.target_alt else "segment_done"
                    elif seg.kind == "descent_to":
                        cl_max = cfg["wing"]["cl_max_cruise"]
                        v_stall = math.sqrt(2.0 * weight_eff / (rho * cfg["wing"]["area"] * cl_max))
                        v_min = profile.level_flight_stall_margin * v_stall
                        v_max = seg.speed_max if seg.speed_max is not None else max(v_min * 2.0, v_min + 1.0)
                        if seg.mode is not None and seg.speed is None:
                            if segment_index in opt_speed_cache:
                                v_next = opt_speed_cache[segment_index]
                            else:
                                v_next = optimize_speed(v_min, v_max, seg.mode, weight_eff, 0.2)
                                opt_speed_cache[segment_index] = v_next
                        elif v_next < v_min:
                            v_next = v_min
                        thrust_scale = seg.thrust_scale if seg.thrust_scale is not None else 0.2
                        if seg.elevator_deg is not None:
                            thrust = available_thrust(v_next, rho) * thrust_scale
                            drag = trim_result["Drag"]
                            gamma_next = math.asin(max(min((thrust - drag) / state.weight, 0.0), -0.3))
                        elif trim_needed:
                            gamma_next, trim_result = trim_with_gamma(v_next, rho, mu, state.weight, bank_rad, thrust_scale)
                            last_trim = trim_result
                        else:
                            thrust = available_thrust(v_next, rho) * thrust_scale
                            drag = trim_result["Drag"]
                            gamma_next = math.asin(max(min((thrust - drag) / state.weight, 0.0), -0.3))
                        h_next = max(state.h + v_next * math.sin(gamma_next) * step_dt, seg.target_alt)
                        x_next = state.x + v_next * math.cos(gamma_next) * step_dt
                        phase = "segment_done" if h_next <= seg.target_alt else seg.kind
                elif seg.kind in ("cruise", "loiter"):
                    cl_max = cfg["wing"]["cl_max_cruise"]
                    v_stall = math.sqrt(2.0 * weight_eff / (rho * cfg["wing"]["area"] * cl_max))
                    v_min = profile.level_flight_stall_margin * v_stall
                    v_max = seg.speed_max if seg.speed_max is not None else max(v_min * 2.0, v_min + 1.0)
                    if seg.mode is not None and seg.speed is None:
                        if segment_index in opt_speed_cache:
                            v_next = opt_speed_cache[segment_index]
                        else:
                            if seg.mode == "max_endurance_eq":
                                v_next = endurance_speed_from_eq(weight_eff, rho, v_min)
                            else:
                                v_next = optimize_speed(v_min, v_max, seg.mode, weight_eff, 1.0)
                            opt_speed_cache[segment_index] = v_next
                    if v_next < v_min:
                        v_next = v_min
                    if v_next > v_max:
                        v_next = v_max
                    gamma_next = 0.0
                    h_next = state.h if seg.target_alt is None else seg.target_alt
                    x_next = state.x + v_next * step_dt
                    segment_time += step_dt
                    segment_distance += v_next * step_dt
                    time_limit = seg.time
                    if seg.kind == "loiter" and seg.mode == "max_endurance_eq" and seg.time is None and seg.distance is None:
                        if segment_index not in segment_time_limit:
                            energy_reserve = cfg["propulsion"]["battery_energy_j"] * power_margin
                            available_energy = max(state.energy_j - energy_reserve, 0.0)
                            power_required = max(trim_result["Drag"], 0.0) * v_next / cfg["propulsion"]["prop_eff"]
                            power_used = max(min(power_required, power_limit), 1e-6)
                            segment_time_limit[segment_index] = available_energy / power_used
                        time_limit = segment_time_limit[segment_index]
                    if seg.distance is not None:
                        phase = "segment_done" if segment_distance >= seg.distance else seg.kind
                    else:
                        phase = "segment_done" if time_limit is not None and segment_time >= time_limit else seg.kind
                elif seg.kind == "landing":
                    cl_max = cfg["wing"]["cl_max_landing"]
                    v_stall = math.sqrt(2.0 * weight_eff / (rho * cfg["wing"]["area"] * cl_max))
                    v_min = profile.landing_stall_margin * v_stall
                    if v_next < v_min:
                        v_next = v_min
                    thrust = 0.0
                    drag = trim_result["Drag"]
                    gamma_next = math.asin(max(min((thrust - drag) / state.weight, 0.0), -0.3))
                    h_next = max(state.h + v_next * math.sin(gamma_next) * step_dt, 0.0)
                    x_next = state.x + v_next * math.cos(gamma_next) * step_dt
                    phase = "landed" if h_next <= 0.0 else seg.kind
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
    
            thrust = available_thrust(max(v_next, 0.1), rho)
            power_used = min(power_limit, thrust * max(v_next, 0.1) / cfg["propulsion"]["prop_eff"])
            energy_next = max(state.energy_j - power_used * step_dt, 0.0)
        
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
        
            if profile.log_interval > 0.0 and state.t >= next_log_time:
                lift = last_aero["Lift"] if last_aero else 0.0
                drag = last_aero["Drag"] if last_aero else 0.0
                elev = last_aero.get("ElevatorDeg", 0.0) if last_aero else 0.0
                thrust = available_thrust(max(state.v, 0.1), rho)
                print(f"[{state.t:7.1f}s] phase={state.phase:<10} h={state.h:7.1f} m  v={state.v:6.2f} m/s  x={state.x:8.1f} m  L={lift:8.1f} N  D={drag:7.1f} N  T={thrust:7.1f} N  elev={elev:6.2f} deg  E={state.energy_j:9.0f} J")
                next_log_time = state.t + profile.log_interval
    
            if prev_phase == "takeoff" and state.phase != "takeoff":
                segment_summaries.append({
                    "segment": segment_label,
                    "time_s": state.t - segment_start_t,
                    "distance_m": state.x - segment_start_x,
                    "energy_used_j": segment_start_energy - state.energy_j,
                })
                segment_start_energy = state.energy_j
                segment_start_x = state.x
                segment_start_t = state.t
                if segments and segment_index < len(segments):
                    current_segment_index = segment_index
                    segment_label = f"{segments[current_segment_index].kind}:{current_segment_index}"
                else:
                    current_segment_index = None
                    segment_label = "mission_end"
            elif current_segment_index is not None and segment_index != prev_segment_index:
                segment_summaries.append({
                    "segment": segment_label,
                    "time_s": state.t - segment_start_t,
                    "distance_m": state.x - segment_start_x,
                    "energy_used_j": segment_start_energy - state.energy_j,
                })
                segment_start_energy = state.energy_j
                segment_start_x = state.x
                segment_start_t = state.t
                if segment_index < len(segments):
                    current_segment_index = segment_index
                    segment_label = f"{segments[current_segment_index].kind}:{current_segment_index}"
                else:
                    current_segment_index = None
                    segment_label = "mission_end"
    
            if state.phase == "landed":
                if segment_start_energy is not None and segment_label != "mission_end":
                    segment_summaries.append({
                        "segment": segment_label,
                        "time_s": state.t - segment_start_t,
                        "distance_m": state.x - segment_start_x,
                        "energy_used_j": segment_start_energy - state.energy_j,
                    })
                break
    
        if return_summary:
            return history, segment_summaries
        return history
    finally:
        cfg["n_span"] = orig_n_span
        if orig_foil_cache is None:
            cfg.pop("foil_cache", None)
        else:
            cfg["foil_cache"] = orig_foil_cache
