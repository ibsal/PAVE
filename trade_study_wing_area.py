import math
import csv
import numpy as np
import matplotlib.pyplot as plt

import CommsNode as comms


def endurance_hours_for_power(power_w, mission_systems_power_w, battery_capacity_wh, landing_margin):
    total_power_w = power_w + mission_systems_power_w
    available_energy_wh = battery_capacity_wh * (1.0 - landing_margin)
    if total_power_w <= 0.0:
        return float("nan")
    return available_energy_wh / total_power_w


def evaluate_point(wing_chord, fixed_span, level_flight_margin=1.25, vguess=20.0, res=40):
    x = list(comms.design_point)
    x[0] = float(fixed_span)
    x[1] = float(wing_chord)
    aircraft, _ = comms.build_aircraft(x)
    vbest, pwr, _ = aircraft.solveBestVelocity(level_flight_margin, vguess=vguess, res=res)
    propulsive_power_elec = (
        pwr / aircraft.pplant.neff if aircraft.pplant.neff > 0.0 else float("inf")
    )
    endurance_h = endurance_hours_for_power(
        propulsive_power_elec,
        comms.mission_systems_power_w,
        comms.battery_capacity_wh,
        comms.landing_margin,
    )
    return aircraft.mwing.area, float(vbest), float(endurance_h)


def main():
    fixed_span = float(comms.design_point[0])
    base_chord = float(comms.design_point[1])

    chord_scales = np.linspace(0.6, 1.4, 19)
    areas = []
    cruise_speeds = []
    endurances = []
    chords = []
    scales = []

    for scale in chord_scales:
        chord = base_chord * float(scale)
        area, vbest, endurance_h = evaluate_point(chord, fixed_span)
        areas.append(area)
        cruise_speeds.append(vbest)
        endurances.append(endurance_h)
        chords.append(chord)
        scales.append(float(scale))

    order = np.argsort(areas)
    areas = np.asarray(areas)[order]
    cruise_speeds = np.asarray(cruise_speeds)[order]
    endurances = np.asarray(endurances)[order]
    chords = np.asarray(chords)[order]
    scales = np.asarray(scales)[order]

    knots_per_mps = getattr(comms, "KNOTS_PER_MPS", 1.9438444924406)
    cruise_knots = cruise_speeds * knots_per_mps

    csv_path = "trade_study_wing_area.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "wing_area_m2",
            "cruise_speed_mps",
            "cruise_speed_kt",
            "endurance_h",
            "wing_span_m",
            "wing_chord_m",
            "chord_scale",
        ])
        for area, v_mps, v_kt, endurance_h, chord, scale in zip(
            areas, cruise_speeds, cruise_knots, endurances, chords, scales
        ):
            writer.writerow([
                f"{area:.6f}",
                f"{v_mps:.6f}",
                f"{v_kt:.6f}",
                f"{endurance_h:.6f}",
                f"{fixed_span:.6f}",
                f"{chord:.6f}",
                f"{scale:.6f}",
            ])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(areas, endurances, marker="o", linestyle="None")
    ax1.set_ylabel("Endurance (h)")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(areas, cruise_knots, marker="o", linestyle="None")
    ax2.set_xlabel("Wing area (m^2)")
    ax2.set_ylabel("Cruise speed (kt)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"Trade study (span fixed at {fixed_span:.3f} m)")
    plt.tight_layout()
    plt.show()

    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    ax3.plot(cruise_knots, endurances, marker="o", linestyle="None")
    ax3.set_xlabel("Cruise speed (kt)")
    ax3.set_ylabel("Endurance (h)")
    ax3.grid(True, linestyle="--", alpha=0.4)
    fig2.suptitle("Endurance vs cruise speed")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
