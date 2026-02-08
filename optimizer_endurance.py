from PyFoil.airfoil_polars import PolarSet
import numpy as np
import scipy.optimize

from CommsNode import Aircraft, Fuselage, HorizontalTail, Powerplant, VerticalTail, Wing

G = 9.80665

def optimize_endurance(
    wingFoil,
    tailFoil,
    altitude,
    batteryElectric,
    fuselages,
    xcg,
    cdomisc,
    baseMass,
    totalMassMax,
    staticMarginMin=0.05,
    staticMarginMax=0.30,
    levelFlightMargin=1.25,
    res=60,
    seed=1,
    maxiter=150,
    popsize=6,
    polish=False,
    local_refine="Powell",
    local_maxiter=200,
    local_only=False,
    x_start=None,
    local_options=None,
    bounds=None,
):
    arealDensityMain = 3.05
    arealDensityH = 1.5
    arealDensityV = 1.5

    boomMassPerM = 0.4
    boomMassFixed = 0.0
    boomLengthMin = 0.05

    evalCount = 0
    bestSeen = {"pwr": 1e30}

    if bounds is None:
        bounds = [
            (4.0, 4.5),
            (0.26, 0.40),
            (0.05, xcg + 0.2),
            (0.60, 1.8),
            (0.18, 0.30),
            (xcg + 0.2, xcg + 1.8),
            (0.20, 1.0),
            (0.08, 0.30),
            (-2.0, 4.0),   # wing incidence (deg)
            (-4.0, 4.0),   # tail incidence (deg)
        ]
    bounds_lo = np.array([b[0] for b in bounds], dtype=float)
    bounds_hi = np.array([b[1] for b in bounds], dtype=float)

    def _clip_to_bounds(x):
        arr = np.array(x, dtype=float).reshape(-1)
        if arr.shape[0] != len(bounds):
            raise ValueError(f"x_start must have {len(bounds)} elements")
        return np.clip(arr, bounds_lo, bounds_hi)

    def _area_and_mac(span, root_chord, mid_chord, tip_chord, mid_pos, symmetric=True):
        area = 2.0 * (
            0.5 * (root_chord + mid_chord) * (mid_pos * 0.5 * span)
            + 0.5 * (mid_chord + tip_chord) * (0.5 * span - mid_pos * 0.5 * span)
        )
        if not symmetric:
            area *= 0.5
        mac = (
            2.0
            * (
                ((mid_pos * 0.5 * span) / 3.0)
                * (root_chord**2 + root_chord * mid_chord + mid_chord**2)
                + ((0.5 * span - mid_pos * 0.5 * span) / 3.0)
                * (mid_chord**2 + mid_chord * tip_chord + tip_chord**2)
            )
        ) / area
        return area, mac

    def build_aircraft(x):
        print("Iteration +1")
        wingSpan = float(x[0])
        wingChord = float(x[1])
        xwqc = float(x[2])

        hSpan = float(x[3])
        hChord = float(x[4])
        xhtqc = float(x[5])

        vHeight = float(x[6])
        vChord = float(x[7])
        xvtqc = xhtqc
        wingIncidence = float(x[8])
        tailIncidence = float(x[9])
        wingIncidence = float(x[8])
        tailIncidence = float(x[9])

        mainWing = Wing(
            wingFoil, altitude, 0.0,
            wingSpan,
            wingChord, wingChord, wingChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0, True,
            xwqc,
            0.0,
            arealDensityMain,
        )

        hwing = HorizontalTail(
            tailFoil, altitude, 0.0,
            hSpan,
            hChord, hChord, hChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            tailIncidence, 0.0, True,
            xhtqc,
            0.0,
            arealDensityH,
            elevatorDeflection=0.0,
            elevatorTau=0.35,
            cd_deltae_k=0.0,
        )

        vwing = VerticalTail(
            tailFoil, altitude, 0.0,
            vHeight,
            vChord, vChord, vChord, 0.5,
            0.0, 0.0, 0.0, 0.5,
            0.0, 0.0,
            xvtqc,
            0.0,
            arealDensityV,
            eta=2.0,
        )
        mainWing.incidence = wingIncidence

        boomLength = max(float(xhtqc - xwqc), float(boomLengthMin))
        boomMass = float(boomMassFixed) + float(boomMassPerM) * float(boomLength)
        fuselages_local = []
        for f in fuselages:
            if isinstance(f, Fuselage) and getattr(f, "width", None) is not None and getattr(f, "height", None) is not None:
                if float(getattr(f, "width")) <= 0.05 and float(getattr(f, "height")) <= 0.05:
                    fnew = Fuselage(
                        boomLength,
                        float(f.width),
                        float(f.height),
                        float(f.pfactor),
                        float(f.roughness),
                        float(f.laminarfraction),
                        float(getattr(f, "qfactor", 1.0)),
                        int(getattr(f, "quantity", 1)),
                    )
                    fuselages_local.append(fnew)
                else:
                    fuselages_local.append(f)
            else:
                fuselages_local.append(f)

        totalMass = float(baseMass) + float(mainWing.mass) + float(hwing.mass) + float(vwing.mass) + float(boomMass)
        weight = totalMass * G

        commsNode = Aircraft(
            altitude,
            20.0,
            batteryElectric,
            mainWing,
            hwing,
            vwing,
            fuselages_local,
            0.0,
            0.0,
            xcg,
            weight,
            cdomisc,
        )

        return commsNode, totalMass

    def objective(x):
        nonlocal evalCount, bestSeen
        evalCount += 1
        x = np.array(x, dtype=float).reshape(-1)
        if x.shape[0] != len(bounds):
            return 1e30
        if np.any(x < bounds_lo) or np.any(x > bounds_hi):
            return 1e30

        wingSpan = float(x[0])
        wingChord = float(x[1])
        xwqc = float(x[2])

        hSpan = float(x[3])
        hChord = float(x[4])
        xhtqc = float(x[5])

        vHeight = float(x[6])
        vChord = float(x[7])
        xvtqc = xhtqc

        if wingSpan <= 0.0 or wingChord <= 0.0 or hSpan <= 0.0 or hChord <= 0.0 or vHeight <= 0.0 or vChord <= 0.0:
            return 1e30

        if xwqc <= 0.0:
            return 1e30

        if xhtqc <= xcg or xvtqc <= xcg:
            return 1e30

        if xhtqc <= xwqc:
            return 1e30

        wing_area, wing_mac = _area_and_mac(
            wingSpan, wingChord, wingChord, wingChord, 0.5, symmetric=True
        )
        htail_area, _ = _area_and_mac(
            hSpan, hChord, hChord, hChord, 0.5, symmetric=True
        )
        vtail_area, _ = _area_and_mac(
            2.0 * vHeight, vChord, vChord, vChord, 0.5, symmetric=True
        )
        if wing_area <= 0.0 or wing_mac <= 0.0:
            return 1e30

        boomLength = max(float(xhtqc - xwqc), float(boomLengthMin))
        boomMass = float(boomMassFixed) + float(boomMassPerM) * float(boomLength)
        totalMass_est = (
            float(baseMass)
            + float(arealDensityMain) * float(wing_area)
            + float(arealDensityH) * float(htail_area)
            + float(arealDensityV) * 2.0 * float(vtail_area)
            + float(boomMass)
        )
        if totalMass_est > float(totalMassMax):
            print("mass reject")
            return 1e30

        htail_volume = (htail_area * (xhtqc - xcg)) / (wing_area * wing_mac)
        if (htail_volume < 0.3) or (htail_volume > 0.9):
            print("Htail reject")
            return 1e30

        vtail_volume = (vtail_area * (xvtqc - xcg)) / (wing_area * wingSpan)
        if (vtail_volume < 0.02) or (vtail_volume > 0.08):
            print("Vtail reject")
            return 1e30

        try:
            commsNode, totalMass = build_aircraft(x)
        except Exception:
            return 1e30
        if (commsNode.horizontalTailVolume(xhtqc) < 0.3) or (commsNode.horizontalTailVolume(xhtqc) > 0.9):
            print("Htail reject")
            return 1e30
        
        if (commsNode.verticalTailVolume(xhtqc) < 0.02 or commsNode.verticalTailVolume(xhtqc) > 0.08):
            print("Vtail reject")
            return 1e30
        
        if totalMass > float(totalMassMax):
            print("mass reject")
            return 1e30

        try:
            vbest, pwr, thrust = commsNode.solveBestVelocity(levelFlightMargin, vguess=20.0, res=res)
        except Exception:
            return 1e30

        if not np.isfinite(pwr) or pwr <= 0.0:
            return 1e30

        if float(pwr) > float(commsNode.pplant.pmax):
            print("pwr reject")
            return 1e30

        try:
            sm = float(commsNode.staticMargin())
            cma = float(commsNode.cm_alpha())
        except Exception:
            return 1e30

        if (sm < float(staticMarginMin)) or (sm > float(staticMarginMax)):
            print("sm reject")
            return 1e30

        if cma > 0:
            print("cma reject")
            return 1e30
        
        

        pwr = float(pwr)

        if pwr < bestSeen["pwr"]:
            bestSeen["pwr"] = pwr
            print(f"[best] eval={evalCount} pwr={pwr:.2f} W x={np.array(x, dtype=float)}")

        print("Valid dp")

        return pwr

    de_result = None
    if local_only:
        if x_start is None:
            raise ValueError("x_start is required when local_only=True")
        xbest = _clip_to_bounds(x_start)
    else:
        init_pop = "latinhypercube"
        if x_start is not None:
            rng = np.random.default_rng(int(seed))
            popsize_total = int(popsize) * len(bounds)
            init_pop = rng.uniform(bounds_lo, bounds_hi, size=(popsize_total, len(bounds)))
            init_pop[0, :] = _clip_to_bounds(x_start)
        de_result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=int(maxiter),
            popsize=int(popsize),
            tol=0.0,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=int(seed),
            polish=bool(polish),
            init=init_pop,
            updating="deferred",
            workers=1,
            disp=True
        )
        xbest = de_result.x

    local_result = None
    if local_refine:
        method = "Powell" if local_refine is True else str(local_refine)
        options = {"maxiter": int(local_maxiter)}
        if local_options:
            options.update(local_options)
        local_result = scipy.optimize.minimize(
            objective,
            xbest,
            method=method,
            options=options,
            bounds=bounds
        )
        if local_result.success and np.isfinite(local_result.fun):
            xbest = _clip_to_bounds(local_result.x)
    commsNode, totalMass = build_aircraft(xbest)
    vbest, pwr, thrust = commsNode.solveBestVelocity(levelFlightMargin, vguess=20.0, res=res)
    sm = float(commsNode.staticMargin())

    if local_result is not None:
        success = bool(local_result.success)
        message = str(local_result.message)
    elif de_result is not None:
        success = bool(de_result.success)
        message = str(de_result.message)
    else:
        success = True
        message = "local_only"

    return {
        "success": success,
        "message": message,
        "power_W": float(pwr),
        "vbest_mps": float(vbest),
        "thrust_N": float(thrust),
        "totalMass_kg": float(totalMass),
        "staticMargin": float(sm),
        "xbest": {
            "wingSpan": float(xbest[0]),
            "wingChord": float(xbest[1]),
            "xwqc": float(xbest[2]),
            "hSpan": float(xbest[3]),
            "hChord": float(xbest[4]),
            "xhtqc": float(xbest[5]),
            "vHeight": float(xbest[6]),
            "vChord": float(xbest[7]),
            "wingIncidence": float(xbest[8]),
            "tailIncidence": float(xbest[9]),
            "boomLength": float(max(float(xbest[5] - xbest[2]), float(boomLengthMin))),
        },
        "de_result": de_result,
        "local_result": local_result,
    }



def _example_run():
    wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="dae21")
    tailFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
    body = Fuselage(1, .3, 0.15, 0.9, 0.00635e-3, 0.3)
    booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
    batteryElectric = Powerplant(7992000, 0.59, 4000)

    prepopulated_best = [4.41080926, 0.27086204,  0.34139652,  1.4461152,   0.23794212,  1.13802129, 0.34202525,  0.24741137,  1.96412508, -0.18642715]
    run_mode = "local_only"  # "de" or "local_only"
    seed_de = True
    local_only = run_mode == "local_only"
    x_start = prepopulated_best if (local_only or seed_de) else None
    bounds = [
        (4.0, 4.5),
        (0.26, 0.35),
        (0.05, 0.45 + 0.2),
        (0.60, 1.8),
        (0.18, 0.30),
        (0.45 + 0.2, 0.45 + 1.8),
        (0.20, 0.9),
        (0.08, 0.30),
        (-2.0, 4.0),   # wing incidence (deg)
        (-4.0, 4.0),   # tail incidence (deg)
    ]
    
    best = optimize_endurance(
        wingFoil=wingFoil,
        tailFoil=tailFoil,
        altitude=200,
        batteryElectric=batteryElectric,
        fuselages=[body, booms, booms],
        xcg=0.45,
        cdomisc=0.01,
        baseMass=17.5,
        totalMassMax=22.6796,
        staticMarginMin=0.05,
        staticMarginMax=0.30,
        levelFlightMargin=1.25,
        res=1,
        seed=7,
        maxiter=10,
        popsize=10,
        polish=False,
        local_refine="L-BFGS-B", #can fuck with method here: L-BFGS-B uses derivatives so only good in vicinity, powell does not use derivative so can be safer
        local_maxiter=200,
        local_only=local_only,
        x_start=x_start,
        bounds=bounds,
    )
    print(best)

if __name__ == "__main__":
    _example_run()
