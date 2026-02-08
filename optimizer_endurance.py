''' Optimizer
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
    polish=True,
):
    arealDensityMain = 3.0
    arealDensityH = 1.5
    arealDensityV = 1.5

    boomMassPerM = 0.4
    boomMassFixed = 0.0
    boomLengthMin = 0.05

    evalCount = 0
    bestSeen = {"pwr": 1e30}

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
            0.0, 0.0, True,
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

        try:
            commsNode, totalMass = build_aircraft(x)
        except Exception:
            return 1e30
        if (commsNode.horizontalTailVolume(xhtqc) < 0.3) or (commsNode.horizontalTailVolume(xhtqc) > 0.9):
            print("Htail reject")
            return 1e30
        
        if (commsNode.verticalTailVolume(xhtqc)<0.02 or commsNode.verticalTailVolume(xhtqc)>0.08):
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

    bounds = [
        (3.0, 4.5),
        (0.3, 0.40),
        (0.05, xcg + 0.2),
        (0.60, 1.8),
        (0.18, 0.30),
        (xcg + 0.2, xcg + 1.8),
        (0.20, 1.0),
        (0.08, 0.30),
    ]

    rng = np.random.default_rng(int(seed))
    x0 = np.array([(lo + hi) * 0.5 for (lo, hi) in bounds], dtype=float)
    x0 += rng.normal(scale=0.05, size=x0.shape) * np.array([(hi - lo) for (lo, hi) in bounds], dtype=float)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    result = scipy.optimize.differential_evolution(
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
        init="latinhypercube",
        updating="deferred",
        workers=1,
        disp=True
    )

    xbest = result.x
    commsNode, totalMass = build_aircraft(xbest)
    out = commsNode.solveBestVelocity(levelFlightMargin, vguess=20.0, res=res)
    sm = float(commsNode.staticMargin())

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "power_W": float(result.fun),
        "vbest_mps": float(out[0]),
        "thrust_N": float(out[2]),
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
            "boomLength": float(max(float(xbest[5] - xbest[2]), float(boomLengthMin))),
        },
        "result": result,
    }


wingFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="psu94097")
tailFoil = PolarSet.from_folder("./PyFoil/polars", airfoil="S9033")
body = Fuselage(1, .3, 0.15, 0.9, 0.00635e-3, 0.3)
booms = Fuselage(1.4, .03, 0.03, 1, 0.00635e-3, 0.05)
batteryElectric = Powerplant(7992000, 0.59, 4000)

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
    res=2,
    seed=6,
    maxiter=30,
    popsize=30,
    polish=True
)
print(best)
'''
