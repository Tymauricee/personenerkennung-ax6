"""Synthetischer Testdatensatz im Format des AX6-Beta-Datensatzes.

ACHTUNG: Nur zum TESTEN der Pipeline. Zahlen auf diesen Daten sind KEINE Ergebnisse!
Sie zeigen nur, dass Code, Splits, Plots und Diagnostik funktionieren.

Modell (bewusst einfach, aber physikalisch plausibel):
* 4 Sensoren (S01 re. Handgelenk, S02 li. Handgelenk, S05 Rumpf, S06 Bein), 100 Hz,
  Beschleunigung in g (inkl. Schwerkraft), Gyroskop in °/s.
* Jede Person hat eine individuelle "Bewegungssignatur" (Kadenz, Schwungamplituden,
  harmonische Form pro Sensor/Freiheitsgrad) – teils populationsweit geteilt.
* Session-Effekte (Tag, Sensoren angelegt): Montage-Rotation (~12°), Gyro-/Acc-Bias,
  Kadenz/Amplitude ("Tagesform"), leichte Formänderung. Innerhalb eines Tages bleiben sie
  über ALLE Aktivitäten gleich (Sensoren werden nicht neu angelegt) – wie bei euch.
* Trial-Effekte: kleine Montage- und Kadenzänderungen.
* Innerhalb eines Trials: langsame Drift + Phasenjitter.
* Default-Struktur wie euer Datensatz: 25 Personen, 1 Tag, 1 Trial pro Aktivität,
  Clips à 10 s; dazu optional ein Retest-Subset (Tag 2) als "Was-wäre-wenn".
* Eingebaute Datenprobleme für den QC-Test: 1 Lücke, 1 fehlende Datei, doppelte
  Zeitstempel, 1 Datei in m/s², Uhr-Offsets zwischen Sensoren.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

SENSORS = {"S01": "Handgelenk rechts", "S02": "Handgelenk links", "S05": "Rumpf", "S06": "Bein"}
ACTIVITIES = {"Gehen": 40, "Treppe": 25, "Hampelmann": 15, "Ball": 24, "Squats": 20,
              "Tuergriff": 18, "Haendewaschen": 15}
DOFS = ["roll", "pitch", "yaw", "ap", "ml", "v"]
_ALPHA = {"roll": [4, 2, 1, .5, .3], "pitch": [6, 2, 1, .5, .3], "yaw": [3, 2, 1, .5, .3],
          "ap": [2, 4, 1.5, 1, .5], "ml": [5, 2, 1, .5, .3], "v": [1, 6, 1.5, 1, .5]}
_NOMINAL_MOUNT = {"S01": (90, 0, 0), "S02": (-90, 0, 180), "S05": (0, -90, 0), "S06": (0, 90, 90)}


# ---------------------------------------------------------------- Hilfsfunktionen
def _shape(rng, dof, pop=None, w_ind=0.35):
    a = rng.dirichlet(_ALPHA[dof])
    p = rng.uniform(0, 2 * np.pi, len(a))
    if pop is not None:
        a = (1 - w_ind) * pop[0] + w_ind * a
        p = np.angle((1 - w_ind) * np.exp(1j * pop[1]) + w_ind * np.exp(1j * p))
    return a / a.sum(), p


def _perturb(shape, rng, w):
    a, p = shape
    a2, p2 = _shape(rng, "pitch") if len(a) == 5 else (a, p)
    a = (1 - w) * a + w * a2
    p = np.angle((1 - w) * np.exp(1j * p) + w * np.exp(1j * p2))
    return a / a.sum(), p


def _harm(phi, shape, rng=None, wander=0.3, fs=100.0):
    """Summe harmonischer Komponenten; mit ``rng`` wandern die Phasen langsam (Schritt-zu-
    Schritt-Variabilität innerhalb eines Trials)."""
    a, p = shape
    h = np.arange(1, len(a) + 1)[:, None]
    ph = p[:, None]
    if rng is not None and wander > 0:
        ph = ph + np.stack([_smooth_noise(len(phi), fs, rng, wander, 2.0) for _ in range(len(a))])
    return (a[:, None] * np.sin(h * phi[None, :] + ph)).sum(0)


def _smooth_noise(n, fs, rng, sigma, tau_s=1.0):
    w = rng.normal(0, 1, n).cumsum()
    w -= np.linspace(w[0], w[-1], n)
    k = max(1, int(fs * tau_s))
    w = np.convolve(w, np.ones(k) / k, mode="same")
    return w / (np.std(w) + 1e-9) * sigma


def _phase(n, fs, f0, rng, drift=0.06, jitter=0.03):
    t = np.arange(n) / fs
    slow = drift * np.sin(2 * np.pi * t / (n / fs) * rng.uniform(0.5, 1.5) + rng.uniform(0, 2 * np.pi))
    f = f0 * (1 + slow + _smooth_noise(n, fs, rng, jitter))
    return 2 * np.pi * np.cumsum(f) / fs


def _gauss(t, centers, width):
    if len(centers) == 0:
        return np.zeros_like(t)
    return np.exp(-0.5 * ((t[None, :] - np.asarray(centers)[:, None]) / width) ** 2).sum(0)


# ---------------------------------------------------------------- Personen/Sessions
def make_person(rng, pop):
    P = {"f_step": rng.uniform(1.75, 1.95), "f_jj": rng.uniform(0.95, 1.25),
         "f_squat": rng.uniform(0.32, 0.48), "f_wash": rng.uniform(2.8, 4.2),
         "t_ball": rng.uniform(2.1, 2.9), "t_door": rng.uniform(4.2, 5.8),
         "swing": {"S01": rng.uniform(22, 32), "S05": rng.uniform(3.0, 5.0), "S06": rng.uniform(24, 30)},
         "lin": {"S01": rng.uniform(.25, .35), "S02": None, "S05": rng.uniform(.2, .3), "S06": rng.uniform(.4, .55)},
         "jj_arm": rng.uniform(55, 75), "jj_jump": rng.uniform(.6, 1.0),
         "sq_knee": rng.uniform(60, 90), "sq_trunk": rng.uniform(15, 35), "sq_arm": rng.uniform(30, 80),
         "ball_A": rng.uniform(60, 100), "ball_w": rng.uniform(.07, .12), "ball_catch": rng.uniform(.8, 1.2),
         "door_rot": rng.uniform(40, 70), "door_w": rng.uniform(.2, .35), "wash_A": rng.uniform(8, 15),
         "mount": {s: rng.normal(0, 6, 3) for s in SENSORS}}
    P["swing"]["S02"] = P["swing"]["S01"] * rng.uniform(.8, 1.2)
    P["lin"]["S02"] = P["lin"]["S01"] * rng.uniform(.85, 1.15)
    shapes = {}
    for act in ACTIVITIES:
        shapes[act] = {}
        for s in SENSORS:
            shapes[act][s] = {}
            for d in DOFS:
                if act == "Treppe":  # Treppe teilt ~70 % der Signatur mit Gehen
                    shapes[act][s][d] = _perturb(shapes["Gehen"][s][d], rng, 0.3)
                else:
                    shapes[act][s][d] = _shape(rng, d, pop[act][s][d])
    P["shape"] = shapes
    return P


def make_session(rng):
    return {"cad": rng.normal(1, .04), "amp": rng.normal(1, .08),
            "mount": {s: rng.normal(0, 10, 3) for s in SENSORS},
            "gyr_bias": {s: rng.normal(0, .8, 3) for s in SENSORS},
            "acc_bias": {s: rng.normal(0, .01, 3) for s in SENSORS},
            "clock": {s: rng.uniform(-.02, .02) for s in SENSORS},
            "shape_mix": 0.15, "seed": int(rng.integers(1 << 31))}


def make_trial(rng):
    return {"cad": rng.normal(1, .02), "amp": rng.normal(1, .05),
            "mount": {s: rng.normal(0, 4, 3) for s in SENSORS}}


# ---------------------------------------------------------------- Bewegung pro Aktivität
def _motion(act, n, fs, P, S, T, rng):
    """Gibt pro Sensor (angles_deg[n,3] = roll,pitch,yaw ; lin_g[n,3] = ap,ml,v) zurück."""
    t = np.arange(n) / fs
    amp, cad = S["amp"] * T["amp"], S["cad"] * T["cad"]
    srng = np.random.default_rng(S["seed"])  # Session-spezifische "Tagesform" der Form
    sh = {s: {d: _perturb(P["shape"][act][s][d], srng, S["shape_mix"]) for d in DOFS} for s in SENSORS}
    out = {}
    sway = {s: np.column_stack([_smooth_noise(n, fs, rng, 1.0, 2.0) for _ in range(3)]) for s in SENSORS}
    zeros = np.zeros(n)
    Hw = lambda ph_, shape_: _harm(ph_, shape_, rng, 0.6, fs)  # noqa: E731

    if act in ("Gehen", "Treppe"):
        st = act == "Treppe"
        phi = _phase(n, fs, P["f_step"] * cad * (0.85 if st else 1) / 2, rng)
        # Wendungen am Gangende (nur Gehen): ~alle 12–18 s eine 180°-Drehung, dabei kleinere Schritte
        turns, tt = [], 3.0
        while not st and tt < n / fs - 2:
            turns.append(tt + rng.uniform(0, 1))
            tt += rng.uniform(12, 18)
        u = _gauss(t, turns, 0.5)
        heading = 180.0 / (0.5 * np.sqrt(2 * np.pi)) * np.cumsum(u) / fs
        mod = (1 - 0.3 * np.clip(u, 0, 1)) * (1 + _smooth_noise(n, fs, rng, 0.12, 3.0))
        for s in SENSORS:
            ph = phi + (np.pi if s == "S02" else 0)
            A = P["swing"][s] * amp * ((0.7 if s in ("S01", "S02") else 1.3) if st else 1) * mod
            L = P["lin"][s] * amp * (1.4 if st else 1) * mod
            H = lambda d: _harm(ph, sh[s][d], rng, 0.6, fs)  # noqa: E731
            ang = np.column_stack([.25 * A * H("roll"), A * H("pitch"), .15 * A * H("yaw") + heading])
            lin = np.column_stack([.6 * L * H("ap"), .4 * L * H("ml"), L * H("v")])
            out[s] = (ang + sway[s], lin)
    elif act == "Hampelmann":
        phi = _phase(n, fs, P["f_jj"] * cad, rng, .03, .02)
        c = (1 - np.cos(phi)) / 2
        for s in SENSORS:
            if s in ("S01", "S02"):
                sign = 1 if s == "S01" else -1
                ang = np.column_stack([sign * P["jj_arm"] * amp * (c + .12 * Hw(phi, sh[s]["roll"])),
                                       10 * amp * Hw(phi, sh[s]["pitch"]), 5 * Hw(phi, sh[s]["yaw"])])
                L = .5 * P["jj_jump"] * amp
            elif s == "S05":
                ang = np.column_stack([3 * Hw(phi, sh[s]["roll"]), 5 * amp * Hw(phi, sh[s]["pitch"]), zeros])
                L = P["jj_jump"] * amp
            else:
                ang = np.column_stack([15 * amp * c + 3 * Hw(phi, sh[s]["roll"]),
                                       8 * Hw(phi, sh[s]["pitch"]), zeros])
                L = 1.2 * P["jj_jump"] * amp
            lin = np.column_stack([.3 * L * Hw(phi, sh[s]["ap"]), .3 * L * Hw(phi, sh[s]["ml"]),
                                   L * Hw(phi, sh[s]["v"])])
            out[s] = (ang + sway[s], lin)
    elif act == "Squats":
        phi = _phase(n, fs, P["f_squat"] * cad, rng, .04, .03)
        c = (1 - np.cos(phi)) / 2
        base = {"S01": P["sq_arm"], "S02": P["sq_arm"], "S05": P["sq_trunk"], "S06": P["sq_knee"]}
        for s in SENSORS:
            A = base[s] * amp
            ang = np.column_stack([3 * Hw(phi, sh[s]["roll"]), A * c + .1 * A * Hw(phi, sh[s]["pitch"]),
                                   2 * Hw(phi, sh[s]["yaw"])])
            lin = .08 * amp * np.column_stack([Hw(phi, sh[s]["ap"]), Hw(phi, sh[s]["ml"]),
                                              Hw(phi, sh[s]["v"])])
            out[s] = (ang + sway[s], lin)
    elif act in ("Ball", "Tuergriff"):
        per = P["t_ball"] if act == "Ball" else P["t_door"]
        ev, tk = [], 1.0
        while tk < n / fs - 1.5:
            ev.append(tk)
            tk += per / cad * rng.uniform(.85, 1.15)
        ev = np.array(ev)
        for s in SENSORS:
            ang = np.zeros((n, 3))
            lin = np.zeros((n, 3))
            if act == "Ball":
                A = P["ball_A"] * amp
                if s == "S01":
                    ang[:, 1] = -.4 * A * _gauss(t, ev - .35, .15) + A * _gauss(t, ev, P["ball_w"])
                    ang[:, 0] = .3 * A * _gauss(t, ev, 1.5 * P["ball_w"])
                    lin[:, 0] = .8 * amp * _gauss(t, ev, P["ball_w"])
                if s in ("S01", "S02"):
                    ang[:, 1] += .4 * A * _gauss(t, ev + P["ball_catch"], .12)
                    lin[:, 0] -= .5 * amp * _gauss(t, ev + P["ball_catch"], .08)
                if s == "S05":
                    ang[:, 2] = 15 * amp * _gauss(t, ev, .2)
                if s == "S06":
                    lin[:, 2] = .15 * amp * _gauss(t, ev - .1, .15)
            else:
                if s == "S01":
                    ang[:, 1] = 45 * amp * _gauss(t, ev, .45)
                    ang[:, 0] = P["door_rot"] * amp * _gauss(t, ev + .3, P["door_w"])
                    lin[:, 0] = .3 * amp * _gauss(t, ev - .2, .2)
                if s == "S05":
                    ang[:, 1] = 6 * amp * _gauss(t, ev, .5)
                if s == "S06":
                    lin[:, 2] = .2 * amp * _gauss(t, ev - .6, .12)
            ang += 3 * np.column_stack([_harm(t * 2 * np.pi * .3, sh[s][d]) for d in ("roll", "pitch", "yaw")])
            out[s] = (ang + sway[s], lin)
    elif act == "Haendewaschen":
        phi = _phase(n, fs, P["f_wash"] * cad, rng, .05, .03)
        env = np.clip(np.sin(2 * np.pi * t / rng.uniform(3.5, 4.5)) * 2 + 1.2, 0, 1)
        for s in SENSORS:
            if s in ("S01", "S02"):
                ph = phi + (np.pi if s == "S02" else 0)
                A = P["wash_A"] * amp
                ang = np.column_stack([.7 * A * env * Hw(ph, sh[s]["roll"]), A * env * Hw(ph, sh[s]["pitch"]) + 40,
                                       .3 * A * env * Hw(ph, sh[s]["yaw"])])
                lin = .15 * amp * env[:, None] * np.column_stack([Hw(ph, sh[s][d]) for d in ("ap", "ml", "v")])
            else:
                ang = np.column_stack([zeros, np.full(n, 20.0 if s == "S05" else 0.0), zeros])
                lin = np.zeros((n, 3))
            out[s] = (ang + sway[s], lin)
    else:
        raise ValueError(act)
    return out


def _to_sensor(ang_deg, lin_g, fs, mount_deg):
    """Segmentwinkel + Linearbeschleunigung -> Accelerometer [g] und Gyroskop [°/s] im Sensorrahmen."""
    R = Rot.from_euler("xyz", ang_deg, degrees=True) * Rot.from_euler("xyz", mount_deg, degrees=True)
    f_world = lin_g + np.array([0.0, 0.0, 1.0])          # spezifische Kraft inkl. Schwerkraft
    omega_world = np.gradient(ang_deg, axis=0) * fs       # Kleinwinkel-Näherung
    Rinv = R.inv()
    return Rinv.apply(f_world), Rinv.apply(omega_world)


# ---------------------------------------------------------------- Hauptfunktion
DURATIONS_DEFAULT = {"Gehen": 60, "Treppe": 40, "Hampelmann": 30, "Ball": 40, "Squats": 30,
                     "Tuergriff": 30, "Haendewaschen": 30}


def generate_dataset(out_dir: str | Path, n_persons: int = 25, n_trials: int = 1,
                     durations: dict | None = None, retest_persons: int = 8,
                     retest_activities: tuple = ("Gehen",), clip_s: float = 10.0, clip_step_s: float | None = None,
                     fs_raw: float = 100.0, seed: int = 7, quirks: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Erzeugt einen Datensatz mit EURER Struktur (Default):

    * ``n_persons`` Personen, alles an EINEM Tag (Session "S1"), ``n_trials`` Trial(s) pro
      Aktivität, Sensoren werden zwischen den Aktivitäten NICHT neu angelegt.
    * optional ein **Retest-Subset** (``retest_persons`` Personen gehen an einem 2. Tag "S2"
      erneut, Sensoren neu angelegt) – so etwas empfehlen wir fürs Datensatz-Team.

    Schreibt ``out_dir/raw/P01/S1/Gehen/T1/S01.csv`` (kontinuierliche Trials) sowie
    ``manifest.csv`` (Trials) und ``manifest_clips.csv`` (jeder Trial in Clips à ``clip_s``
    Sekunden zerlegt, über start_s/end_s – so wie euer "1 Trial -> mehrere Clips").
    """
    out_dir = Path(out_dir)
    acts = durations or DURATIONS_DEFAULT
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 1)
    pop = {a: {s: {d: _shape(prng, d) for d in DOFS} for s in SENSORS} for a in ACTIVITIES}
    devices = [f"AX6-{i:02d}" for i in range(1, 8)]
    rows, clip_rows = [], []
    clip_step_s = clip_step_s or clip_s
    q_gap = ("P02", "S1", "Gehen", "T1", "S05")
    q_missing = ("P03", "S1", "Squats", "T1", "S06")
    q_dup = ("P04", "S1", "Hampelmann", "T1", "S01")
    q_units = ("P05", "S1", "Ball", "T1", "S02")

    def write_trial(pid, sid, act, tid, dur, P, S, dev, clock):
        T = make_trial(rng)
        n = int(dur * fs_raw)
        motion = _motion(act, n, fs_raw, P, S, T, rng)
        t_true = clock + np.arange(n) / fs_raw
        for s in SENSORS:
            key = (pid, sid, act, tid, s)
            rel = (Path("raw") / pid / sid / act / tid / f"{s}.csv").as_posix()
            base = {"path": rel, "person": pid, "session": sid, "activity": act, "trial": tid,
                    "sensor": s, "position": SENSORS[s], "device_id": dev[s]}
            rows.append(base)
            # Clips: ab 1 s nach Trialstart, volle Clips à clip_s (Schrittweite clip_step_s)
            k, c0 = 1, clock + 1.0
            while c0 + clip_s <= clock + dur - 0.5:
                clip_rows.append({**base, "clip": f"C{k:02d}", "start_s": round(c0, 3), "end_s": round(c0 + clip_s, 3)})
                k, c0 = k + 1, c0 + clip_step_s
            if quirks and key == q_missing:
                continue  # Datei fehlt (steht aber im Manifest) -> Inventar muss das finden
            ang, lin = motion[s]
            mount = np.array(_NOMINAL_MOUNT[s]) + P["mount"][s] + S["mount"][s] + T["mount"][s]
            acc, gyr = _to_sensor(ang, lin, fs_raw, mount)
            acc = acc + S["acc_bias"][s] + rng.normal(0, .03, acc.shape)
            gyr = gyr + S["gyr_bias"][s] + rng.normal(0, 3.0, gyr.shape)
            a0, a1 = int(rng.integers(0, 30)), n - int(rng.integers(0, 30))  # Geräte starten versetzt
            df = pd.DataFrame({"time_s": t_true[a0:a1] + S["clock"][s],
                               "acc_x": acc[a0:a1, 0], "acc_y": acc[a0:a1, 1], "acc_z": acc[a0:a1, 2],
                               "gyr_x": gyr[a0:a1, 0], "gyr_y": gyr[a0:a1, 1], "gyr_z": gyr[a0:a1, 2]})
            if quirks and key == q_gap:
                tt = df["time_s"] - df["time_s"].iloc[0]
                df = df[(tt < 10.0) | (tt > 10.4)]
            if quirks and key == q_dup:
                df = pd.concat([df, df.sample(5, random_state=1)]).sort_values("time_s", kind="stable")
            if quirks and key == q_units:
                df[["acc_x", "acc_y", "acc_z"]] *= 9.80665
            f = out_dir / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, index=False, float_format="%.5f")

    for pi in range(1, n_persons + 1):
        pid = f"P{pi:02d}"
        P = make_person(rng, pop)
        S = make_session(rng)
        dev = dict(zip(SENSORS, rng.permutation(devices)[:4]))  # Geräte rotieren -> kein Person-Gerät-Confound
        clock = 0.0
        for act, dur in acts.items():
            for ti in range(1, n_trials + 1):
                write_trial(pid, "S1", act, f"T{ti}", dur, P, S, dev, clock)
                clock += dur + 10.0
        if pi <= retest_persons:  # zweiter Tag: Sensoren neu angelegt, neue Tagesform
            S2 = make_session(rng)
            dev2 = dict(zip(SENSORS, rng.permutation(devices)[:4]))
            clock = 0.0
            for act in retest_activities:
                write_trial(pid, "S2", act, "T1", acts[act], P, S2, dev2, clock)
                clock += acts[act] + 10.0
        if verbose and pi % 5 == 0:
            print(f"[synth] {pi}/{n_persons} Personen")
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(clip_rows).to_csv(out_dir / "manifest_clips.csv", index=False)
    (out_dir / "README.txt").write_text(
        "SYNTHETISCHE TESTDATEN – nur zum Testen der Pipeline, keine echten Messungen!\n"
        f"{n_persons} Personen, 1 Tag (S1), {n_trials} Trial(s) pro Aktivität, 4 Sensoren, {fs_raw:.0f} Hz.\n"
        f"Retest-Subset: {retest_persons} Personen an Tag 2 (S2): {', '.join(retest_activities)}.\n"
        f"manifest_clips.csv: Trials in Clips à {clip_s}s (Schritt {clip_step_s}s).\n"
        "Eingebaute Probleme (für QC): 1 Lücke, 1 fehlende Datei, doppelte Zeitstempel, 1 Datei in m/s².\n",
        encoding="utf-8")
    return man
