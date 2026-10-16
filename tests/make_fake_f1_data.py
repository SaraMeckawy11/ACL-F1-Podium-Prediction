"""Generate a FAKE Formula 1 dataset with the same 14-table schema as the Kaggle/Ergast data.

This exists only so the notebook can be smoke-tested end to end without the real data
(e.g. in CI or on a machine without Kaggle access). The numbers are random and mean
nothing: never report results computed from this data.

It deliberately reproduces the quirks the notebook has to handle:
\\N sentinels, lap-time strings, grid 0 (pit-lane starts), non-qualifiers, shared drives
(duplicate raceId/driverId rows before 1958), the Indianapolis 500 (1950-1960),
qualifying data only from 1994, q2/q3 only from 2006, and one orphan foreign key.

Usage:  python tests/make_fake_f1_data.py --out /tmp/fake_f1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N = "\\N"

STATUS = [
    (1, "Finished"), (2, "Disqualified"), (3, "Accident"), (4, "Collision"), (5, "Engine"),
    (6, "Gearbox"), (7, "Transmission"), (8, "Clutch"), (9, "Hydraulics"), (10, "Electrical"),
    (11, "+1 Lap"), (12, "+2 Laps"), (13, "+3 Laps"), (20, "Spun off"), (21, "Radiator"),
    (22, "Suspension"), (23, "Brakes"), (29, "Puncture"), (31, "Retired"), (54, "Withdrew"),
    (61, "Wheel nut"), (62, "Not classified"), (75, "Power loss"), (77, "107% Rule"),
    (81, "Did not qualify"), (97, "Did not prequalify"), (101, "Turbo"), (130, "Collision damage"),
    (131, "Power Unit"), (132, "ERS"), (137, "Damage"), (141, "Cooling system"),
]
MECH_IDS = [5, 6, 7, 8, 9, 10, 21, 22, 23, 75, 101, 131, 132, 141]
CRASH_IDS = [3, 4, 20, 130, 137, 29, 31, 61]

CIRCUITS = [
    ("silverstone", "Silverstone Circuit", "Silverstone", "UK"),
    ("monaco", "Circuit de Monaco", "Monte-Carlo", "Monaco"),
    ("monza", "Autodromo Nazionale di Monza", "Monza", "Italy"),
    ("spa", "Circuit de Spa-Francorchamps", "Spa", "Belgium"),
    ("nurburgring", "Nürburgring", "Nürburg", "Germany"),
    ("hockenheimring", "Hockenheimring", "Hockenheim", "Germany"),
    ("magny_cours", "Circuit de Nevers Magny-Cours", "Magny Cours", "France"),
    ("interlagos", "Autódromo José Carlos Pace", "São Paulo", "Brazil"),
    ("catalunya", "Circuit de Barcelona-Catalunya", "Montmeló", "Spain"),
    ("hungaroring", "Hungaroring", "Budapest", "Hungary"),
    ("suzuka", "Suzuka Circuit", "Suzuka", "Japan"),
    ("albert_park", "Albert Park Grand Prix Circuit", "Melbourne", "Australia"),
    ("villeneuve", "Circuit Gilles Villeneuve", "Montreal", "Canada"),
    ("americas", "Circuit of the Americas", "Austin", "USA"),
    ("watkins_glen", "Watkins Glen", "New York State", "United States"),
    ("rodriguez", "Autódromo Hermanos Rodríguez", "Mexico City", "Mexico"),
    ("red_bull_ring", "Red Bull Ring", "Spielberg", "Austria"),
    ("zandvoort", "Circuit Park Zandvoort", "Zandvoort", "Netherlands"),
    ("galvez", "Autódromo Juan y Oscar Gálvez", "Buenos Aires", "Argentina"),
    ("bahrain", "Bahrain International Circuit", "Sakhir", "Bahrain"),
    ("shanghai", "Shanghai International Circuit", "Shanghai", "China"),
    ("yas_marina", "Yas Marina Circuit", "Abu Dhabi", "UAE"),
    ("marina_bay", "Marina Bay Street Circuit", "Marina Bay", "Singapore"),
    ("kyalami", "Kyalami", "Midrand", "South Africa"),
    ("anderstorp", "Scandinavian Raceway", "Anderstorp", "Sweden"),
    ("bremgarten", "Circuit Bremgarten", "Bern", "Switzerland"),
    ("indianapolis", "Indianapolis Motor Speedway", "Indianapolis", "USA"),
]
NATS = ["British", "German", "Italian", "French", "American", "Brazilian", "Spanish", "Finnish",
        "Australian", "Dutch", "Japanese", "Mexican", "Canadian", "Belgian", "Austrian",
        "Monegasque", "Argentine", "Swiss", "Swedish", "New Zealander", "Thai", "Chinese", "Danish"]


def fmt_lap(sec):
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:06.3f}" if m else f"{s:.3f}"


def fmt_race(ms):
    h, rem = divmod(ms / 1000, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


def main(out: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    circuits = pd.DataFrame(
        [(i + 1, ref, name, loc, ctry, round(rng.uniform(-40, 60), 4), round(rng.uniform(-120, 140), 4),
          N if i % 5 == 0 else int(rng.integers(0, 800)), f"http://en.wikipedia.org/wiki/{ref}")
         for i, (ref, name, loc, ctry) in enumerate(CIRCUITS)],
        columns=["circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt", "url"])
    indy_id = int(circuits.loc[circuits.circuitRef == "indianapolis", "circuitId"].iloc[0])
    normal_circuits = circuits.loc[circuits.circuitId != indy_id, "circuitId"].to_numpy()

    n_cons = 140
    constructors = pd.DataFrame({
        "constructorId": np.arange(1, n_cons + 1),
        "constructorRef": [f"team_{i}" for i in range(1, n_cons + 1)],
        "name": [f"Team {i}" for i in range(1, n_cons + 1)],
        "nationality": rng.choice(NATS, n_cons),
        "url": [f"http://en.wikipedia.org/wiki/Team_{i}" for i in range(1, n_cons + 1)],
    })
    cons_strength = rng.normal(0, 1, n_cons + 1)

    n_drv = 900
    driver_skill = rng.normal(0, 0.6, n_drv + 1)
    driver_nat = rng.choice(NATS, n_drv + 1)
    drv_rows = []

    races, results, quali, dstand, cstand, cres, laps, pits, sprints, seasons = ([] for _ in range(10))
    race_id = result_id = quali_id = ds_id = cs_id = cr_id = sprint_id = 0
    next_driver = 1
    active = []
    for year in range(1950, 2025):
        seasons.append((year, f"http://en.wikipedia.org/wiki/{year}_Formula_One_season"))
        n_races = int(round(7 + (24 - 7) * (year - 1950) / 74))
        if year >= 2010:
            n_entrants = 20
        elif 1980 <= year <= 1995:
            n_entrants = int(rng.integers(26, 35))
        else:
            n_entrants = int(rng.integers(18, 26))
        # driver turnover
        active = [d for d in active if rng.random() > 0.15]
        while len(active) < n_entrants:
            active.append(next_driver)
            dob = pd.Timestamp(year - int(rng.integers(20, 32)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
            drv_rows.append((next_driver, f"driver_{next_driver}", N if year < 2014 else int(rng.integers(2, 99)),
                             N if year < 1990 else f"D{next_driver:02d}"[-3:], f"First{next_driver}",
                             f"Last{next_driver}", dob.strftime("%Y-%m-%d"), driver_nat[next_driver],
                             f"http://en.wikipedia.org/wiki/Driver_{next_driver}"))
            next_driver += 1
        entrants = list(rng.permutation(active)[:n_entrants])
        n_teams = (n_entrants + 1) // 2
        team_ids = rng.choice(np.arange(1, n_cons + 1), n_teams, replace=False)
        cons_strength[team_ids] = 0.8 * cons_strength[team_ids] + rng.normal(0, 0.5, n_teams)
        team_of = {d: int(team_ids[i // 2]) for i, d in enumerate(entrants)}

        season_pts, season_wins = {}, {}
        cseason_pts, cseason_wins = {}, {}
        chosen = list(rng.choice(normal_circuits, n_races, replace=n_races > len(normal_circuits)))
        if 1950 <= year <= 1960:
            chosen[min(3, n_races - 1)] = indy_id
        sprint_rounds = set(rng.choice(np.arange(3, n_races + 1), 3, replace=False)) if year >= 2021 else set()
        p_ret = 0.45 - (0.45 - 0.10) * (year - 1950) / 74
        for rnd, cid in enumerate(chosen, start=1):
            race_id += 1
            date = pd.Timestamp(year, 3, 1) + pd.Timedelta(days=int(rnd * 240 / n_races))
            cname = circuits.loc[circuits.circuitId == cid, "name"].iloc[0]
            rname = "Indianapolis 500" if cid == indy_id else f"{cname.split()[0]} Grand Prix"
            sprint = rnd in sprint_rounds
            races.append((race_id, year, rnd, cid, rname, date.strftime("%Y-%m-%d"),
                          N if year < 2005 else "14:00:00", f"http://en.wikipedia.org/wiki/{year}_{rnd}",
                          N, N, N, N, N, N,
                          N if year < 2022 else (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                          N if year < 2022 else "15:00:00",
                          (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if sprint else N,
                          "11:00:00" if sprint else N))
            perf = {d: cons_strength[team_of[d]] + driver_skill[d] + rng.normal(0, 0.5) for d in entrants}
            qtime = {d: 80 - 0.6 * perf[d] + rng.normal(0, 0.3) for d in entrants}
            qorder = sorted(entrants, key=lambda d: qtime[d])
            max_grid = 26
            starters, nonstart = qorder[:max_grid], qorder[max_grid:]
            if year >= 1994:
                for pos, d in enumerate(qorder, start=1):
                    quali_id += 1
                    q1 = N if rng.random() < 0.01 else fmt_lap(qtime[d])
                    q2 = fmt_lap(qtime[d] - 0.3) if year >= 2006 and pos <= 15 else N
                    q3 = fmt_lap(qtime[d] - 0.5) if year >= 2006 and pos <= 10 else N
                    quali.append((quali_id, race_id, d, team_of[d], int(rng.integers(1, 40)), pos, q1, q2, q3))
            # penalties and pit-lane starts
            gkey = {d: i + (5 if rng.random() < 0.05 else 0) + rng.random() * 0.1 for i, d in enumerate(starters)}
            grid_order = sorted(starters, key=lambda d: gkey[d])
            grid = {d: i + 1 for i, d in enumerate(grid_order)}
            for d in starters:
                if rng.random() < 0.01:
                    grid[d] = 0
            total_laps = 60
            race_score = {d: perf[d] - 0.12 * (grid[d] or len(starters)) + rng.normal(0, 0.8) for d in starters}
            retired = {d for d in starters if rng.random() < p_ret}
            fin = sorted([d for d in starters if d not in retired], key=lambda d: -race_score[d])
            ret_laps = {d: int(rng.integers(0, total_laps)) for d in retired}
            ret = sorted(retired, key=lambda d: -ret_laps[d])
            order = fin + ret + nonstart
            pts_table = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] if year >= 2010 else [9, 6, 4, 3, 2, 1]
            win_ms = int(5_600_000 + rng.normal(0, 100_000))
            race_rows = []
            for po, d in enumerate(order, start=1):
                result_id += 1
                if d in nonstart:
                    sid = 97 if (year < 1992 and po == len(order)) else 81
                    row = [result_id, race_id, d, team_of[d], N, 0, N, "F", po, 0, 0, N, N, N, N, N, N, sid]
                elif d in retired:
                    sid = int(rng.choice(MECH_IDS if rng.random() < 0.6 else CRASH_IDS))
                    row = [result_id, race_id, d, team_of[d], int(rng.integers(1, 40)), grid[d], N, "R", po, 0,
                           ret_laps[d], N, N, N, N, N, N, sid]
                else:
                    lapped = po > 8 and rng.random() < 0.5
                    pts = pts_table[po - 1] if po <= len(pts_table) else 0
                    gap_ms = (po - 1) * int(rng.integers(3000, 9000))
                    has_fl = year >= 2004
                    fl = 78 + rng.normal(0, 1)
                    row = [result_id, race_id, d, team_of[d], int(rng.integers(1, 40)), grid[d], po, str(po), po, pts,
                           total_laps - (1 if lapped else 0),
                           fmt_race(win_ms) if po == 1 else (N if lapped else f"+{gap_ms / 1000:.3f}"),
                           N if lapped else win_ms + gap_ms,
                           int(rng.integers(30, 60)) if has_fl else N, po if has_fl else N,
                           fmt_lap(fl) if has_fl else N, f"{200 + rng.normal(0, 5):.3f}" if has_fl else N,
                           11 if lapped else 1]
                    season_pts[d] = season_pts.get(d, 0) + pts
                    cseason_pts[team_of[d]] = cseason_pts.get(team_of[d], 0) + pts
                    if po == 1:
                        season_wins[d] = season_wins.get(d, 0) + 1
                        cseason_wins[team_of[d]] = cseason_wins.get(team_of[d], 0) + 1
                race_rows.append(row)
            # shared drives before 1958: a retired driver takes over a running car
            if year <= 1957 and retired and fin and rng.random() < 0.3:
                d = sorted(retired)[0]
                donor = next(r for r in race_rows if r[2] == fin[min(2, len(fin) - 1)])
                result_id += 1
                dup = list(donor)
                dup[0], dup[2] = result_id, d
                race_rows.append(dup)
            results.extend(race_rows)
            for d in set(season_pts) | {d for d in starters}:
                season_pts.setdefault(d, 0)
            ranked = sorted(season_pts, key=lambda d: (-season_pts[d], -season_wins.get(d, 0)))
            for pos, d in enumerate(ranked, start=1):
                ds_id += 1
                dstand.append((ds_id, race_id, d, season_pts[d], pos, str(pos), season_wins.get(d, 0)))
            for t in set(team_of.values()):
                cr_id += 1
                cres.append((cr_id, race_id, t, sum(r[9] for r in race_rows if r[3] == t), N))
                cseason_pts.setdefault(t, 0)
            if year >= 1958:
                cranked = sorted(cseason_pts, key=lambda t: -cseason_pts[t])
                for pos, t in enumerate(cranked, start=1):
                    cs_id += 1
                    cstand.append((cs_id, race_id, t, cseason_pts[t], pos, str(pos), cseason_wins.get(t, 0)))
            if year >= 2022:
                for d in fin[:10]:
                    for lap in range(1, 6):
                        t = 80 + rng.normal(0, 1)
                        laps.append((race_id, d, lap, int(rng.integers(1, 20)), fmt_lap(t), int(t * 1000)))
            if year >= 2011:
                for d in starters:
                    for stop in range(1, int(rng.integers(1, 3)) + 1):
                        dur = 22 + abs(rng.normal(0, 2)) if rng.random() > 0.01 else 1004.718
                        pits.append((race_id, d, stop, int(rng.integers(5, 55)),
                                     f"14:{int(rng.integers(10, 59))}:{int(rng.integers(10, 59))}",
                                     fmt_lap(dur) if dur > 60 else f"{dur:.3f}", int(dur * 1000)))
            if sprint:
                for po, d in enumerate(sorted(starters, key=lambda d: -perf[d]), start=1):
                    sprint_id += 1
                    sprints.append((sprint_id, race_id, d, team_of[d], int(rng.integers(1, 40)), po, po, str(po), po,
                                    max(0, 9 - po), 17, N, N, N, N, N, 1))

    drivers = pd.DataFrame(drv_rows, columns=["driverId", "driverRef", "number", "code", "forename", "surname",
                                              "dob", "nationality", "url"])
    res_cols = ["resultId", "raceId", "driverId", "constructorId", "number", "grid", "position", "positionText",
                "positionOrder", "points", "laps", "time", "milliseconds", "fastestLap", "rank", "fastestLapTime",
                "fastestLapSpeed", "statusId"]
    quali_df = pd.DataFrame(quali, columns=["qualifyId", "raceId", "driverId", "constructorId", "number", "position",
                                            "q1", "q2", "q3"])
    # one deliberate orphan foreign key, to exercise the reconciliation check
    quali_df.loc[len(quali_df)] = [quali_id + 1, race_id, 99999, 1, 1, 30, N, N, N]
    tables = {
        "circuits": circuits,
        "constructors": constructors,
        "constructor_results": pd.DataFrame(cres, columns=["constructorResultsId", "raceId", "constructorId",
                                                           "points", "status"]),
        "constructor_standings": pd.DataFrame(cstand, columns=["constructorStandingsId", "raceId", "constructorId",
                                                               "points", "position", "positionText", "wins"]),
        "drivers": drivers,
        "driver_standings": pd.DataFrame(dstand, columns=["driverStandingsId", "raceId", "driverId", "points",
                                                          "position", "positionText", "wins"]),
        "races": pd.DataFrame(races, columns=["raceId", "year", "round", "circuitId", "name", "date", "time", "url",
                                              "fp1_date", "fp1_time", "fp2_date", "fp2_time", "fp3_date", "fp3_time",
                                              "quali_date", "quali_time", "sprint_date", "sprint_time"]),
        "results": pd.DataFrame(results, columns=res_cols),
        "qualifying": quali_df,
        "lap_times": pd.DataFrame(laps, columns=["raceId", "driverId", "lap", "position", "time", "milliseconds"]),
        "pit_stops": pd.DataFrame(pits, columns=["raceId", "driverId", "stop", "lap", "time", "duration",
                                                 "milliseconds"]),
        "sprint_results": pd.DataFrame(sprints, columns=[c for c in res_cols if c != "rank"]),
        "seasons": pd.DataFrame(seasons, columns=["year", "url"]),
        "status": pd.DataFrame(STATUS, columns=["statusId", "status"]),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:24s} {len(df):>7,d} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.seed)
