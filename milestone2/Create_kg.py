"""ACL Milestone 2 - Formula 1 Knowledge Graph.

Builds the F1 knowledge graph in Neo4j from 9 CSV files that sit in the same
folder as this script:
    drivers, constructors, circuits, seasons, status, races,
    results, qualifying, pit_stops

Neo4j credentials are read from config.txt (URI, USERNAME, PASSWORD), by key.
No database name is hardcoded: the session uses the server's default database.

The script is idempotent: uniqueness constraints + MERGE on every id, so running
it again does not create duplicate nodes or relationships.

Usage:  python Create_kg.py
"""

import csv
import os
import sys
from datetime import date

from neo4j import GraphDatabase

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 5000
MISSING = {"", "\\N"}  # the literal \N and empty cells are missing values

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def read_config(path):
    """Read KEY=VALUE lines into a dict. Lines are matched by key, not order."""
    cfg = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            cfg[key.strip().upper()] = value
    for key in ("URI", "USERNAME", "PASSWORD"):
        if key not in cfg:
            sys.exit(f"config.txt is missing the {key}= line")
    return cfg


def find_file(name):
    """Look next to the script first, then in the current folder."""
    for folder in (SCRIPT_DIR, os.getcwd()):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    sys.exit(f"Cannot find {name} (looked in {SCRIPT_DIR} and {os.getcwd()})")


# --------------------------------------------------------------------------
# Type conversion (missing values -> property is omitted)
# --------------------------------------------------------------------------


def to_int(v):
    try:
        return int(v)
    except ValueError:
        return int(float(v))


def to_date(v):
    return date.fromisoformat(v[:10])


CONVERTERS = {"int": to_int, "float": float, "str": str, "date": to_date}


def clean(value, kind):
    """Return the converted value, or None if the value is missing."""
    if value is None:
        return None
    if value.strip() in MISSING:
        return None
    try:
        return CONVERTERS[kind](value.strip() if kind != "str" else value)
    except (ValueError, TypeError):
        print(f"  warning: could not convert {value!r} to {kind}; property omitted")
        return None


def build_props(row, spec):
    """spec = [(csv_column, property_name, type)]. Missing values are skipped."""
    props = {}
    for col, prop, kind in spec:
        v = clean(row.get(col), kind)
        if v is not None:
            props[prop] = v
    return props


def read_csv(name):
    with open(find_file(name), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Schema: property specs (csv column, property name, type)
# --------------------------------------------------------------------------

DRIVER = [
    ("driverId", "driverId", "int"),
    ("driverRef", "driverRef", "str"),
    ("number", "number", "int"),
    ("code", "code", "str"),
    ("forename", "forename", "str"),
    ("surname", "surname", "str"),
    ("dob", "dob", "date"),
    ("nationality", "nationality", "str"),
]
CONSTRUCTOR = [
    ("constructorId", "constructorId", "int"),
    ("constructorRef", "constructorRef", "str"),
    ("name", "constructor_name", "str"),
    ("nationality", "constructor_nationality", "str"),
]
CIRCUIT = [
    ("circuitId", "circuitId", "int"),
    ("circuitRef", "circuitRef", "str"),
    ("name", "circuit_name", "str"),
    ("location", "circuit_loc", "str"),
    ("country", "circuit_country", "str"),
    ("lat", "lat", "float"),
    ("lng", "lng", "float"),
    ("alt", "alt", "int"),
]
SEASON = [("year", "year", "int")]
STATUS = [("statusId", "statusId", "int"), ("status", "status", "str")]
RACE = [
    ("raceId", "raceId", "int"),
    ("year", "year", "int"),
    ("round", "round", "int"),
    ("name", "name", "str"),
    ("date", "date", "date"),
    ("time", "time", "str"),
]
RESULT = [
    ("resultId", "resultId", "int"),
    ("number", "number", "int"),
    ("grid", "grid", "int"),
    ("position", "position", "int"),
    ("positionText", "positionText", "str"),
    ("positionOrder", "positionOrder", "int"),
    ("points", "points", "float"),
    ("laps", "laps", "int"),
    ("time", "time", "str"),
    ("milliseconds", "milliseconds", "int"),
    ("fastestLap", "fastestLap", "int"),
    ("rank", "rank", "int"),
    ("fastestLapTime", "fastestLapTime", "str"),
    ("fastestLapSpeed", "fastestLapSpeed", "float"),
]
QUALIFYING = [
    ("qualifyId", "qualifyId", "int"),
    ("number", "number", "int"),
    ("position", "position", "int"),
    ("q1", "q1", "str"),
    ("q2", "q2", "str"),
    ("q3", "q3", "str"),
]
PITSTOP = [
    ("stop", "stop", "int"),
    ("lap", "lap", "int"),
    ("time", "time", "str"),
    ("duration", "duration", "str"),
    ("milliseconds", "milliseconds", "int"),
]

CONSTRAINTS = [
    ("Driver", "driverId"),
    ("Constructor", "constructorId"),
    ("Circuit", "circuitId"),
    ("Season", "year"),
    ("Status", "statusId"),
    ("Race", "raceId"),
    ("Result", "resultId"),
    ("Qualifying", "qualifyId"),
    ("PitStop", "eventId"),
]

# --------------------------------------------------------------------------
# Neo4j helpers
# --------------------------------------------------------------------------


def run_batches(session, query, rows, label):
    for i in range(0, len(rows), BATCH_SIZE):
        session.run(query, rows=rows[i : i + BATCH_SIZE]).consume()
    print(f"  {label}: {len(rows)} rows")


def create_constraints(session):
    for label, key in CONSTRAINTS:
        name = f"{label.lower()}_{key}_unique"
        try:
            session.run(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
            ).consume()
        except Exception as e:  # e.g. an equal constraint exists under another name
            print(f"  note: constraint on {label}.{key} not created ({e.__class__.__name__})")
    session.run("CALL db.awaitIndexes(300)").consume()


def merge_nodes(session, label, key, rows):
    """rows = [{'id': ..., 'props': {...}}]"""
    query = (
        "UNWIND $rows AS row "
        f"MERGE (n:{label} {{{key}: row.id}}) "
        "SET n += row.props"
    )
    run_batches(session, query, rows, f"{label} nodes")


def merge_rels(session, src, src_key, rel, dst, dst_key, rows):
    """rows = [{'src': ..., 'dst': ...}]  ->  (src)-[:rel]->(dst)"""
    rows = [r for r in rows if r["src"] is not None and r["dst"] is not None]
    query = (
        "UNWIND $rows AS row "
        f"MATCH (a:{src} {{{src_key}: row.src}}) "
        f"MATCH (b:{dst} {{{dst_key}: row.dst}}) "
        f"MERGE (a)-[:{rel}]->(b)"
    )
    run_batches(session, query, rows, f"({src})-[:{rel}]->({dst})")


def node_rows(csv_rows, spec, key):
    rows = []
    for r in csv_rows:
        props = build_props(r, spec)
        if key in props:
            rows.append({"id": props[key], "props": props})
    return rows


def rel_rows(csv_rows, src_col, dst_col):
    return [
        {"src": clean(r.get(src_col), "int"), "dst": clean(r.get(dst_col), "int")}
        for r in csv_rows
    ]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build(session):
    print("Creating constraints ...")
    create_constraints(session)

    print("Loading CSV files ...")
    drivers = read_csv("drivers.csv")
    constructors = read_csv("constructors.csv")
    circuits = read_csv("circuits.csv")
    seasons = read_csv("seasons.csv")
    statuses = read_csv("status.csv")
    races = read_csv("races.csv")
    results = read_csv("results.csv")
    qualifying = read_csv("qualifying.csv")
    pit_stops = read_csv("pit_stops.csv")

    # 1) Parent (entity) nodes
    print("Creating entity nodes ...")
    merge_nodes(session, "Driver", "driverId", node_rows(drivers, DRIVER, "driverId"))
    merge_nodes(session, "Constructor", "constructorId",
                node_rows(constructors, CONSTRUCTOR, "constructorId"))
    merge_nodes(session, "Circuit", "circuitId", node_rows(circuits, CIRCUIT, "circuitId"))
    merge_nodes(session, "Season", "year", node_rows(seasons, SEASON, "year"))
    merge_nodes(session, "Status", "statusId", node_rows(statuses, STATUS, "statusId"))
    merge_nodes(session, "Race", "raceId", node_rows(races, RACE, "raceId"))

    print("Linking races ...")
    merge_rels(session, "Race", "raceId", "AT_CIRCUIT", "Circuit", "circuitId",
               rel_rows(races, "raceId", "circuitId"))
    merge_rels(session, "Race", "raceId", "IN_SEASON", "Season", "year",
               rel_rows(races, "raceId", "year"))

    # 2) Fact nodes and their links
    print("Creating Result facts ...")
    merge_nodes(session, "Result", "resultId", node_rows(results, RESULT, "resultId"))
    for rel, dst, dst_key, col in [
        ("FOR_RACE", "Race", "raceId", "raceId"),
        ("BY_DRIVER", "Driver", "driverId", "driverId"),
        ("FOR_CONSTRUCTOR", "Constructor", "constructorId", "constructorId"),
        ("HAS_STATUS", "Status", "statusId", "statusId"),
    ]:
        merge_rels(session, "Result", "resultId", rel, dst, dst_key,
                   rel_rows(results, "resultId", col))

    print("Creating Qualifying facts ...")
    merge_nodes(session, "Qualifying", "qualifyId",
                node_rows(qualifying, QUALIFYING, "qualifyId"))
    for rel, dst, dst_key, col in [
        ("FOR_RACE", "Race", "raceId", "raceId"),
        ("BY_DRIVER", "Driver", "driverId", "driverId"),
        ("FOR_CONSTRUCTOR", "Constructor", "constructorId", "constructorId"),
    ]:
        merge_rels(session, "Qualifying", "qualifyId", rel, dst, dst_key,
                   rel_rows(qualifying, "qualifyId", col))

    print("Creating PitStop facts ...")
    pit_nodes, pit_race, pit_driver = [], [], []
    for r in pit_stops:
        race_id = clean(r.get("raceId"), "int")
        driver_id = clean(r.get("driverId"), "int")
        stop = clean(r.get("stop"), "int")
        if race_id is None or driver_id is None or stop is None:
            continue
        event_id = f"{race_id}:{driver_id}:{stop}"
        props = build_props(r, PITSTOP)
        props["eventId"] = event_id
        pit_nodes.append({"id": event_id, "props": props})
        pit_race.append({"src": event_id, "dst": race_id})
        pit_driver.append({"src": event_id, "dst": driver_id})
    merge_nodes(session, "PitStop", "eventId", pit_nodes)
    merge_rels(session, "PitStop", "eventId", "FOR_RACE", "Race", "raceId", pit_race)
    merge_rels(session, "PitStop", "eventId", "BY_DRIVER", "Driver", "driverId", pit_driver)


def print_counts(session):
    nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"Done. Graph has {nodes} nodes and {rels} relationships.")


def main():
    cfg = read_config(find_file("config.txt"))
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    try:
        driver.verify_connectivity()
        with driver.session() as session:  # default database, none hardcoded
            build(session)
            print_counts(session)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
