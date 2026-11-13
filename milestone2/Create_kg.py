"""ACL Milestone 2 - Formula 1 Knowledge Graph.

Reads 9 CSV files from the same folder as this script and builds the
knowledge graph in Neo4j. Credentials come from config.txt (URI, USERNAME,
PASSWORD). Running the script again does not create duplicates, because every
node and relationship is created with MERGE on a unique id.

Usage:  python Create_kg.py
"""

import csv
import os
from datetime import date

from neo4j import GraphDatabase

FOLDER = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 5000


# ---------------------------------------------------------------- helpers

def read_config():
    """Read config.txt by key (KEY=VALUE), so extra lines like DATABASE= are fine."""
    config = {}
    with open(os.path.join(FOLDER, "config.txt"), encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key.strip()] = value.strip()
    return config


def read_csv(name):
    with open(os.path.join(FOLDER, name), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def convert(value, kind):
    """Convert a CSV value to its type. Missing values (\\N or empty) give None."""
    if value is None or value.strip() in ("", "\\N"):
        return None
    value = value.strip()
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "date":
        return date.fromisoformat(value)
    return value  # string


def props(row, columns):
    """Build the property map of one node. Missing values are left out."""
    result = {}
    for prop, (column, kind) in columns.items():
        value = convert(row[column], kind)
        if value is not None:
            result[prop] = value
    return result


def ids(row, *columns):
    """The ids of the parent nodes a fact node links to, as integers."""
    return {column: convert(row[column], "int") for column in columns}


def load(session, query, rows, label):
    """Send the rows to Neo4j in batches (one UNWIND query per batch)."""
    for i in range(0, len(rows), BATCH_SIZE):
        session.run(query, rows=rows[i:i + BATCH_SIZE]).consume()
    print(f"{label}: {len(rows)}")


# ---------------------------------------------------------------- schema
# property name: (CSV column, type)

DRIVER = {
    "driverId": ("driverId", "int"), "driverRef": ("driverRef", "str"),
    "number": ("number", "int"), "code": ("code", "str"),
    "forename": ("forename", "str"), "surname": ("surname", "str"),
    "dob": ("dob", "date"), "nationality": ("nationality", "str"),
}
CONSTRUCTOR = {
    "constructorId": ("constructorId", "int"), "constructorRef": ("constructorRef", "str"),
    "constructor_name": ("name", "str"), "constructor_nationality": ("nationality", "str"),
}
CIRCUIT = {
    "circuitId": ("circuitId", "int"), "circuitRef": ("circuitRef", "str"),
    "circuit_name": ("name", "str"), "circuit_loc": ("location", "str"),
    "circuit_country": ("country", "str"), "lat": ("lat", "float"),
    "lng": ("lng", "float"), "alt": ("alt", "int"),
}
SEASON = {"year": ("year", "int")}
STATUS = {"statusId": ("statusId", "int"), "status": ("status", "str")}
RACE = {
    "raceId": ("raceId", "int"), "year": ("year", "int"), "round": ("round", "int"),
    "name": ("name", "str"), "date": ("date", "date"), "time": ("time", "str"),
}
RESULT = {
    "resultId": ("resultId", "int"), "number": ("number", "int"), "grid": ("grid", "int"),
    "position": ("position", "int"), "positionText": ("positionText", "str"),
    "positionOrder": ("positionOrder", "int"), "points": ("points", "float"),
    "laps": ("laps", "int"), "time": ("time", "str"), "milliseconds": ("milliseconds", "int"),
    "fastestLap": ("fastestLap", "int"), "rank": ("rank", "int"),
    "fastestLapTime": ("fastestLapTime", "str"), "fastestLapSpeed": ("fastestLapSpeed", "float"),
}
QUALIFYING = {
    "qualifyId": ("qualifyId", "int"), "number": ("number", "int"),
    "position": ("position", "int"), "q1": ("q1", "str"), "q2": ("q2", "str"), "q3": ("q3", "str"),
}
PITSTOP = {
    "stop": ("stop", "int"), "lap": ("lap", "int"), "time": ("time", "str"),
    "duration": ("duration", "str"), "milliseconds": ("milliseconds", "int"),
}

# ---------------------------------------------------------------- Cypher

CONSTRAINTS = [
    "CREATE CONSTRAINT driver_id IF NOT EXISTS FOR (n:Driver) REQUIRE n.driverId IS UNIQUE",
    "CREATE CONSTRAINT constructor_id IF NOT EXISTS FOR (n:Constructor) REQUIRE n.constructorId IS UNIQUE",
    "CREATE CONSTRAINT circuit_id IF NOT EXISTS FOR (n:Circuit) REQUIRE n.circuitId IS UNIQUE",
    "CREATE CONSTRAINT season_year IF NOT EXISTS FOR (n:Season) REQUIRE n.year IS UNIQUE",
    "CREATE CONSTRAINT status_id IF NOT EXISTS FOR (n:Status) REQUIRE n.statusId IS UNIQUE",
    "CREATE CONSTRAINT race_id IF NOT EXISTS FOR (n:Race) REQUIRE n.raceId IS UNIQUE",
    "CREATE CONSTRAINT result_id IF NOT EXISTS FOR (n:Result) REQUIRE n.resultId IS UNIQUE",
    "CREATE CONSTRAINT qualifying_id IF NOT EXISTS FOR (n:Qualifying) REQUIRE n.qualifyId IS UNIQUE",
    "CREATE CONSTRAINT pitstop_id IF NOT EXISTS FOR (n:PitStop) REQUIRE n.eventId IS UNIQUE",
]

# Entity nodes: each row is the node's property map.
DRIVER_QUERY = """
UNWIND $rows AS row
MERGE (d:Driver {driverId: row.driverId})
SET d += row
"""
CONSTRUCTOR_QUERY = """
UNWIND $rows AS row
MERGE (c:Constructor {constructorId: row.constructorId})
SET c += row
"""
CIRCUIT_QUERY = """
UNWIND $rows AS row
MERGE (c:Circuit {circuitId: row.circuitId})
SET c += row
"""
SEASON_QUERY = """
UNWIND $rows AS row
MERGE (s:Season {year: row.year})
"""
STATUS_QUERY = """
UNWIND $rows AS row
MERGE (s:Status {statusId: row.statusId})
SET s += row
"""

# Race and fact nodes: each row has the property map plus the parent ids.
RACE_QUERY = """
UNWIND $rows AS row
MERGE (r:Race {raceId: row.props.raceId})
SET r += row.props
WITH r, row
MATCH (c:Circuit {circuitId: row.circuitId})
MATCH (s:Season {year: row.props.year})
MERGE (r)-[:AT_CIRCUIT]->(c)
MERGE (r)-[:IN_SEASON]->(s)
"""
RESULT_QUERY = """
UNWIND $rows AS row
MERGE (res:Result {resultId: row.props.resultId})
SET res += row.props
WITH res, row
MATCH (r:Race {raceId: row.raceId})
MATCH (d:Driver {driverId: row.driverId})
MATCH (c:Constructor {constructorId: row.constructorId})
MATCH (s:Status {statusId: row.statusId})
MERGE (res)-[:FOR_RACE]->(r)
MERGE (res)-[:BY_DRIVER]->(d)
MERGE (res)-[:FOR_CONSTRUCTOR]->(c)
MERGE (res)-[:HAS_STATUS]->(s)
"""
QUALIFYING_QUERY = """
UNWIND $rows AS row
MERGE (q:Qualifying {qualifyId: row.props.qualifyId})
SET q += row.props
WITH q, row
MATCH (r:Race {raceId: row.raceId})
MATCH (d:Driver {driverId: row.driverId})
MATCH (c:Constructor {constructorId: row.constructorId})
MERGE (q)-[:FOR_RACE]->(r)
MERGE (q)-[:BY_DRIVER]->(d)
MERGE (q)-[:FOR_CONSTRUCTOR]->(c)
"""
PITSTOP_QUERY = """
UNWIND $rows AS row
MERGE (p:PitStop {eventId: row.props.eventId})
SET p += row.props
WITH p, row
MATCH (r:Race {raceId: row.raceId})
MATCH (d:Driver {driverId: row.driverId})
MERGE (p)-[:FOR_RACE]->(r)
MERGE (p)-[:BY_DRIVER]->(d)
"""


def pit_stop_row(r):
    row = {"props": props(r, PITSTOP), **ids(r, "raceId", "driverId")}
    row["props"]["eventId"] = f"{row['raceId']}:{row['driverId']}:{row['props']['stop']}"
    return row


# ---------------------------------------------------------------- main

def main():
    config = read_config()
    with GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"])) as driver:
        with driver.session() as session:  # default database (no name hardcoded)
            for query in CONSTRAINTS:
                session.run(query).consume()

            # 1) parent (entity) nodes first
            load(session, DRIVER_QUERY, [props(r, DRIVER) for r in read_csv("drivers.csv")], "Driver")
            load(session, CONSTRUCTOR_QUERY, [props(r, CONSTRUCTOR) for r in read_csv("constructors.csv")], "Constructor")
            load(session, CIRCUIT_QUERY, [props(r, CIRCUIT) for r in read_csv("circuits.csv")], "Circuit")
            load(session, SEASON_QUERY, [props(r, SEASON) for r in read_csv("seasons.csv")], "Season")
            load(session, STATUS_QUERY, [props(r, STATUS) for r in read_csv("status.csv")], "Status")
            load(session, RACE_QUERY,
                 [{"props": props(r, RACE), **ids(r, "circuitId")} for r in read_csv("races.csv")], "Race")

            # 2) fact nodes and their relationships
            load(session, RESULT_QUERY,
                 [{"props": props(r, RESULT), **ids(r, "raceId", "driverId", "constructorId", "statusId")}
                  for r in read_csv("results.csv")], "Result")
            load(session, QUALIFYING_QUERY,
                 [{"props": props(r, QUALIFYING), **ids(r, "raceId", "driverId", "constructorId")}
                  for r in read_csv("qualifying.csv")], "Qualifying")
            load(session, PITSTOP_QUERY, [pit_stop_row(r) for r in read_csv("pit_stops.csv")], "PitStop")

            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"Done: {nodes} nodes, {rels} relationships.")


if __name__ == "__main__":
    main()
