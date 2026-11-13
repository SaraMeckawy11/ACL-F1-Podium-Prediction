"""Check the KG built by Create_kg.py and run the queries in queries.txt.

Not a deliverable; a helper for testing.  Usage:  python verify_kg.py
"""

import os
import re
import sys

from neo4j import GraphDatabase

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Create_kg import find_file, read_config  # noqa: E402

EXPECTED_NODES, EXPECTED_RELS = 51113, 163510


def load_queries(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^//\s*(\d+)\.\s*Query\s*\d+\s*$", text, flags=re.M)
    return [(int(parts[i]), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def main():
    cfg = read_config(find_file("config.txt"))
    driver = GraphDatabase.driver(cfg["URI"], auth=(cfg["USERNAME"], cfg["PASSWORD"]))
    with driver.session() as s:
        n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"nodes {n} (expected {EXPECTED_NODES})  rels {r} (expected {EXPECTED_RELS})")
        for row in s.run("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY l"):
            print(f"  {row['l']:<12}{row['c']}")
        for row in s.run(
            "MATCH (a)-[r]->(b) RETURN labels(a)[0] AS a, type(r) AS t, labels(b)[0] AS b, "
            "count(*) AS c ORDER BY a, t"
        ):
            print(f"  ({row['a']})-[:{row['t']}]->({row['b']}) {row['c']}")
        # every fact node has exactly its listed relationships
        for label, k in [("Result", 4), ("Qualifying", 3), ("PitStop", 2), ("Race", 2)]:
            bad = s.run(
                f"MATCH (n:{label}) WITH n, COUNT {{ (n)-->() }} AS k WHERE k <> $k RETURN count(n) AS c",
                k=k,
            ).single()["c"]
            print(f"  {label} nodes without exactly {k} outgoing rels: {bad}")
        # property types
        for row in s.run(
            "MATCH (n) WITH labels(n)[0] AS l, n UNWIND keys(n) AS k "
            "RETURN l, k, collect(DISTINCT valueType(n[k])) AS types ORDER BY l, k"
        ):
            print(f"  {row['l']}.{row['k']}: {row['types']}")
        bad = s.run("MATCH (n) UNWIND keys(n) AS k WITH n[k] AS v "
                    "WHERE v = '\\\\N' OR v = '' RETURN count(*) AS c").single()["c"]
        print(f"  properties stored as '\\N' or '': {bad}")

        for num, q in load_queries(os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.txt")):
            res = s.run(q)
            rows = [rec.data() for rec in res]
            print(f"\n=== Query {num}: columns {res.keys()}  rows {len(rows)}")
            for row in rows[:25]:
                print("  ", row)
    driver.close()


if __name__ == "__main__":
    main()
