# ACL Milestone 2 — Formula 1 Knowledge Graph

Builds a Neo4j knowledge graph from 9 tables of the
[Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
and answers 5 verification questions in Cypher.

| file | what |
|---|---|
| `Create_kg.py` | **deliverable** — builds the KG (official `neo4j` Python driver) |
| `config.txt` | **deliverable** — `URI`, `USERNAME`, `PASSWORD` (read by key) |
| `queries.txt` | **deliverable** — the 5 Cypher queries (UTF-8, `//1. Query 1` format) |
| `verify_kg.py` | helper (not submitted): prints node/relationship counts, property types and runs `queries.txt` |

## How to run
1. Start Neo4j 5 (Desktop, Docker or the server zip) and note the `neo4j` password.
2. Put these 9 CSVs **in this folder** (next to `Create_kg.py`):
   `drivers, constructors, circuits, seasons, status, races, results, qualifying, pit_stops` (`.csv`).
   They are git-ignored.
3. Edit `config.txt` and set your own `PASSWORD=` (do not commit it).
4. Run:
   ```bash
   pip install neo4j
   python Create_kg.py
   python verify_kg.py      # optional check
   ```
   Expected: **51,113 nodes and 163,510 relationships**. Running `Create_kg.py` again gives the same numbers.

## Design notes
- Parent nodes (Driver, Constructor, Circuit, Season, Status, Race) are created first, then the fact nodes
  (Result, Qualifying, PitStop) and their links.
- Idempotent: a uniqueness constraint on every id + `MERGE` for nodes and relationships, batched with `UNWIND`.
- Missing values (`\N` or empty cells) are never stored; the property is left out.
- Types: ids and whole numbers are integers, `points/lat/lng/fastestLapSpeed` are floats,
  `Driver.dob` and `Race.date` are Neo4j dates, everything else is a string. `PitStop.eventId` = `"raceId:driverId:stop"`.
- No database name is hardcoded; the session uses the server's default database.
- Q4 computes `Avg_Finish` as `sum / count` (not `avg()`), so equal averages compare equal and ties sort by wins.

## Results on the full dataset (tested on Neo4j Community 5.26)
1. 2021: Max Verstappen 388.5 pts / 10 wins, Lewis Hamilton 385.5 / 8, Valtteri Bottas 219 / 1, … (21 drivers; Grand Prix points only, no sprints).
2. 2014–2021 reliability: Mercedes 298/320 = 0.931, Alfa Romeo 0.908, Ferrari 0.866, … Toro Rosso 0.744 (11 constructors).
3. Hamilton vs Bottas: Mercedes, 100 shared races, Hamilton ahead 75, Bottas ahead 25.
4. Monaco before 2020: Ayrton Senna 10 starts / 3.80 / 6 wins, Lewis Hamilton 13 / 4.00 / 3, Phil Hill 5 / 4.20 / 0, … (106 drivers).
5. 6 drivers changed primary constructor 2020 → 2021: Esteban Ocon, Sergio Pérez, Daniel Ricciardo, Carlos Sainz, Lance Stroll, Sebastian Vettel.
