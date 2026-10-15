# TripPilot

An agentic trip-planning app. An LLM agent plans itineraries by calling tools
(Google Places, Google Routes), and a deterministic validator checks every plan
for schedule conflicts and route feasibility before it reaches the user.

## Design

```
user request
   │
   ▼
agent (Microsoft Foundry Agent Service, tool calling)
   ├─ search_places / get_place   → Google Places API (New)
   ├─ travel time                 → Google Routes API
   ▼
draft itinerary (structured JSON)
   ▼
deterministic validator ── violations ──► fed back to the agent to repair
   ▼
feasible itinerary
```

The validator is plain code, not an LLM. It checks:

| Check | Example |
|---|---|
| Unknown place | a `place_id` that no tool call returned (hallucinated stop) |
| Duplicate place | visiting a place twice, or revisiting a place after a replan |
| Invalid times | departure not after arrival |
| Trip window | first stop before the trip starts, last stop after it ends |
| Overlap | arriving at a stop before leaving the previous one |
| Travel time | not enough time between stops for the chosen travel mode |
| No route | no route exists between two stops |
| Opening hours | visiting outside the place's opening hours |
| Budget | estimated total cost above the budget |

The agent reaches the validator through a `submit_itinerary` tool. The
planner, not the model, decides whether the final itinerary is accepted.

## Layout

```
src/trippilot/
  models.py          data types: Place, TripRequest, Itinerary, opening hours
  validation.py      deterministic itinerary validator
  tools/places.py    Google Places API (New) client
  tools/routes.py    Google Routes API (route matrix) client
  agent/toolbox.py   tool schemas and execution, including submit_itinerary
  agent/planner.py   tool-call loop with validate-and-repair limits
  agent/foundry.py   Microsoft Foundry Agent Service backend
  offline.py         Uppsala sample data for running without Google APIs
tests/               unit tests; no network or API keys needed
```

## Running

```bash
# tests
python -m unittest discover -s tests -t .

# plan a trip (needs `pip install .[foundry]`, `az login`, and .env filled in)
PYTHONPATH=src python -m trippilot --offline --preferences history fika
```

## Evaluation

Two scenario sets, 30 scenarios each. Every score is computed by code: the validator decides
feasibility, and each scenario adds deterministic expectations (e.g. no outdoor stops after rain).

| Set | Data |
|---|---|
| `eval/scenarios_real.json` | `data/uppsala_snapshot.json`: 36 real places with Google opening hours and Routes API travel times for walk, bicycle, transit and drive, recorded once with `python -m trippilot.snapshot` |
| `eval/scenarios.json` | a small hand-written sample with made-up hours and straight-line travel times; kept for unit tests only, not used for reported results |

The snapshot makes runs reproducible and free, and lets the Harbor verifier score without an API
key. Tags that drive preference matching are derived from Google's place types and corrected by
hand in `data/tag_overrides.json`.

```bash
# in-process benchmark (tools: batched or legacy)
PYTHONPATH=src python -m trippilot.evaluation --scenarios eval/scenarios_real.json --out eval/results/my-run

# the same scenarios as Harbor tasks, scored inside Docker
PYTHONPATH=src python -m trippilot.harbor_tasks --scenarios eval/scenarios_real.json --out eval/harbor/tasks
uv tool install --python 3.12 harbor --with-editable ".[foundry]" --with opentelemetry-sdk
# PYTHONUTF8=1: on Windows, make Harbor read task files as UTF-8
PYTHONUTF8=1 harbor run -p eval/harbor/tasks -a trippilot.harbor_agent:TripPilotAgent -n 2 -o eval/harbor/jobs -y
```

## Results

All on `eval/scenarios_real.json` (30 scenarios, recorded Google data for Uppsala), gpt-5-mini,
one run per scenario per configuration. Raw per-scenario results and OpenTelemetry spans are in
`eval/results/`.

| Run | First attempt feasible | Final feasible | Task success | Turns | Input tokens |
|---|---|---|---|---|---|
| `real-batched` | 87% | 100% | 100% | 4.4 | 19,540 |
| `trippilot-batched` | 70% | 100% | 97% | 4.9 | 21,855 |
| `real-legacy` (unbatched tools) | 87% | 100% | 90% | 6.7 | 27,243 |
| `trippilot-legacy` (unbatched tools) | 93% | 100% | 100% | 7.5 | 28,541 |

- Across both batched runs (60 plans), 22% of first submissions were infeasible, all because they
  left too little travel time; every final itinerary passed the validator.
- Batched tools vs. unbatched, pooled over both pairs of runs: 35% fewer turns, 26% fewer input tokens.
- The first-attempt rate varied from 87% to 70% between identical runs, so single-run numbers
  should be read with that spread in mind.
- Harbor, same scenarios scored inside Docker: 29/30 (the miss planned one stop where two were required).
- `real-*` runs predate two tag corrections in `data/tag_overrides.json` (Stadsträdgården: garden;
  Church of Old Uppsala: viking), which affect task success only, not feasibility or token counts.
- The project was renamed from Itinera to TripPilot on 2026-10-15; `real-*` runs use the former name.

## Roadmap

- [x] Phase 1: data model, validator, Google tool clients, unit tests
- [x] Phase 2: Foundry agent with tool calling; validate-and-repair loop
- [x] Phase 3: replanning (locks started stops), recommendations, Google APIs, Streamlit UI (`streamlit run app.py`)
- [x] Phase 4a: OpenTelemetry tracing (GenAI conventions; `--trace FILE`, UI writes `traces/spans.jsonl`)
- [x] Phase 4b: evaluation scenarios and metrics; Harbor tasks with an in-container verifier
- [ ] Phase 5: deploy, demo
