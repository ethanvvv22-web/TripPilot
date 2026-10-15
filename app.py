"""TripPilot web UI.

    streamlit run app.py
"""

import json
import os
import sys
from datetime import date, time, timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from trippilot.agent import INSTRUCTIONS, TOOL_SPECS, TripPlanner  # noqa: E402
from trippilot.agent.foundry import FoundryAgent  # noqa: E402
from trippilot.agent.toolbox import ToolCallRecord  # noqa: E402
from trippilot.models import TravelMode, TripRequest, google_day  # noqa: E402
from trippilot.dataset import load_snapshot  # noqa: E402
from trippilot.telemetry import setup_tracing  # noqa: E402
from trippilot.tools import PlacesClient, RoutesClient  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

MODE_LABELS = {TravelMode.WALK: "Walk", TravelMode.TRANSIT: "Public transport",
               TravelMode.BICYCLE: "Bicycle", TravelMode.DRIVE: "Car"}


@st.cache_resource
def tracing():
    # Once per server process: every agent run is exported to traces/spans.jsonl.
    os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True)
    setup_tracing(jsonl_path=os.path.join(ROOT, "traces", "spans.jsonl"))


tracing()


@st.cache_resource
def snapshot():
    return load_snapshot(os.path.join(ROOT, "data", "uppsala_snapshot.json"))


@st.cache_resource
def google_clients(key: str):
    # Cached across reruns so the clients' response caches are reused.
    return PlacesClient(key), RoutesClient(key)


def next_saturday() -> date:
    today = date.today()
    return today + timedelta(days=(5 - today.weekday()) % 7 or 7)


def describe(record: ToolCallRecord) -> str:
    args, out = record.arguments, record.output
    if "error" in out and not record.name.startswith("submit_"):
        return f"⚠️ `{record.name}` failed: {out['error']}"
    if record.name == "search_places":
        return f"🔎 Searched **{', '.join(args.get('queries', []))}** → {len(out.get('places', []))} places"
    if record.name == "get_place_details":
        return f"📍 Checked details of **{out.get('name', args.get('place_id'))}**"
    if record.name == "get_travel_times":
        return f"🧭 Travel-time matrix for {len(args.get('place_ids', []))} places"
    if record.name.startswith("submit_"):
        if out.get("accepted"):
            return "✅ **Accepted by the deterministic checks**"
        problems = [p["message"] for p in out.get("violations", [])] + out.get("problems", [])
        problems = problems or [out.get("error", "unknown error")]
        return f"❌ **Rejected** ({len(problems)} problems): " + "; ".join(problems)
    return f"`{record.name}`"


def run_agent(label: str, task):
    """Create a Foundry agent, run `task(planner)` while streaming tool calls, then clean up."""
    with st.status(label, expanded=True) as status:
        agent = FoundryAgent(
            endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            instructions=INSTRUCTIONS,
            tool_specs=TOOL_SPECS,
        )
        try:
            planner = TripPlanner(agent, *data_source(), on_tool_call=lambda r: status.write(describe(r)))
            result = task(planner)
        finally:
            agent.delete()
        ok = result.status == "accepted"
        status.update(label="Done" if ok else f"Ended: {result.status}",
                      state="complete" if ok else "error", expanded=False)
    return result


def show_submissions(trace):
    for n, record in enumerate([r for r in trace if r.name.startswith("submit_")], 1):
        out = record.output
        if out.get("accepted"):
            st.success(f"Attempt {n}: accepted")
            continue
        problems = [f"`{p['type']}` {p['message']}" for p in out.get("violations", [])]
        problems += out.get("problems", []) or ([out["error"]] if "error" in out else [])
        with st.container(border=True):
            st.markdown(f"**Attempt {n}: rejected**")
            for p in problems:
                st.markdown(f"- {p}")


def show_trace(trace):
    with st.expander(f"Tool-call trace ({len(trace)} calls)"):
        st.dataframe(pd.DataFrame([
            {"#": i, "Tool": r.name, "Arguments": json.dumps(r.arguments, ensure_ascii=False)[:120],
             "ms": round(r.duration_ms, 1)}
            for i, r in enumerate(trace, 1)
        ]), hide_index=True, width="stretch")


def show_map(points, connect: bool = True):
    df = pd.DataFrame(points)
    layers = []
    if connect:
        layers.append(pdk.Layer("PathLayer", [{"path": [[p["lng"], p["lat"]] for p in points]}], get_path="path",
                                get_width=4, width_units="pixels", get_color=[90, 90, 220, 180]))
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=df.lat.mean(), longitude=df.lng.mean(), zoom=12.5),
        layers=layers + [
            pdk.Layer("ScatterplotLayer", df, get_position="[lng, lat]", get_radius=9,
                      radius_units="pixels", get_fill_color=[220, 60, 60, 220], pickable=True),
            pdk.Layer("TextLayer", df, get_position="[lng, lat]", get_text="label", get_size=13,
                      get_alignment_baseline="'bottom'", get_pixel_offset=[0, -12]),
        ],
        tooltip={"text": "{label}"},
    ))


# ---------------------------------------------------------------- sidebar
st.set_page_config(page_title="TripPilot", page_icon="🗺️", layout="wide")
st.title("🗺️ TripPilot")
st.caption("An LLM agent plans, replans and recommends; deterministic checks verify every answer before you see it.")

google_key = os.getenv("GOOGLE_MAPS_API_KEY", "")

with st.sidebar.form("trip_form"):
    st.header("Your trip")
    source = st.radio("Data source", ["Google (live)", "Recorded Google snapshot (Uppsala)"],
                      index=0 if google_key else 1)
    live = source.startswith("Google")
    city = st.text_input("City", "Stockholm" if live else "Uppsala", disabled=not live)
    start_place = st.text_input("Start location", "Sergels torg, Stockholm" if live else "Uppsala Central Station",
                                disabled=not live)
    trip_date = st.date_input("Date", next_saturday())
    c1, c2 = st.columns(2)
    start_time = c1.time_input("From", time(9, 0))
    end_time = c2.time_input("To", time(17, 0))
    mode = st.selectbox("Travel mode", list(MODE_LABELS), format_func=MODE_LABELS.get)
    budget = st.number_input("Budget (0 = no limit)", min_value=0, value=600, step=50)
    preferences = st.text_input("Preferences", "history, fika")
    plan_clicked = st.form_submit_button("Plan my day", type="primary", width="stretch")

if live and not google_key:
    st.error("GOOGLE_MAPS_API_KEY is missing from .env; use the recorded snapshot or add the key.")
    st.stop()


def data_source():
    """(places provider, travel-time function) for the selected data source."""
    if live:
        places, routes = google_clients(google_key)
        return places, routes.travel_minutes
    return snapshot().provider(), snapshot().travel_minutes


def build_request(start: time) -> TripRequest:
    if live:
        found = data_source()[0].search_text(start_place, max_results=1)
        if not found:
            st.error(f"Could not find the start location '{start_place}'.")
            st.stop()
        location = found[0].location
    else:
        location = snapshot().start  # travel times exist only between snapshot points
    return TripRequest(
        city=city, date=trip_date, start_time=start, end_time=end_time,
        start_location=location, travel_mode=mode, budget=budget or None,
        preferences=[p.strip() for p in preferences.split(",") if p.strip()],
    )


if plan_clicked:
    if end_time <= start_time:
        st.error("The end time must be after the start time.")
        st.stop()
    request = build_request(start_time)
    st.session_state.request = request
    st.session_state.previous = None
    st.session_state.result = run_agent("The agent is planning your day…", lambda p: p.plan(request))

tab_plan, tab_rec = st.tabs(["🗓️ Plan & replan", "💡 Recommendations"])

# ---------------------------------------------------------------- plan & replan
with tab_plan:
    result = st.session_state.get("result")
    request = st.session_state.get("request")
    if result is None:
        st.info("Fill in your trip on the left and press **Plan my day**.")
    else:
        stops = result.itinerary.stops if result.itinerary else []
        locked = len(result.locked_stops)
        total_cost = sum(s.estimated_cost for s in stops)

        m = st.columns(6)
        m[0].metric("Status", result.status)
        m[1].metric("Submissions", result.submissions)
        m[2].metric("Agent turns", result.turns)
        m[3].metric("Tool calls", len(result.trace))
        m[4].metric("Tokens (in/out)", f"{result.input_tokens / 1000:.1f}k / {result.output_tokens / 1000:.1f}k")
        m[5].metric("Est. cost", f"{total_cost:g}" + (f" / {request.budget:g}" if request.budget else ""))

        previous = st.session_state.get("previous")
        if previous is not None and previous.itinerary is not None:
            before = [s.name for s in previous.itinerary.stops[locked:]]
            after = [s.name for s in stops[locked:]]
            kept = [n for n in after if n in before]
            st.info(
                f"**Replanned.** Kept {len(kept)}: {', '.join(kept) or '—'} · "
                f"Removed: {', '.join(n for n in before if n not in after) or '—'} · "
                f"Added: {', '.join(n for n in after if n not in before) or '—'}"
            )

        left, right = st.columns(2)
        with left:
            st.subheader("Itinerary")
            day = google_day(request.date)
            rows = []
            for i, s in enumerate(stops, 1):
                place = result.places.get(s.place_id)
                rows.append({
                    "#": i,
                    "": "🔒" if i <= locked else "",
                    "Time": f"{s.arrival:%H:%M}–{s.departure:%H:%M}",
                    "Place": s.name,
                    "Open that day": place.hours_on(day) if place else "?",
                    "Rating": place.rating if place else None,
                    "Est. cost": s.estimated_cost,
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                if locked:
                    st.caption("🔒 Already started before the change; kept as they were.")
            for v in result.violations:
                st.warning(v.message)

        with right:
            st.subheader("Map")
            points = [{"label": "Start", "lat": request.start_location.lat, "lng": request.start_location.lng}]
            for i, s in enumerate(stops, 1):
                place = result.places.get(s.place_id)
                if place:
                    points.append({"label": f"{i}. {s.name}", "lat": place.location.lat, "lng": place.location.lng})
            show_map(points)
            st.caption("Lines connect stops in order; actual routes follow streets.")

        st.subheader("Agent's summary")
        st.markdown(result.summary or "_No summary._")

        if result.itinerary is not None:
            st.subheader("Change of plans")
            with st.form("replan_form"):
                c1, c2 = st.columns([1, 3])
                default_now = stops[0].departure if stops else request.start_time
                now = c1.time_input("Current time", default_now)
                change = c2.text_input("What changed?", "It's raining — swap outdoor stops for indoor ones.")
                replan_clicked = st.form_submit_button("Replan the rest of the day")
            if replan_clicked:
                try:
                    new = run_agent("The agent is replanning…", lambda p: p.replan(
                        request, result.itinerary, now, change, result.places))
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.session_state.previous = result
                    st.session_state.result = new
                    st.rerun()

        st.subheader("Validate-and-repair history")
        show_submissions(result.trace)
        show_trace(result.trace)

# ---------------------------------------------------------------- recommendations
with tab_rec:
    with st.form("recommend_form"):
        c1, c2 = st.columns([3, 1])
        query = c1.text_input("What are you looking for?", "A cosy café for fika")
        rec_time = c2.time_input("From", time(14, 0))
        rec_clicked = st.form_submit_button("Recommend")
    if rec_clicked:
        rec_request = build_request(rec_time)
        st.session_state.rec_request = rec_request
        st.session_state.rec = run_agent("The agent is looking for places…", lambda p: p.recommend(rec_request, query))

    rec = st.session_state.get("rec")
    if rec is not None:
        rec_request = st.session_state.rec_request
        m = st.columns(4)
        m[0].metric("Status", rec.status)
        m[1].metric("Submissions", rec.submissions)
        m[2].metric("Agent turns", rec.turns)
        m[3].metric("Tool calls", len(rec.trace))

        day = google_day(rec_request.date)
        for r in rec.recommendations:
            with st.container(border=True):
                rating = f" · ⭐ {r.place.rating}" if r.place.rating else ""
                st.markdown(f"**{r.place.name}**{rating} · open {r.place.hours_on(day)}")
                st.write(r.reason)
                if r.place.address:
                    st.caption(r.place.address)
        for p in rec.problems:
            st.warning(p)
        if rec.recommendations:
            show_map([{"label": r.place.name, "lat": r.place.location.lat, "lng": r.place.location.lng}
                      for r in rec.recommendations], connect=False)
        st.markdown(rec.summary or "")
        show_submissions(rec.trace)
        show_trace(rec.trace)
