"""Dashboard: customer issues from Swiggy app reviews, how they were merged,
and how well each stage of the pipeline works.

It reads only the aggregated outputs committed to the repository; no review text.
Run from the repository root:
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py
"""
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/Kkumar-007/app-review-issue-tracker"

# Colours: one accent, quiet greys (reference palette, light mode)
ACCENT = "#2a78d6"
MUTED = "#c3c2b7"
INK_2 = "#52514e"
GRID = "#e1e0d9"

st.set_page_config(page_title="Customer issues in Swiggy app reviews", layout="wide")


def style(chart):
    return (chart.configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor=GRID,
                                 domainColor=MUTED, tickColor=MUTED, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12, orient="top")
            .configure_view(strokeWidth=0))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


@st.cache_data
def load():
    trends = pd.read_csv(ROOT / "output_v3" / "topic_trends.csv")
    days = [c for c in trends.columns if c not in ("topic", "total")]
    totals = pd.read_csv(ROOT / "output_v2" / "daily_totals.csv", dtype={"day": str})
    mapping = pd.read_csv(ROOT / "output_v3" / "topic_mapping.csv")
    spikes = pd.read_csv(ROOT / "output_v3" / "spikes.csv")
    triage = pd.read_csv(ROOT / "analysis" / "summary_triage_scores.csv")
    position = pd.read_csv(ROOT / "analysis" / "summary_triage_position.csv")
    stars = pd.read_csv(ROOT / "analysis" / "summary_triage_stars.csv")
    samples = {name: pd.read_csv(ROOT / "analysis" / f"merge_sample_{tag}.csv", keep_default_na=False,
                                 encoding="utf-8-sig")
               for name, tag in [("v2", "output_v2"), ("v3, floor 0.60 (final)", "output_v3"),
                                 ("v3, floor 0.70 (tested)", "output_v3_floor070")]}
    tables = {"v1": ROOT / "output" / "topic_trends.csv", "v2": ROOT / "output_v2" / "topic_trends.csv",
              "v3, floor 0.60 (final)": ROOT / "output_v3" / "topic_trends.csv",
              "v3, floor 0.70 (tested)": ROOT / "output_v3_floor070" / "topic_trends.csv"}
    topic_counts = []
    for name, path in tables.items():
        t = pd.read_csv(path)
        total = t.select_dtypes("number").drop(columns=["total"], errors="ignore").sum(axis=1)
        topic_counts.append({"version": name, "topics": len(t), "seen_once": (total == 1).mean()})
    return trends, days, totals, mapping, spikes, triage, position, stars, samples, pd.DataFrame(topic_counts)


trends, days, totals, mapping, spikes, triage, position, stars, samples, topic_counts = load()
day_labels = {d: f"{pd.Timestamp(d).day} {pd.Timestamp(d):%b}" for d in days}   # portable (no %-d on Windows)
actionable = totals.set_index("day")["actionable"].reindex(days)

st.title("Customer issues in Swiggy app reviews")
st.markdown(
    f"What customers complained about in Swiggy's Google Play reviews, 1–9 October 2025. "
    f"A local Llama 3.1 8B filters reviews, extracts specific issues and merges them into topics; "
    f"every stage is checked against hand labels. [Code and method]({REPO_URL})")

tab_issues, tab_merges, tab_eval = st.tabs(["Issues", "How phrases were merged", "How well it works"])

# ---------------------------------------------------------------- Issues
with tab_issues:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews read", f"{totals['reviews'].sum():,}")
    c2.metric("Actionable", f"{totals['actionable'].sum():,} ({totals['actionable'].sum() / totals['reviews'].sum():.0%})")
    c3.metric("Topics", f"{len(trends):,}")
    c4.metric(f"Largest topic ({int(trends.iloc[0]['total'])} mentions)", trends.iloc[0]["topic"])

    f1, f2 = st.columns([2, 1])
    query = f1.text_input("Search topics", placeholder="e.g. refund, delivery partner, coupon")
    min_mentions = f2.slider("Minimum mentions over 9 days", 1, 50, 10)
    view = trends[trends["total"] >= min_mentions]
    if query:
        view = view[view["topic"].str.contains(query, case=False, regex=False)]
    view = view.assign(daily=view[days].values.tolist(),
                       per_100=(view["total"] / actionable.sum() * 100).round(1))
    st.dataframe(
        view[["topic", "total", "per_100", "daily"]], hide_index=True, width="stretch", height=380,
        column_config={
            "topic": st.column_config.TextColumn("Topic", width="large"),
            "total": st.column_config.NumberColumn("Mentions", format="%d"),
            "per_100": st.column_config.NumberColumn("Per 100 actionable reviews", format="%.1f"),
            "daily": st.column_config.LineChartColumn("Daily mentions, 1–9 Oct", y_min=0),
        })
    st.caption(f"{len(view)} of {len(trends)} topics shown. A review can mention more than one issue.")

    st.subheader("One topic, day by day")
    choice = st.selectbox("Topic", view["topic"].tolist() or trends["topic"].tolist())
    row = trends.set_index("topic").loc[choice]
    daily = pd.DataFrame({"day": [day_labels[d] for d in days], "date": days,
                          "mentions": [int(row[d]) for d in days],
                          "share": [row[d] / actionable[d] for d in days],
                          "actionable": actionable.values})
    base = alt.Chart(daily).encode(
        x=alt.X("day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share:Q", title="Share of that day's actionable reviews", axis=alt.Axis(format="%"),
                scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("day:N", title="Day"), alt.Tooltip("mentions:Q", title="Mentions"),
                 alt.Tooltip("actionable:Q", title="Actionable reviews that day"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")])
    chart = (base.mark_line(color=ACCENT, strokeWidth=2)
             + base.mark_point(color=ACCENT, filled=True, size=70)).properties(
        height=260, title=alt.Title(f"{choice}: {int(row['total'])} mentions over 9 days", anchor="start",
                                    color="#0b0b0b", fontSize=14))
    st.altair_chart(style(chart), width="stretch")
    st.caption("Shown as a share, not a count: on a day with more reviews, every issue gets more mentions.")

    merged_in = mapping[(mapping["canonical"] == choice) & (mapping["phrase"] != choice)]
    with st.expander(f"Phrases merged into “{choice}” ({len(merged_in)})"):
        st.dataframe(merged_in[["phrase", "mentions", "merged_similarity"]].sort_values("mentions", ascending=False),
                     hide_index=True, width="stretch",
                     column_config={"phrase": "Phrase", "mentions": "Mentions",
                                    "merged_similarity": st.column_config.NumberColumn("Similarity", format="%.2f")})

    st.subheader("Flagged rises")
    if len(spikes):
        st.dataframe(spikes.assign(day=spikes["day"].map(lambda d: day_labels.get(d, d))), hide_index=True,
                     width="stretch",
                     column_config={"day": "Day", "topic": "Topic", "observed": "Mentions",
                                    "expected": st.column_config.NumberColumn("Expected", format="%.1f"),
                                    "z": st.column_config.NumberColumn("z", format="%.1f")})
    st.caption("A day is flagged when an issue is well above what its share of the previous 5 days predicts "
               "for that day's volume. With 9 days and dozens of topics, a few flags can appear by chance; "
               "treat these as a demonstration of the method.")

# ---------------------------------------------------------------- Merge log
with tab_merges:
    st.markdown(f"Extraction produced {len(mapping):,} distinct issue phrases. Each was merged into an existing "
                "topic or started a new one, most frequent first. This is the record of every decision.")
    counts = mapping["decision"].value_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric("Merged after a yes from the LLM", f"{counts.get('merged', 0):,}")
    m2.metric("New topic: nothing similar enough", f"{counts.get('new: below floor', 0) + counts.get('new: first topic', 0):,}")
    m3.metric("New topic: LLM said not the same", f"{counts.get('new: rejected by LLM', 0):,}")

    g1, g2 = st.columns([2, 1])
    q = g1.text_input("Search phrases or topics", placeholder="e.g. cold food")
    kinds = g2.multiselect("Decision", sorted(mapping["decision"].unique()), default=["merged"])
    log = mapping if not kinds else mapping[mapping["decision"].isin(kinds)]
    if q:
        log = log[log["phrase"].str.contains(q, case=False, regex=False)
                  | log["canonical"].str.contains(q, case=False, regex=False)]
    st.dataframe(log[["phrase", "mentions", "canonical", "decision", "merged_similarity", "top_similarity"]],
                 hide_index=True, width="stretch", height=420,
                 column_config={"phrase": st.column_config.TextColumn("Phrase", width="medium"),
                                "mentions": "Mentions",
                                "canonical": st.column_config.TextColumn("Topic it ended up in", width="medium"),
                                "decision": "Decision",
                                "merged_similarity": st.column_config.NumberColumn("Similarity to topic", format="%.2f"),
                                "top_similarity": st.column_config.NumberColumn("Closest topic's similarity",
                                                                                format="%.2f")})
    st.caption("Phrases below a cosine similarity of 0.60 to every existing topic start a new topic without an "
               "LLM call. Above it, the LLM answers a yes/no question for up to 3 candidates.")

# ---------------------------------------------------------------- Evaluation
with tab_eval:
    st.markdown("Every figure here comes from hand labels: 200 randomly sampled reviews for triage, and 100 "
                "randomly sampled merge decisions per version. Ranges are 95% intervals. One annotator, with a "
                "strict rule: borderline cases count as wrong.")

    st.subheader("Triage: is this review actionable?")
    tri = triage.copy()
    fmt = lambda p, lo, hi: f"{p:.0%} ({lo:.0%}–{hi:.0%})"
    st.dataframe(pd.DataFrame({
        "Version": tri["system"],
        "Precision": [fmt(r.precision, r.precision_lo, r.precision_hi) for r in tri.itertuples()],
        "Recall": [fmt(r.recall, r.recall_lo, r.recall_hi) for r in tri.itertuples()],
        "Accuracy": tri["accuracy"].map("{:.0%}".format)}), hide_index=True, width="stretch")
    v1r = tri.set_index("system").loc["v1"]
    st.caption(f"v1's accuracy ({v1r['accuracy']:.0%}) is below the {(v1r['tn'] + v1r['fp']) / v1r['labelled_reviews']:.0%} "
               "you'd get by calling every review not actionable.")

    a, b = st.columns(2)
    pos = position.assign(position=position["position"].map(lambda p: f"Position {p}"))
    pos_chart = alt.Chart(pos).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("position:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("version:N", sort=["v1", "v2"], scale=alt.Scale(paddingInner=0.06)),
        y=alt.Y("actionable_rate:Q", title="Labelled actionable", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("version:N", sort=["v1", "v2"], scale=alt.Scale(domain=["v1", "v2"], range=[MUTED, ACCENT]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("version:N", title="Version"), alt.Tooltip("position:N", title="Batch position"),
                 alt.Tooltip("actionable_rate:Q", title="Actionable", format=".1%"),
                 alt.Tooltip("reviews:Q", title="Reviews", format=",")]
    ).properties(height=280, title=alt.Title("v1's label depended on where a review sat in its batch",
                                             anchor="start", color="#0b0b0b", fontSize=14))
    a.altair_chart(style(pos_chart), width="stretch")
    a.caption("Reviews are labelled in batches of 4. Reviews without a rescue keyword, so this is the model alone.")

    st_df = stars.assign(stars=stars["stars"].map(lambda s: f"{s}★"))
    star_chart = alt.Chart(st_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("stars:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("version:N", sort=["v1", "v2"], scale=alt.Scale(paddingInner=0.06)),
        y=alt.Y("actionable_rate:Q", title="Labelled actionable", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("version:N", sort=["v1", "v2"], scale=alt.Scale(domain=["v1", "v2"], range=[MUTED, ACCENT]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("version:N", title="Version"), alt.Tooltip("stars:N", title="Rating"),
                 alt.Tooltip("actionable_rate:Q", title="Actionable", format=".1%"),
                 alt.Tooltip("reviews:Q", title="Reviews", format=",")]
    ).properties(height=280, title=alt.Title("v2's labels follow the review: complaints are low-star",
                                             anchor="start", color="#0b0b0b", fontSize=14))
    b.altair_chart(style(star_chart), width="stretch")
    b.caption("Same reviews. v1 labelled about a third actionable at every rating.")

    st.subheader("Merging: is this phrase really the same issue as its topic?")
    rows, bands = [], []
    for name, s in samples.items():
        ok = s["correct"].astype(str).str.strip().str.upper().str[:1].map({"Y": True, "N": False})
        s = s.assign(ok=ok).dropna(subset=["ok"])
        k, n = int(s["ok"].sum()), len(s)
        p, lo, hi = wilson(k, n)
        weighted = (s["ok"].astype(bool) * s["mentions"]).sum() / s["mentions"].sum()
        tc = topic_counts.set_index("version").loc[name]
        rows.append({"Version": name, "Merge precision": f"{p:.0%} ({lo:.0%}–{hi:.0%})",
                     "Weighted by mentions": f"{weighted:.0%}", "Topics": f"{int(tc['topics']):,}",
                     "Topics seen once": f"{tc['seen_once']:.0%}"})
        if name.startswith("v3, floor 0.60"):
            s = s.assign(band=pd.cut(pd.to_numeric(s["similarity"]), [0.6, 0.7, 0.8, 1.01], right=False,
                                     labels=["0.6–0.7", "0.7–0.8", "0.8 and above"]))
            for band, g in s.groupby("band", observed=True):
                bp, blo, bhi = wilson(int(g["ok"].sum()), len(g))
                bands.append({"band": str(band), "precision": bp, "lo": blo, "hi": bhi, "merges": len(g)})
    v1 = topic_counts.set_index("version").loc["v1"]
    rows.insert(0, {"Version": "v1", "Merge precision": "not measured", "Weighted by mentions": "–",
                    "Topics": f"{int(v1['topics']):,}", "Topics seen once": f"{v1['seen_once']:.0%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("v2 merged 98% of phrases into 46 topics. v3 adds a similarity floor and yes/no checks. "
               "Raising the floor to 0.70 was tested and gave no gain on a fresh sample, so 0.60 stays.")

    band_df = pd.DataFrame(bands)
    band_base = alt.Chart(band_df).encode(
        y=alt.Y("band:N", sort=None, title="Similarity of the merge"),
        tooltip=[alt.Tooltip("band:N", title="Similarity"), alt.Tooltip("precision:Q", title="Right", format=".0%"),
                 alt.Tooltip("lo:Q", title="95% low", format=".0%"), alt.Tooltip("hi:Q", title="95% high", format=".0%"),
                 alt.Tooltip("merges:Q", title="Labelled merges")])
    band_chart = (band_base.mark_rule(color=MUTED, strokeWidth=2).encode(
                      x=alt.X("lo:Q", title="Share of merges that were right", axis=alt.Axis(format="%"),
                              scale=alt.Scale(domain=[0, 1])), x2="hi:Q")
                  + band_base.mark_point(color=ACCENT, filled=True, size=110).encode(x="precision:Q")
                  + band_base.mark_text(align="left", dx=10, dy=-10, color=INK_2, fontSize=12).encode(
                      x="precision:Q", text=alt.Text("precision:Q", format=".0%"))
                  ).properties(height=200, title=alt.Title(
                      "Low-similarity merges were mostly wrong (v3, floor 0.60)", anchor="start",
                      color="#0b0b0b", fontSize=14))
    st.altair_chart(style(band_chart), width="stretch")
    st.caption("Dot: share right in the labelled sample. Line: 95% interval. This is what prompted the 0.70 test.")

st.divider()
st.caption(f"Data: Swiggy reviews from Google Play, India, English-language setting, 1–9 Oct 2025. "
           f"Only aggregated outputs are published; review text and reviewer names are not. [Repository]({REPO_URL})")
