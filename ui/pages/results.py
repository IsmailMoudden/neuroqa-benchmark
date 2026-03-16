import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from labels import METRIC_INFO
from ui.db import db_list_runs, db_load_run, db_save_run, db_delete_run
from ui.storage import load_results, RESULTS_DIR


def render():
    st.title("Benchmark Results")

    past_runs = db_list_runs()
    if not past_runs:
        raw_json, summary_json = load_results()
        if summary_json:
            db_save_run(raw_json, summary_json, label="(imported)")
            past_runs = db_list_runs()

    if not past_runs:
        st.warning("No results yet. Run the benchmark first.")
        st.stop()

    run_options = {r["id"]: f"#{r['id']} · {r['run_at']} · {r['label']}" for r in past_runs}
    if len(past_runs) > 1:
        col_sel, col_del = st.columns([5, 1])
        with col_sel:
            selected_run_id = st.selectbox(
                "Select a past run",
                options=list(run_options.keys()),
                format_func=lambda x: run_options[x],
                key="selected_run_id",
            )
        with col_del:
            st.write("")
            if st.button("Delete run", icon=":material/delete:", key="del_run"):
                db_delete_run(selected_run_id)
                st.success("Run deleted.")
                st.rerun()
    else:
        selected_run_id = past_runs[0]["id"]
        st.caption(f"Last run: {past_runs[0]['run_at']} · {past_runs[0]['label']}")

    raw, summary = db_load_run(selected_run_id)
    if not summary:
        st.warning("No results found for this run.")
        st.stop()

    df_summary   = pd.DataFrame(summary)
    df_raw       = pd.DataFrame(raw)
    n_strategies = len(df_summary)
    n_questions  = df_raw["q_id"].nunique() if not df_raw.empty else 0

    if n_strategies > 1:
        best = df_summary.loc[df_summary["composite_score"].idxmax()]
        st.success(
            f"Best strategy: **{best['strategy_id']} — {best['strategy_name']}**  "
            f"(Composite score: {best['composite_score']:.3f})",
            icon=":material/trophy:",
        )

    with st.expander("What do the metrics mean?", icon=":material/help:"):
        cols = st.columns(3)
        for idx, (key, info) in enumerate(METRIC_INFO.items()):
            with cols[idx % 3]:
                st.markdown(f"**{info['label']}**")
                st.caption(info["help"])

    st.markdown("### Strategy Summary")
    display_cols = {
        "strategy_id":    "ID",
        "strategy_name":  "Strategy",
        "n_chunks":       "# Chunks",
        "avg_tokens":     "Avg Tokens",
        "recall_at_5":    METRIC_INFO["recall_at_5"]["label"],
        "mrr":            METRIC_INFO["mrr"]["label"],
        "f1":             METRIC_INFO["f1"]["label"],
        "faithfulness":   METRIC_INFO["faithfulness"]["label"],
        "relevance":      METRIC_INFO["relevance"]["label"],
        "cost_per_query": "Cost / Query ($)",
        "composite_score": "Composite Score",
    }
    df_disp  = df_summary.rename(columns=display_cols)[list(display_cols.values())]
    num_cols = [
        METRIC_INFO["recall_at_5"]["label"], METRIC_INFO["mrr"]["label"],
        METRIC_INFO["f1"]["label"], METRIC_INFO["faithfulness"]["label"],
        METRIC_INFO["relevance"]["label"], "Composite Score",
    ]

    def highlight_best(s):
        if s.nunique() <= 1:
            return [""] * len(s)
        return ["background-color: #d4edda; font-weight: bold" if v == s.max() else "" for v in s]

    styled = df_disp.style.apply(highlight_best, subset=num_cols).format(
        {c: "{:.3f}" for c in num_cols} | {"Cost / Query ($)": "{:.5f}", "Avg Tokens": "{:.1f}"}
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    if n_strategies > 1:
        st.markdown("### Performance Radar")
        st.caption("Each axis is one metric (0 = worst, 1 = best). A larger filled area means a stronger strategy overall.")
        metrics_radar = ["recall_at_5", "mrr", "f1", "faithfulness", "relevance"]
        radar_labels  = [METRIC_INFO[m]["label"] for m in metrics_radar]
        fig_radar = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, row in df_summary.iterrows():
            vals = [row[m] for m in metrics_radar]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=f"{row['strategy_id']} — {row['strategy_name']}",
                line_color=colors[i % len(colors)],
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=420, margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, width="stretch")

        st.markdown("### Metric Comparison")
        metric_choice = st.selectbox(
            "Select a metric to compare",
            options=list(METRIC_INFO.keys()),
            format_func=lambda x: METRIC_INFO[x]["label"],
        )
        st.caption(METRIC_INFO[metric_choice]["help"])
        fig_bar = px.bar(
            df_summary, x="strategy_id", y=metric_choice, color="strategy_id",
            text=df_summary[metric_choice].round(3),
            labels={"strategy_id": "Strategy", metric_choice: METRIC_INFO[metric_choice]["label"]},
            color_discrete_sequence=px.colors.qualitative.Set2, height=350,
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, width="stretch")

    if not df_raw.empty:
        st.markdown("### Question-by-Question Results")
        st.caption("Select a question to see the generated answer, scores, and which passages were retrieved.")

        selected_qid = st.selectbox("Question", options=sorted(df_raw["q_id"].unique()), key="res_qid")
        q_rows = df_raw[df_raw["q_id"] == selected_qid]

        q_text     = q_rows.iloc[0]["question"]
        q_expected = q_rows.iloc[0].get("expected", "")
        st.markdown(f"**Question:** {q_text}")
        if q_expected:
            st.markdown(f"**Expected answer:** {q_expected}")
        st.divider()

        for _, row in q_rows.iterrows():
            with st.expander(f"Strategy {row['strategy_id']} — {row['strategy_name']}", expanded=True):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(METRIC_INFO["recall_at_5"]["label"], f"{row['recall_at_5']:.2f}")
                m2.metric(METRIC_INFO["mrr"]["label"],          f"{row['mrr']:.2f}")
                m3.metric(METRIC_INFO["f1"]["label"],           f"{row['f1']:.2f}")
                m4.metric(METRIC_INFO["faithfulness"]["label"], f"{row['faithfulness']:.2f}")
                m5.metric(METRIC_INFO["relevance"]["label"],    f"{row['relevance']:.2f}")

                st.markdown("**Generated answer:**")
                st.info(row["answer"] or "_(no answer)_")

                chunks = row.get("retrieved_chunks", [])
                if chunks:
                    st.markdown(f"**Retrieved passages** ({len(chunks)} used to generate the answer):")
                    for idx_c, c in enumerate(chunks, 1):
                        chunk_id   = c.get("id", "")   if isinstance(c, dict) else c
                        chunk_text = c.get("text", "") if isinstance(c, dict) else ""
                        with st.expander(f"Passage {idx_c} — {chunk_id}"):
                            st.write(chunk_text or "_(text not available — re-run benchmark to populate)_")

        if n_strategies > 1 and n_questions > 1:
            st.markdown("### F1 Heatmap — all questions × all strategies")
            pivot = df_raw.pivot_table(index="q_id", columns="strategy_id", values="f1", aggfunc="mean")
            fig_heat = px.imshow(
                pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                labels={"color": "F1 Score"}, aspect="auto",
                height=max(400, len(pivot) * 28),
            )
            st.plotly_chart(fig_heat, width="stretch")

    st.divider()
    xlsx_path = RESULTS_DIR / "benchmark_results.xlsx"
    if xlsx_path.exists():
        with open(xlsx_path, "rb") as f:
            st.download_button(
                label="Download Excel Report",
                icon=":material/download:",
                data=f,
                file_name="benchmark_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
