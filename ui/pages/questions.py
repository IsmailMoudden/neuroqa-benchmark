import copy
import streamlit as st
from labels import DIFFICULTY_BADGE, DIFFICULTY_COLOR, TYPE_LABELS
from sample_questions import SAMPLE_QUESTIONS
from ui.storage import list_source_docs, save_questions, load_questions


def render():
    st.title("Questions")

    available_docs = list_source_docs()
    active_ids     = {q["id"] for q in st.session_state.questions}

    tab_active, tab_library = st.tabs(
        [f"Active questions ({len(st.session_state.questions)})", "Sample library"]
    )

    with tab_active:
        col_add, col_reload, col_save, _ = st.columns([1, 1, 1, 4])
        with col_add:
            if st.button("Add blank question", icon=":material/add:", width="stretch"):
                new_id = f"Q{len(st.session_state.questions)+1:02d}"
                st.session_state.questions.append({
                    "id": new_id, "type": "factual", "difficulty": "medium",
                    "source_doc": available_docs[0] if available_docs else "",
                    "question": "", "expected": "", "keywords": [],
                })
        with col_reload:
            if st.button("Reload from file", icon=":material/refresh:", width="stretch"):
                st.session_state.questions = load_questions()
                st.success("Questions reloaded")
        with col_save:
            if st.button("Save to file", icon=":material/save:", width="stretch", type="primary"):
                save_questions(st.session_state.questions)
                st.success("Questions saved")

        if not available_docs:
            st.warning("No source documents found. Go to the Documents page to upload .docx files first.", icon=":material/warning:")

        st.divider()

        if not st.session_state.questions:
            st.info("No active questions yet. Add your own above, or pick from the **Sample library** tab.", icon=":material/info:")

        to_delete = None
        for i, q in enumerate(st.session_state.questions):
            diff  = q.get("difficulty", "medium")
            label = f"**{q['id']}** — {q['question'][:80] or '(empty question)'}  `{DIFFICULTY_BADGE[diff]}`"
            with st.expander(label, expanded=False):
                c1, c2, c3 = st.columns([2, 2, 2])
                with c1:
                    q["id"] = st.text_input("ID", value=q["id"], key=f"id_{i}")
                with c2:
                    q["type"] = st.selectbox(
                        "Question type",
                        options=list(TYPE_LABELS.keys()),
                        index=list(TYPE_LABELS.keys()).index(q.get("type", "factual")),
                        format_func=lambda x: TYPE_LABELS[x],
                        key=f"type_{i}",
                        help="Factual: simple fact lookup. Definition: explain a term. "
                             "Procedural: step-by-step process. Multi-hop: requires combining passages. "
                             "Causal: why/how something happens.",
                    )
                with c3:
                    q["difficulty"] = st.selectbox(
                        "Difficulty",
                        options=["easy", "medium", "hard"],
                        index=["easy", "medium", "hard"].index(q.get("difficulty", "medium")),
                        format_func=lambda x: DIFFICULTY_BADGE[x],
                        key=f"diff_{i}",
                    )

                current_doc = q.get("source_doc", "")
                doc_options = available_docs if available_docs else ([current_doc] if current_doc else [])
                if doc_options:
                    idx = doc_options.index(current_doc) if current_doc in doc_options else 0
                    q["source_doc"] = st.selectbox("Source document", options=doc_options, index=idx, key=f"src_{i}")
                else:
                    st.caption("No documents available — upload files in the Documents page.")
                    q["source_doc"] = current_doc

                q["question"] = st.text_area("Question", value=q["question"], height=80, key=f"q_{i}")
                q["expected"] = st.text_area(
                    "Expected answer", value=q.get("expected", ""), height=100, key=f"exp_{i}",
                    placeholder="(optional) — leave blank if you have no reference answer. F1 Score will not be computed.",
                )
                new_kw = st.text_input(
                    "Keywords", value=", ".join(q.get("keywords", [])), key=f"kw_{i}",
                    placeholder="comma-separated, e.g. CLS, settlement, payment",
                    help="Words that should appear in a relevant passage. Used to evaluate retrieval.",
                )
                q["keywords"] = [k.strip() for k in new_kw.split(",") if k.strip()]

                if st.button("Remove from benchmark", icon=":material/delete:", key=f"del_{i}"):
                    to_delete = i

        if to_delete is not None:
            st.session_state.questions.pop(to_delete)
            save_questions(st.session_state.questions)
            st.rerun()

        if st.session_state.questions:
            st.divider()
            if st.button("Save all changes", icon=":material/save:", type="primary", width="stretch"):
                save_questions(st.session_state.questions)
                st.success("All changes saved")

    with tab_library:
        st.caption("Example questions based on CLS settlement documentation. Enable the ones you want to include.")

        fc1, fc2, _ = st.columns([2, 2, 4])
        with fc1:
            filter_type = st.selectbox(
                "Filter by type",
                options=["All"] + list(TYPE_LABELS.keys()),
                format_func=lambda x: "All types" if x == "All" else TYPE_LABELS[x],
                key="lib_filter_type",
            )
        with fc2:
            filter_diff = st.selectbox(
                "Filter by difficulty",
                options=["All", "easy", "medium", "hard"],
                format_func=lambda x: "All difficulties" if x == "All" else DIFFICULTY_BADGE[x],
                key="lib_filter_diff",
            )

        filtered = [
            q for q in SAMPLE_QUESTIONS
            if (filter_type == "All" or q["type"] == filter_type)
            and (filter_diff == "All" or q["difficulty"] == filter_diff)
        ]

        st.divider()
        added_any = False
        for sq in filtered:
            is_active = sq["id"] in active_ids
            diff      = sq.get("difficulty", "medium")
            col_toggle, col_info = st.columns([1, 8])
            with col_toggle:
                enabled = st.toggle(label=sq["id"], value=is_active, key=f"sample_{sq['id']}")
            with col_info:
                st.markdown(
                    f"**{sq['id']}** &nbsp; "
                    f":{DIFFICULTY_COLOR[diff]}[{DIFFICULTY_BADGE[diff]}] &nbsp; "
                    f"`{TYPE_LABELS.get(sq['type'], sq['type'])}`  \n"
                    f"{sq['question']}"
                )

            if enabled and not is_active:
                st.session_state.questions.append(copy.deepcopy(sq))
                active_ids.add(sq["id"])
                added_any = True
            elif not enabled and is_active:
                st.session_state.questions = [q for q in st.session_state.questions if q["id"] != sq["id"]]
                active_ids.discard(sq["id"])
                added_any = True

        if added_any:
            save_questions(st.session_state.questions)
            st.rerun()
