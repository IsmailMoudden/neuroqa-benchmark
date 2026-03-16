import streamlit as st
from ui.storage import SOURCES_DIR, list_source_docs


def render():
    st.title("Source Documents")
    st.caption("Upload the `.docx` files you want to evaluate. They will be stored in the `sources/` folder.")

    uploaded = st.file_uploader("Upload one or more Word documents", type=["docx"], accept_multiple_files=True)
    if uploaded:
        saved, skipped = [], []
        for f in uploaded:
            dest = SOURCES_DIR / f.name
            if dest.exists():
                skipped.append(f.name)
            else:
                dest.write_bytes(f.read())
                saved.append(f.name)
        if saved:
            st.success(f"Uploaded: {', '.join(saved)}")
        if skipped:
            st.info(f"Already exists: {', '.join(skipped)}")

    st.divider()
    docs = list_source_docs()
    st.markdown(f"### Documents in sources/  `{len(docs)} file{'s' if len(docs) != 1 else ''}`")

    if not docs:
        st.warning("No documents yet. Upload at least one .docx file before running the benchmark.")
    else:
        to_delete = None
        for doc in docs:
            col_name, col_size, col_del = st.columns([5, 2, 1])
            with col_name:
                st.markdown(f":material/description: &nbsp; {doc}")
            with col_size:
                st.caption(f"{(SOURCES_DIR / doc).stat().st_size / 1024:.1f} KB")
            with col_del:
                if st.button("Remove", key=f"del_doc_{doc}", icon=":material/delete:"):
                    to_delete = doc
        if to_delete:
            (SOURCES_DIR / to_delete).unlink()
            st.success(f"Removed {to_delete}")
            st.rerun()
