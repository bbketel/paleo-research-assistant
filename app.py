"""
Paleontology Research Assistant -- Streamlit UI
V0.3: Four UI improvements:
  1. st.form so Enter key submits the query.
  2. st.status with sequential progress steps replacing the generic spinner.
  3. Sidebar with tool description and source tier legend.
  4. Colored HTML badge rendering for [T1]-[T4] tier labels in Sources.
"""

import re
import streamlit as st
from agent import run_agent

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Paleontology Research Assistant",
    page_icon="🦕",
    layout="centered",
)

# -- Tier badge rendering ------------------------------------------------------
# Inline HTML spans used in _render_message() for the Sources section.
# T1/T2 (green/blue) are visually distinct from T3/T4 (orange/gray).
_TIER_BADGE = {
    "1": '<span style="background:#2e7d32;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.78em;font-weight:700;letter-spacing:0.02em">T1</span>',
    "2": '<span style="background:#1565c0;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.78em;font-weight:700;letter-spacing:0.02em">T2</span>',
    "3": '<span style="background:#e65100;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.78em;font-weight:700;letter-spacing:0.02em">T3</span>',
    "4": '<span style="background:#616161;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.78em;font-weight:700;letter-spacing:0.02em">T4</span>',
}

_TIER_LABEL_RE = re.compile(r"\[T([1-4])\]")


def _replace_badge(m: re.Match) -> str:
    return _TIER_BADGE[m.group(1)]


def _render_message(content: str) -> None:
    """
    Render a chat message with tier badge HTML substitution.

    If the content contains any [T1]-[T4] labels, replaces them with colored
    HTML badge spans and renders with unsafe_allow_html=True. Falls back to
    plain st.markdown for messages without tier labels (user turns, errors).
    """
    if _TIER_LABEL_RE.search(content):
        st.markdown(_TIER_LABEL_RE.sub(_replace_badge, content), unsafe_allow_html=True)
    else:
        st.markdown(content)


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Paleontology Research Assistant** is a live research tool that answers "
        "questions about fossil life, evolutionary biology, and palaeontology by "
        "combining a real-time **web search** with academic abstracts from "
        "**Semantic Scholar**. Findings are synthesised into a structured summary "
        "with cited sources.\n\n"
        "Follow-up questions build on prior answers within the same session. "
        "The conversation resets when you refresh the page."
    )

    st.divider()
    st.markdown("**Source tier labels**")
    st.markdown(
        '<span style="background:#2e7d32;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.82em;font-weight:700">T1</span>'
        " &nbsp;Peer-reviewed journal",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="background:#1565c0;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.82em;font-weight:700">T2</span>'
        " &nbsp;Institutional / museum",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="background:#e65100;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.82em;font-weight:700">T3</span>'
        " &nbsp;Science journalism",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="background:#616161;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.82em;font-weight:700">T4</span>'
        " &nbsp;General news or blog",
        unsafe_allow_html=True,
    )

# -- Title ---------------------------------------------------------------------
st.title("🦕 Paleontology Research Assistant")
st.caption("Ask a research question. Follow-up questions build on prior answers.")

# -- Session state -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Input form ----------------------------------------------------------------
# st.form enables Enter-key submission. Both actions are form_submit_buttons so
# they coexist in the same layout. Enter fires the first button (Research).
with st.form("query_form", clear_on_submit=True):
    query = st.text_input(
        label="Research question or topic",
        placeholder="e.g. What are the latest findings on Spinosaurus locomotion?",
    )
    col_submit, col_clear = st.columns([5, 1])
    with col_submit:
        submit = st.form_submit_button(
            "Research", type="primary", use_container_width=True
        )
    with col_clear:
        clear = st.form_submit_button("Clear", use_container_width=True)

# -- Handle clear --------------------------------------------------------------
if clear:
    st.session_state.messages = []
    st.rerun()

# -- Handle submit -------------------------------------------------------------
if submit and query.strip():
    with st.status("Researching...", expanded=True) as status:
        st.write("📚 Searching academic databases (Semantic Scholar)...")
        st.write("🌐 Running live web search...")
        st.write("🔬 Synthesizing findings and scoring sources...")
        result = run_agent(query.strip(), st.session_state.messages)
        status.update(label="Research complete", state="complete", expanded=False)
    st.session_state.messages.append({"role": "user", "content": query.strip()})
    st.session_state.messages.append({"role": "assistant", "content": result})

elif submit and not query.strip():
    st.warning("Enter a research question to continue.")

# -- Render conversation -------------------------------------------------------
if st.session_state.messages:
    st.divider()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _render_message(msg["content"])
