"""
Paleontology Research Assistant -- Streamlit UI
V0.2: Multi-turn conversation. Conversation history lives in session_state,
      is passed to the agent on each call, and resets on page refresh.
"""

import streamlit as st
from agent import run_agent

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Paleontology Research Assistant",
    page_icon="🦕",
    layout="centered",
)

st.title("🦕 Paleontology Research Assistant")
st.caption("Ask a research question. Follow-up questions build on prior answers.")

# -- Session state -------------------------------------------------------------
# messages: [{"role": "user"/"assistant", "content": str}, ...]
# User turns store the raw query; assistant turns store the response text.
# This list is passed directly to run_agent as conversation history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Input ---------------------------------------------------------------------
query = st.text_input(
    label="Research question or topic",
    placeholder="e.g. What are the latest findings on Spinosaurus locomotion?",
)

col_submit, col_clear = st.columns([5, 1])
with col_submit:
    submit = st.button("Research", type="primary", use_container_width=True)
with col_clear:
    clear = st.button("Clear", use_container_width=True)

# -- Handle clear --------------------------------------------------------------
if clear:
    st.session_state.messages = []
    st.rerun()

# -- Handle submit -------------------------------------------------------------
# Process before rendering so the new response appears in this same re-run.
if submit and query.strip():
    with st.spinner("Researching..."):
        result = run_agent(query.strip(), st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": query.strip()})
    st.session_state.messages.append({"role": "assistant", "content": result})

elif submit and not query.strip():
    st.warning("Enter a research question to continue.")

# -- Render conversation -------------------------------------------------------
if st.session_state.messages:
    st.divider()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
