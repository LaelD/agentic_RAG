from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm

# def _format_sources(context_docs: List[Any]) -> List[str]:
#     return [
#         str((meta.get("source") or "Unknown"))
#         for doc in (context_docs or [])
#         if (meta := (getattr(doc, "metadata", None) or {})) is not None
#     ]


st.set_page_config(page_title="Smart Agriculture Helper", layout="centered")
st.title("Smart Agriculture Helper")

# with st.sidebar:
#     st.subheader("Session")
#     if st.button("Clear chat", use_container_width=True):
#         st.session_state.pop("messages", None)
#         st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about smart agriculture. I’ll retrieve relevant context."
            
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # if msg.get("sources"):
        #     with st.expander("Sources"):
        #         for s in msg["sources"]:
        #             st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about smart agriculture")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                # sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            # if sources:
            #     with st.expander("Sources"):
            #         for s in sources:
            #             st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
