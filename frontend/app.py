"""
Streamlit chat frontend for the Study Notes RAG Assistant.

Run with: streamlit run app.py
Requires the backend to be running (see ../backend).
"""
import streamlit as st

from api_client import ask_question, ask_question_with_image, check_health

st.set_page_config(page_title="Study Notes RAG Assistant", page_icon="📚")
st.title("📚 Study Notes RAG Assistant")
st.caption("Ask a question about your study notes and get a grounded, cited answer.")

if "history" not in st.session_state:
    st.session_state.history = []

if not check_health():
    st.warning("⚠️ Can't reach the backend. Make sure it's running (see README).")

with st.sidebar:
    st.header("Extended Track: Image input")
    uploaded_image = st.file_uploader("Upload a scanned notes page (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded page", use_column_width=True)

question = st.chat_input("Ask a question about your study notes...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                if uploaded_image is not None:
                    result = ask_question_with_image(
                        question, uploaded_image.getvalue(), uploaded_image.name
                    )
                    detections = result.get("detections") or []
                    if detections:
                        st.info("Detected in image: " + ", ".join(detections))
                else:
                    result = ask_question(question)

                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    st.markdown("**Sources:** " + ", ".join(sources))

                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as exc:  # noqa: BLE001
                error_msg = f"❌ Sorry, something went wrong reaching the backend: {exc}"
                st.error(error_msg)
                st.session_state.history.append({"role": "assistant", "content": error_msg})
