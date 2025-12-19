import streamlit as st
import requests

st.title("Agentic Application")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

text_input = st.chat_input("Enter your message or upload file below")

uploaded_file = st.file_uploader("Upload file (image, pdf, audio)", type=['jpg', 'png', 'pdf', 'mp3', 'wav', 'm4a'])

if text_input:
    st.session_state.messages.append({"role": "user", "content": text_input})
    with st.chat_message("user"):
        st.markdown(text_input)

    # Process
    data = {'query': text_input}  # assuming query is the input
    files = {}
    if uploaded_file:
        files['file'] = uploaded_file

    try:
        response = requests.post("http://localhost:8000/process", data=data, files=files)
        result = response.json()
        extracted = result.get('extracted', '')
        response_text = result.get('response', '')
        content = f"Extracted: {extracted}\n\n{response_text}"
        st.session_state.messages.append({"role": "assistant", "content": content})
        with st.chat_message("assistant"):
            st.markdown(content)
    except Exception as e:
        st.error(f"Error: {e}")

    st.rerun()