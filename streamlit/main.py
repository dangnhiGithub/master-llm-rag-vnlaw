import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CHATBOT RAG app", layout="wide")
st.title("CHATBOT RAG app")
st.write('Welcome to the CHATBOT RAG app!')

messages = st.container()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.update(messages=[]))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt:= st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.chat_message('user').write(prompt)
    messages.chat_message("agent retriever", avatar="🔍").write("Đây là câu trả lời từ agent retriever")
    messages.chat_message("agent reader", avatar="📚").write("Đây là câu trả lời từ agent reader")
    messages.chat_message("agent generator", avatar="🧠").write("Đây là câu trả lời từ agent generator")
    messages.chat_message("ai").write("Đây là câu trả lời cuối cùng")
    # Add a button to clear chat history
    
