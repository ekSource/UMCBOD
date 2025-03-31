import streamlit as st
from rag_query import query_rag

st.title("United Methodist Church Discipline Assistant")
st.write("Ask questions about The Book of Discipline 2020-2024")

# User input
query = st.text_input("Enter your question:", "Can you tell me the history of United Methodist Church")

# Process query and display response
if st.button("Submit"):
    with st.spinner("Fetching response..."):
        response = query_rag(query)
        st.markdown(response)

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("Enter a question about The United Methodist Church's Bood Of Discipline. The assistant will retrieve relevant sections and explain them in simple language, citing parts, paragraphs, and titles as needed.")
