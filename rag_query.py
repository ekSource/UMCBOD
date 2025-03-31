from openai import OpenAI
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Initialize OpenAI client with Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load vector store
def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=st.secrets["OPENAI_API_KEY"])
    return LangChainFAISS.load_local(
        folder_path="faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

vector_store = load_vector_store()

def query_rag(query, top_k=5):
    results = vector_store.similarity_search(query, k=top_k)
    context = ""
    for i, doc in enumerate(results):
        meta = doc.metadata
        context += f"Source: {meta['source']}\n"
        if meta["part"]:
            context += f"Part: {meta['part']}\n"
        context += f"Heading: {meta['heading']}\n"
        if meta["title"]:
            context += f"Title: {meta['title']}\n"
        if meta["sub_title"]:
            context += f"Sub-title: {meta['sub_title']}\n"
        if meta["paragraph_number"]:
            context += f"Paragraph {meta['paragraph_number']}"
        if meta["paragraph_title"]:
            context += f": {meta['paragraph_title']}\n"
        elif meta["paragraph_number"]:
            context += "\n"
        if meta["sub_para_title"]:
            context += f"Sub-paragraph: {meta['sub_para_title']}\n"
        context += f"Text: {doc.page_content}\n\n"
    
    prompt = f"""
    User Query: {query}
    Retrieved Context:
    {context}
    
    Provide a clear, easy-to-understand explanation based on the context. Include direct quotes with citations (e.g., 'Part I, Paragraph 1: Article I. Declaration of Union') where relevant. Structure the response by grouping information from the same 'part' together.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content
