# from langchain_core import documents
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
import wikipedia

    
st.set_page_config(page_title="Context-Aware Chatbot", layout="wide")
st.title("Context-Aware Chatbot")
st.markdown("Query Wikipedia topics or upload your own documents to chat with Llama-3.")

with st.sidebar:
    st.header("Setting")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")

    st.divider()

    st.header("Data Sources")
    wiki_topics=st.multiselect(
        "Wikipedia Topics to Index",
        ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Artificial Intelligence'],
        default='Machine Learning'
    )
    
    upload_files=st.file_uploader("Upload Text Files", type=['txt'], accept_multiple_files=True)

    process_btn=st.button("Build/Update knowledge base")


def build_vector_store(topics, files):
    documents=[]

    for topic in topics:
        page=wikipedia.page(topic, auto_suggest=False)
        documents.append(Document(
             page_content=page.content[:5000],
             metadata={"source": page.url, "title": page.title}
        ))

    for file in files:
        content=file.read().decode('utf-8')
        documents.append(Document(
            page_content=content,
            metadata={"source": file.name, "title": file.name}
        ))

    if not documents:
        return None
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks=splitter.split_documents(documents)

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_store=FAISS.from_documents(chunks, embeddings)
    return vector_store


if process_btn:
    if not groq_api_key:
        st.error("Enter your Groq API Key in the sidebar!")
    else:
        with st.spinner("Building vector store..."):
            st.session_state.vector_store=build_vector_store(wiki_topics,upload_files)
            st.success("Knowledge Base Ready!")

if "messages" not in st.session_state:
    st.session_state.messages=[]
if "memory" not in st.session_state:
    st.session_state.memory=ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the loaded topics..."):
    if "vector_store"  not in st.session_state:
        st.error("Please process documents in the sidebar first!")
    elif not groq_api_key:
        st.error("Missing API Key!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm=ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key,temperature=0.5)

            qa_chain=ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(),
                memory=st.session_state.memory,
                return_source_documents=True,
                verbose=False
            )

            response=qa_chain.invoke({"question": prompt})
            full_response=response["answer"]

            sources=set([doc.metadata.get("title") for doc in response["source_documents"]])
            if sources:
                full_response += f"\n\n**Sources:** {', ' .join(sources)}"
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
