#!/usr/bin/env python
# coding: utf-8

# In[11]:


pip install streamlit faiss-cpu numpy langchain sentence-transformers transformers pdfminer.six unstructured unstructured-inference unstructured-pytesseract


# In[12]:


import os
import streamlit as st
import pickle
import time
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
FILE_PATH = "faiss_store.pkl"

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# User Input: Dynamic URL Entry
urls = st.sidebar.text_area("Enter article URLs (one per line)").split("\n")
process_url_clicked = st.sidebar.button("Process URLs")

# Load Hugging Face LLM (Replaces OpenAI)
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load Sentence Transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Process URLs and store FAISS index
def process_urls(urls, embedding_model):
    if not any(urls):
        st.error("Please enter at least one valid URL.")
        return

    try:
        st.text("Fetching articles... ⏳")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        st.text("Splitting text into chunks... ⏳")
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        docs = text_splitter.split_documents(data)

        st.text("Generating embeddings... ⏳")
        embeddings = np.array([embedding_model.encode(doc.page_content) for doc in docs])

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        with open(FILE_PATH, "wb") as f:
            pickle.dump((index, docs), f)

        st.text("Processing complete! ✅")
        time.sleep(2)

    except ModuleNotFoundError:
        st.error("Missing dependencies. Install using:")
        st.code("pip install unstructured pdfminer.six unstructured-inference unstructured-pytesseract")
    except Exception as e:
        st.error(f"Error processing URLs: {e}")

# Retrieve answers from FAISS
def query_llm(question, generator, embedding_model):
    if not os.path.exists(FILE_PATH):
        return "No data available. Please process URLs first."

    try:
        with open(FILE_PATH, "rb") as f:
            index, docs = pickle.load(f)

        question_embedding = embedding_model.encode(question).reshape(1, -1)
        D, I = index.search(question_embedding, k=3)

        relevant_texts = " ".join([docs[i].page_content for i in I[0]])

        prompt = f"Context: {relevant_texts}\n\nQuestion: {question} Answer:"
        response = generator(prompt, max_length=200, do_sample=True)

        return response[0]['generated_text']

    except FileNotFoundError:
        return "Error: FAISS index not found. Please process URLs first."
    except Exception as e:
        return f"Error: {str(e)}"

# Load models
generator = load_llm()
embedding_model = load_embedding_model()

# Process URLs
if process_url_clicked:
    process_urls(urls, embedding_model)

# Question Input
query = st.text_input("Ask a question:")
if query:
    answer = query_llm(query, generator, embedding_model)
    st.subheader("Answer:")
    st.write(answer) 


# In[ ]:





# In[ ]:





# In[ ]:




