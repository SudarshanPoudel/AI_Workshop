import json
import streamlit as st
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional
from pydantic import BaseModel
from uuid import uuid4
import os
from dotenv import load_dotenv

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       
    chunk_overlap=200,   
    separators=["\n---\n", "\n\n", "\n", ".", " "]  
)


load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)
class Response(BaseModel):
    answer: str
    filename: Optional[str] = None
    page_no: Optional[int] = None

structured_model = model.with_structured_output(schema=Response)

def load_pdf(path, filename):
    loader = PyMuPDFLoader(file_path=path)
    documents = loader.load()
    
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["file_path"] = filename

    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)




prompt_template = """
You are a helpful and intelligent study assistant. You are given a list of JSON objects as context, each representing extracted text from a PDF.

Each object contains:
- 'text': the actual content
- 'filename': the PDF file name
- 'page_no': the page number

Your task is to answer the student's question based **primarily** on the 'text' fields in the context. You may reason and infer the answer if the exact wording is not available, as long as your answer is clearly supported by the content.

# Context:
{context}

# Question:
{question}

# Instructions:
- Use only the 'text' field from the context entries for answering the question, but include the most relevant 'filename' and 'page_no' you used.
- Even if the original text is technical or unclear, rewrite the answer in a simple, **student-friendly** way that is easy to understand.
- You can use **Markdown formatting** (headings, bullet points, code blocks, tables, etc.) to make the answer more readable and structured.
- You **may infer** or **summarize** answers from the content to help students understand, even if the answer is not a perfect match.
- Avoid saying you don’t know unless the question is entirely unrelated to the context.
- Return a JSON object with:
  - "answer": your helpful, clear, Markdown-formatted answer
  - "filename": the filename of the most relevant entry you used
  - "page_no": the corresponding page number
  
Respond ONLY with a valid JSON object. Do not include any explanation, formatting, or extra text."""




def get_answer(question):
    
    docs = vector_store.similarity_search(query=question, k=3)
    similar_chunks = vector_store.similarity_search(
        query=question,
        k=5,
    )

    context_list = []
    for chunk in similar_chunks:
        context_list.append({
            "text": chunk.page_content,
            "page_no": chunk.metadata["page"] + 1,
            "filename": chunk.metadata["file_path"]
        })

    context = json.dumps(context_list, indent=2)

    prompt = prompt_template.format(context=context, question=question)
    resp = structured_model.invoke(prompt)
    return resp