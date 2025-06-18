from langgraph.graph import StateGraph
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import TypedDict, List
import os

load_dotenv()

class GraphState(TypedDict):
    query: str
    docs: List[Document]
    summary: str
    analysis: str
    recommendation: str

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectordb = Chroma(
    collection_name="IEEE",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

def retrieve_docs_node(state):
    query = state["query"]
    docs = vectordb.similarity_search(query, k=5)
    return {"docs": docs} 

def summarize_node(state):
    chunks = "\n\n".join([doc.page_content for doc in state["docs"]])
    prompt = f"Summarize the following for PQM application:\n\n{chunks}"
    summary = llm.invoke(prompt)
    return {"summary": summary}

def analyze_node(state):
    summary_or_docs = state.get("summary") or "\n\n".join([doc.page_content for doc in state["docs"]])
    prompt = f"Based on the following context, provide PQM insights:\n\n{summary_or_docs}"
    analysis = llm.invoke(prompt)
    return {"analysis": analysis}

def recommend_node(state):
    prompt = f"Given this PQM analysis:{state['analysis']}\n and the user prompt:{state['query']} \n give some recomendations."
    recommendation = llm.invoke(prompt)
    return {"recommendation": recommendation}

builder = StateGraph(GraphState)
builder.add_node("retrieve_docs", retrieve_docs_node)
builder.add_node("summarize", summarize_node)
builder.add_node("analyze", analyze_node)
builder.add_node("recommend", recommend_node)

builder.set_entry_point("retrieve_docs")
builder.add_edge("retrieve_docs", "summarize")
builder.add_edge("summarize", "analyze")
builder.add_edge("analyze", "recommend")
builder.set_finish_point("recommend")

graph = builder.compile()

output = graph.invoke({"query": "disturbance:Sag\n duration:0.2 s: voltage_pu :1.2"})

print(" Analysis:\n", output["analysis"])
print(" Recommendation:\n", output["recommendation"])
