from langgraph.graph import StateGraph
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import TypedDict, List
import os
from report2pdf import get_report

# Load environment variables
load_dotenv()
report_path = "reports"

# Define graph state
class GraphState(TypedDict):
    query: str
    docs: List[Document]
    summary: str
    analysis: str
    recommendation: str
    report: str 
    node: str
    conv: str

# Initialize embedding model and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectordb = Chroma(
    collection_name="IEEE",
    embedding_function=embeddings,
    persist_directory="chroma_langchain_db",
)

# Load LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# Nodes
def retrieve_docs_node(state):
    query = state["query"]
    docs = vectordb.similarity_search(query, k=5)
    return {"docs": docs}

def summarize_node(state):
    chunks = "\n\n".join([doc.page_content for doc in state["docs"]])
    prompt_template = PromptTemplate(
        input_variables=["prompt", "chunks"],
        template="""
        You are a summarizer node used in a RAG system. Summarize the best info for the user from the retrieved chunks, relevant to the prompt.

        chunks:
        {chunks}

        prompt:
        {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"], chunks=chunks)
    summary = llm.invoke(prompt_text)
    return {"summary": summary}

def analyze_node(state):
    summary_or_docs = state.get("summary") or "\n\n".join([doc.page_content for doc in state["docs"]])
    prompt_template = PromptTemplate(
        input_variables=["prompt", "data", "report_sample"],
        template="""
        You are an analyzer node. Your job is to analyze the power quality event data and give insights in the format below:

        Report Format:
        {report_sample}

        Retrieved Data:
        {data}

        Prompt:
        {prompt}
        """
    )
    with open("src/AI_agent/sample_texts/report_sample.txt", "r") as f:
        report_sample = f.read()

    prompt_text = prompt_template.format(prompt=state["query"], data=summary_or_docs, report_sample=report_sample)
    analysis = llm.invoke(prompt_text)
    return {"analysis": analysis}

def report_node(state):
    with open("src/AI_agent/sample_texts/report_sample.txt", "r") as f:
        report_sample = f.read()

    prompt_template = PromptTemplate(
        input_variables=["prompt", "data", "report_sample"],
        template="""
        You are a report generator node. Generate a report based on the retrieved data and prompt.

        Report Format:
        {report_sample}

        Retrieved Data:
        {data}

        Prompt:
        {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"], data=state['summary'], report_sample=report_sample)
    report = llm.invoke(prompt_text)
    print(report)
    return {"report": report}

def routing(state):
    if state["query"] =="":
        query = input()
    else:
        query = state["query"]
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        This is a Power Quality Management Agent.
        You are a chat node which routes the user query to one of the following nodes:
        1) chatnode
        2) report
        3) prevfaults
        4) systeminfo
        5) analyze
        Just return the node name only.
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=query)
    result = llm.invoke(prompt_text)
    print(f"router trigerred {result}")
    return {"node": result,"query":query}

def chatbot(state):
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        You are a chatbot made to answer casual queries for a Power Quality Management Agent.
        this agent can generate report,do chat,give info about the power system and analyze.
        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"])
    conv = llm.invoke(prompt_text)
    print(conv)
    state["query"] = ""
    return {"conv": conv}

# Router function
def router_node(state):
    query = state["node"]
    if "chatnode" in query.lower():
        return "chatnode"
    elif "report" in query.lower():
        return "retrieve_docs"
    elif "analyze" in query.lower():
        return "analyze"
        

def system_info(state):
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="""
        You are a node in a Power Quality Management Agent your work is to use the fetched info and utlize this info with the use prompt to answer the query .
        mainly the queries realted to the poower system will be asked to you.

        Prompt: {prompt}
        """
    )
    prompt_text = prompt_template.format(prompt=state["query"])
    conv = llm.invoke(prompt_text)
    print(conv)
    state["query"] = ""
    return {"conv": conv}
builder = StateGraph(GraphState)

builder.add_node("chatnode", chatbot)
builder.add_node("router", routing)
builder.add_node("retrieve_docs", retrieve_docs_node)
builder.add_node("summarize", summarize_node)
builder.add_node("analyze", analyze_node)
builder.add_node("recommend", report_node)
builder.add_node("retrieve_docs_1", retrieve_docs_node)
builder.add_node("summarize_1", summarize_node)
builder.add_node("infosys",system_info)
builder.set_entry_point("router")

builder.add_conditional_edges("router", router_node, {
    "chatnode": "chatnode",
    "retrieve_docs": "retrieve_docs",
    "analyze":"retrieve_docs_1",

})
builder.add_edge("retrieve_docs_1", "summarize_1")
builder.add_edge("summarize_1", "analyze")

builder.add_edge("retrieve_docs", "summarize")
builder.add_edge("summarize", "recommend")  # Or "analyze" if you want analysis step
builder.add_edge('chatnode',"router")
builder.set_finish_point("recommend")
builder.set_finish_point("analyze")

graph = builder.compile()
output = graph.invoke({
    "query": """
    Task: Generate a report  of power quality disturbances.

    Input Data:
    - Time Duration: 2025-06-21 14:00 to 2025-06-21 16:00
    - Disturbance Type(s): Voltage Swell
    - Measurement Points: Main LT Panel (Incoming Feeder)
    - Sampling Frequency: 10 kHz
    - Nominal Voltage: 230V
    - Events Detected:
      - Time: 14:27:55
      - Type: Voltage Swell
      - Duration: 140 ms
      - Amplitude: Voltage Rised from 230V to 335V
      - Recovery: Returned to nominal within 160 ms
    """
})
# output = graph.invoke({"query":""})
# print(output["conv"])
print(graph.get_graph().draw_mermaid())
print(output['report'])
get_report(output["report"], report_path)
