import os 
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph,END

load_dotenv()

AGENT_REASON="agent_reason"
RAG= "rag"
LAST = -1

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

model=init_chat_model("gpt-5.2",model_provider="openai")

@tool (response_format="content")
def retrieve_context(query:str):
    """Retrieve relevant documents to help answer user queries about Smart Agriculture"""
    #retrive 8 most similar documents
    retrived_docs=vectorstore.as_retriever().invoke(query, k=8)
    return retrived_docs

tools = [ retrieve_context ]

llm = ChatOpenAI(model="gpt-5.2", temperature=0).bind_tools(tools)

system_prompt = (
        "You are a helpful AI assistant that answers questions about smart agriculture. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": system_prompt}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: MessagesState) -> str:
    if not state["messages"][-1].tool_calls:
        return END
    return RAG

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(RAG, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    RAG:RAG})

flow.add_edge(RAG, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# print("Hello ReAct LangGraph with Function Calling")
# res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
# print(res["messages"][LAST].content)








def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
    """
    res = app.invoke({
        "messages": [HumanMessage(content=query)]
    })
    return {
        "answer": res["messages"][-1].content
    }
  

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)