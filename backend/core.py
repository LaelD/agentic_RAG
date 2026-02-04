import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()


class RAGAgent:
    """Smart Agriculture RAG Agent with LangGraph orchestration."""
    
    # Constants
    AGENT_REASON = "agent_reason"
    RAG = "rag"
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        model_name: str = "gpt-5.2",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0,
        retrieval_k: int = 5,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the RAG Agent.
        """
        
        self.index_name = index_name or os.getenv("INDEX_NAME")
        self.model_name = model_name
        self.temperature = temperature
        self.retrieval_k = retrieval_k
        
        # Initialize embeddings and vectorstore
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
        )
        
        # System prompt
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant that answers questions about smart agriculture. "
            "You have access to a tool that retrieves relevant documentation. "
            "Use the tool to find relevant information before answering questions. "
            "If you cannot find the answer in the retrieved documentation, say so."
        )
        
        # Build the graph
        self._build_graph()
    
    def _create_retrieve_tool(self):
        """Create the retrieval tool with access to vectorstore."""
        vectorstore = self.vectorstore
        retrieval_k = self.retrieval_k
        
        @tool(response_format="content")
        def retrieve_context(query: str) -> str:
            """Retrieve relevant documents to help answer user queries about Smart Agriculture"""
            docs = vectorstore.as_retriever(search_kwargs={"k": retrieval_k}).invoke(query)
            return "\n\n".join(
                f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
                for d in docs
            )
        
        return retrieve_context
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Create tools
        retrieve_tool = self._create_retrieve_tool()
        self.tools = [retrieve_tool]
        
        # Create LLM with tools
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        ).bind_tools(self.tools)
        
        # Create tool node
        self.tool_node = ToolNode(self.tools)
        
        # Build graph
        flow = StateGraph(MessagesState)
        
        flow.add_node(self.AGENT_REASON, self._run_agent_reasoning)
        flow.set_entry_point(self.AGENT_REASON)
        flow.add_node(self.RAG, self.tool_node)
        
        flow.add_conditional_edges(
            self.AGENT_REASON,
            self._should_continue,
            {END: END, self.RAG: self.RAG}
        )
        flow.add_edge(self.RAG, self.AGENT_REASON)
        
        self.app = flow.compile()
        
        # Optional: Generate flow diagram
        try:
            self.app.get_graph().draw_mermaid_png(output_file_path="flow.png")
        except Exception:
            pass
    
    def _run_agent_reasoning(self, state: MessagesState) -> MessagesState:
        """Run the agent reasoning node."""
        response = self.llm.invoke([
            {"role": "system", "content": self.system_prompt},
            *state["messages"]
        ])
        return {"messages": [response]}
    
    def _should_continue(self, state: MessagesState) -> str:
        """Decide whether to continue to tools or end."""
        if not state["messages"][-1].tool_calls:
            return END
        return self.RAG
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Run a query through the RAG agent.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the answer
        """
        result = self.app.invoke({"messages": [HumanMessage(content=query)]})
        return {"answer": result["messages"][-1].content}


# Singleton instance for backward compatibility
_default_agent: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    """Get or create the default RAG agent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = RAGAgent()
    return _default_agent


def run_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to run a query using the default agent.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing the answer
    """
    return get_agent().query(query)


# Alias for backward compatibility
run_llm = run_query


