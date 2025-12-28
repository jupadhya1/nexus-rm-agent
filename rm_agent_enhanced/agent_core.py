"""
Enhanced RM Assistant Agent with MCP Integration
Core agent implementation
"""

import asyncio
import logging
from typing import Any, Generator, Optional, Sequence, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver

try:
    import mlflow
    from databricks_langchain import (
        ChatDatabricks,
        DatabricksFunctionClient,
        UCFunctionToolkit,
        set_uc_function_client,
    )
    from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
    DATABRICKS_AVAILABLE = True
    mlflow.langchain.autolog()
except ImportError:
    DATABRICKS_AVAILABLE = False
    ChatDatabricks = None
    ChatAgentState = dict
    ChatAgentToolNode = ToolNode

logger = logging.getLogger(__name__)


# System Prompts
ENHANCED_SYSTEM_PROMPT = """You are an advanced AI assistant for Relationship Managers (RMs) at a major bank.

## Your Capabilities
You have access to specialized tools and can help with:
- **Customer Management**: Look up customer profiles, assess risk ratings, analyze portfolios
- **Product Recommendations**: Search and compare unit trusts, match products to customer profiles
- **Compliance**: Verify KYC status, screen sanctions, check regulatory requirements
- **Analytics**: Perform calculations, currency conversions, portfolio analysis
- **Document Analysis**: Search and summarize product documentation

## Guidelines
1. **Always verify customer information** before making recommendations
2. **Check compliance status** for sensitive operations
3. **Consider risk suitability** when recommending products
4. **Use appropriate tools** - don't make up data
5. **Be transparent** about limitations and uncertainties
6. **Document actions** for audit trail

## Response Style
- Be professional but approachable
- Provide clear, actionable information
- Explain your reasoning when making recommendations
- Flag any compliance concerns proactively
- Summarize key points for quick reference

## Scope Limitations
If a question is outside banking, investment, or customer service domains, politely redirect.
Never provide personal financial advice without proper disclaimers.
"""


def create_enhanced_agent(
    model: LanguageModelLike = None,
    tools: Union[Sequence[BaseTool], ToolNode] = None,
    system_prompt: Optional[str] = None,
    enable_checkpointing: bool = True,
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
) -> CompiledGraph:
    """
    Create an enhanced tool-calling agent
    
    Args:
        model: Language model to use
        tools: List of tools or ToolNode
        system_prompt: Custom system prompt
        enable_checkpointing: Enable conversation checkpointing
        model_endpoint: Databricks model endpoint name
    
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize model
    if model is None:
        if ChatDatabricks:
            model = ChatDatabricks(endpoint=model_endpoint)
        else:
            raise ValueError("Model required when databricks_langchain not available")
    
    # Use default tools if none provided
    if tools is None:
        tools = []
    
    # Use default system prompt
    if system_prompt is None:
        system_prompt = ENHANCED_SYSTEM_PROMPT
    
    # Bind tools to model
    if tools:
        model_with_tools = model.bind_tools(tools)
    else:
        model_with_tools = model
    
    # Preprocessor to add system prompt
    def preprocess(state: dict) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state.get("messages", []))
        return messages
    
    preprocessor = RunnableLambda(preprocess)
    model_runnable = preprocessor | model_with_tools
    
    # Model node
    def call_model(state: dict, config: RunnableConfig) -> dict:
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    # Routing function
    def should_continue(state: dict) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        elif isinstance(last_message, dict) and last_message.get("tool_calls"):
            return "continue"
        
        return "end"
    
    # Build workflow
    workflow = StateGraph(ChatAgentState if DATABRICKS_AVAILABLE else dict)
    
    workflow.add_node("agent", RunnableLambda(call_model))
    
    if tools:
        if DATABRICKS_AVAILABLE:
            workflow.add_node("tools", ChatAgentToolNode(tools))
        else:
            workflow.add_node("tools", ToolNode(tools))
    
    workflow.set_entry_point("agent")
    
    if tools:
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)
    
    # Compile with optional checkpointing
    if enable_checkpointing:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    return workflow.compile()


def create_agent_with_uc_tools(
    catalog: str = "workspace",
    schema: str = "rm_agent",
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    additional_tools: list = None
) -> CompiledGraph:
    """
    Create agent with Unity Catalog function tools
    """
    if not DATABRICKS_AVAILABLE:
        raise ImportError("databricks_langchain required for UC tools")
    
    # Initialize UC client
    client = DatabricksFunctionClient()
    set_uc_function_client(client)
    
    # Load UC functions as tools
    uc_tool_names = [f"{catalog}.{schema}.*"]
    uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
    
    tools = list(uc_toolkit.tools)
    if additional_tools:
        tools.extend(additional_tools)
    
    # Create model
    llm = ChatDatabricks(endpoint=model_endpoint)
    
    return create_enhanced_agent(
        model=llm,
        tools=tools,
        enable_checkpointing=True
    )
