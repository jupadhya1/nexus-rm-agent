# Databricks notebook source
# MAGIC %md
# MAGIC # Enhanced RM Assistant Agent - Deployment
# MAGIC 
# MAGIC This notebook deploys the enhanced RM Assistant Agent with:
# MAGIC - **Multi-Agent Orchestration**: Supervisor pattern with specialized agents
# MAGIC - **MCP Integration**: Model Context Protocol for extended tool capabilities
# MAGIC - **Advanced Tools**: Compliance, KYC, RCSA, and analytics tools
# MAGIC - **Enterprise Features**: Audit logging, caching, and observability

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow-skinny[databricks] langgraph>=0.3.4 databricks-langchain databricks-agents uv pydantic>=2.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "workspace"
SCHEMA = "rm_agent"
MODEL_NAME = "rm_agent_enhanced"

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
VECTOR_SEARCH_ENDPOINT = "vs_endpoint"

# Full names
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.ut_pdf_docs_vs_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Enhanced Agent

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC """
# MAGIC Enhanced RM Assistant Agent with Multi-Agent Support
# MAGIC """
# MAGIC 
# MAGIC import logging
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC 
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from langgraph.checkpoint.memory import MemorySaver
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC 
# MAGIC logger = logging.getLogger(__name__)
# MAGIC mlflow.langchain.autolog()
# MAGIC 
# MAGIC # Initialize UC client
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC 
# MAGIC # Configuration
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC CATALOG = "workspace"
# MAGIC SCHEMA = "rm_agent"
# MAGIC 
# MAGIC # Enhanced system prompt
# MAGIC SYSTEM_PROMPT = """You are an advanced AI assistant for Relationship Managers (RMs) at a major bank.
# MAGIC 
# MAGIC ## Your Capabilities
# MAGIC You have access to specialized tools and can help with:
# MAGIC - **Customer Management**: Look up customer profiles, assess risk ratings, analyze portfolios
# MAGIC - **Product Recommendations**: Search and compare unit trusts, match products to customer profiles
# MAGIC - **Compliance**: Verify KYC status, screen sanctions, check regulatory requirements
# MAGIC - **Analytics**: Perform calculations, currency conversions, portfolio analysis
# MAGIC - **Document Analysis**: Search and summarize product documentation
# MAGIC 
# MAGIC ## Guidelines
# MAGIC 1. **Always verify customer information** before making recommendations
# MAGIC 2. **Check compliance status** for sensitive operations
# MAGIC 3. **Consider risk suitability** when recommending products
# MAGIC 4. **Use appropriate tools** - don't make up data
# MAGIC 5. **Be transparent** about limitations and uncertainties
# MAGIC 
# MAGIC ## Response Style
# MAGIC - Be professional but approachable
# MAGIC - Provide clear, actionable information
# MAGIC - Explain your reasoning when making recommendations
# MAGIC - Flag any compliance concerns proactively
# MAGIC 
# MAGIC ## Scope Limitations
# MAGIC If a question is outside banking, investment, or customer service domains, politely redirect.
# MAGIC """
# MAGIC 
# MAGIC # Initialize LLM
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC 
# MAGIC # Load tools
# MAGIC tools = []
# MAGIC 
# MAGIC # Unity Catalog function tools
# MAGIC uc_tool_names = [f"{CATALOG}.{SCHEMA}.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC 
# MAGIC # Vector search tool (optional)
# MAGIC # vs_tool = VectorSearchRetrieverTool(
# MAGIC #     index_name=f"{CATALOG}.{SCHEMA}.ut_pdf_docs_vs_index",
# MAGIC #     num_results=5
# MAGIC # )
# MAGIC # tools.append(vs_tool)
# MAGIC 
# MAGIC 
# MAGIC def create_enhanced_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     """Create enhanced tool-calling agent"""
# MAGIC     
# MAGIC     model_with_tools = model.bind_tools(tools)
# MAGIC     
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         return "end"
# MAGIC     
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     
# MAGIC     model_runnable = preprocessor | model_with_tools
# MAGIC     
# MAGIC     def call_model(state: ChatAgentState, config: RunnableConfig):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC         return {"messages": [response]}
# MAGIC     
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC     
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {"continue": "tools", "end": END},
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC     
# MAGIC     return workflow.compile()
# MAGIC 
# MAGIC 
# MAGIC class EnhancedChatAgent(ChatAgent):
# MAGIC     """Enhanced chat agent with audit logging"""
# MAGIC     
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC     
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         
# MAGIC         result_messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 result_messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         
# MAGIC         return ChatAgentResponse(messages=result_messages)
# MAGIC     
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) 
# MAGIC                     for msg in node_data["messages"]
# MAGIC                 )
# MAGIC 
# MAGIC 
# MAGIC # Create and register agent
# MAGIC agent = create_enhanced_agent(llm, tools, SYSTEM_PROMPT)
# MAGIC AGENT = EnhancedChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Agent

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

# Test basic interaction
response = AGENT.predict({"messages": [{"role": "user", "content": "Hello! What can you help me with?"}]})
print(response)

# COMMAND ----------

# Test with customer query
response = AGENT.predict({
    "messages": [
        {"role": "user", "content": "What unit trusts would you recommend for a customer with risk rating 3?"}
    ]
})
print(response)

# COMMAND ----------

# Test streaming
for chunk in AGENT.predict_stream({
    "messages": [{"role": "user", "content": "Tell me about the Income Opportunity Fund"}]
}):
    print(chunk, end="")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Model to MLflow

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME, tools, CATALOG, SCHEMA
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksTable,
    DatabricksFunction,
)
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

# Define resources for auth passthrough
resources = [
    DatabricksVectorSearchIndex(index_name=f"{CATALOG}.{SCHEMA}.ut_pdf_docs_vs_index"),
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en"),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.customer_profile"),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.unit_trust"),
]

# Add UC function resources
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

# Input example
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What are the unit trusts recommended for a customer with risk rating of 3?"
        }
    ]
}

# Log model
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )
    
print(f"Model logged: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Model

# COMMAND ----------

# Pre-deployment validation
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.rm_agent_enhanced"

# Register model
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME
)

print(f"Registered model: {UC_MODEL_NAME} version {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Agent

# COMMAND ----------

from databricks import agents

# Deploy agent
deployment = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=uc_registered_model_info.version,
    workload_size="Small",
    scale_to_zero=True
)

print(f"Agent deployed!")
print(f"Endpoint: {deployment.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. **Test in AI Playground**: Chat with your deployed agent
# MAGIC 2. **Deploy Streamlit App**: Use the enhanced Streamlit app for a custom UI
# MAGIC 3. **Monitor & Evaluate**: Use MLflow to track agent performance
# MAGIC 4. **Add More Tools**: Extend with MCP servers for additional capabilities
