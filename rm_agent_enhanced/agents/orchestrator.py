"""
Supervisor Agent and Multi-Agent Orchestrator
Implements the supervisor pattern for coordinating multiple agents
"""

import json
import logging
from datetime import datetime
from typing import Optional, Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .base import BaseAgent, AgentConfig, AgentRole, AgentState
from .specialists import (
    CustomerSpecialistAgent,
    ProductSpecialistAgent,
    ComplianceOfficerAgent,
    AnalyticsExpertAgent,
    DocumentAnalystAgent,
)

try:
    from databricks_langchain import ChatDatabricks
except ImportError:
    ChatDatabricks = None

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor agent that orchestrates multiple specialized agents
    Uses ReAct pattern for reasoning about agent delegation
    """
    
    SUPERVISOR_PROMPT = """You are a Supervisor Agent orchestrating a team of specialized agents to help Relationship Managers.

Your team consists of:
1. customer_specialist - Handles customer data, profiles, and risk assessment
2. product_specialist - Provides investment product information and recommendations  
3. compliance_officer - Ensures regulatory compliance and risk management
4. analytics_expert - Performs data analysis and calculations
5. document_analyst - Analyzes and extracts information from documents

Given the user's request, determine which agent(s) should handle the task.
You can delegate to multiple agents in sequence if needed.

Response format (return valid JSON only):
{{
    "reasoning": "Your analysis of the request and delegation strategy",
    "next_agent": "agent_name or FINISH if task is complete",
    "instructions": "Specific instructions for the next agent",
    "requires_tools": ["list", "of", "expected", "tools"]
}}

If the task requires information from multiple agents, route through them sequentially.
When all necessary information is gathered, respond with "next_agent": "FINISH".

Current context: {context}
Agent history: {agent_history}
Completed tasks: {completed_tasks}
"""
    
    def __init__(
        self,
        agents: dict[str, BaseAgent],
        llm: LanguageModelLike = None,
        model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    ):
        self.agents = agents
        self.agent_names = list(agents.keys())
        
        if llm:
            self.llm = llm
        elif ChatDatabricks:
            self.llm = ChatDatabricks(endpoint=model_endpoint, temperature=0.0)
        else:
            raise ImportError("databricks_langchain required or provide custom LLM")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SUPERVISOR_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ])
    
    async def route(self, state: dict) -> dict:
        """Determine which agent should handle the current state"""
        messages = state.get("messages", [])
        context = state.get("context", {})
        agent_history = state.get("agent_history", [])
        completed_tasks = state.get("completed_tasks", [])
        
        prompt_messages = self.prompt.format_messages(
            messages=messages,
            context=json.dumps(context, default=str),
            agent_history=json.dumps(agent_history, default=str),
            completed_tasks=json.dumps(completed_tasks, default=str)
        )
        
        response = await self.llm.ainvoke(prompt_messages)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            decision = json.loads(content.strip())
            
            next_agent = decision.get("next_agent", "FINISH")
            instructions = decision.get("instructions", "")
            reasoning = decision.get("reasoning", "")
            
            return {
                "current_agent": next_agent,
                "context": {
                    **context,
                    "supervisor_reasoning": reasoning,
                    "agent_instructions": instructions
                }
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse supervisor response: {e}")
            return {
                "current_agent": "FINISH",
                "context": {
                    **context,
                    "supervisor_response": response.content
                }
            }


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent workflows using LangGraph
    Supports supervisor pattern with specialized agents
    """
    
    def __init__(
        self,
        model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
        worker_model_endpoint: str = "databricks-meta-llama-3-1-8b-instruct",
        tools: list = None,
        enable_checkpointing: bool = True
    ):
        self.model_endpoint = model_endpoint
        self.worker_model_endpoint = worker_model_endpoint
        self.tools = tools or []
        self.enable_checkpointing = enable_checkpointing
        
        self.agents: dict[str, BaseAgent] = {}
        self.supervisor: Optional[SupervisorAgent] = None
        self.graph: Optional[CompiledGraph] = None
        
        if enable_checkpointing:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None
        
        self._setup_agents()
        self._build_graph()
    
    def _setup_agents(self):
        """Initialize specialized agents"""
        agent_classes = {
            "customer_specialist": CustomerSpecialistAgent,
            "product_specialist": ProductSpecialistAgent,
            "compliance_officer": ComplianceOfficerAgent,
            "analytics_expert": AnalyticsExpertAgent,
            "document_analyst": DocumentAnalystAgent,
        }
        
        for name, agent_class in agent_classes.items():
            self.agents[name] = agent_class.create(
                tools=self.tools,
                model_endpoint=self.worker_model_endpoint
            )
        
        self.supervisor = SupervisorAgent(
            agents=self.agents,
            model_endpoint=self.model_endpoint
        )
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        from typing import TypedDict, Annotated
        from operator import add
        
        class GraphState(TypedDict):
            messages: Annotated[Sequence[dict], add]
            current_agent: str
            context: dict
            agent_history: Annotated[list, add]
            completed_tasks: Annotated[list, add]
            iteration_count: int
        
        workflow = StateGraph(GraphState)
        
        # Supervisor node
        async def supervisor_node(state: GraphState) -> GraphState:
            result = await self.supervisor.route(state)
            return result
        
        workflow.add_node("supervisor", supervisor_node)
        
        # Agent nodes
        for name, agent in self.agents.items():
            async def agent_node(state: GraphState, agent=agent) -> GraphState:
                result = await agent.invoke(state)
                return {
                    **result,
                    "completed_tasks": [{
                        "agent": agent.name,
                        "timestamp": datetime.now().isoformat()
                    }],
                    "iteration_count": state.get("iteration_count", 0) + 1
                }
            
            workflow.add_node(name, agent_node)
        
        # Tool node if tools provided
        if self.tools:
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
        
        # Entry point
        workflow.set_entry_point("supervisor")
        
        # Router function
        def router(state: GraphState) -> str:
            current = state.get("current_agent", "FINISH")
            iteration = state.get("iteration_count", 0)
            
            if iteration >= 15:
                return END
            if current == "FINISH" or current == END:
                return END
            if current in self.agents:
                return current
            return END
        
        # Conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            router,
            {name: name for name in self.agents.keys()} | {END: END}
        )
        
        # All agents route back to supervisor
        for name in self.agents.keys():
            workflow.add_edge(name, "supervisor")
        
        self.graph = workflow.compile(checkpointer=self.checkpointer)
    
    async def run(
        self,
        messages: list,
        config: RunnableConfig = None,
        thread_id: str = None
    ) -> dict:
        """Run the multi-agent workflow"""
        initial_state = {
            "messages": messages,
            "current_agent": "",
            "context": {},
            "agent_history": [],
            "completed_tasks": [],
            "iteration_count": 0
        }
        
        if thread_id and self.enable_checkpointing:
            config = config or {}
            config["configurable"] = config.get("configurable", {})
            config["configurable"]["thread_id"] = thread_id
        
        result = await self.graph.ainvoke(initial_state, config=config)
        return result
    
    async def stream(
        self,
        messages: list,
        config: RunnableConfig = None,
        thread_id: str = None
    ):
        """Stream the multi-agent workflow"""
        initial_state = {
            "messages": messages,
            "current_agent": "",
            "context": {},
            "agent_history": [],
            "completed_tasks": [],
            "iteration_count": 0
        }
        
        if thread_id and self.enable_checkpointing:
            config = config or {}
            config["configurable"] = config.get("configurable", {})
            config["configurable"]["thread_id"] = thread_id
        
        async for event in self.graph.astream(
            initial_state, 
            config=config, 
            stream_mode="updates"
        ):
            yield event
    
    def get_graph_visualization(self) -> str:
        """Get Mermaid diagram of the workflow"""
        return self.graph.get_graph().draw_mermaid()


def create_rm_orchestrator(
    tools: list = None,
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    enable_checkpointing: bool = True
) -> MultiAgentOrchestrator:
    """Factory function to create a configured RM Assistant orchestrator"""
    return MultiAgentOrchestrator(
        model_endpoint=model_endpoint,
        tools=tools,
        enable_checkpointing=enable_checkpointing
    )
