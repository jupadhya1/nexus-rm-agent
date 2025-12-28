"""
Base Agent Classes for Multi-Agent System
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

try:
    from databricks_langchain import ChatDatabricks
except ImportError:
    ChatDatabricks = None

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the multi-agent system"""
    SUPERVISOR = "supervisor"
    CUSTOMER_SPECIALIST = "customer_specialist"
    PRODUCT_SPECIALIST = "product_specialist"
    COMPLIANCE_OFFICER = "compliance_officer"
    ANALYTICS_EXPERT = "analytics_expert"
    DOCUMENT_ANALYST = "document_analyst"
    GENERAL_ASSISTANT = "general_assistant"


@dataclass
class AgentConfig:
    """Configuration for individual agent"""
    name: str
    role: AgentRole
    description: str
    system_prompt: str
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    tools: list = field(default_factory=list)
    max_iterations: int = 10
    temperature: float = 0.1


@dataclass
class AgentState:
    """State container for multi-agent workflow"""
    messages: list = field(default_factory=list)
    current_agent: str = ""
    task_queue: list = field(default_factory=list)
    completed_tasks: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    iteration_count: int = 0
    max_iterations: int = 15
    final_response: str = ""
    agent_history: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "current_agent": self.current_agent,
            "task_queue": self.task_queue,
            "completed_tasks": self.completed_tasks,
            "context": self.context,
            "iteration_count": self.iteration_count,
            "final_response": self.final_response
        }


class BaseAgent:
    """Base class for specialized agents"""
    
    def __init__(
        self,
        config: AgentConfig,
        llm: LanguageModelLike = None,
        tools: list = None
    ):
        self.config = config
        self.name = config.name
        self.role = config.role
        self.tools = tools or config.tools
        
        # Initialize LLM
        if llm:
            self.llm = llm
        elif ChatDatabricks:
            self.llm = ChatDatabricks(
                endpoint=config.model_endpoint,
                temperature=config.temperature
            )
        else:
            raise ImportError("databricks_langchain required or provide custom LLM")
        
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
        
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup the agent's prompt template"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad", optional=True)
        ])
    
    async def invoke(self, state: dict, config: RunnableConfig = None) -> dict:
        """Process the current state and return updated state"""
        messages = state.get("messages", [])
        
        # Build prompt
        prompt_messages = self.prompt.format_messages(
            messages=messages,
            agent_scratchpad=[]
        )
        
        # Get response
        response = await self.llm.ainvoke(prompt_messages, config=config)
        
        return {
            "messages": [response],
            "agent_history": [{
                "agent": self.name,
                "role": self.role.value,
                "timestamp": datetime.now().isoformat()
            }]
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, role={self.role.value})"
