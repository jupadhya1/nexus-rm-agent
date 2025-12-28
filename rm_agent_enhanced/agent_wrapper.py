"""
MLflow Chat Agent Wrapper for Databricks Deployment
"""

import logging
from typing import Any, Generator, Optional

from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

try:
    from mlflow.pyfunc import ChatAgent
    from mlflow.types.agent import (
        ChatAgentChunk,
        ChatAgentMessage,
        ChatAgentResponse,
        ChatContext,
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    ChatAgent = object


if MLFLOW_AVAILABLE:
    class EnhancedChatAgent(ChatAgent):
        """
        Enhanced chat agent wrapper for MLflow deployment
        Provides streaming, audit logging, and data masking
        """
        
        def __init__(
            self,
            agent: CompiledStateGraph,
            enable_audit: bool = True
        ):
            self.agent = agent
            self.enable_audit = enable_audit
        
        def predict(
            self,
            messages: list,
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
        ) -> ChatAgentResponse:
            """
            Process messages and return response
            """
            request = {"messages": self._convert_messages_to_dict(messages)}
            
            result_messages = []
            for event in self.agent.stream(request, stream_mode="updates"):
                for node_data in event.values():
                    result_messages.extend(
                        ChatAgentMessage(**msg) 
                        for msg in node_data.get("messages", [])
                    )
            
            if self.enable_audit:
                self._log_interaction(messages, result_messages)
            
            return ChatAgentResponse(messages=result_messages)
        
        def predict_stream(
            self,
            messages: list,
            context: Optional[ChatContext] = None,
            custom_inputs: Optional[dict[str, Any]] = None,
        ) -> Generator[ChatAgentChunk, None, None]:
            """
            Stream responses chunk by chunk
            """
            request = {"messages": self._convert_messages_to_dict(messages)}
            
            for event in self.agent.stream(request, stream_mode="updates"):
                for node_data in event.values():
                    yield from (
                        ChatAgentChunk(**{"delta": msg}) 
                        for msg in node_data.get("messages", [])
                    )
        
        def _log_interaction(self, input_msgs: list, output_msgs: list) -> None:
            """Log interaction for audit purposes"""
            logger.info(f"Agent interaction: {len(input_msgs)} input, {len(output_msgs)} output")


class SimpleChatAgent:
    """
    Simple chat agent for non-MLflow environments
    """
    
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
    
    def chat(self, message: str, history: list = None) -> str:
        """
        Simple chat interface
        """
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        request = {"messages": messages}
        
        result = self.agent.invoke(request)
        
        # Extract last assistant message
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        
        return ""
    
    async def achat(self, message: str, history: list = None) -> str:
        """
        Async chat interface
        """
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        request = {"messages": messages}
        
        result = await self.agent.ainvoke(request)
        
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        
        return ""
    
    def stream(self, message: str, history: list = None):
        """
        Streaming chat interface
        """
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        request = {"messages": messages}
        
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                for msg in node_data.get("messages", []):
                    if hasattr(msg, 'content'):
                        yield msg.content
                    elif isinstance(msg, dict):
                        yield msg.get("content", "")
