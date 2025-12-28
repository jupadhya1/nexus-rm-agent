"""
Enhanced RM Assistant Agent with MCP Integration
================================================

An advanced AI agent for Relationship Managers built on Databricks with:
- Multi-Agent Orchestration using LangGraph
- MCP (Model Context Protocol) integration
- Advanced Compliance and Banking Tools
- Enterprise Features (audit, caching, observability)
"""

__version__ = "1.0.0"
__author__ = "RM Agent Team"

from .agent_core import create_enhanced_agent, create_agent_with_uc_tools
from .agent_wrapper import SimpleChatAgent

__all__ = [
    "create_enhanced_agent",
    "create_agent_with_uc_tools",
    "SimpleChatAgent",
]
