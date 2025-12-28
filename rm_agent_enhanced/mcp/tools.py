"""
MCP Tool Adapter for LangGraph/LangChain Integration
Provides seamless bridging between MCP tools and LangChain tool ecosystem
"""

import asyncio
import json
import logging
from typing import Any, Callable, Optional, Type, Union
from functools import wraps

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, create_model

from .client import MCPClientManager, MCPTool

logger = logging.getLogger(__name__)


def mcp_tool_to_pydantic_schema(mcp_tool: MCPTool) -> Type[BaseModel]:
    """
    Convert MCP tool input schema to Pydantic model for LangChain
    """
    input_schema = mcp_tool.input_schema
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    
    field_definitions = {}
    for name, prop in properties.items():
        field_type = _json_type_to_python(prop.get("type", "string"))
        description = prop.get("description", "")
        default = ... if name in required else None
        field_definitions[name] = (field_type, Field(default=default, description=description))
    
    if not field_definitions:
        # Empty schema - create a model with no required fields
        field_definitions["_placeholder"] = (Optional[str], Field(default=None, description="No arguments required"))
    
    model_name = f"{mcp_tool.name.replace('-', '_').title()}Args"
    return create_model(model_name, **field_definitions)


def _json_type_to_python(json_type: str) -> type:
    """Convert JSON schema type to Python type"""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_mapping.get(json_type, str)


class MCPToolWrapper(BaseTool):
    """
    LangChain tool wrapper for MCP tools with async support
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    mcp_client: MCPClientManager
    mcp_tool_name: str
    return_direct: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Synchronous run - wraps async"""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Asynchronous run"""
        try:
            # Remove placeholder argument if present
            kwargs.pop("_placeholder", None)
            
            result = await self.mcp_client.call_tool(self.mcp_tool_name, kwargs)
            
            # Format result
            if isinstance(result, dict):
                if "content" in result:
                    # Handle MCP content format
                    contents = result["content"]
                    if isinstance(contents, list):
                        return "\n".join(
                            c.get("text", str(c)) 
                            for c in contents 
                            if isinstance(c, dict)
                        )
                return json.dumps(result, indent=2)
            return str(result)
            
        except Exception as e:
            logger.error(f"MCP tool '{self.mcp_tool_name}' failed: {e}")
            return f"Error: {str(e)}"


class MCPToolRegistry:
    """
    Registry for managing MCP tools and their LangChain wrappers
    """
    
    def __init__(self, mcp_client: MCPClientManager):
        self.mcp_client = mcp_client
        self._wrapped_tools: dict[str, MCPToolWrapper] = {}
        self._tool_metadata: dict[str, dict] = {}
    
    def register_all_tools(self) -> list[MCPToolWrapper]:
        """
        Register all MCP tools as LangChain tools
        """
        mcp_tools = self.mcp_client.get_all_tools()
        wrapped = []
        
        for mcp_tool in mcp_tools:
            try:
                wrapper = self._wrap_tool(mcp_tool)
                self._wrapped_tools[mcp_tool.name] = wrapper
                self._tool_metadata[mcp_tool.name] = {
                    "server": mcp_tool.server_name,
                    "original_schema": mcp_tool.input_schema
                }
                wrapped.append(wrapper)
                logger.info(f"Registered MCP tool: {mcp_tool.name}")
            except Exception as e:
                logger.error(f"Failed to register MCP tool '{mcp_tool.name}': {e}")
        
        return wrapped
    
    def _wrap_tool(self, mcp_tool: MCPTool) -> MCPToolWrapper:
        """
        Wrap a single MCP tool
        """
        args_schema = mcp_tool_to_pydantic_schema(mcp_tool)
        
        return MCPToolWrapper(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=args_schema,
            mcp_client=self.mcp_client,
            mcp_tool_name=mcp_tool.name
        )
    
    def get_tool(self, name: str) -> Optional[MCPToolWrapper]:
        """Get a specific wrapped tool by name"""
        return self._wrapped_tools.get(name)
    
    def get_all_tools(self) -> list[MCPToolWrapper]:
        """Get all wrapped tools"""
        return list(self._wrapped_tools.values())
    
    def get_tools_by_server(self, server_name: str) -> list[MCPToolWrapper]:
        """Get tools from a specific MCP server"""
        return [
            tool for name, tool in self._wrapped_tools.items()
            if self._tool_metadata.get(name, {}).get("server") == server_name
        ]


class MCPResourceTool(BaseTool):
    """
    LangChain tool for reading MCP resources
    """
    name: str = "read_mcp_resource"
    description: str = "Read content from an MCP resource by URI"
    mcp_client: MCPClientManager
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, uri: str) -> str:
        """Synchronous run"""
        return asyncio.run(self._arun(uri))
    
    async def _arun(self, uri: str) -> str:
        """Read a resource by URI"""
        try:
            # Find which server has this resource
            for resource in self.mcp_client.get_all_resources():
                if resource.uri == uri:
                    client = self.mcp_client._clients[resource.server_name]
                    result = await client.read_resource(uri)
                    
                    if isinstance(result, dict) and "contents" in result:
                        contents = result["contents"]
                        if isinstance(contents, list):
                            return "\n".join(
                                c.get("text", str(c)) 
                                for c in contents
                            )
                    return str(result)
            
            return f"Resource not found: {uri}"
            
        except Exception as e:
            logger.error(f"Failed to read resource '{uri}': {e}")
            return f"Error: {str(e)}"


class MCPPromptTool(BaseTool):
    """
    LangChain tool for getting MCP prompts
    """
    name: str = "get_mcp_prompt"
    description: str = "Get a prompt template from MCP server"
    mcp_client: MCPClientManager
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, prompt_name: str, arguments: dict = None) -> str:
        """Synchronous run"""
        return asyncio.run(self._arun(prompt_name, arguments))
    
    async def _arun(self, prompt_name: str, arguments: dict = None) -> str:
        """Get a prompt by name"""
        try:
            for prompt in self.mcp_client.get_all_prompts():
                if prompt.name == prompt_name:
                    client = self.mcp_client._clients[prompt.server_name]
                    result = await client.get_prompt(prompt_name, arguments)
                    
                    if isinstance(result, dict) and "messages" in result:
                        messages = result["messages"]
                        return "\n".join(
                            f"{m.get('role', 'user')}: {m.get('content', {}).get('text', '')}"
                            for m in messages
                        )
                    return str(result)
            
            return f"Prompt not found: {prompt_name}"
            
        except Exception as e:
            logger.error(f"Failed to get prompt '{prompt_name}': {e}")
            return f"Error: {str(e)}"


def create_mcp_toolkit(mcp_client: MCPClientManager) -> list[BaseTool]:
    """
    Create a complete toolkit of MCP tools for LangChain/LangGraph
    
    Returns:
        List of LangChain tools wrapping all MCP capabilities
    """
    tools = []
    
    # Register all MCP tools
    registry = MCPToolRegistry(mcp_client)
    tools.extend(registry.register_all_tools())
    
    # Add resource tool if resources available
    if mcp_client.get_all_resources():
        tools.append(MCPResourceTool(mcp_client=mcp_client))
    
    return tools
