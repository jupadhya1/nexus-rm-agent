"""
Advanced MCP (Model Context Protocol) Client Implementation
Supports multiple transports, tool execution, resource management, and prompts
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
from enum import Enum
import subprocess
import sys
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    """MCP Transport types"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    input_schema: dict
    server_name: str
    
    def to_langchain_tool_schema(self) -> dict:
        """Convert to LangChain tool format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }


@dataclass
class MCPResource:
    """Represents an MCP resource"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server_name: str = ""


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template"""
    name: str
    description: Optional[str] = None
    arguments: list = field(default_factory=list)
    server_name: str = ""


@dataclass
class MCPMessage:
    """MCP JSON-RPC message"""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[dict] = None
    
    def to_dict(self) -> dict:
        d = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            d["id"] = self.id
        if self.method:
            d["method"] = self.method
        if self.params:
            d["params"] = self.params
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, data: str) -> "MCPMessage":
        parsed = json.loads(data)
        return cls(**{k: v for k, v in parsed.items() if k in cls.__dataclass_fields__})


class MCPTransportBase(ABC):
    """Base class for MCP transports"""
    
    @abstractmethod
    async def connect(self) -> None:
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        pass
    
    @abstractmethod
    async def send(self, message: MCPMessage) -> None:
        pass
    
    @abstractmethod
    async def receive(self) -> MCPMessage:
        pass


class StdioTransport(MCPTransportBase):
    """STDIO transport for MCP communication"""
    
    def __init__(self, command: str, args: list = None, env: dict = None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: Optional[asyncio.subprocess.Process] = None
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Start the MCP server process"""
        import os
        env = {**os.environ, **self.env}
        
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        logger.info(f"Started MCP server: {self.command} {' '.join(self.args)}")
    
    async def disconnect(self) -> None:
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            logger.info("MCP server stopped")
    
    async def send(self, message: MCPMessage) -> None:
        """Send a message to the server"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Transport not connected")
        
        async with self._write_lock:
            data = message.to_json() + "\n"
            self.process.stdin.write(data.encode())
            await self.process.stdin.drain()
    
    async def receive(self) -> MCPMessage:
        """Receive a message from the server"""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Transport not connected")
        
        async with self._read_lock:
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError("Connection closed")
            return MCPMessage.from_json(line.decode().strip())


class SSETransport(MCPTransportBase):
    """Server-Sent Events transport for MCP communication"""
    
    def __init__(self, url: str, headers: dict = None):
        self.url = url
        self.headers = headers or {}
        self._session = None
        self._response = None
        self._message_queue = asyncio.Queue()
    
    async def connect(self) -> None:
        """Connect to SSE endpoint"""
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required for SSE transport: pip install aiohttp")
        
        self._session = aiohttp.ClientSession()
        self._response = await self._session.get(
            self.url,
            headers={**self.headers, "Accept": "text/event-stream"}
        )
        
        # Start background task to read events
        asyncio.create_task(self._read_events())
        logger.info(f"Connected to SSE endpoint: {self.url}")
    
    async def _read_events(self) -> None:
        """Background task to read SSE events"""
        async for line in self._response.content:
            line = line.decode().strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    message = MCPMessage.from_json(data)
                    await self._message_queue.put(message)
    
    async def disconnect(self) -> None:
        """Disconnect from SSE endpoint"""
        if self._response:
            self._response.close()
        if self._session:
            await self._session.close()
    
    async def send(self, message: MCPMessage) -> None:
        """Send message via POST request"""
        if not self._session:
            raise RuntimeError("Transport not connected")
        
        async with self._session.post(
            self.url,
            json=message.to_dict(),
            headers=self.headers
        ) as response:
            response.raise_for_status()
    
    async def receive(self) -> MCPMessage:
        """Receive message from queue"""
        return await self._message_queue.get()


class MCPServerClient:
    """Client for a single MCP server"""
    
    def __init__(
        self,
        name: str,
        transport: MCPTransportBase,
        timeout: float = 30.0
    ):
        self.name = name
        self.transport = transport
        self.timeout = timeout
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._initialized = False
        self._reader_task: Optional[asyncio.Task] = None
    
    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id
    
    async def connect(self) -> None:
        """Connect to the MCP server and initialize"""
        await self.transport.connect()
        self._reader_task = asyncio.create_task(self._read_messages())
        await self._initialize()
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if self._reader_task:
            self._reader_task.cancel()
        await self.transport.disconnect()
    
    async def _read_messages(self) -> None:
        """Background task to read incoming messages"""
        while True:
            try:
                message = await self.transport.receive()
                
                # Handle response to pending request
                if message.id is not None and message.id in self._pending_requests:
                    future = self._pending_requests.pop(message.id)
                    if message.error:
                        future.set_exception(Exception(message.error.get("message", "Unknown error")))
                    else:
                        future.set_result(message.result)
                
                # Handle notifications
                elif message.method:
                    await self._handle_notification(message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading MCP message: {e}")
    
    async def _handle_notification(self, message: MCPMessage) -> None:
        """Handle server notifications"""
        if message.method == "notifications/tools/list_changed":
            await self._refresh_tools()
        elif message.method == "notifications/resources/list_changed":
            await self._refresh_resources()
        elif message.method == "notifications/prompts/list_changed":
            await self._refresh_prompts()
    
    async def _send_request(self, method: str, params: dict = None) -> Any:
        """Send a request and wait for response"""
        request_id = self._next_request_id()
        message = MCPMessage(id=request_id, method=method, params=params or {})
        
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future
        
        try:
            await self.transport.send(message)
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request {method} timed out")
    
    async def _initialize(self) -> None:
        """Initialize connection with capability negotiation"""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "rm-agent-mcp-client",
                "version": "1.0.0"
            }
        })
        
        # Send initialized notification
        await self.transport.send(MCPMessage(method="notifications/initialized"))
        
        # Refresh capabilities
        await self._refresh_tools()
        await self._refresh_resources()
        await self._refresh_prompts()
        
        self._initialized = True
        logger.info(f"MCP server '{self.name}' initialized with capabilities: {result}")
    
    async def _refresh_tools(self) -> None:
        """Refresh available tools"""
        result = await self._send_request("tools/list")
        self._tools = {
            tool["name"]: MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
                server_name=self.name
            )
            for tool in result.get("tools", [])
        }
        logger.info(f"Server '{self.name}' has {len(self._tools)} tools")
    
    async def _refresh_resources(self) -> None:
        """Refresh available resources"""
        try:
            result = await self._send_request("resources/list")
            self._resources = {
                res["uri"]: MCPResource(
                    uri=res["uri"],
                    name=res["name"],
                    description=res.get("description"),
                    mime_type=res.get("mimeType"),
                    server_name=self.name
                )
                for res in result.get("resources", [])
            }
        except Exception as e:
            logger.debug(f"Resources not supported by server '{self.name}': {e}")
    
    async def _refresh_prompts(self) -> None:
        """Refresh available prompts"""
        try:
            result = await self._send_request("prompts/list")
            self._prompts = {
                prompt["name"]: MCPPrompt(
                    name=prompt["name"],
                    description=prompt.get("description"),
                    arguments=prompt.get("arguments", []),
                    server_name=self.name
                )
                for prompt in result.get("prompts", [])
            }
        except Exception as e:
            logger.debug(f"Prompts not supported by server '{self.name}': {e}")
    
    async def call_tool(self, name: str, arguments: dict = None) -> Any:
        """Call an MCP tool"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in server '{self.name}'")
        
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments or {}
        })
        return result
    
    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource"""
        result = await self._send_request("resources/read", {"uri": uri})
        return result
    
    async def get_prompt(self, name: str, arguments: dict = None) -> Any:
        """Get an MCP prompt"""
        result = await self._send_request("prompts/get", {
            "name": name,
            "arguments": arguments or {}
        })
        return result
    
    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools.values())
    
    @property
    def resources(self) -> list[MCPResource]:
        return list(self._resources.values())
    
    @property
    def prompts(self) -> list[MCPPrompt]:
        return list(self._prompts.values())


class MCPClientManager:
    """
    Manages multiple MCP server connections with unified tool access
    """
    
    def __init__(self):
        self._clients: dict[str, MCPServerClient] = {}
        self._tool_mapping: dict[str, str] = {}  # tool_name -> server_name
    
    async def add_server(
        self,
        name: str,
        command: str = None,
        args: list = None,
        env: dict = None,
        url: str = None,
        transport_type: MCPTransport = MCPTransport.STDIO,
        timeout: float = 30.0
    ) -> None:
        """Add and connect to an MCP server"""
        if transport_type == MCPTransport.STDIO:
            if not command:
                raise ValueError("Command required for STDIO transport")
            transport = StdioTransport(command, args, env)
        elif transport_type == MCPTransport.SSE:
            if not url:
                raise ValueError("URL required for SSE transport")
            transport = SSETransport(url)
        else:
            raise ValueError(f"Unsupported transport: {transport_type}")
        
        client = MCPServerClient(name, transport, timeout)
        await client.connect()
        
        self._clients[name] = client
        
        # Update tool mapping
        for tool in client.tools:
            self._tool_mapping[tool.name] = name
        
        logger.info(f"Added MCP server '{name}' with {len(client.tools)} tools")
    
    async def remove_server(self, name: str) -> None:
        """Disconnect and remove an MCP server"""
        if name in self._clients:
            await self._clients[name].disconnect()
            del self._clients[name]
            
            # Update tool mapping
            self._tool_mapping = {
                k: v for k, v in self._tool_mapping.items() if v != name
            }
    
    async def call_tool(self, name: str, arguments: dict = None) -> Any:
        """Call a tool by name (routes to correct server)"""
        if name not in self._tool_mapping:
            raise ValueError(f"Tool '{name}' not found in any connected server")
        
        server_name = self._tool_mapping[name]
        return await self._clients[server_name].call_tool(name, arguments)
    
    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers"""
        tools = []
        for client in self._clients.values():
            tools.extend(client.tools)
        return tools
    
    def get_all_resources(self) -> list[MCPResource]:
        """Get all resources from all connected servers"""
        resources = []
        for client in self._clients.values():
            resources.extend(client.resources)
        return resources
    
    def get_langchain_tools(self) -> list:
        """Convert MCP tools to LangChain tool format"""
        from langchain_core.tools import StructuredTool
        
        lc_tools = []
        for tool in self.get_all_tools():
            async def call_fn(arguments: dict = None, tool_name=tool.name) -> str:
                result = await self.call_tool(tool_name, arguments)
                return json.dumps(result) if isinstance(result, dict) else str(result)
            
            lc_tool = StructuredTool.from_function(
                func=lambda **kwargs, tn=tool.name: asyncio.run(self.call_tool(tn, kwargs)),
                name=tool.name,
                description=tool.description,
                args_schema=None  # Will be inferred from input_schema
            )
            lc_tools.append(lc_tool)
        
        return lc_tools
    
    async def close_all(self) -> None:
        """Disconnect from all servers"""
        for name in list(self._clients.keys()):
            await self.remove_server(name)
    
    @asynccontextmanager
    async def session(self):
        """Context manager for MCP client session"""
        try:
            yield self
        finally:
            await self.close_all()


# Factory function for creating MCP client from config
async def create_mcp_client_from_config(config: dict) -> MCPClientManager:
    """Create MCPClientManager from configuration dictionary"""
    manager = MCPClientManager()
    
    for server_config in config.get("servers", []):
        if not server_config.get("enabled", True):
            continue
            
        await manager.add_server(
            name=server_config["name"],
            command=server_config.get("command"),
            args=server_config.get("args", []),
            env=server_config.get("env", {}),
            url=server_config.get("url"),
            transport_type=MCPTransport(server_config.get("transport", "stdio")),
            timeout=server_config.get("timeout", 30.0)
        )
    
    return manager
