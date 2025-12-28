"""
MCP (Model Context Protocol) Module
Provides comprehensive MCP integration for the RM Assistant Agent
"""

from .client import (
    MCPClientManager,
    MCPServerClient,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPTransport,
    StdioTransport,
    SSETransport,
    create_mcp_client_from_config,
)

from .tools import (
    MCPToolWrapper,
    MCPToolRegistry,
    MCPResourceTool,
    MCPPromptTool,
    create_mcp_toolkit,
    mcp_tool_to_pydantic_schema,
)

from .banking_servers import (
    ComplianceToolServer,
    RCSAToolServer,
    DocumentAnalysisToolServer,
    RiskLevel,
    TransactionType,
    CustomerRiskProfile,
    TransactionAlert,
)

__all__ = [
    # Client classes
    "MCPClientManager",
    "MCPServerClient",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPTransport",
    "StdioTransport",
    "SSETransport",
    "create_mcp_client_from_config",
    
    # Tool classes
    "MCPToolWrapper",
    "MCPToolRegistry",
    "MCPResourceTool",
    "MCPPromptTool",
    "create_mcp_toolkit",
    "mcp_tool_to_pydantic_schema",
    
    # Banking servers
    "ComplianceToolServer",
    "RCSAToolServer",
    "DocumentAnalysisToolServer",
    "RiskLevel",
    "TransactionType",
    "CustomerRiskProfile",
    "TransactionAlert",
]
