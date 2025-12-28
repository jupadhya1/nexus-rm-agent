"""
Advanced Configuration for RM Assistant Agent with MCP Integration
Supports multi-agent orchestration, compliance tools, and enterprise features
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import os


class AgentMode(Enum):
    """Agent execution modes"""
    SINGLE = "single"
    MULTI_AGENT = "multi_agent"
    SUPERVISOR = "supervisor"
    HIERARCHICAL = "hierarchical"


class ComplianceLevel(Enum):
    """Compliance strictness levels"""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    STRICT = "strict"


@dataclass
class DatabricksConfig:
    """Databricks connection configuration"""
    catalog: str = "workspace"
    schema: str = "rm_agent"
    vector_search_endpoint: str = "vs_endpoint"
    llm_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    embedding_endpoint: str = "databricks-gte-large-en"
    
    @property
    def full_schema(self) -> str:
        return f"{self.catalog}.{self.schema}"


@dataclass
class MCPServerConfig:
    """MCP Server Configuration"""
    name: str
    transport: str = "stdio"  # stdio, sse, websocket
    command: Optional[str] = None
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    url: Optional[str] = None  # For SSE/WebSocket transports
    enabled: bool = True


@dataclass
class MCPConfig:
    """Model Context Protocol Configuration"""
    enabled: bool = True
    servers: list = field(default_factory=list)
    tool_timeout_seconds: int = 30
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    
    def __post_init__(self):
        if not self.servers:
            self.servers = self._default_servers()
    
    def _default_servers(self) -> list:
        """Default MCP server configurations"""
        return [
            MCPServerConfig(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
            ),
            MCPServerConfig(
                name="sqlite",
                command="uvx",
                args=["mcp-server-sqlite", "--db-path", "/data/rm_agent.db"],
            ),
            MCPServerConfig(
                name="memory",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
            ),
        ]


@dataclass
class AgentConfig:
    """Agent orchestration configuration"""
    mode: AgentMode = AgentMode.MULTI_AGENT
    max_iterations: int = 15
    max_execution_time_seconds: int = 120
    enable_human_in_loop: bool = False
    enable_checkpointing: bool = True
    checkpoint_namespace: str = "rm_agent"
    
    # Sub-agent configurations
    enable_customer_agent: bool = True
    enable_product_agent: bool = True
    enable_compliance_agent: bool = True
    enable_analytics_agent: bool = True
    enable_document_agent: bool = True
    
    # LLM settings per agent type
    supervisor_model: str = "databricks-meta-llama-3-3-70b-instruct"
    worker_model: str = "databricks-meta-llama-3-1-8b-instruct"
    
    # Parallel execution
    enable_parallel_tools: bool = True
    max_parallel_calls: int = 5


@dataclass
class ComplianceConfig:
    """Compliance and audit configuration"""
    level: ComplianceLevel = ComplianceLevel.ENHANCED
    enable_audit_logging: bool = True
    enable_pii_detection: bool = True
    enable_data_masking: bool = True
    enable_regulatory_checks: bool = True
    
    # Regulatory frameworks
    mas_tn_enabled: bool = True  # MAS Technology Notice
    pdpa_enabled: bool = True     # Personal Data Protection Act
    fatf_enabled: bool = True     # FATF AML/CFT
    
    # Risk thresholds
    high_value_transaction_threshold: float = 100000.0
    suspicious_activity_threshold: float = 50000.0
    pep_screening_enabled: bool = True
    sanctions_screening_enabled: bool = True


@dataclass
class VectorSearchConfig:
    """Vector search configuration"""
    index_name: str = "ut_pdf_docs_vs_index"
    embedding_dimension: int = 1024
    similarity_metric: str = "cosine"
    num_results: int = 5
    score_threshold: float = 0.7
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    backend: str = "redis"  # redis, memory, disk
    redis_url: str = "redis://localhost:6379"
    default_ttl_seconds: int = 3600
    max_entries: int = 10000
    
    # Semantic cache for LLM responses
    semantic_cache_enabled: bool = True
    semantic_similarity_threshold: float = 0.95


@dataclass 
class ObservabilityConfig:
    """Observability and monitoring configuration"""
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    
    # MLflow integration
    mlflow_tracking_uri: str = "databricks"
    mlflow_experiment_name: str = "/Shared/rm_agent_experiments"
    
    # LangSmith integration (optional)
    langsmith_enabled: bool = False
    langsmith_project: str = "rm-agent"
    
    # OpenTelemetry
    otel_enabled: bool = True
    otel_endpoint: str = "http://localhost:4317"


@dataclass
class Settings:
    """Main settings container"""
    databricks: DatabricksConfig = field(default_factory=DatabricksConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    vector_search: VectorSearchConfig = field(default_factory=VectorSearchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables"""
        settings = cls()
        
        # Override from environment
        settings.databricks.catalog = os.getenv("DATABRICKS_CATALOG", settings.databricks.catalog)
        settings.databricks.schema = os.getenv("DATABRICKS_SCHEMA", settings.databricks.schema)
        settings.environment = os.getenv("ENVIRONMENT", settings.environment)
        settings.debug = os.getenv("DEBUG", "true").lower() == "true"
        
        return settings


# Global settings instance
settings = Settings.from_env()
