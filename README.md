# Functional Specification: Enhanced RM Assistant Agent with MCP Integration

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Module Specifications](#4-module-specifications)
5. [Data Flow & Processing](#5-data-flow--processing)
6. [Integration Points](#6-integration-points)
7. [Security & Compliance](#7-security--compliance)
8. [Deployment Architecture](#8-deployment-architecture)
9. [API Reference](#9-api-reference)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Executive Summary

### 1.1 Purpose

The Enhanced RM Assistant Agent is an enterprise-grade AI system designed to augment Relationship Managers (RMs) in Corporate & Institutional Banking (CIB) operations. It provides intelligent automation for:

- Customer profile management and risk assessment
- Investment product recommendations with suitability checks
- Regulatory compliance verification (KYC, AML, Sanctions)
- Risk and Control Self-Assessment (RCSA) automation
- Document analysis and information retrieval

### 1.2 Key Differentiators

| Feature | Description | Business Value |
|---------|-------------|----------------|
| **Multi-Agent Orchestration** | Supervisor pattern with specialized agents | Complex query handling, domain expertise |
| **MCP Integration** | Model Context Protocol for extensible tools | Future-proof architecture, third-party integrations |
| **Compliance-First Design** | Built-in audit logging, PII masking | Regulatory adherence, audit readiness |
| **Enterprise Caching** | Semantic + LRU caching layers | Reduced latency, cost optimization |

### 1.3 Target Users

- **Primary:** Relationship Managers (RMs) in CIB
- **Secondary:** Compliance Officers, Risk Analysts
- **Tertiary:** Operations Support, Product Teams

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Streamlit   │  │   AI        │  │    REST     │  │   Slack/     │   │
│  │  Web App     │  │  Playground │  │    API      │  │   Teams Bot  │   │
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘   │
│         └──────────────────┼────────────────┼────────────────┘           │
│                            ▼                ▼                             │
├────────────────────────────────────────────────────────────────────────────┤
│                           GATEWAY LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                 Databricks Model Serving Endpoint                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │ Rate Limit  │  │   Auth &    │  │   Request   │                  │  │
│  │  │   Control   │  │   AuthZ     │  │   Routing   │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│                         ORCHESTRATION LAYER                                │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     SUPERVISOR AGENT                                 │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │  • Query Analysis & Intent Classification                      │  │  │
│  │  │  • Agent Selection & Task Routing                              │  │  │
│  │  │  • Response Synthesis & Quality Control                        │  │  │
│  │  │  • Conversation State Management                               │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│         ┌────────────┬─────────────┼─────────────┬────────────┐           │
│         ▼            ▼             ▼             ▼            ▼           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │ Customer  │ │ Product   │ │Compliance │ │ Analytics │ │ Document  │   │
│  │Specialist │ │Specialist │ │ Officer   │ │  Expert   │ │ Analyst   │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │
│        └─────────────┴─────────────┴─────────────┴─────────────┘         │
│                                    │                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                            TOOL LAYER                                      │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  UC Function   │  │   MCP Tool     │  │   Native       │               │
│  │    Tools       │  │   Adapters     │  │   Tools        │               │
│  │                │  │                │  │                │               │
│  │ • lookup_cust  │  │ • Compliance   │  │ • Currency     │               │
│  │ • lookup_ut    │  │   Server       │  │   Conversion   │               │
│  │ • vector_search│  │ • RCSA Server  │  │ • Portfolio    │               │
│  │ • convert_usd  │  │ • Document     │  │   Analysis     │               │
│  │                │  │   Server       │  │ • Risk Calc    │               │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘               │
│          └───────────────────┼───────────────────┘                        │
│                              ▼                                             │
├────────────────────────────────────────────────────────────────────────────┤
│                            DATA LAYER                                      │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  Unity Catalog │  │  Vector Search │  │   External     │               │
│  │    Tables      │  │     Index      │  │   Systems      │               │
│  │                │  │                │  │                │               │
│  │ • customer_    │  │ • ut_pdf_docs  │  │ • Sanctions    │               │
│  │   profile      │  │   _vs_index    │  │   Lists        │               │
│  │ • unit_trust   │  │                │  │ • PEP Database │               │
│  │ • audit_log    │  │                │  │ • FX Rates     │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | LangGraph | Multi-agent workflow management |
| LLM | Llama 3.3 70B | Natural language understanding and generation |
| Vector Search | Databricks VS | Semantic document retrieval |
| Tools | Unity Catalog Functions | Governed data access |
| MCP | Custom Implementation | Extensible tool integration |
| Storage | Delta Tables | Structured data persistence |
| Caching | Custom LRU + Semantic | Performance optimization |

---

## 3. Architecture Deep Dive

### 3.1 Multi-Agent Orchestration

#### 3.1.1 Supervisor Agent

The Supervisor Agent is the central coordinator that:

1. **Analyzes incoming queries** to determine intent and complexity
2. **Selects appropriate specialist agents** based on task requirements
3. **Manages conversation state** across multi-turn interactions
4. **Synthesizes final responses** from specialist outputs

**Decision Logic:**

```python
class SupervisorDecision:
    """
    Supervisor routing decision structure
    """
    reasoning: str          # Explanation of routing decision
    next_agent: str         # Target agent or "FINISH"
    instructions: str       # Specific instructions for agent
    requires_tools: list    # Expected tools to be used
```

**Routing Rules:**

| Query Pattern | Target Agent | Example |
|---------------|--------------|---------|
| Customer lookup, profile, balance | customer_specialist | "Look up Brian Long's profile" |
| Product recommendation, unit trust, fund | product_specialist | "Recommend funds for risk level 3" |
| KYC, AML, sanctions, compliance | compliance_officer | "Check sanctions status for client" |
| Calculate, analyze, convert, metrics | analytics_expert | "Convert 100 GBP to USD" |
| Document, factsheet, summary | document_analyst | "Summarize the Global Growth Fund" |
| Complex multi-domain | Sequential routing | "Recommend compliant products for customer X" |

#### 3.1.2 Specialist Agents

Each specialist agent has:
- **Domain-specific system prompt** defining expertise boundaries
- **Curated tool access** limited to relevant functions
- **Output formatting guidelines** for consistent responses

**Customer Specialist Agent:**
```
Responsibilities:
├── Customer profile retrieval
├── Risk rating interpretation
├── Account balance queries
├── Investment history analysis
└── Cross-sell opportunity identification

Tools Available:
├── lookup_customer_profile
├── assess_customer_risk
└── analyze_portfolio
```

**Product Specialist Agent:**
```
Responsibilities:
├── Unit trust information retrieval
├── Product comparison and ranking
├── Risk-return analysis
├── Suitability matching
└── Fee structure explanation

Tools Available:
├── search_investment_products
├── compare_products
├── lookup_unit_trust
└── search_documents
```

**Compliance Officer Agent:**
```
Responsibilities:
├── KYC status verification
├── Sanctions screening
├── PEP identification
├── AML risk assessment
├── Regulatory limit checking
└── SAR report generation

Tools Available:
├── check_compliance_status
├── assess_product_suitability
├── screen_customer_sanctions
├── check_pep_status
└── check_regulatory_limits
```

**Analytics Expert Agent:**
```
Responsibilities:
├── Numerical calculations
├── Currency conversions
├── Portfolio metrics
├── Risk score computation
├── Statistical analysis
└── Trend identification

Tools Available:
├── convert_currency
├── analyze_portfolio
├── calculate_residual_risk
└── Custom calculation functions
```

**Document Analyst Agent:**
```
Responsibilities:
├── Document search and retrieval
├── Content summarization
├── Entity extraction
├── Document comparison
├── Table data extraction
└── Factsheet analysis

Tools Available:
├── search_documents
├── extract_document_entities
├── summarize_document
├── compare_documents
└── extract_table_data
```

### 3.2 MCP (Model Context Protocol) Integration

#### 3.2.1 MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Client Manager                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ MCP Server 1  │  │ MCP Server 2  │  │ MCP Server N  │        │
│  │ (Compliance)  │  │ (RCSA)        │  │ (Documents)   │        │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘        │
│          │                  │                  │                 │
│          └──────────────────┼──────────────────┘                 │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Transport Layer                            ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ││
│  │  │   STDIO     │  │    SSE      │  │  WebSocket  │          ││
│  │  │  Transport  │  │  Transport  │  │  Transport  │          ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘          ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 MCP Protocol Handler                         ││
│  │  • JSON-RPC 2.0 Message Processing                          ││
│  │  • Request/Response Correlation                              ││
│  │  • Notification Handling                                     ││
│  │  • Capability Negotiation                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Tool Registry                                ││
│  │  • Tool Discovery & Registration                             ││
│  │  • Schema Conversion (MCP → LangChain)                       ││
│  │  • Execution Routing                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 MCP Protocol Implementation

**Connection Lifecycle:**

```
1. INITIALIZE
   Client → Server: initialize(protocolVersion, capabilities, clientInfo)
   Server → Client: {serverInfo, capabilities}
   Client → Server: notifications/initialized

2. DISCOVER
   Client → Server: tools/list
   Server → Client: {tools: [{name, description, inputSchema}]}
   
   Client → Server: resources/list
   Server → Client: {resources: [{uri, name, mimeType}]}

3. EXECUTE
   Client → Server: tools/call(name, arguments)
   Server → Client: {content: [{type, text}]}

4. TERMINATE
   Client closes transport connection
```

**Message Format:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "check_customer_kyc_status",
    "arguments": {
      "customer_id": "CUST0001"
    }
  }
}
```

#### 3.2.3 Banking MCP Servers

**Compliance Tool Server:**

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `check_customer_kyc_status` | Verify KYC status | customer_id | KYC status, documents, next review |
| `screen_customer_sanctions` | Screen against sanctions lists | name, country, dob | Match status, lists checked |
| `check_pep_status` | Check PEP status | name, customer_id | PEP category, related PEP |
| `calculate_customer_risk_score` | Calculate risk score | customer_id, include_transactions | Score, breakdown, recommendations |
| `generate_sar_report` | Generate SAR draft | customer_id, transaction_ids, narrative | Report ID, template sections |
| `check_regulatory_limits` | Check reporting thresholds | amount, currency, jurisdiction | Alerts, reporting requirements |

**RCSA Tool Server:**

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `assess_control_effectiveness` | Assess control effectiveness | control_id, period | Rating, score, findings |
| `identify_control_gaps` | Find control gaps | process_id, risk_category | Gaps, coverage analysis |
| `generate_rcsa_summary` | Generate RCSA report | business_unit, period | Summary, KRIs, remediation |
| `map_risks_to_controls` | Map risks to controls | risk_ids | Mappings, coverage status |
| `calculate_residual_risk` | Calculate residual risk | inherent_risk, control_effectiveness | Residual score, level |

**Document Analysis Server:**

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `extract_document_entities` | Extract entities from text | document_content, entity_types | Entities by type |
| `compare_documents` | Compare two documents | document_a, document_b, type | Similarity score, differences |
| `classify_document` | Classify document type | document_content, scheme | Type, confidence, categories |
| `extract_table_data` | Extract table data | document_content, format | Tables, rows, columns |
| `summarize_document` | Summarize document | document_content, length, focus | Summary, key points |

### 3.3 Tool Layer Architecture

#### 3.3.1 Unity Catalog Function Tools

**Function Registration Pattern:**

```sql
CREATE OR REPLACE FUNCTION {catalog}.{schema}.{function_name}(
    param1 TYPE COMMENT 'Parameter description'
)
RETURNS {return_type}
LANGUAGE {SQL|PYTHON}
COMMENT 'Function description for LLM'
AS
$$
    -- Implementation
$$;
```

**Registered Functions:**

| Function | Type | Description |
|----------|------|-------------|
| `unit_trust_vector_search` | SQL | Vector search on UT documents |
| `lookup_customer_info` | SQL | Customer profile lookup |
| `lookup_ut_info` | SQL | Unit trust details by risk |
| `convert_to_usd` | Python | Currency conversion |

#### 3.3.2 Native LangChain Tools

**Tool Implementation Pattern:**

```python
class BankingTool(BaseTool):
    name: str = "tool_name"
    description: str = "Tool description for LLM"
    args_schema: Type[BaseModel] = ToolArgsModel
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution"""
        pass
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution"""
        pass
```

**Native Tools:**

| Tool | Category | Purpose |
|------|----------|---------|
| `CustomerProfileTool` | Customer | Profile lookup with history |
| `CustomerRiskAssessmentTool` | Customer | Comprehensive risk assessment |
| `ProductSearchTool` | Product | Semantic product search |
| `ProductComparisonTool` | Product | Side-by-side comparison |
| `ComplianceCheckTool` | Compliance | Multi-type compliance check |
| `SuitabilityAssessmentTool` | Compliance | Product suitability |
| `PortfolioAnalysisTool` | Analytics | Portfolio metrics |
| `CurrencyConversionTool` | Analytics | FX conversion |
| `DocumentSearchTool` | Document | Semantic doc search |

---

## 4. Module Specifications

### 4.1 Configuration Module (`config/`)

#### 4.1.1 Settings Classes

```python
@dataclass
class Settings:
    databricks: DatabricksConfig      # Catalog, schema, endpoints
    mcp: MCPConfig                    # MCP server configurations
    agent: AgentConfig                # Agent behavior settings
    compliance: ComplianceConfig      # Compliance thresholds
    vector_search: VectorSearchConfig # VS settings
    cache: CacheConfig                # Caching parameters
    observability: ObservabilityConfig # Logging, tracing
```

**Key Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent.max_iterations` | 15 | Maximum agent loop iterations |
| `agent.max_execution_time_seconds` | 120 | Timeout for agent execution |
| `compliance.high_value_threshold` | 100,000 | High-value transaction threshold |
| `cache.default_ttl_seconds` | 3600 | Cache entry lifetime |
| `vector_search.num_results` | 5 | Default VS results |

### 4.2 MCP Module (`mcp/`)

#### 4.2.1 Client Classes

```
mcp/
├── client.py
│   ├── MCPTransport (ABC)
│   │   ├── StdioTransport
│   │   └── SSETransport
│   ├── MCPServerClient
│   │   ├── connect()
│   │   ├── call_tool()
│   │   ├── read_resource()
│   │   └── get_prompt()
│   └── MCPClientManager
│       ├── add_server()
│       ├── remove_server()
│       ├── call_tool()
│       └── get_all_tools()
│
├── tools.py
│   ├── MCPToolWrapper (BaseTool)
│   ├── MCPToolRegistry
│   ├── MCPResourceTool
│   └── create_mcp_toolkit()
│
└── banking_servers.py
    ├── ComplianceToolServer
    ├── RCSAToolServer
    └── DocumentAnalysisToolServer
```

#### 4.2.2 Tool Adapter Flow

```
MCP Tool Definition          LangChain Tool
┌─────────────────┐         ┌─────────────────┐
│ name: "tool_x"  │         │ name: "tool_x"  │
│ description:... │   ───►  │ description:... │
│ inputSchema: {  │         │ args_schema:    │
│   properties:{} │         │   PydanticModel │
│ }               │         │ _run(): str     │
└─────────────────┘         └─────────────────┘
```

### 4.3 Agents Module (`agents/`)

#### 4.3.1 Class Hierarchy

```
agents/
├── base.py
│   ├── AgentRole (Enum)
│   ├── AgentConfig (dataclass)
│   ├── AgentState (dataclass)
│   └── BaseAgent
│       ├── __init__(config, llm, tools)
│       ├── invoke(state) → dict
│       └── _setup_prompt()
│
├── specialists.py
│   ├── CustomerSpecialistAgent(BaseAgent)
│   ├── ProductSpecialistAgent(BaseAgent)
│   ├── ComplianceOfficerAgent(BaseAgent)
│   ├── AnalyticsExpertAgent(BaseAgent)
│   ├── DocumentAnalystAgent(BaseAgent)
│   ├── RCSASpecialistAgent(BaseAgent)
│   └── KYCSpecialistAgent(BaseAgent)
│
└── orchestrator.py
    ├── SupervisorAgent
    │   ├── __init__(agents, llm)
    │   └── route(state) → dict
    └── MultiAgentOrchestrator
        ├── __init__(model_endpoint, tools)
        ├── run(messages) → dict
        ├── stream(messages) → Generator
        └── get_graph_visualization() → str
```

#### 4.3.2 State Schema

```python
class GraphState(TypedDict):
    messages: Annotated[Sequence[dict], add]  # Conversation messages
    current_agent: str                         # Active agent name
    context: dict                              # Shared context
    agent_history: Annotated[list, add]       # Agent execution log
    completed_tasks: Annotated[list, add]     # Completed task log
    iteration_count: int                       # Loop counter
```

### 4.4 Tools Module (`tools/`)

#### 4.4.1 Banking Tools

```python
# Customer Tools
CustomerProfileTool      # Full profile with optional history
CustomerRiskAssessmentTool  # Risk score + compliance + transactions

# Product Tools
ProductSearchTool        # Semantic search with filters
ProductComparisonTool    # Multi-product comparison

# Compliance Tools
ComplianceCheckTool      # KYC/AML/Sanctions/PEP checks
SuitabilityAssessmentTool  # Product-customer suitability

# Analytics Tools
PortfolioAnalysisTool    # Allocation, risk metrics, recommendations
CurrencyConversionTool   # Multi-currency conversion

# Document Tools
DocumentSearchTool       # Vector search on documents
```

#### 4.4.2 UC Tools Generator

```python
class VectorSearchToolGenerator:
    """Generate tools from VS indexes"""
    create_search_tool(index_name, tool_name, description)
    create_hybrid_search_tool(index_name, table_name, ...)

class UCTableToolGenerator:
    """Generate tools from UC tables"""
    create_lookup_tool(table_name, lookup_column, ...)
    create_aggregation_tool(table_name, group_by, ...)

class GraphRAGToolGenerator:
    """Generate tools for knowledge graph queries"""
    create_entity_search_tool(entity_type, ...)
    create_path_finding_tool(...)
```

### 4.5 Utils Module (`utils/`)

#### 4.5.1 Caching

```python
class LRUCache:
    """Thread-safe LRU cache with TTL"""
    get(key) → Optional[Any]
    set(key, value) → None
    stats → dict  # hits, misses, hit_rate

class SemanticCache:
    """Cache with embedding similarity matching"""
    get(query) → Optional[Any]  # Exact + semantic match
    set(query, response) → None
```

#### 4.5.2 Logging

```python
class AuditLogger:
    """Compliance audit logging"""
    log(action, user_id, customer_id, details)
    log_tool_call(tool_name, arguments, result)
    log_customer_access(customer_id, access_type, user_id)
    get_entries(filters) → list[AuditLogEntry]

class StructuredLogger:
    """JSON structured logging"""
    set_context(**kwargs)  # Persistent fields
    info/warning/error/debug(message, **extra)
```

#### 4.5.3 Utilities

```python
# Decorators
@cached(ttl_seconds=3600)      # Sync function caching
@async_cached(ttl_seconds=3600) # Async function caching
@timed                          # Execution timing
@retry(max_attempts=3)          # Retry with backoff
@rate_limited(limiter, tokens)  # Rate limiting

# Classes
DataMasker        # PII masking in logs/responses
RateLimiter       # Token bucket rate limiter
```

---

## 5. Data Flow & Processing

### 5.1 Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY PROCESSING FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

User Query: "What unit trusts would you recommend for customer Brian Long?"

Step 1: INTAKE
┌─────────────────────────────────────────────────────────────────────────┐
│ • Receive query via API/UI                                              │
│ • Validate input format                                                 │
│ • Check rate limits                                                     │
│ • Create audit log entry                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 2: SUPERVISOR ANALYSIS
┌─────────────────────────────────────────────────────────────────────────┐
│ Supervisor Agent Analysis:                                              │
│ • Intent: Product recommendation                                        │
│ • Entities: customer_name="Brian Long"                                  │
│ • Required data: Customer profile, risk rating                          │
│ • Required checks: Suitability, compliance                              │
│                                                                         │
│ Decision: Route to customer_specialist first                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 3: CUSTOMER SPECIALIST
┌─────────────────────────────────────────────────────────────────────────┐
│ Tool Call: lookup_customer_profile(customer_name="Brian Long")          │
│                                                                         │
│ Result:                                                                 │
│ {                                                                       │
│   "customer_id": "CUST0006",                                           │
│   "name": "Brian Long",                                                │
│   "risk_rating": 3,                                                    │
│   "account_balance": 9104.05,                                          │
│   "customer_since": "2020-03-24"                                       │
│ }                                                                       │
│                                                                         │
│ Agent Response: "Customer Brian Long has risk rating 3 (moderate)"      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 4: SUPERVISOR RE-ROUTING
┌─────────────────────────────────────────────────────────────────────────┐
│ Context Updated: customer_risk_rating = 3                               │
│ Decision: Route to product_specialist                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 5: PRODUCT SPECIALIST
┌─────────────────────────────────────────────────────────────────────────┐
│ Tool Call: search_investment_products(                                  │
│   query="suitable unit trusts",                                        │
│   max_risk_rating=3                                                    │
│ )                                                                       │
│                                                                         │
│ Result:                                                                 │
│ {                                                                       │
│   "products": [                                                        │
│     {"id": "UT02", "name": "Income Opportunity Fund", "risk": 3},      │
│     {"id": "UT03", "name": "Balanced Advantage Fund", "risk": 3},      │
│     {"id": "UT04", "name": "Equity Income Fund", "risk": 2}            │
│   ]                                                                     │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 6: SUPERVISOR RE-ROUTING
┌─────────────────────────────────────────────────────────────────────────┐
│ Context Updated: product_recommendations available                      │
│ Decision: Route to compliance_officer for suitability check             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 7: COMPLIANCE OFFICER
┌─────────────────────────────────────────────────────────────────────────┐
│ Tool Call: check_compliance_status(customer_id="CUST0006")              │
│                                                                         │
│ Result: { "kyc_status": "verified", "overall_status": "CLEAR" }         │
│                                                                         │
│ Tool Call: assess_product_suitability(                                  │
│   customer_id="CUST0006",                                              │
│   product_id="UT02"                                                    │
│ )                                                                       │
│                                                                         │
│ Result: { "suitability_status": "SUITABLE", "risk_match": true }        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 8: SUPERVISOR SYNTHESIS
┌─────────────────────────────────────────────────────────────────────────┐
│ All required information gathered                                       │
│ Decision: FINISH - Synthesize final response                            │
│                                                                         │
│ Final Response:                                                         │
│ "Based on Brian Long's moderate risk profile (rating 3), I recommend:   │
│                                                                         │
│ 1. **Income Opportunity Fund (UT02)** - Balanced fund with 0.99%        │
│    expense ratio, suitable for moderate risk tolerance                  │
│                                                                         │
│ 2. **Balanced Advantage Fund (UT03)** - Balanced fund with 0.55%        │
│    expense ratio, good for capital preservation with growth             │
│                                                                         │
│ 3. **Equity Income Fund (UT04)** - Lower risk (rating 2), bond fund     │
│    with 1.01% expense ratio for conservative allocation                 │
│                                                                         │
│ All recommendations have passed suitability checks. Customer's KYC      │
│ status is verified and current."                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tool Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOL EXECUTION FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   Agent decides     │
                    │   to call tool      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Tool Router       │
                    │   identifies tool   │
                    │   type              │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  UC Function │     │  MCP Tool   │     │ Native Tool │
    │    Tool      │     │             │     │             │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Execute via │     │ Send MCP    │     │ Execute     │
    │ Spark SQL   │     │ tools/call  │     │ Python      │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Query UC    │     │ MCP Server  │     │ Local       │
    │ Table/Index │     │ processes   │     │ computation │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Result Formatter  │
                    │   (JSON output)     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Audit Logger      │
                    │   records call      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Return to Agent   │
                    └─────────────────────┘
```

---

## 6. Integration Points

### 6.1 Databricks Platform Integration

| Integration | Method | Purpose |
|-------------|--------|---------|
| Unity Catalog | SQL Functions | Governed tool execution |
| Vector Search | REST API | Semantic document retrieval |
| Model Serving | MLflow | Agent deployment |
| Delta Tables | Spark SQL | Structured data access |
| Secrets | dbutils.secrets | Credential management |
| Apps | Streamlit | Custom UI deployment |

### 6.2 External System Integration

| System | Protocol | Use Case |
|--------|----------|----------|
| Sanctions Lists | REST/Batch | OFAC, UN, EU screening |
| PEP Database | REST | Political exposure check |
| FX Rate Service | REST | Live exchange rates |
| Document Storage | S3/ADLS | PDF factsheet access |
| Audit System | Kafka/REST | Compliance log shipping |

### 6.3 MCP Server Integration

```yaml
# MCP Server Configuration
servers:
  - name: compliance
    transport: stdio
    command: python
    args: ["-m", "compliance_mcp_server"]
    
  - name: external_data
    transport: sse
    url: https://mcp.internal.bank.com/data
    headers:
      Authorization: Bearer ${MCP_TOKEN}
```

---

## 7. Security & Compliance

### 7.1 Authentication & Authorization

| Layer | Mechanism | Details |
|-------|-----------|---------|
| API Gateway | OAuth 2.0 / SAML | SSO integration |
| Model Serving | Service Principal | Databricks auth passthrough |
| UC Functions | ACLs | Function-level permissions |
| Data Tables | Row-level security | Customer data isolation |

### 7.2 Data Protection

**PII Handling:**
```python
# Automatic masking in logs
DataMasker.mask("Email: john@bank.com") 
# → "Email: jo**********om"

# Sensitive field detection
DataMasker.mask_dict({
    "customer_id": "CUST001",
    "ssn": "123-45-6789",
    "account_number": "1234567890"
})
# → {"customer_id": "CUST001", "ssn": "***MASKED***", "account_number": "***MASKED***"}
```

**Encryption:**
- Data at rest: AES-256 (Delta Lake encryption)
- Data in transit: TLS 1.3
- Secrets: Databricks Secrets with scope isolation

### 7.3 Audit Trail

**Logged Events:**
- All tool invocations with parameters
- Customer data access with user attribution
- Compliance check results
- Agent routing decisions
- Response generation

**Audit Log Schema:**
```json
{
  "timestamp": "2024-12-28T10:30:00Z",
  "action": "tool_call",
  "user_id": "rm_user_123",
  "customer_id": "CUST0006",
  "details": {
    "tool": "lookup_customer_profile",
    "arguments": {"customer_name": "Brian Long"},
    "result_summary": "Profile retrieved successfully",
    "execution_time_ms": 145
  },
  "status": "success",
  "session_id": "sess_abc123"
}
```

### 7.4 Regulatory Compliance

| Regulation | Implementation |
|------------|----------------|
| MAS TRM | Audit logging, access controls, data protection |
| PDPA | PII masking, consent tracking, data minimization |
| AML/CFT | Transaction monitoring, SAR generation, sanctions screening |
| FAA | Suitability assessment, risk disclosure |

---

## 8. Deployment Architecture

### 8.1 Databricks Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATABRICKS WORKSPACE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    MODEL SERVING LAYER                              │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │              rm-agent-enhanced-endpoint                       │  │ │
│  │  │  • Workload: Small (auto-scale)                              │  │ │
│  │  │  • Scale to Zero: Enabled                                     │  │ │
│  │  │  • Concurrency: 4 requests                                    │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    UNITY CATALOG                                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │   Models    │  │  Functions  │  │   Tables    │                 │ │
│  │  │             │  │             │  │             │                 │ │
│  │  │ rm_agent_   │  │ lookup_*    │  │ customer_   │                 │ │
│  │  │ enhanced    │  │ convert_*   │  │ profile     │                 │ │
│  │  │             │  │ vector_*    │  │ unit_trust  │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    VECTOR SEARCH                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  Index: ut_pdf_docs_vs_index                                  │  │ │
│  │  │  Endpoint: vs_endpoint                                        │  │ │
│  │  │  Embedding: databricks-gte-large-en                          │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    DATABRICKS APPS                                  │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  rm-assistant-app (Streamlit)                                 │  │ │
│  │  │  • Custom chat UI                                             │  │ │
│  │  │  • SSO authentication                                         │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 MLflow Model Registration

```python
# Model logging
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
            DatabricksVectorSearchIndex(index_name=VS_INDEX),
            DatabricksTable(table_name=CUSTOMER_TABLE),
            DatabricksFunction(function_name=UC_FUNCTION),
        ],
        pip_requirements=[...],
    )

# Model registration
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name="workspace.rm_agent.rm_agent_enhanced"
)
```

### 8.3 Scaling Configuration

| Parameter | Development | Production |
|-----------|-------------|------------|
| Workload Size | Small | Medium/Large |
| Min Instances | 0 | 1 |
| Max Instances | 1 | 10 |
| Scale to Zero | Yes | Optional |
| Concurrency | 4 | 8-16 |

---

## 9. API Reference

### 9.1 Chat Endpoint

**Request:**
```json
POST /serving-endpoints/{endpoint}/invocations

{
  "messages": [
    {"role": "user", "content": "Look up customer Brian Long"}
  ],
  "max_tokens": 500
}
```

**Response:**
```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "I found Brian Long's profile...",
      "tool_calls": [...]
    }
  ]
}
```

### 9.2 Streaming Endpoint

**Request:**
```json
POST /serving-endpoints/{endpoint}/invocations
Content-Type: application/json
Accept: text/event-stream

{
  "messages": [...],
  "stream": true
}
```

**Response (SSE):**
```
data: {"delta": {"content": "I"}}
data: {"delta": {"content": " found"}}
data: {"delta": {"content": " Brian"}}
...
data: [DONE]
```

---

## 10. Configuration Reference

### 10.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABRICKS_CATALOG` | Yes | workspace | Unity Catalog name |
| `DATABRICKS_SCHEMA` | Yes | rm_agent | Schema name |
| `LLM_ENDPOINT` | Yes | - | Model serving endpoint |
| `VS_ENDPOINT` | No | vs_endpoint | Vector search endpoint |
| `SERVING_ENDPOINT` | Yes | - | Agent serving endpoint |
| `LOG_LEVEL` | No | INFO | Logging verbosity |
| `CACHE_TTL_SECONDS` | No | 3600 | Cache lifetime |
| `MAX_ITERATIONS` | No | 15 | Agent loop limit |

### 10.2 Agent Configuration

```python
AgentConfig(
    mode=AgentMode.MULTI_AGENT,       # SINGLE, MULTI_AGENT, SUPERVISOR
    max_iterations=15,                 # Loop limit
    max_execution_time_seconds=120,    # Timeout
    enable_human_in_loop=False,        # HITL for sensitive ops
    enable_checkpointing=True,         # Conversation persistence
    supervisor_model="llama-3-3-70b",  # Orchestrator model
    worker_model="llama-3-1-8b",       # Specialist model
    enable_parallel_tools=True,        # Parallel tool execution
    max_parallel_calls=5               # Max concurrent tools
)
```

### 10.3 Compliance Configuration

```python
ComplianceConfig(
    level=ComplianceLevel.ENHANCED,
    enable_audit_logging=True,
    enable_pii_detection=True,
    enable_data_masking=True,
    enable_regulatory_checks=True,
    high_value_transaction_threshold=100000.0,
    suspicious_activity_threshold=50000.0,
    pep_screening_enabled=True,
    sanctions_screening_enabled=True
)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| RM | Relationship Manager - bank employee managing client relationships |
| CIB | Corporate & Institutional Banking division |
| MCP | Model Context Protocol - standard for LLM tool integration |
| UC | Unity Catalog - Databricks data governance layer |
| VS | Vector Search - semantic similarity search |
| KYC | Know Your Customer - identity verification |
| AML | Anti-Money Laundering - financial crime prevention |
| PEP | Politically Exposed Person - high-risk individual |
| SAR | Suspicious Activity Report - regulatory filing |
| RCSA | Risk and Control Self-Assessment |
| LangGraph | LangChain library for agent orchestration |

---

## Appendix B: Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2024 | Initial release with MCP integration |

---

*End of Functional Specification*
