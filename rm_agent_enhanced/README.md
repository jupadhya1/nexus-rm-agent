# Enhanced RM Assistant Agent with MCP Integration

An advanced AI agent for Relationship Managers (RMs) built on Databricks with:
- **Multi-Agent Orchestration** using LangGraph supervisor pattern
- **MCP (Model Context Protocol)** integration for extended tool capabilities
- **Advanced Compliance Tools** for KYC, AML, and RCSA automation
- **Enterprise Features** including audit logging, caching, and observability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       RM Assistant System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │  Streamlit   │───▶│        Supervisor Agent              │   │
│  │     UI       │    │   (LangGraph Orchestration)          │   │
│  └──────────────┘    └──────────────┬───────────────────────┘   │
│                                      │                           │
│           ┌──────────────────────────┼──────────────────────┐   │
│           ▼                          ▼                      ▼   │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐ │
│  │   Customer     │    │    Product     │    │   Compliance   │ │
│  │  Specialist    │    │   Specialist   │    │    Officer     │ │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘ │
│          │                     │                     │          │
│          └─────────────────────┼─────────────────────┘          │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Tool Layer                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │UC Tools  │  │MCP Tools │  │Compliance│  │Analytics │  │   │
│  │  │(Functions│  │(External │  │  Tools   │  │  Tools   │  │   │
│  │  │ Tables)  │  │ Servers) │  │          │  │          │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Data Layer                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │ Customer │  │  Unit    │  │  Vector  │  │  Audit   │  │   │
│  │  │ Profiles │  │  Trusts  │  │  Index   │  │   Logs   │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
rm_agent_enhanced/
├── config/
│   └── settings.py          # Configuration management
├── mcp/
│   ├── __init__.py
│   ├── client.py            # MCP client implementation
│   ├── tools.py             # MCP-LangChain tool adapter
│   └── banking_servers.py   # Banking-specific MCP servers
├── agents/
│   ├── __init__.py
│   ├── base.py              # Base agent classes
│   ├── specialists.py       # Specialized agents
│   └── orchestrator.py      # Multi-agent orchestrator
├── tools/
│   ├── __init__.py
│   ├── banking_tools.py     # Banking-specific tools
│   └── uc_tools.py          # Unity Catalog integration
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Caching, logging, utilities
├── notebooks/
│   └── 01-Deploy-Enhanced-Agent.py
├── streamlit_app/
│   ├── app.py               # Enhanced Streamlit UI
│   ├── requirements.txt
│   └── app.yaml
├── agent_core.py            # Core agent implementation
├── agent_wrapper.py         # MLflow chat agent wrapper
└── README.md
```

## 🚀 Features

### Multi-Agent Orchestration
- **Supervisor Pattern**: Central coordinator routes tasks to specialized agents
- **Specialized Agents**:
  - Customer Specialist: Profile lookup, risk assessment
  - Product Specialist: Investment recommendations
  - Compliance Officer: KYC, AML, regulatory checks
  - Analytics Expert: Data analysis, calculations
  - Document Analyst: Document processing

### MCP Integration
- Full Model Context Protocol support
- Multiple transport types (STDIO, SSE)
- Tool, Resource, and Prompt management
- Banking-specific MCP servers

### Advanced Tools
- **Customer Tools**: Profile lookup, risk assessment, portfolio analysis
- **Product Tools**: Search, comparison, suitability assessment
- **Compliance Tools**: KYC check, sanctions screening, PEP verification
- **Analytics Tools**: Currency conversion, portfolio metrics

### Enterprise Features
- **Audit Logging**: Complete interaction tracking for compliance
- **Data Masking**: PII protection in logs and responses
- **Caching**: LRU and semantic caching for performance
- **Rate Limiting**: Token bucket rate limiter
- **Retry Logic**: Exponential backoff for resilience

## 🛠️ Installation

### Prerequisites
- Databricks workspace with Unity Catalog
- Python 3.10+
- Access to Model Serving endpoints

### Setup

1. **Clone and upload to Databricks**:
```bash
# Upload the rm_agent_enhanced folder to your Databricks workspace
```

2. **Install dependencies**:
```python
%pip install mlflow databricks-langchain langgraph pydantic
```

3. **Configure settings** in `config/settings.py`:
```python
CATALOG = "your_catalog"
SCHEMA = "your_schema"
LLM_ENDPOINT = "your-llm-endpoint"
```

4. **Run deployment notebook**:
```python
# Execute notebooks/01-Deploy-Enhanced-Agent.py
```

## 📖 Usage

### Basic Agent
```python
from agent_core import create_enhanced_agent
from tools.banking_tools import get_all_tools

# Create agent
agent = create_enhanced_agent(
    tools=get_all_tools(),
    enable_checkpointing=True
)

# Run
result = agent.invoke({
    "messages": [{"role": "user", "content": "Look up customer Brian Long"}]
})
```

### Multi-Agent Orchestration
```python
from agents import MultiAgentOrchestrator

# Create orchestrator
orchestrator = MultiAgentOrchestrator(
    model_endpoint="databricks-meta-llama-3-3-70b-instruct",
    tools=tools
)

# Run with supervisor
result = await orchestrator.run(
    messages=[{"role": "user", "content": "Recommend products for customer with risk rating 3"}]
)
```

### MCP Integration
```python
from mcp import MCPClientManager

# Create MCP client
mcp_client = MCPClientManager()

# Add MCP server
await mcp_client.add_server(
    name="compliance",
    command="python",
    args=["-m", "compliance_server"]
)

# Get tools
mcp_tools = mcp_client.get_all_tools()
```

## 🔧 Configuration

### Environment Variables
```bash
DATABRICKS_CATALOG=workspace
DATABRICKS_SCHEMA=rm_agent
LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
SERVING_ENDPOINT=rm-agent-endpoint
```

### Settings Class
```python
from config.settings import Settings

settings = Settings.from_env()
print(settings.databricks.catalog)
print(settings.agent.max_iterations)
print(settings.compliance.level)
```

## 📊 Monitoring

### Audit Logging
```python
from utils import AuditLogger

logger = AuditLogger()
logger.log_customer_access(
    customer_id="CUST001",
    access_type="profile_view",
    user_id="rm_user_1"
)
```

### Performance Metrics
```python
from utils import timer, timed

@timed
def my_function():
    # Function execution is automatically timed
    pass
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_agents.py -v
```

## 📝 License

Apache 2.0 - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📚 Resources

- [Databricks AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
