"""
Agents Module - Multi-Agent System for RM Assistant
"""

from .base import (
    AgentRole,
    AgentConfig,
    AgentState,
    BaseAgent,
)

from .specialists import (
    CustomerSpecialistAgent,
    ProductSpecialistAgent,
    ComplianceOfficerAgent,
    AnalyticsExpertAgent,
    DocumentAnalystAgent,
    RCSASpecialistAgent,
    KYCSpecialistAgent,
)

from .orchestrator import (
    SupervisorAgent,
    MultiAgentOrchestrator,
    create_rm_orchestrator,
)

__all__ = [
    # Base classes
    "AgentRole",
    "AgentConfig",
    "AgentState",
    "BaseAgent",
    
    # Specialist agents
    "CustomerSpecialistAgent",
    "ProductSpecialistAgent",
    "ComplianceOfficerAgent",
    "AnalyticsExpertAgent",
    "DocumentAnalystAgent",
    "RCSASpecialistAgent",
    "KYCSpecialistAgent",
    
    # Orchestration
    "SupervisorAgent",
    "MultiAgentOrchestrator",
    "create_rm_orchestrator",
]
