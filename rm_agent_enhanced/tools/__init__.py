"""
Tools Module - Banking and UC Tools for RM Assistant
"""

from .banking_tools import (
    # Argument models
    CustomerLookupArgs,
    ProductSearchArgs,
    RiskAssessmentArgs,
    CurrencyConversionArgs,
    DocumentSearchArgs,
    ComplianceCheckArgs,
    PortfolioAnalysisArgs,
    
    # Tools
    CustomerProfileTool,
    CustomerRiskAssessmentTool,
    ProductSearchTool,
    ProductComparisonTool,
    ComplianceCheckTool,
    SuitabilityAssessmentTool,
    PortfolioAnalysisTool,
    CurrencyConversionTool,
    DocumentSearchTool,
    
    # Utilities
    get_all_tools,
    get_tools_by_category,
)

from .uc_tools import (
    uc_function,
    generate_uc_function_sql,
    VectorSearchToolGenerator,
    UCTableToolGenerator,
    GraphRAGToolGenerator,
    create_uc_toolkit,
)

__all__ = [
    # Banking tools
    "CustomerLookupArgs",
    "ProductSearchArgs",
    "RiskAssessmentArgs",
    "CurrencyConversionArgs",
    "DocumentSearchArgs",
    "ComplianceCheckArgs",
    "PortfolioAnalysisArgs",
    "CustomerProfileTool",
    "CustomerRiskAssessmentTool",
    "ProductSearchTool",
    "ProductComparisonTool",
    "ComplianceCheckTool",
    "SuitabilityAssessmentTool",
    "PortfolioAnalysisTool",
    "CurrencyConversionTool",
    "DocumentSearchTool",
    "get_all_tools",
    "get_tools_by_category",
    
    # UC tools
    "uc_function",
    "generate_uc_function_sql",
    "VectorSearchToolGenerator",
    "UCTableToolGenerator",
    "GraphRAGToolGenerator",
    "create_uc_toolkit",
]
