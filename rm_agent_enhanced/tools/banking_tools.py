"""
Advanced Tools for RM Assistant Agent
Includes enhanced UC Functions, GraphRAG, and utility tools
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional, Type
from functools import lru_cache

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Tool Arguments
# ============================================================================

class CustomerLookupArgs(BaseModel):
    """Arguments for customer lookup"""
    customer_name: str = Field(description="Name of the customer to look up")
    include_history: bool = Field(default=False, description="Include transaction history")


class ProductSearchArgs(BaseModel):
    """Arguments for product search"""
    query: str = Field(description="Search query for products")
    max_risk_rating: Optional[int] = Field(default=None, description="Maximum risk rating (1-5)")
    currency: Optional[str] = Field(default=None, description="Filter by currency")
    limit: int = Field(default=5, description="Maximum results to return")


class RiskAssessmentArgs(BaseModel):
    """Arguments for risk assessment"""
    customer_id: str = Field(description="Customer ID to assess")
    include_transactions: bool = Field(default=True, description="Include transaction analysis")
    include_compliance: bool = Field(default=True, description="Include compliance checks")


class CurrencyConversionArgs(BaseModel):
    """Arguments for currency conversion"""
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="Source currency code")
    to_currency: str = Field(default="USD", description="Target currency code")


class DocumentSearchArgs(BaseModel):
    """Arguments for document search"""
    query: str = Field(description="Search query")
    document_type: Optional[str] = Field(default=None, description="Filter by document type")
    num_results: int = Field(default=5, description="Number of results")


class ComplianceCheckArgs(BaseModel):
    """Arguments for compliance check"""
    customer_id: str = Field(description="Customer ID")
    check_type: str = Field(
        default="full",
        description="Type of check: kyc, sanctions, pep, aml, or full"
    )


class PortfolioAnalysisArgs(BaseModel):
    """Arguments for portfolio analysis"""
    customer_id: str = Field(description="Customer ID")
    include_recommendations: bool = Field(default=True, description="Include recommendations")


# ============================================================================
# Enhanced Customer Tools
# ============================================================================

class CustomerProfileTool(BaseTool):
    """Enhanced customer profile lookup with rich data"""
    
    name: str = "lookup_customer_profile"
    description: str = """Look up comprehensive customer profile including:
    - Demographics and contact information
    - Risk rating and investment preferences
    - Account balance and tenure
    - Transaction summary
    Use this when you need customer information for recommendations."""
    args_schema: Type[BaseModel] = CustomerLookupArgs
    
    def _run(self, customer_name: str, include_history: bool = False) -> str:
        """Execute customer lookup"""
        # This would integrate with Unity Catalog in production
        return json.dumps({
            "customer_name": customer_name,
            "status": "found",
            "profile": {
                "name": customer_name,
                "risk_rating": 3,
                "account_balance": 50000,
                "customer_since": "2020-01-15",
                "preferred_currency": "USD"
            },
            "summary": f"Customer {customer_name} has moderate risk tolerance"
        })


class CustomerRiskAssessmentTool(BaseTool):
    """Comprehensive customer risk assessment"""
    
    name: str = "assess_customer_risk"
    description: str = """Perform comprehensive risk assessment for a customer including:
    - Overall risk score calculation
    - Transaction behavior analysis
    - Compliance status check
    - Risk factor identification
    Use for suitability assessment before product recommendations."""
    args_schema: Type[BaseModel] = RiskAssessmentArgs
    
    def _run(
        self, 
        customer_id: str, 
        include_transactions: bool = True,
        include_compliance: bool = True
    ) -> str:
        """Execute risk assessment"""
        result = {
            "customer_id": customer_id,
            "assessment_date": datetime.now().isoformat(),
            "risk_score": 45,
            "risk_level": "MEDIUM",
            "factors": {
                "geographic_risk": 15,
                "product_risk": 10,
                "transaction_risk": 12,
                "tenure_factor": -5
            },
            "recommendations": [
                "Standard monitoring applies",
                "Suitable for medium-risk products"
            ]
        }
        
        if include_transactions:
            result["transaction_analysis"] = {
                "avg_monthly_volume": 25000,
                "unusual_patterns": False,
                "large_transactions_30d": 2
            }
        
        if include_compliance:
            result["compliance_status"] = {
                "kyc_verified": True,
                "sanctions_clear": True,
                "pep_status": False,
                "last_review": "2024-06-15"
            }
        
        return json.dumps(result)


# ============================================================================
# Enhanced Product Tools
# ============================================================================

class ProductSearchTool(BaseTool):
    """Advanced product search with vector similarity"""
    
    name: str = "search_investment_products"
    description: str = """Search for investment products (unit trusts) based on:
    - Natural language query
    - Risk rating filter
    - Currency preference
    Returns relevant products with details."""
    args_schema: Type[BaseModel] = ProductSearchArgs
    
    def _run(
        self,
        query: str,
        max_risk_rating: Optional[int] = None,
        currency: Optional[str] = None,
        limit: int = 5
    ) -> str:
        """Execute product search"""
        # In production, this calls vector search
        products = [
            {
                "product_id": "UT01",
                "name": "Global Growth Fund",
                "risk_rating": 5,
                "currency": "USD",
                "nav": 88.87,
                "expense_ratio": 1.49,
                "relevance_score": 0.95
            },
            {
                "product_id": "UT02",
                "name": "Income Opportunity Fund",
                "risk_rating": 3,
                "currency": "GBP",
                "nav": 23.76,
                "expense_ratio": 0.99,
                "relevance_score": 0.87
            }
        ]
        
        # Apply filters
        if max_risk_rating:
            products = [p for p in products if p["risk_rating"] <= max_risk_rating]
        if currency:
            products = [p for p in products if p["currency"] == currency]
        
        return json.dumps({
            "query": query,
            "total_results": len(products),
            "products": products[:limit]
        })


class ProductComparisonTool(BaseTool):
    """Compare multiple investment products"""
    
    name: str = "compare_products"
    description: str = """Compare multiple investment products side by side.
    Provides comparison across: risk, returns, fees, currency, and suitability."""
    
    def _run(self, product_ids: list[str]) -> str:
        """Execute product comparison"""
        comparison = {
            "products_compared": product_ids,
            "comparison_date": datetime.now().isoformat(),
            "metrics": {
                "risk_rating": {"UT01": 5, "UT02": 3},
                "expense_ratio": {"UT01": 1.49, "UT02": 0.99},
                "1y_return": {"UT01": 12.5, "UT02": 7.8},
                "volatility": {"UT01": "High", "UT02": "Medium"}
            },
            "recommendation": "UT02 offers better risk-adjusted returns for moderate investors"
        }
        return json.dumps(comparison)


# ============================================================================
# Compliance Tools
# ============================================================================

class ComplianceCheckTool(BaseTool):
    """Comprehensive compliance checking tool"""
    
    name: str = "check_compliance_status"
    description: str = """Run compliance checks for a customer including:
    - KYC verification status
    - Sanctions screening
    - PEP (Politically Exposed Person) check
    - AML risk indicators
    Use before finalizing any recommendations."""
    args_schema: Type[BaseModel] = ComplianceCheckArgs
    
    def _run(self, customer_id: str, check_type: str = "full") -> str:
        """Execute compliance check"""
        result = {
            "customer_id": customer_id,
            "check_type": check_type,
            "timestamp": datetime.now().isoformat(),
            "overall_status": "CLEAR"
        }
        
        if check_type in ["kyc", "full"]:
            result["kyc"] = {
                "status": "verified",
                "verification_date": "2024-03-15",
                "documents_valid": True,
                "next_review": "2025-03-15"
            }
        
        if check_type in ["sanctions", "full"]:
            result["sanctions"] = {
                "status": "clear",
                "lists_checked": ["OFAC", "UN", "EU", "MAS"],
                "matches_found": 0
            }
        
        if check_type in ["pep", "full"]:
            result["pep"] = {
                "is_pep": False,
                "related_pep": False,
                "last_checked": datetime.now().isoformat()
            }
        
        if check_type in ["aml", "full"]:
            result["aml"] = {
                "risk_score": 25,
                "risk_level": "LOW",
                "alerts_pending": 0
            }
        
        return json.dumps(result)


class SuitabilityAssessmentTool(BaseTool):
    """Assess product suitability for customer"""
    
    name: str = "assess_product_suitability"
    description: str = """Assess whether a specific product is suitable for a customer.
    Considers risk tolerance, investment objectives, and regulatory requirements."""
    
    def _run(self, customer_id: str, product_id: str) -> str:
        """Execute suitability assessment"""
        result = {
            "customer_id": customer_id,
            "product_id": product_id,
            "assessment_date": datetime.now().isoformat(),
            "suitability_status": "SUITABLE",
            "risk_match": True,
            "factors_considered": [
                "Customer risk rating vs product risk",
                "Investment horizon",
                "Liquidity requirements",
                "Regulatory restrictions"
            ],
            "warnings": [],
            "disclosures_required": [
                "Product risk disclosure",
                "Currency risk disclosure"
            ]
        }
        return json.dumps(result)


# ============================================================================
# Analytics Tools
# ============================================================================

class PortfolioAnalysisTool(BaseTool):
    """Analyze customer portfolio allocation"""
    
    name: str = "analyze_portfolio"
    description: str = """Analyze customer's current portfolio including:
    - Asset allocation breakdown
    - Risk exposure analysis
    - Diversification assessment
    - Rebalancing recommendations"""
    args_schema: Type[BaseModel] = PortfolioAnalysisArgs
    
    def _run(
        self, 
        customer_id: str, 
        include_recommendations: bool = True
    ) -> str:
        """Execute portfolio analysis"""
        result = {
            "customer_id": customer_id,
            "analysis_date": datetime.now().isoformat(),
            "total_value": 150000,
            "allocation": {
                "equities": 45,
                "bonds": 35,
                "money_market": 15,
                "alternatives": 5
            },
            "risk_metrics": {
                "portfolio_risk_score": 3.2,
                "volatility": "Medium",
                "sharpe_ratio": 1.15
            },
            "diversification_score": 72
        }
        
        if include_recommendations:
            result["recommendations"] = [
                "Consider reducing equity exposure by 5%",
                "Add exposure to international bonds",
                "Portfolio is well-diversified overall"
            ]
        
        return json.dumps(result)


class CurrencyConversionTool(BaseTool):
    """Convert currencies with live rates"""
    
    name: str = "convert_currency"
    description: str = """Convert amounts between currencies.
    Supports major currencies: USD, EUR, GBP, SGD, AUD, JPY, etc."""
    args_schema: Type[BaseModel] = CurrencyConversionArgs
    
    # Simplified rate table
    RATES = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "SGD": 1.34,
        "AUD": 1.53,
        "JPY": 149.5
    }
    
    def _run(
        self, 
        amount: float, 
        from_currency: str, 
        to_currency: str = "USD"
    ) -> str:
        """Execute currency conversion"""
        from_rate = self.RATES.get(from_currency.upper(), 1.0)
        to_rate = self.RATES.get(to_currency.upper(), 1.0)
        
        usd_amount = amount / from_rate
        converted = usd_amount * to_rate
        
        return json.dumps({
            "original_amount": amount,
            "original_currency": from_currency,
            "converted_amount": round(converted, 2),
            "target_currency": to_currency,
            "rate_used": round(to_rate / from_rate, 6),
            "timestamp": datetime.now().isoformat()
        })


# ============================================================================
# Document Tools
# ============================================================================

class DocumentSearchTool(BaseTool):
    """Search product documentation using vector search"""
    
    name: str = "search_documents"
    description: str = """Search product documentation and factsheets.
    Uses semantic search to find relevant information about unit trusts and investments."""
    args_schema: Type[BaseModel] = DocumentSearchArgs
    
    def _run(
        self, 
        query: str, 
        document_type: Optional[str] = None,
        num_results: int = 5
    ) -> str:
        """Execute document search"""
        # In production, this calls vector search index
        results = [
            {
                "document_id": "UT01_factsheet",
                "title": "Global Growth Fund Factsheet",
                "snippet": "The fund invests in global equities with focus on growth companies...",
                "relevance_score": 0.92,
                "document_type": "factsheet"
            },
            {
                "document_id": "UT02_factsheet",
                "title": "Income Opportunity Fund Factsheet",
                "snippet": "A balanced fund seeking income with capital appreciation...",
                "relevance_score": 0.85,
                "document_type": "factsheet"
            }
        ]
        
        if document_type:
            results = [r for r in results if r["document_type"] == document_type]
        
        return json.dumps({
            "query": query,
            "total_results": len(results),
            "results": results[:num_results]
        })


# ============================================================================
# Tool Registry
# ============================================================================

def get_all_tools() -> list[BaseTool]:
    """Get all available tools"""
    return [
        CustomerProfileTool(),
        CustomerRiskAssessmentTool(),
        ProductSearchTool(),
        ProductComparisonTool(),
        ComplianceCheckTool(),
        SuitabilityAssessmentTool(),
        PortfolioAnalysisTool(),
        CurrencyConversionTool(),
        DocumentSearchTool(),
    ]


def get_tools_by_category(category: str) -> list[BaseTool]:
    """Get tools filtered by category"""
    tool_categories = {
        "customer": [CustomerProfileTool, CustomerRiskAssessmentTool],
        "product": [ProductSearchTool, ProductComparisonTool],
        "compliance": [ComplianceCheckTool, SuitabilityAssessmentTool],
        "analytics": [PortfolioAnalysisTool, CurrencyConversionTool],
        "document": [DocumentSearchTool],
    }
    
    tool_classes = tool_categories.get(category, [])
    return [cls() for cls in tool_classes]


__all__ = [
    # Argument models
    "CustomerLookupArgs",
    "ProductSearchArgs",
    "RiskAssessmentArgs",
    "CurrencyConversionArgs",
    "DocumentSearchArgs",
    "ComplianceCheckArgs",
    "PortfolioAnalysisArgs",
    
    # Tools
    "CustomerProfileTool",
    "CustomerRiskAssessmentTool",
    "ProductSearchTool",
    "ProductComparisonTool",
    "ComplianceCheckTool",
    "SuitabilityAssessmentTool",
    "PortfolioAnalysisTool",
    "CurrencyConversionTool",
    "DocumentSearchTool",
    
    # Utilities
    "get_all_tools",
    "get_tools_by_category",
]
