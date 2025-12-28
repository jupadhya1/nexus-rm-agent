"""
Banking-Specific MCP Servers for RM Assistant
Implements domain-specific tools for compliance, KYC, and risk management
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum
import asyncio
import re

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionType(Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    TRANSFER = "transfer"
    TRADE = "trade"


@dataclass
class CustomerRiskProfile:
    customer_id: str
    name: str
    risk_score: float  # 0-100
    risk_level: RiskLevel
    pep_status: bool
    sanctions_match: bool
    adverse_media: bool
    source_of_funds_verified: bool
    last_review_date: datetime
    next_review_date: datetime
    risk_factors: list
    recommendations: list


@dataclass
class TransactionAlert:
    alert_id: str
    customer_id: str
    transaction_id: str
    alert_type: str
    risk_score: float
    amount: float
    currency: str
    description: str
    created_at: datetime
    status: str


class ComplianceToolServer:
    """
    MCP Server implementation for compliance tools
    Provides KYC, AML, and regulatory compliance capabilities
    """
    
    def __init__(self):
        self.tools = self._register_tools()
    
    def _register_tools(self) -> dict:
        return {
            "check_customer_kyc_status": {
                "description": "Check the KYC verification status for a customer",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID to check"
                        }
                    },
                    "required": ["customer_id"]
                },
                "handler": self.check_customer_kyc_status
            },
            "screen_customer_sanctions": {
                "description": "Screen customer against sanctions lists (OFAC, UN, EU, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_name": {
                            "type": "string",
                            "description": "Customer name for screening"
                        },
                        "country": {
                            "type": "string",
                            "description": "Customer country of residence"
                        },
                        "dob": {
                            "type": "string",
                            "description": "Date of birth (YYYY-MM-DD)"
                        }
                    },
                    "required": ["customer_name"]
                },
                "handler": self.screen_customer_sanctions
            },
            "check_pep_status": {
                "description": "Check if customer is a Politically Exposed Person (PEP)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_name": {
                            "type": "string",
                            "description": "Customer name"
                        },
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID"
                        }
                    },
                    "required": ["customer_name"]
                },
                "handler": self.check_pep_status
            },
            "calculate_customer_risk_score": {
                "description": "Calculate comprehensive risk score for a customer",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID"
                        },
                        "include_transaction_analysis": {
                            "type": "boolean",
                            "description": "Include transaction pattern analysis"
                        }
                    },
                    "required": ["customer_id"]
                },
                "handler": self.calculate_customer_risk_score
            },
            "generate_sar_report": {
                "description": "Generate Suspicious Activity Report (SAR) draft",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "Customer ID"
                        },
                        "transaction_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of suspicious transaction IDs"
                        },
                        "narrative": {
                            "type": "string",
                            "description": "Description of suspicious activity"
                        }
                    },
                    "required": ["customer_id", "narrative"]
                },
                "handler": self.generate_sar_report
            },
            "check_regulatory_limits": {
                "description": "Check if a transaction exceeds regulatory reporting thresholds",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Transaction amount"
                        },
                        "currency": {
                            "type": "string",
                            "description": "Currency code"
                        },
                        "transaction_type": {
                            "type": "string",
                            "description": "Type of transaction"
                        },
                        "jurisdiction": {
                            "type": "string",
                            "description": "Regulatory jurisdiction (SG, US, UK, etc.)"
                        }
                    },
                    "required": ["amount", "currency"]
                },
                "handler": self.check_regulatory_limits
            }
        }
    
    async def check_customer_kyc_status(self, customer_id: str) -> dict:
        """Check KYC verification status"""
        # In production, this would query the actual KYC database
        return {
            "customer_id": customer_id,
            "kyc_status": "verified",
            "verification_level": "enhanced",
            "last_verification_date": "2024-06-15",
            "next_review_date": "2025-06-15",
            "documents_on_file": [
                "passport",
                "utility_bill",
                "bank_statement",
                "source_of_wealth_declaration"
            ],
            "verification_flags": [],
            "risk_tier": "medium"
        }
    
    async def screen_customer_sanctions(
        self, 
        customer_name: str, 
        country: str = None, 
        dob: str = None
    ) -> dict:
        """Screen against sanctions lists"""
        return {
            "customer_name": customer_name,
            "screening_date": datetime.now().isoformat(),
            "lists_checked": ["OFAC", "UN", "EU", "UK_HMT", "SG_MAS"],
            "match_found": False,
            "potential_matches": [],
            "screening_score": 0.0,
            "recommendation": "CLEAR - No sanctions matches found"
        }
    
    async def check_pep_status(
        self, 
        customer_name: str, 
        customer_id: str = None
    ) -> dict:
        """Check PEP status"""
        return {
            "customer_name": customer_name,
            "pep_status": False,
            "pep_category": None,
            "related_pep": False,
            "close_associate": False,
            "screening_sources": ["World-Check", "Dow Jones", "Refinitiv"],
            "last_checked": datetime.now().isoformat(),
            "recommendation": "No PEP indicators found"
        }
    
    async def calculate_customer_risk_score(
        self, 
        customer_id: str, 
        include_transaction_analysis: bool = True
    ) -> dict:
        """Calculate comprehensive risk score"""
        base_risk_factors = {
            "geographic_risk": 15,
            "product_risk": 10,
            "channel_risk": 8,
            "customer_type_risk": 12,
            "tenure_risk": 5
        }
        
        transaction_risk = 0
        if include_transaction_analysis:
            transaction_risk = 8  # Would be calculated from actual transactions
        
        total_score = sum(base_risk_factors.values()) + transaction_risk
        
        if total_score < 30:
            risk_level = "LOW"
        elif total_score < 50:
            risk_level = "MEDIUM"
        elif total_score < 70:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            "customer_id": customer_id,
            "risk_score": total_score,
            "risk_level": risk_level,
            "risk_breakdown": {
                **base_risk_factors,
                "transaction_behavior_risk": transaction_risk
            },
            "risk_factors": [
                f for f, s in base_risk_factors.items() if s >= 10
            ],
            "recommendations": [
                "Conduct periodic review as per policy",
                "Monitor transaction patterns"
            ],
            "next_review_date": (datetime.now() + timedelta(days=365)).isoformat()
        }
    
    async def generate_sar_report(
        self, 
        customer_id: str, 
        transaction_ids: list = None, 
        narrative: str = ""
    ) -> dict:
        """Generate SAR report draft"""
        return {
            "report_id": f"SAR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "DRAFT",
            "customer_id": customer_id,
            "transaction_ids": transaction_ids or [],
            "narrative": narrative,
            "filing_requirements": {
                "jurisdiction": "Singapore",
                "regulatory_body": "MAS STRO",
                "filing_deadline": (datetime.now() + timedelta(days=15)).isoformat()
            },
            "required_fields_complete": False,
            "missing_fields": [
                "authorized_signatory",
                "compliance_officer_review"
            ],
            "template_sections": [
                "subject_information",
                "suspicious_activity_description",
                "transaction_details",
                "supporting_documentation"
            ]
        }
    
    async def check_regulatory_limits(
        self, 
        amount: float, 
        currency: str, 
        transaction_type: str = None,
        jurisdiction: str = "SG"
    ) -> dict:
        """Check regulatory reporting thresholds"""
        # Thresholds by jurisdiction (in local currency)
        thresholds = {
            "SG": {"CTR": 20000, "STR": 0},  # SGD
            "US": {"CTR": 10000, "SAR": 5000},  # USD
            "UK": {"SAR": 0},  # GBP
            "EU": {"CTR": 10000}  # EUR
        }
        
        jur_thresholds = thresholds.get(jurisdiction, {"CTR": 10000})
        
        alerts = []
        for report_type, threshold in jur_thresholds.items():
            if amount >= threshold:
                alerts.append({
                    "report_type": report_type,
                    "threshold": threshold,
                    "exceeded_by": amount - threshold,
                    "action_required": f"File {report_type} within regulatory timeframe"
                })
        
        return {
            "amount": amount,
            "currency": currency,
            "jurisdiction": jurisdiction,
            "thresholds_checked": jur_thresholds,
            "reporting_required": len(alerts) > 0,
            "alerts": alerts,
            "recommendation": alerts[0]["action_required"] if alerts else "No reporting required"
        }


class RCSAToolServer:
    """
    MCP Server for Risk and Control Self-Assessment (RCSA) tools
    """
    
    def __init__(self):
        self.tools = self._register_tools()
    
    def _register_tools(self) -> dict:
        return {
            "assess_control_effectiveness": {
                "description": "Assess the effectiveness of a specific control",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "control_id": {
                            "type": "string",
                            "description": "Control identifier"
                        },
                        "assessment_period": {
                            "type": "string",
                            "description": "Assessment period (e.g., 'Q4 2024')"
                        }
                    },
                    "required": ["control_id"]
                },
                "handler": self.assess_control_effectiveness
            },
            "identify_control_gaps": {
                "description": "Identify gaps in control coverage for a process",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "process_id": {
                            "type": "string",
                            "description": "Process identifier"
                        },
                        "risk_category": {
                            "type": "string",
                            "description": "Risk category to analyze"
                        }
                    },
                    "required": ["process_id"]
                },
                "handler": self.identify_control_gaps
            },
            "generate_rcsa_summary": {
                "description": "Generate RCSA summary report for a business unit",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "business_unit": {
                            "type": "string",
                            "description": "Business unit code"
                        },
                        "period": {
                            "type": "string",
                            "description": "Reporting period"
                        },
                        "include_remediation": {
                            "type": "boolean",
                            "description": "Include remediation plans"
                        }
                    },
                    "required": ["business_unit"]
                },
                "handler": self.generate_rcsa_summary
            },
            "map_risks_to_controls": {
                "description": "Map risks to their corresponding controls",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "risk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of risk identifiers"
                        }
                    },
                    "required": ["risk_ids"]
                },
                "handler": self.map_risks_to_controls
            },
            "calculate_residual_risk": {
                "description": "Calculate residual risk after control application",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "inherent_risk_score": {
                            "type": "number",
                            "description": "Inherent risk score (1-5)"
                        },
                        "control_effectiveness": {
                            "type": "number",
                            "description": "Control effectiveness percentage (0-100)"
                        }
                    },
                    "required": ["inherent_risk_score", "control_effectiveness"]
                },
                "handler": self.calculate_residual_risk
            }
        }
    
    async def assess_control_effectiveness(
        self, 
        control_id: str, 
        assessment_period: str = None
    ) -> dict:
        """Assess control effectiveness"""
        return {
            "control_id": control_id,
            "assessment_period": assessment_period or "Q4 2024",
            "effectiveness_rating": "Effective",
            "effectiveness_score": 85,
            "design_effectiveness": 90,
            "operating_effectiveness": 80,
            "test_results": {
                "samples_tested": 25,
                "exceptions_found": 2,
                "exception_rate": 0.08
            },
            "key_findings": [
                "Control operating as designed",
                "Minor timing delays noted"
            ],
            "recommendations": [
                "Enhance monitoring frequency",
                "Update control documentation"
            ]
        }
    
    async def identify_control_gaps(
        self, 
        process_id: str, 
        risk_category: str = None
    ) -> dict:
        """Identify control gaps"""
        return {
            "process_id": process_id,
            "risk_category": risk_category or "all",
            "gaps_identified": [
                {
                    "gap_id": "GAP001",
                    "description": "Missing automated reconciliation control",
                    "risk_exposure": "Medium",
                    "recommendation": "Implement automated daily reconciliation",
                    "priority": "High"
                },
                {
                    "gap_id": "GAP002",
                    "description": "Insufficient segregation of duties",
                    "risk_exposure": "High",
                    "recommendation": "Implement dual approval workflow",
                    "priority": "Critical"
                }
            ],
            "coverage_analysis": {
                "total_risks": 15,
                "risks_with_controls": 12,
                "coverage_percentage": 80
            }
        }
    
    async def generate_rcsa_summary(
        self, 
        business_unit: str, 
        period: str = None,
        include_remediation: bool = True
    ) -> dict:
        """Generate RCSA summary"""
        return {
            "business_unit": business_unit,
            "period": period or "2024",
            "summary": {
                "total_risks": 45,
                "high_risks": 5,
                "medium_risks": 15,
                "low_risks": 25,
                "total_controls": 78,
                "effective_controls": 65,
                "partially_effective": 10,
                "ineffective": 3
            },
            "risk_profile_change": "Stable",
            "key_risk_indicators": [
                {"kri": "Transaction Error Rate", "status": "Green", "trend": "Improving"},
                {"kri": "Compliance Breach Count", "status": "Green", "trend": "Stable"},
                {"kri": "Control Exceptions", "status": "Amber", "trend": "Deteriorating"}
            ],
            "remediation_status": {
                "open_actions": 12,
                "overdue_actions": 2,
                "completed_this_period": 8
            } if include_remediation else None
        }
    
    async def map_risks_to_controls(self, risk_ids: list) -> dict:
        """Map risks to controls"""
        mappings = []
        for risk_id in risk_ids:
            mappings.append({
                "risk_id": risk_id,
                "controls": [
                    {
                        "control_id": f"CTRL-{risk_id}-001",
                        "control_type": "Preventive",
                        "effectiveness": "Effective"
                    },
                    {
                        "control_id": f"CTRL-{risk_id}-002",
                        "control_type": "Detective",
                        "effectiveness": "Partially Effective"
                    }
                ],
                "control_coverage": "Adequate"
            })
        
        return {
            "mappings": mappings,
            "analysis": {
                "risks_analyzed": len(risk_ids),
                "total_controls_mapped": len(risk_ids) * 2,
                "adequately_controlled": len(risk_ids)
            }
        }
    
    async def calculate_residual_risk(
        self, 
        inherent_risk_score: float, 
        control_effectiveness: float
    ) -> dict:
        """Calculate residual risk"""
        # Residual Risk = Inherent Risk × (1 - Control Effectiveness)
        residual_score = inherent_risk_score * (1 - control_effectiveness / 100)
        
        if residual_score < 1:
            residual_level = "Low"
        elif residual_score < 2.5:
            residual_level = "Medium"
        elif residual_score < 4:
            residual_level = "High"
        else:
            residual_level = "Critical"
        
        return {
            "inherent_risk_score": inherent_risk_score,
            "control_effectiveness": control_effectiveness,
            "residual_risk_score": round(residual_score, 2),
            "residual_risk_level": residual_level,
            "risk_reduction": round((1 - residual_score / inherent_risk_score) * 100, 1),
            "within_appetite": residual_score < 2.5,
            "recommendation": "Additional controls recommended" if residual_score >= 2.5 else "Risk within appetite"
        }


class DocumentAnalysisToolServer:
    """
    MCP Server for document analysis and extraction
    """
    
    def __init__(self):
        self.tools = self._register_tools()
    
    def _register_tools(self) -> dict:
        return {
            "extract_document_entities": {
                "description": "Extract key entities from a document (names, amounts, dates, etc.)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_content": {
                            "type": "string",
                            "description": "Document text content"
                        },
                        "entity_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of entities to extract"
                        }
                    },
                    "required": ["document_content"]
                },
                "handler": self.extract_document_entities
            },
            "compare_documents": {
                "description": "Compare two documents for differences and similarities",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_a": {"type": "string"},
                        "document_b": {"type": "string"},
                        "comparison_type": {
                            "type": "string",
                            "enum": ["semantic", "structural", "detailed"]
                        }
                    },
                    "required": ["document_a", "document_b"]
                },
                "handler": self.compare_documents
            },
            "classify_document": {
                "description": "Classify a document by type and category",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_content": {"type": "string"},
                        "classification_scheme": {"type": "string"}
                    },
                    "required": ["document_content"]
                },
                "handler": self.classify_document
            },
            "extract_table_data": {
                "description": "Extract structured data from tables in documents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_content": {"type": "string"},
                        "table_format": {
                            "type": "string",
                            "enum": ["json", "csv", "markdown"]
                        }
                    },
                    "required": ["document_content"]
                },
                "handler": self.extract_table_data
            },
            "summarize_document": {
                "description": "Generate a summary of a document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_content": {"type": "string"},
                        "summary_length": {
                            "type": "string",
                            "enum": ["brief", "detailed", "comprehensive"]
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["document_content"]
                },
                "handler": self.summarize_document
            }
        }
    
    async def extract_document_entities(
        self, 
        document_content: str, 
        entity_types: list = None
    ) -> dict:
        """Extract entities from document"""
        # Simplified entity extraction (in production, use NER model)
        entities = {
            "amounts": re.findall(r'\$[\d,]+(?:\.\d{2})?', document_content),
            "dates": re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', document_content),
            "percentages": re.findall(r'\d+(?:\.\d+)?%', document_content),
            "emails": re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', document_content)
        }
        
        return {
            "entities": entities,
            "entity_count": sum(len(v) for v in entities.values()),
            "extraction_confidence": 0.85
        }
    
    async def compare_documents(
        self, 
        document_a: str, 
        document_b: str,
        comparison_type: str = "semantic"
    ) -> dict:
        """Compare two documents"""
        # Simplified comparison
        words_a = set(document_a.lower().split())
        words_b = set(document_b.lower().split())
        
        common = words_a & words_b
        only_a = words_a - words_b
        only_b = words_b - words_a
        
        similarity = len(common) / max(len(words_a | words_b), 1)
        
        return {
            "similarity_score": round(similarity, 3),
            "common_terms_count": len(common),
            "unique_to_a": len(only_a),
            "unique_to_b": len(only_b),
            "comparison_type": comparison_type,
            "summary": f"Documents are {similarity*100:.1f}% similar"
        }
    
    async def classify_document(
        self, 
        document_content: str, 
        classification_scheme: str = "banking"
    ) -> dict:
        """Classify document type"""
        # Simplified classification based on keywords
        content_lower = document_content.lower()
        
        if "unit trust" in content_lower or "fund" in content_lower:
            doc_type = "investment_product"
        elif "kyc" in content_lower or "verification" in content_lower:
            doc_type = "kyc_document"
        elif "transaction" in content_lower:
            doc_type = "transaction_record"
        elif "risk" in content_lower:
            doc_type = "risk_assessment"
        else:
            doc_type = "general"
        
        return {
            "document_type": doc_type,
            "classification_scheme": classification_scheme,
            "confidence": 0.78,
            "suggested_categories": [doc_type, "financial_document"],
            "keywords_detected": ["fund", "risk", "investment"][:3]
        }
    
    async def extract_table_data(
        self, 
        document_content: str, 
        table_format: str = "json"
    ) -> dict:
        """Extract table data"""
        # Would use actual table extraction in production
        return {
            "tables_found": 1,
            "extracted_data": [
                {
                    "table_index": 0,
                    "rows": 5,
                    "columns": 3,
                    "data": [
                        ["Header1", "Header2", "Header3"],
                        ["Value1", "Value2", "Value3"]
                    ]
                }
            ],
            "format": table_format
        }
    
    async def summarize_document(
        self, 
        document_content: str, 
        summary_length: str = "brief",
        focus_areas: list = None
    ) -> dict:
        """Summarize document"""
        word_count = len(document_content.split())
        
        target_lengths = {
            "brief": 50,
            "detailed": 150,
            "comprehensive": 300
        }
        
        # Would use actual summarization model in production
        return {
            "original_word_count": word_count,
            "summary_word_count": target_lengths.get(summary_length, 100),
            "summary": f"[Summary of {word_count} word document...]",
            "key_points": [
                "Key point 1 extracted from document",
                "Key point 2 extracted from document",
                "Key point 3 extracted from document"
            ],
            "focus_areas_addressed": focus_areas or []
        }


# Export all server classes
__all__ = [
    "ComplianceToolServer",
    "RCSAToolServer", 
    "DocumentAnalysisToolServer",
    "RiskLevel",
    "TransactionType",
    "CustomerRiskProfile",
    "TransactionAlert"
]
