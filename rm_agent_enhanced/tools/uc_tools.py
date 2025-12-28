"""
Databricks Unity Catalog Integration Tools
Provides tools that integrate with UC Functions, Vector Search, and Tables
"""

import logging
from typing import Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# UC Function Decorators
# ============================================================================

def uc_function(
    catalog: str = "workspace",
    schema: str = "rm_agent",
    comment: str = ""
):
    """
    Decorator to register a Python function as a Unity Catalog function
    
    Usage:
        @uc_function(catalog="workspace", schema="rm_agent")
        def my_tool(param1: str, param2: int) -> str:
            '''Tool description'''
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Store UC metadata
        wrapper._uc_metadata = {
            "catalog": catalog,
            "schema": schema,
            "function_name": func.__name__,
            "comment": comment or func.__doc__ or "",
            "full_name": f"{catalog}.{schema}.{func.__name__}"
        }
        
        return wrapper
    return decorator


def generate_uc_function_sql(func: Callable) -> str:
    """
    Generate SQL to create a UC function from a decorated Python function
    """
    if not hasattr(func, '_uc_metadata'):
        raise ValueError("Function must be decorated with @uc_function")
    
    metadata = func._uc_metadata
    annotations = func.__annotations__
    
    # Build parameter list
    params = []
    for name, type_hint in annotations.items():
        if name == 'return':
            continue
        sql_type = _python_type_to_sql(type_hint)
        params.append(f"{name} {sql_type}")
    
    return_type = _python_type_to_sql(annotations.get('return', str))
    
    sql = f"""
CREATE OR REPLACE FUNCTION {metadata['full_name']}(
    {', '.join(params)}
)
RETURNS {return_type}
LANGUAGE PYTHON
COMMENT '{metadata['comment']}'
AS $$
{_get_function_body(func)}
$$;
"""
    return sql.strip()


def _python_type_to_sql(python_type) -> str:
    """Convert Python type hint to SQL type"""
    type_mapping = {
        str: "STRING",
        int: "LONG",
        float: "DOUBLE",
        bool: "BOOLEAN",
        list: "ARRAY<STRING>",
        dict: "MAP<STRING, STRING>",
    }
    
    # Handle Optional types
    origin = getattr(python_type, '__origin__', None)
    if origin is type(None):
        return "STRING"
    
    return type_mapping.get(python_type, "STRING")


def _get_function_body(func: Callable) -> str:
    """Extract function body for UC function"""
    import inspect
    source = inspect.getsource(func)
    # Remove decorator and def line, return body
    lines = source.split('\n')
    body_lines = []
    in_body = False
    for line in lines:
        if in_body:
            body_lines.append(line)
        elif line.strip().startswith('def '):
            in_body = True
    return '\n'.join(body_lines)


# ============================================================================
# Vector Search Tools Generator
# ============================================================================

class VectorSearchToolGenerator:
    """
    Generate LangChain tools from Databricks Vector Search indexes
    """
    
    def __init__(
        self,
        endpoint_name: str = "vs_endpoint",
        default_num_results: int = 5
    ):
        self.endpoint_name = endpoint_name
        self.default_num_results = default_num_results
    
    def create_search_tool(
        self,
        index_name: str,
        tool_name: str,
        description: str,
        columns: list[str] = None,
        filters: dict = None
    ):
        """
        Create a search tool for a specific vector search index
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        class SearchArgs(BaseModel):
            query: str = Field(description="Search query")
            num_results: int = Field(
                default=self.default_num_results,
                description="Number of results to return"
            )
        
        def search_fn(query: str, num_results: int = self.default_num_results) -> str:
            """Execute vector search"""
            try:
                # In production, use actual Databricks Vector Search client
                # from databricks.vector_search.client import VectorSearchClient
                
                # Placeholder response
                import json
                return json.dumps({
                    "index": index_name,
                    "query": query,
                    "num_results": num_results,
                    "results": [
                        {"content": f"Result for: {query}", "score": 0.95}
                    ]
                })
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return f"Error: {str(e)}"
        
        return StructuredTool.from_function(
            func=search_fn,
            name=tool_name,
            description=description,
            args_schema=SearchArgs
        )
    
    def create_hybrid_search_tool(
        self,
        index_name: str,
        table_name: str,
        tool_name: str,
        description: str
    ):
        """
        Create a hybrid search tool combining vector and keyword search
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        class HybridSearchArgs(BaseModel):
            query: str = Field(description="Search query")
            keyword_filter: Optional[str] = Field(
                default=None,
                description="Additional keyword filter"
            )
            num_results: int = Field(default=5, description="Number of results")
        
        def hybrid_search(
            query: str,
            keyword_filter: Optional[str] = None,
            num_results: int = 5
        ) -> str:
            """Execute hybrid search"""
            import json
            
            # Combine vector search with SQL filtering
            return json.dumps({
                "index": index_name,
                "table": table_name,
                "query": query,
                "keyword_filter": keyword_filter,
                "results": [
                    {
                        "content": f"Hybrid result for: {query}",
                        "vector_score": 0.92,
                        "keyword_match": True
                    }
                ]
            })
        
        return StructuredTool.from_function(
            func=hybrid_search,
            name=tool_name,
            description=description,
            args_schema=HybridSearchArgs
        )


# ============================================================================
# UC Table Tools Generator
# ============================================================================

class UCTableToolGenerator:
    """
    Generate tools for Unity Catalog table operations
    """
    
    def __init__(self, catalog: str = "workspace", schema: str = "rm_agent"):
        self.catalog = catalog
        self.schema = schema
    
    def create_lookup_tool(
        self,
        table_name: str,
        lookup_column: str,
        return_columns: list[str],
        tool_name: str,
        description: str
    ):
        """Create a lookup tool for a UC table"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        class LookupArgs(BaseModel):
            lookup_value: str = Field(description=f"Value to lookup in {lookup_column}")
        
        def lookup_fn(lookup_value: str) -> str:
            """Execute table lookup"""
            import json
            
            # In production, use actual Spark SQL
            # full_table = f"{self.catalog}.{self.schema}.{table_name}"
            # df = spark.sql(f"SELECT {','.join(return_columns)} FROM {full_table} WHERE {lookup_column} = '{lookup_value}'")
            
            return json.dumps({
                "table": table_name,
                "lookup_column": lookup_column,
                "lookup_value": lookup_value,
                "result": {col: f"value_{col}" for col in return_columns}
            })
        
        return StructuredTool.from_function(
            func=lookup_fn,
            name=tool_name,
            description=description,
            args_schema=LookupArgs
        )
    
    def create_aggregation_tool(
        self,
        table_name: str,
        group_by: str,
        aggregations: dict[str, str],
        tool_name: str,
        description: str
    ):
        """Create an aggregation tool for a UC table"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        class AggArgs(BaseModel):
            filter_value: Optional[str] = Field(
                default=None,
                description="Optional filter value"
            )
        
        def agg_fn(filter_value: Optional[str] = None) -> str:
            """Execute aggregation"""
            import json
            
            return json.dumps({
                "table": table_name,
                "group_by": group_by,
                "aggregations": aggregations,
                "filter": filter_value,
                "result": [
                    {group_by: "group1", **{k: 100 for k in aggregations.keys()}}
                ]
            })
        
        return StructuredTool.from_function(
            func=agg_fn,
            name=tool_name,
            description=description,
            args_schema=AggArgs
        )


# ============================================================================
# GraphRAG Integration
# ============================================================================

class GraphRAGToolGenerator:
    """
    Generate tools for GraphRAG (Knowledge Graph + RAG) queries
    """
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
    
    def create_entity_search_tool(
        self,
        entity_type: str,
        tool_name: str,
        description: str
    ):
        """Create entity search tool for knowledge graph"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        class EntitySearchArgs(BaseModel):
            query: str = Field(description="Entity search query")
            max_hops: int = Field(default=2, description="Maximum graph traversal hops")
        
        def entity_search(query: str, max_hops: int = 2) -> str:
            """Search for entities in knowledge graph"""
            import json
            
            # In production, use Neo4j driver
            return json.dumps({
                "entity_type": entity_type,
                "query": query,
                "max_hops": max_hops,
                "entities": [
                    {
                        "id": "entity_1",
                        "type": entity_type,
                        "properties": {"name": f"Entity matching: {query}"},
                        "relationships": [
                            {"type": "RELATED_TO", "target": "entity_2"}
                        ]
                    }
                ]
            })
        
        return StructuredTool.from_function(
            func=entity_search,
            name=tool_name,
            description=description,
            args_schema=EntitySearchArgs
        )
    
    def create_path_finding_tool(
        self,
        tool_name: str,
        description: str
    ):
        """Create path finding tool for relationship analysis"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        class PathFindingArgs(BaseModel):
            source_entity: str = Field(description="Source entity ID or name")
            target_entity: str = Field(description="Target entity ID or name")
            relationship_types: Optional[list[str]] = Field(
                default=None,
                description="Filter by relationship types"
            )
        
        def find_path(
            source_entity: str,
            target_entity: str,
            relationship_types: Optional[list[str]] = None
        ) -> str:
            """Find path between entities"""
            import json
            
            return json.dumps({
                "source": source_entity,
                "target": target_entity,
                "paths": [
                    {
                        "length": 2,
                        "nodes": [source_entity, "intermediate", target_entity],
                        "relationships": ["CONNECTS_TO", "RELATES_TO"]
                    }
                ]
            })
        
        return StructuredTool.from_function(
            func=find_path,
            name=tool_name,
            description=description,
            args_schema=PathFindingArgs
        )


# ============================================================================
# Tool Factory
# ============================================================================

def create_uc_toolkit(
    catalog: str = "workspace",
    schema: str = "rm_agent"
) -> list:
    """
    Create a complete toolkit of UC-integrated tools
    """
    tools = []
    
    # Vector search tools
    vs_gen = VectorSearchToolGenerator()
    tools.append(vs_gen.create_search_tool(
        index_name=f"{catalog}.{schema}.ut_pdf_docs_vs_index",
        tool_name="search_unit_trust_documents",
        description="Search unit trust documentation for relevant information"
    ))
    
    # Table lookup tools
    table_gen = UCTableToolGenerator(catalog, schema)
    tools.append(table_gen.create_lookup_tool(
        table_name="customer_profile",
        lookup_column="Name",
        return_columns=["CustomerID", "Name", "RiskRating", "AccountBalance"],
        tool_name="lookup_customer_by_name",
        description="Look up customer information by name"
    ))
    
    tools.append(table_gen.create_lookup_tool(
        table_name="unit_trust",
        lookup_column="ProductID",
        return_columns=["ProductID", "ProductName", "RiskRating", "Currency", "TotalAssets"],
        tool_name="lookup_unit_trust",
        description="Look up unit trust details by product ID"
    ))
    
    return tools


__all__ = [
    "uc_function",
    "generate_uc_function_sql",
    "VectorSearchToolGenerator",
    "UCTableToolGenerator",
    "GraphRAGToolGenerator",
    "create_uc_toolkit",
]
