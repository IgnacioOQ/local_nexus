"""
Text2SQL module for converting natural language queries to SQL.

This module enables natural language querying of the DuckDB data warehouse
by generating SQL from user questions using schema introspection and LLM.
"""

import re
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class SQLResult:
    """Result of a Text2SQL operation."""
    success: bool
    sql: Optional[str] = None
    data: Optional[list] = None
    columns: Optional[list[str]] = None
    error: Optional[str] = None
    row_count: int = 0


@dataclass
class TableSchema:
    """Schema information for a table."""
    name: str
    columns: list[dict]  # [{"name": str, "type": str}, ...]
    row_count: int
    sample_data: list[dict]  # First few rows as dicts


class SchemaIntrospector:
    """
    Introspects DuckDB schema to provide context for SQL generation.

    Extracts table names, column types, and sample data to help
    the LLM generate accurate SQL queries.
    """

    # System tables to exclude from introspection
    SYSTEM_TABLES = {'metadata_registry', 'telemetry_log'}

    def __init__(self, db_connection):
        """
        Initialize with a DuckDB connection.

        Args:
            db_connection: Active DuckDB connection
        """
        self.conn = db_connection

    def get_all_tables(self) -> list[str]:
        """Get list of all user tables (excluding system tables)."""
        try:
            result = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            tables = [row[0] for row in result]
            # Filter out system tables
            return [t for t in tables if t not in self.SYSTEM_TABLES]
        except Exception:
            return []

    def get_table_schema(self, table_name: str, sample_rows: int = 3) -> Optional[TableSchema]:
        """
        Get detailed schema for a specific table.

        Args:
            table_name: Name of the table
            sample_rows: Number of sample rows to include

        Returns:
            TableSchema object or None if table doesn't exist
        """
        try:
            # Get column information
            columns_result = self.conn.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()

            if not columns_result:
                return None

            columns = [{"name": row[0], "type": row[1]} for row in columns_result]

            # Get row count
            count_result = self.conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()
            row_count = count_result[0] if count_result else 0

            # Get sample data
            sample_result = self.conn.execute(
                f"SELECT * FROM \"{table_name}\" LIMIT {sample_rows}"
            ).fetchall()

            column_names = [c["name"] for c in columns]
            sample_data = [dict(zip(column_names, row)) for row in sample_result]

            return TableSchema(
                name=table_name,
                columns=columns,
                row_count=row_count,
                sample_data=sample_data
            )
        except Exception as e:
            return None

    def get_full_schema_context(self, max_tables: int = 10) -> str:
        """
        Generate a formatted schema context string for LLM prompts.

        Args:
            max_tables: Maximum number of tables to include

        Returns:
            Formatted string describing the database schema
        """
        tables = self.get_all_tables()[:max_tables]

        if not tables:
            return "No tables available in the database."

        context_parts = ["DATABASE SCHEMA:\n"]

        for table_name in tables:
            schema = self.get_table_schema(table_name)
            if schema:
                context_parts.append(f"\nTable: {schema.name} ({schema.row_count} rows)")
                context_parts.append("Columns:")
                for col in schema.columns:
                    context_parts.append(f"  - {col['name']}: {col['type']}")

                if schema.sample_data:
                    context_parts.append("Sample data:")
                    for i, row in enumerate(schema.sample_data[:2]):
                        # Truncate long values
                        truncated = {k: str(v)[:50] + "..." if len(str(v)) > 50 else v
                                   for k, v in row.items()}
                        context_parts.append(f"  Row {i+1}: {truncated}")

        return "\n".join(context_parts)


class Text2SQL:
    """
    Converts natural language queries to SQL for DuckDB.

    Features:
    - Schema-aware SQL generation
    - Query validation before execution
    - Safe execution with error handling
    - Support for pluggable LLM backends
    """

    # SQL keywords that indicate potentially dangerous operations
    DANGEROUS_KEYWORDS = {
        'drop', 'delete', 'truncate', 'alter', 'create', 'insert',
        'update', 'grant', 'revoke', 'exec', 'execute'
    }

    def __init__(
        self,
        db_connection,
        llm_func: Optional[Callable[[str], str]] = None,
        read_only: bool = True
    ):
        """
        Initialize the Text2SQL converter.

        Args:
            db_connection: Active DuckDB connection
            llm_func: Function that takes a prompt and returns LLM response.
                     Signature: (prompt: str) -> str
            read_only: If True, reject write operations (default: True)
        """
        self.conn = db_connection
        self.introspector = SchemaIntrospector(db_connection)
        self.llm_func = llm_func
        self.read_only = read_only

    def _build_prompt(self, question: str, schema_context: str) -> str:
        """Build the prompt for SQL generation."""
        return f"""You are a SQL expert. Generate a DuckDB SQL query to answer the user's question.

{schema_context}

RULES:
1. Return ONLY the SQL query, no explanations or markdown
2. Use DuckDB SQL syntax
3. Only use tables and columns that exist in the schema above
4. Use double quotes for column/table names with special characters
5. For aggregations, always include appropriate GROUP BY clauses
6. Limit results to 100 rows unless the user specifies otherwise

USER QUESTION: {question}

SQL QUERY:"""

    def _extract_sql(self, llm_response: str) -> str:
        """Extract SQL from LLM response, handling various formats."""
        response = llm_response.strip()

        # Remove markdown code blocks if present
        if "```sql" in response.lower():
            match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)

        # Remove any leading/trailing whitespace and semicolons
        response = response.strip().rstrip(';')

        return response

    def _validate_sql(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety and correctness.

        Args:
            sql: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        sql_lower = sql.lower()

        # Check for dangerous keywords if read_only mode
        if self.read_only:
            for keyword in self.DANGEROUS_KEYWORDS:
                # Use word boundary matching to avoid false positives
                if re.search(rf'\b{keyword}\b', sql_lower):
                    return False, f"Write operation '{keyword}' not allowed in read-only mode"

        # Must start with SELECT for read-only queries
        if self.read_only and not sql_lower.strip().startswith('select'):
            return False, "Only SELECT queries allowed in read-only mode"

        # Basic syntax validation - try to prepare the query
        try:
            # Use EXPLAIN to validate without executing
            self.conn.execute(f"EXPLAIN {sql}")
            return True, None
        except Exception as e:
            return False, f"SQL syntax error: {str(e)}"

    def generate_sql(self, question: str) -> tuple[Optional[str], Optional[str]]:
        """
        Generate SQL from a natural language question.

        Args:
            question: Natural language question

        Returns:
            Tuple of (sql_query, error_message)
        """
        if not self.llm_func:
            return None, "No LLM function configured"

        # Get schema context
        schema_context = self.introspector.get_full_schema_context()

        if "No tables available" in schema_context:
            return None, "No tables available in the database"

        # Build and send prompt
        prompt = self._build_prompt(question, schema_context)

        try:
            llm_response = self.llm_func(prompt)
            sql = self._extract_sql(llm_response)

            # Validate the generated SQL
            is_valid, error = self._validate_sql(sql)
            if not is_valid:
                return None, error

            return sql, None

        except Exception as e:
            return None, f"LLM error: {str(e)}"

    def execute_sql(self, sql: str) -> SQLResult:
        """
        Execute a SQL query safely.

        Args:
            sql: SQL query to execute

        Returns:
            SQLResult with data or error
        """
        # Validate first
        is_valid, error = self._validate_sql(sql)
        if not is_valid:
            return SQLResult(success=False, error=error)

        try:
            result = self.conn.execute(sql)

            # Get column names
            columns = [desc[0] for desc in result.description] if result.description else []

            # Fetch data
            data = result.fetchall()

            return SQLResult(
                success=True,
                sql=sql,
                data=data,
                columns=columns,
                row_count=len(data)
            )

        except Exception as e:
            return SQLResult(success=False, sql=sql, error=str(e))

    def query(self, question: str) -> SQLResult:
        """
        End-to-end: generate SQL from question and execute it.

        Args:
            question: Natural language question

        Returns:
            SQLResult with data or error
        """
        # Generate SQL
        sql, error = self.generate_sql(question)

        if error:
            return SQLResult(success=False, error=error)

        # Execute and return
        result = self.execute_sql(sql)
        result.sql = sql  # Ensure SQL is included
        return result

    def get_available_tables(self) -> list[str]:
        """Get list of available tables for querying."""
        return self.introspector.get_all_tables()

    def get_table_info(self, table_name: str) -> Optional[TableSchema]:
        """Get schema information for a specific table."""
        return self.introspector.get_table_schema(table_name)


def create_gemini_sql_generator() -> Callable[[str], str]:
    """
    Create a SQL generator function using Gemini.

    Returns:
        Function that takes a prompt and returns LLM response
    """
    from src.core.llm import init_gemini, DEFAULT_MODEL
    from google import genai
    import os

    init_gemini()

    def generate(prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    return generate


if __name__ == "__main__":
    # Quick test with in-memory database
    import duckdb

    # Create test database
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE sales (
            id INTEGER,
            product VARCHAR,
            amount DECIMAL(10,2),
            sale_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO sales VALUES
        (1, 'Widget A', 100.00, '2024-01-15'),
        (2, 'Widget B', 250.00, '2024-01-20'),
        (3, 'Widget A', 150.00, '2024-02-01')
    """)

    # Test schema introspection
    introspector = SchemaIntrospector(conn)
    print("Tables:", introspector.get_all_tables())
    print("\nSchema Context:")
    print(introspector.get_full_schema_context())

    # Test SQL validation (without LLM)
    text2sql = Text2SQL(conn)

    # Valid query
    result = text2sql.execute_sql("SELECT product, SUM(amount) FROM sales GROUP BY product")
    print(f"\nQuery result: {result}")

    # Invalid query (write operation)
    result = text2sql.execute_sql("DELETE FROM sales")
    print(f"Delete attempt: {result}")
