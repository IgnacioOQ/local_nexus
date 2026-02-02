"""
Unified RAG Engine for Local Nexus.

This module orchestrates retrieval from multiple sources:
- VectorStore (ChromaDB) for unstructured document search
- Text2SQL (DuckDB) for structured data queries
- KnowledgeGraphMetadata for query expansion and guided retrieval
- Hybrid queries combining both sources

Key Features:
1. Smart Retrieval (Query Decomposition) - breaks complex questions into sub-queries
2. Batch Vector Search - 82% latency reduction via parallel queries
3. KG-Guided Retrieval - expands queries based on entity-data links

Inspired by mcmp_chatbot RAGEngine patterns.
"""

import functools
from typing import Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RetrievalResult:
    """Container for retrieval results from any source."""
    source: str  # 'vector', 'sql', 'hybrid'
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class EngineResponse:
    """Complete response from the unified engine."""
    answer: str
    query_type: str
    sources: list[RetrievalResult] = field(default_factory=list)
    sql_query: Optional[str] = None
    decomposed_queries: list[str] = field(default_factory=list)
    error: Optional[str] = None


class UnifiedEngine:
    """
    Main orchestrator for the unified RAG + Data Warehouse system.

    Features:
    - Query decomposition with LRU caching for complex questions
    - Multi-source retrieval (VectorStore + DuckDB)
    - KG-guided query expansion for finding related sources
    - Batch vector search for 82% latency reduction
    - Context assembly for LLM response generation
    - Deduplication of retrieved content

    Usage:
        engine = UnifiedEngine(
            vector_store=vs,
            db_connection=conn,
            kg_metadata=kg,
            llm_func=my_llm_function
        )
        response = engine.query("What are the total sales?")
    """

    def __init__(
        self,
        vector_store=None,
        db_connection=None,
        kg_metadata=None,
        llm_func: Optional[Callable[[str], str]] = None,
        enable_decomposition: bool = True,
        max_context_tokens: int = 4000
    ):
        """
        Initialize the unified engine.

        Args:
            vector_store: VectorStore instance for document retrieval
            db_connection: DuckDB connection for SQL queries
            kg_metadata: KnowledgeGraphMetadata for query expansion
            llm_func: Function for LLM calls (prompt: str) -> str
            enable_decomposition: Whether to decompose complex queries
            max_context_tokens: Approximate max tokens for context (chars/4)
        """
        self.vector_store = vector_store
        self.db_connection = db_connection
        self.kg_metadata = kg_metadata
        self.llm_func = llm_func
        self.enable_decomposition = enable_decomposition
        self.max_context_chars = max_context_tokens * 4  # Rough estimate

        # Initialize components lazily
        self._router = None
        self._text2sql = None

    @property
    def router(self):
        """Lazy initialization of QueryRouter."""
        if self._router is None:
            from src.core.query_router import QueryRouter
            self._router = QueryRouter()
        return self._router

    @property
    def text2sql(self):
        """Lazy initialization of Text2SQL."""
        if self._text2sql is None and self.db_connection:
            from src.core.text2sql import Text2SQL
            self._text2sql = Text2SQL(
                self.db_connection,
                llm_func=self.llm_func,
                read_only=True
            )
        return self._text2sql

    def _decompose_query_impl(self, question: str) -> list[str]:
        """
        Decompose a complex question into simpler sub-questions.

        Uses LLM to break down multi-part questions.
        """
        if not self.llm_func:
            return [question]

        prompt = f"""Break down this question into 1-3 simpler sub-questions that can be answered independently.
If the question is already simple, return just the original question.

Question: "{question}"

Return ONLY the sub-questions, one per line, no numbering or bullets.
If no decomposition needed, return the original question."""

        try:
            response = self.llm_func(prompt)
            lines = [
                line.strip()
                for line in response.strip().split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*', '•'))
            ]
            # Remove numbering if present
            cleaned = []
            for line in lines:
                # Remove patterns like "1.", "1)", "1:"
                import re
                cleaned_line = re.sub(r'^\d+[\.\)\:]\s*', '', line)
                if cleaned_line:
                    cleaned.append(cleaned_line)

            return cleaned[:3] if cleaned else [question]
        except Exception:
            return [question]

    @functools.lru_cache(maxsize=128)
    def decompose_query(self, question: str) -> tuple:
        """
        Cached query decomposition.

        Args:
            question: User's natural language question

        Returns:
            Tuple of sub-questions (tuple for hashability/caching)
        """
        if not self.enable_decomposition:
            return (question,)
        return tuple(self._decompose_query_impl(question))

    def _retrieve_from_vector_store(
        self,
        queries: list[str],
        top_k: int = 3
    ) -> list[RetrievalResult]:
        """
        Retrieve documents from vector store.

        Uses batch queries for efficiency.
        """
        if not self.vector_store:
            return []

        try:
            # Batch query for all sub-questions
            results = self.vector_store.query(
                query_texts=queries,
                n_results=top_k
            )

            # Flatten and deduplicate results
            seen_ids = set()
            retrieval_results = []

            for query_idx, query in enumerate(queries):
                if query_idx >= len(results.get('documents', [])):
                    continue

                docs = results['documents'][query_idx]
                ids = results['ids'][query_idx]
                metadatas = results.get('metadatas', [[]])[query_idx]
                distances = results.get('distances', [[]])[query_idx]

                for i, (doc, doc_id) in enumerate(zip(docs, ids)):
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    meta = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 0.0

                    retrieval_results.append(RetrievalResult(
                        source='vector',
                        content=doc,
                        metadata={
                            'id': doc_id,
                            'query': query,
                            **meta
                        },
                        score=1.0 - distance  # Convert distance to similarity
                    ))

            return retrieval_results

        except Exception as e:
            return []

    def _retrieve_from_sql(
        self, 
        question: str,
        table_hints: Optional[list[str]] = None
    ) -> list[RetrievalResult]:
        """
        Retrieve data from DuckDB via Text2SQL.
        
        Args:
            question: Natural language question
            table_hints: Optional list of table names suggested by KG
        """
        if not self.text2sql:
            return []

        try:
            # Expand question with table hints from KG
            expanded_question = question
            if table_hints:
                hint_text = ", ".join(table_hints)
                expanded_question = f"{question} (Consider tables: {hint_text})"
            
            result = self.text2sql.query(expanded_question)

            if not result.success:
                return [RetrievalResult(
                    source='sql',
                    content=f"SQL Error: {result.error}",
                    metadata={'error': True}
                )]

            # Format SQL results as text
            if result.data:
                # Create a formatted table
                headers = result.columns or []
                rows = result.data[:20]  # Limit rows

                content_parts = [f"SQL Query: {result.sql}"]
                content_parts.append(f"Results ({result.row_count} rows):")

                if headers:
                    content_parts.append(" | ".join(str(h) for h in headers))
                    content_parts.append("-" * 50)

                for row in rows:
                    content_parts.append(" | ".join(str(v) for v in row))

                if result.row_count > 20:
                    content_parts.append(f"... and {result.row_count - 20} more rows")

                return [RetrievalResult(
                    source='sql',
                    content='\n'.join(content_parts),
                    metadata={
                        'sql': result.sql,
                        'row_count': result.row_count,
                        'columns': headers
                    },
                    score=1.0
                )]

            return [RetrievalResult(
                source='sql',
                content=f"SQL Query: {result.sql}\nNo results found.",
                metadata={'sql': result.sql, 'row_count': 0}
            )]

        except Exception as e:
            return [RetrievalResult(
                source='sql',
                content=f"SQL Error: {str(e)}",
                metadata={'error': True}
            )]

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        force_type: Optional[str] = None
    ) -> tuple[str, list[RetrievalResult]]:
        """
        Retrieve relevant information based on query type.

        Uses KG metadata layer to expand queries before retrieval.

        Args:
            question: User's natural language question
            top_k: Number of results per source
            force_type: Override query classification ('structured', 'unstructured', 'hybrid')

        Returns:
            Tuple of (query_type, list of RetrievalResults)
        """
        from src.core.query_router import QueryType

        # Classify query
        if force_type:
            query_type = force_type
        else:
            classification = self.router.classify(question)
            query_type = classification.value

        # ★ KG-Guided Query Expansion ★
        # Expand query using Knowledge Graph before retrieval
        retrieval_plan = None
        sql_hints = []
        extra_vector_queries = []
        
        if self.kg_metadata:
            retrieval_plan = self.kg_metadata.expand_query(question, query_type)
            sql_hints = retrieval_plan.sql_hints
            extra_vector_queries = retrieval_plan.vector_queries[1:]  # Skip original query
            
            # Upgrade to hybrid if KG found SQL-related links for unstructured query
            if retrieval_plan.expanded and sql_hints and query_type == 'unstructured':
                query_type = 'hybrid'
                print(f"DEBUG: KG expanded to hybrid, sql_hints={sql_hints}")

        results = []

        if query_type == 'structured':
            results = self._retrieve_from_sql(question, table_hints=sql_hints)

        elif query_type == 'unstructured':
            # Decompose and retrieve using batch queries (82% latency improvement)
            sub_queries = list(self.decompose_query(question))
            # Add KG-suggested queries
            all_queries = sub_queries + extra_vector_queries
            results = self._retrieve_from_vector_store(all_queries, top_k)

        elif query_type == 'hybrid':
            # Get both types of results
            sub_queries = list(self.decompose_query(question))
            all_queries = sub_queries + extra_vector_queries
            vector_results = self._retrieve_from_vector_store(all_queries, top_k)
            sql_results = self._retrieve_from_sql(question, table_hints=sql_hints)
            results = vector_results + sql_results

        return query_type, results

    def _assemble_context(self, results: list[RetrievalResult]) -> str:
        """
        Assemble retrieved results into context for LLM.

        Respects max_context_chars limit.
        """
        if not results:
            return "No relevant information found."

        context_parts = []
        current_chars = 0

        for i, result in enumerate(results):
            section = f"[Source {i+1}: {result.source}]\n{result.content}\n"

            if current_chars + len(section) > self.max_context_chars:
                # Truncate if needed
                remaining = self.max_context_chars - current_chars
                if remaining > 100:
                    section = section[:remaining] + "...[truncated]"
                    context_parts.append(section)
                break

            context_parts.append(section)
            current_chars += len(section)

        return "\n".join(context_parts)

    def _generate_answer(
        self,
        question: str,
        context: str,
        query_type: str
    ) -> str:
        """
        Generate final answer using LLM.
        """
        if not self.llm_func:
            return f"Context retrieved but no LLM configured.\n\n{context}"

        type_instructions = {
            'structured': "The context contains data from database queries. Provide precise, data-driven answers.",
            'unstructured': "The context contains document excerpts. Synthesize information to answer the question.",
            'hybrid': "The context contains both database results and document excerpts. Combine both to provide a comprehensive answer."
        }

        prompt = f"""Answer the following question using ONLY the provided context.
{type_instructions.get(query_type, '')}

If the context doesn't contain enough information to answer, say so clearly.
Be concise and direct.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        try:
            return self.llm_func(prompt)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query(
        self,
        question: str,
        top_k: int = 3,
        force_type: Optional[str] = None,
        return_sources: bool = True
    ) -> EngineResponse:
        """
        End-to-end query processing.
        """
        import time
        start_t = time.time()
        print(f"DEBUG: Engine.query start: '{question}'")
        try:
            # Get decomposed queries for reporting
            print("DEBUG: Decomposing query...")
            decomposed = list(self.decompose_query(question))
            print(f"DEBUG: Decomposed into {len(decomposed)} sub-queries")

            # Retrieve
            print("DEBUG: Starting retrieval...")
            query_type, results = self.retrieve(question, top_k, force_type)
            print(f"DEBUG: Retrieval done. Type: {query_type}, Results: {len(results)}")

            # Assemble context
            context = self._assemble_context(results)
            print(f"DEBUG: Context assembled ({len(context)} chars)")

            # Generate answer
            print("DEBUG: Generating answer with LLM...")
            answer = self._generate_answer(question, context, query_type)
            print(f"DEBUG: Answer generated in {time.time() - start_t:.2f}s")

            # Extract SQL if present
            sql_query = None
            for r in results:
                if r.source == 'sql' and 'sql' in r.metadata:
                    sql_query = r.metadata['sql']
                    break

            return EngineResponse(
                answer=answer,
                query_type=query_type,
                sources=results if return_sources else [],
                sql_query=sql_query,
                decomposed_queries=decomposed if len(decomposed) > 1 else []
            )

        except Exception as e:
            print(f"ERROR: Engine.query failed: {e}")
            import traceback
            traceback.print_exc()
            return EngineResponse(
                answer="",
                query_type="error",
                error=str(e)
            )

    def get_stats(self) -> dict:
        """Get engine statistics."""
        stats = {
            'decomposition_enabled': self.enable_decomposition,
            'has_vector_store': self.vector_store is not None,
            'has_db_connection': self.db_connection is not None,
            'has_llm': self.llm_func is not None,
            'decomposition_cache_info': self.decompose_query.cache_info()._asdict()
        }

        if self.vector_store:
            try:
                vs_stats = self.vector_store.get_stats()
                stats['vector_store'] = vs_stats
            except Exception:
                pass

        if self.text2sql:
            try:
                tables = self.text2sql.get_available_tables()
                stats['available_tables'] = tables
            except Exception:
                pass

        if self.kg_metadata:
            try:
                kg_stats = self.kg_metadata.get_stats()
                stats['kg_metadata'] = kg_stats
            except Exception:
                pass

        return stats

    def clear_cache(self):
        """Clear the query decomposition cache."""
        self.decompose_query.cache_clear()


def create_engine_from_defaults(
    db_path: str = "data/warehouse.db",
    vector_db_path: str = "data/vectordb",
    use_gemini: bool = True
) -> UnifiedEngine:
    """
    Factory function to create a UnifiedEngine with default configuration.

    Args:
        db_path: Path to DuckDB database
        vector_db_path: Path to ChromaDB storage
        use_gemini: Whether to use Gemini LLM

    Returns:
        Configured UnifiedEngine instance
    """
    import duckdb

    # Initialize database
    conn = duckdb.connect(db_path)

    # Initialize vector store (may fail if chromadb not installed)
    vector_store = None
    try:
        from src.core.vector_store import VectorStore
        vector_store = VectorStore(db_path=vector_db_path)
    except ImportError:
        pass

    # Initialize LLM
    llm_func = None
    if use_gemini:
        try:
            from src.core.llm import init_gemini, DEFAULT_MODEL
            import google.generativeai as genai
            import os

            if init_gemini():
                def gemini_call(prompt: str) -> str:
                    # Respect env var if set, else use default safe model
                    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text

                llm_func = gemini_call
        except ImportError:
            pass

    return UnifiedEngine(
        vector_store=vector_store,
        db_connection=conn,
        llm_func=llm_func
    )


if __name__ == "__main__":
    # Quick test with mock components
    print("Unified Engine Test")
    print("=" * 50)

    # Create engine without external dependencies
    engine = UnifiedEngine(
        vector_store=None,
        db_connection=None,
        llm_func=lambda p: "Mock LLM response"
    )

    # Test query classification
    test_queries = [
        "How many customers do we have?",
        "What is our refund policy?",
        "Which customers complained about shipping and spent over $1000?",
    ]

    for q in test_queries:
        query_type, _ = engine.retrieve(q)
        decomposed = engine.decompose_query(q)
        print(f"\nQuery: {q}")
        print(f"Type: {query_type}")
        print(f"Decomposed: {decomposed}")

    print(f"\nStats: {engine.get_stats()}")
