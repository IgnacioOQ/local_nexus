"""
Unit tests for the vector store and document ingestion modules (Phase 1).
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestVectorStore:
    """Tests for the VectorStore class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def vector_store(self, temp_db):
        """Create a VectorStore instance with temp storage."""
        from src.core.vector_store import VectorStore
        return VectorStore(db_path=temp_db, collection_name="test_collection")
    
    def test_add_single_document(self, vector_store):
        """Test adding a single document."""
        doc_id = vector_store.add_document(
            "Machine learning is a subset of AI.",
            metadata={"type": "definition"}
        )
        
        assert doc_id is not None
        assert vector_store.get_stats()["count"] == 1
    
    def test_add_multiple_documents(self, vector_store):
        """Test adding multiple documents in batch."""
        contents = [
            "Python is a programming language.",
            "JavaScript runs in browsers.",
            "Rust is memory-safe."
        ]
        metadatas = [{"lang": "python"}, {"lang": "js"}, {"lang": "rust"}]
        
        ids = vector_store.add_documents(contents, metadatas)
        
        assert len(ids) == 3
        assert vector_store.get_stats()["count"] == 3
    
    def test_query_returns_relevant_results(self, vector_store):
        """Test that queries return semantically relevant results."""
        vector_store.add_documents([
            "Machine learning uses algorithms to learn from data.",
            "Cooking requires fresh ingredients.",
            "Deep learning is a type of machine learning."
        ])
        
        results = vector_store.query("What is ML?", n_results=2)
        
        assert len(results["documents"][0]) == 2
        # First result should be about ML, not cooking
        assert "learning" in results["documents"][0][0].lower() or \
               "learning" in results["documents"][0][1].lower()
    
    def test_batch_query(self, vector_store):
        """Test batch querying multiple queries at once."""
        vector_store.add_documents([
            "Python is great for data science.",
            "JavaScript is used for web development.",
        ])
        
        results = vector_store.query(
            ["Python programming", "Web development"],
            n_results=1
        )
        
        # Should have results for both queries
        assert len(results["documents"]) == 2
    
    def test_metadata_filter(self, vector_store):
        """Test filtering by metadata."""
        vector_store.add_documents(
            ["Doc about Python", "Doc about JavaScript"],
            [{"lang": "python"}, {"lang": "javascript"}]
        )
        
        results = vector_store.query(
            "programming",
            n_results=5,
            where={"lang": "python"}
        )
        
        assert len(results["documents"][0]) == 1
        assert "Python" in results["documents"][0][0]
    
    def test_deduplication(self, vector_store):
        """Test that duplicate content is deduplicated."""
        # Add same content twice
        id1 = vector_store.add_document("Same content here")
        id2 = vector_store.add_document("Same content here")
        
        # IDs should be the same (content-based)
        assert id1 == id2
        # Should only have 1 document
        assert vector_store.get_stats()["count"] == 1
    
    def test_delete(self, vector_store):
        """Test deleting documents."""
        ids = vector_store.add_documents(["Doc 1", "Doc 2"])
        assert vector_store.get_stats()["count"] == 2
        
        vector_store.delete([ids[0]])
        assert vector_store.get_stats()["count"] == 1
    
    def test_clear(self, vector_store):
        """Test clearing all documents."""
        vector_store.add_documents(["Doc 1", "Doc 2", "Doc 3"])
        assert vector_store.get_stats()["count"] == 3
        
        vector_store.clear()
        assert vector_store.get_stats()["count"] == 0


class TestDocumentChunker:
    """Tests for the DocumentChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker with small sizes for testing."""
        from src.core.document_ingestion import DocumentChunker
        return DocumentChunker(chunk_size=100, chunk_overlap=10)
    
    def test_short_text_single_chunk(self, chunker):
        """Test that short text stays as single chunk."""
        text = "This is a short text."
        chunks = chunker.chunk_by_size(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_long_text_multiple_chunks(self, chunker):
        """Test that long text is split into multiple chunks."""
        text = "This is a sentence. " * 20  # ~400 chars
        chunks = chunker.chunk_by_size(text)
        
        assert len(chunks) > 1
    
    def test_markdown_header_chunking(self, chunker):
        """Test chunking markdown by headers."""
        md_text = """# Heading 1
Content under heading 1.

## Heading 2
Content under heading 2.

## Heading 3
Content under heading 3."""
        
        chunks = chunker.chunk_by_headers(md_text)
        
        assert len(chunks) >= 3  # At least one chunk per section


class TestDocumentIngester:
    """Tests for the DocumentIngester class."""
    
    @pytest.fixture
    def temp_setup(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = os.path.join(tmpdir, "vectordb")
            docs_dir = os.path.join(tmpdir, "docs")
            os.makedirs(docs_dir)
            yield {"db_dir": db_dir, "docs_dir": docs_dir}
    
    @pytest.fixture
    def ingester(self, temp_setup):
        """Create an ingester with temp storage."""
        from src.core.vector_store import VectorStore
        from src.core.document_ingestion import DocumentIngester
        
        vs = VectorStore(db_path=temp_setup["db_dir"])
        return DocumentIngester(vs), temp_setup
    
    def test_ingest_txt_file(self, ingester):
        """Test ingesting a plain text file."""
        ing, setup = ingester
        
        # Create test file
        txt_path = Path(setup["docs_dir"]) / "test.txt"
        txt_path.write_text("This is a test document with some content.")
        
        result = ing.ingest_file(txt_path)
        
        assert result["success"] is True
        assert result["chunks"] >= 1
    
    def test_ingest_md_file(self, ingester):
        """Test ingesting a markdown file."""
        ing, setup = ingester
        
        md_path = Path(setup["docs_dir"]) / "test.md"
        md_path.write_text("""# Title
First section content.

## Section 2
Second section content.""")
        
        result = ing.ingest_file(md_path)
        
        assert result["success"] is True
        assert "md" in result["file"]
    
    def test_ingest_nonexistent_file(self, ingester):
        """Test that nonexistent file returns error."""
        ing, _ = ingester
        
        result = ing.ingest_file("/nonexistent/path.txt")
        
        assert result["success"] is False
        assert "not found" in result["error"].lower()
    
    def test_ingest_unsupported_extension(self, ingester):
        """Test that unsupported extension returns error."""
        ing, setup = ingester
        
        unsupported = Path(setup["docs_dir"]) / "test.xyz"
        unsupported.write_text("content")
        
        result = ing.ingest_file(unsupported)
        
        assert result["success"] is False
        assert "unsupported" in result["error"].lower()
    
    def test_ingest_directory(self, ingester):
        """Test ingesting all files in a directory."""
        ing, setup = ingester
        docs_dir = Path(setup["docs_dir"])
        
        # Create multiple test files
        (docs_dir / "doc1.txt").write_text("Document 1 content.")
        (docs_dir / "doc2.txt").write_text("Document 2 content.")
        (docs_dir / "doc3.md").write_text("# Markdown\nContent here.")
        
        results = ing.ingest_directory(docs_dir, extensions=[".txt", ".md"])
        
        assert len(results) == 3
        assert all(r["success"] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
