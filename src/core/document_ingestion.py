"""
Document ingestion pipeline for the unified RAG system.

Handles processing of various document types (TXT, MD, PDF, DOCX)
and ingests them into the vector store with appropriate chunking.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
import hashlib
import os


class DocumentChunker:
    """
    Splits documents into chunks suitable for vector storage.
    
    Supports:
    - Size-based chunking with overlap
    - Header-based chunking for Markdown files
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_size(self, text: str) -> list[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(sep)
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_by_headers(self, text: str) -> list[str]:
        """
        Split Markdown text by headers, then by size if needed.
        
        Args:
            text: Markdown text to chunk
            
        Returns:
            List of text chunks
        """
        import re
        
        # Split by markdown headers (##, ###, etc.)
        header_pattern = r'^(#{1,6}\s+.+)$'
        lines = text.split('\n')
        
        sections = []
        current_section = []
        current_header = ""
        
        for line in lines:
            if re.match(header_pattern, line):
                # Save previous section
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append((current_header, section_text))
                
                current_header = line.strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        # Save last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append((current_header, section_text))
        
        # Further chunk large sections
        chunks = []
        for header, section_text in sections:
            if len(section_text) <= self.chunk_size:
                chunks.append(section_text)
            else:
                # Use size-based chunking for large sections
                sub_chunks = self.chunk_by_size(section_text)
                chunks.extend(sub_chunks)
        
        return chunks


class DocumentIngester:
    """
    Ingests documents into the vector store.
    
    Supports:
    - Plain text (.txt)
    - Markdown (.md)
    - PDF (.pdf) - requires pypdf
    - Word documents (.docx) - requires python-docx
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx'}
    
    def __init__(self, vector_store, chunker: Optional[DocumentChunker] = None):
        """
        Initialize the ingester.
        
        Args:
            vector_store: VectorStore instance to add documents to
            chunker: Optional DocumentChunker (created with defaults if not provided)
        """
        self.vs = vector_store
        self.chunker = chunker or DocumentChunker()
    
    def _generate_doc_id(self, file_path: Path, content: str) -> str:
        """Generate a stable ID for a document."""
        key = f"{file_path.name}:{hashlib.md5(content.encode()).hexdigest()[:8]}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _read_file(self, file_path: Path) -> str:
        """
        Read file content based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        ext = file_path.suffix.lower()
        
        if ext in {'.txt', '.md'}:
            return file_path.read_text(encoding='utf-8')
        
        elif ext == '.pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError("pypdf not installed. Run: pip install pypdf")
        
        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(str(file_path))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except ImportError:
                raise ImportError("python-docx not installed. Run: pip install python-docx")
        
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def ingest_file(self, file_path: str | Path) -> dict:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with ingestion results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return {"success": False, "error": f"Unsupported extension: {file_path.suffix}"}
        
        try:
            # Read content
            content = self._read_file(file_path)
            
            if not content.strip():
                return {"success": False, "error": "Empty file"}
            
            # Determine chunking strategy
            if file_path.suffix.lower() == '.md':
                chunks = self.chunker.chunk_by_headers(content)
            else:
                chunks = self.chunker.chunk_by_size(content)
            
            if not chunks:
                return {"success": False, "error": "No chunks generated"}
            
            # Prepare metadata
            base_metadata = {
                "source": str(file_path.name),
                "type": file_path.suffix.lower().lstrip('.'),
                "ingested_at": datetime.now().isoformat(),
                "full_path": str(file_path.absolute())
            }
            
            # Generate IDs for each chunk
            doc_id_base = self._generate_doc_id(file_path, content)
            chunk_ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(chunks))]
            
            # Add chunk index to metadata
            metadatas = []
            for i in range(len(chunks)):
                meta = base_metadata.copy()
                meta["chunk_index"] = i
                meta["total_chunks"] = len(chunks)
                metadatas.append(meta)
            
            # Add to vector store
            self.vs.add_documents(
                contents=chunks,
                metadatas=metadatas,
                doc_ids=chunk_ids
            )
            
            return {
                "success": True,
                "file": str(file_path.name),
                "chunks": len(chunks),
                "ids": chunk_ids
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ingest_directory(
        self, 
        directory: str | Path, 
        extensions: Optional[list[str]] = None,
        recursive: bool = True
    ) -> list[dict]:
        """
        Ingest all supported files in a directory.
        
        Args:
            directory: Directory path
            extensions: Optional list of extensions to include (e.g., ['.md', '.txt'])
            recursive: Whether to search subdirectories
            
        Returns:
            List of ingestion results
        """
        directory = Path(directory)
        
        if not directory.is_dir():
            return [{"success": False, "error": f"Not a directory: {directory}"}]
        
        if extensions is None:
            extensions = list(self.SUPPORTED_EXTENSIONS)
        
        # Normalize extensions
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in extensions]
        
        results = []
        
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                result = self.ingest_file(file_path)
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Quick test
    from src.core.vector_store import VectorStore
    
    vs = VectorStore()
    ingester = DocumentIngester(vs)
    
    # Test with a markdown file if it exists
    test_file = Path("README.md")
    if test_file.exists():
        result = ingester.ingest_file(test_file)
        print(f"Ingestion result: {result}")
        print(f"Vector store stats: {vs.get_stats()}")
