import duckdb
import os
from datetime import datetime
import uuid

class DatabaseManager:
    _instance = None
    
    def __new__(cls, db_path="data/warehouse.db"):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance._initialize_db()
        return cls._instance

    def _initialize_db(self):
        """Initialize the database connection and schema."""
        self.conn = duckdb.connect(self.db_path)
        
        # Create Metadata Registry Table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata_registry (
                file_id VARCHAR PRIMARY KEY,
                filename VARCHAR,
                upload_timestamp TIMESTAMP,
                file_hash VARCHAR,
                row_count INTEGER
            )
        """)
        
        # Create Telemetry Log Table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_log (
                log_id VARCHAR PRIMARY KEY,
                user_id VARCHAR,
                timestamp TIMESTAMP,
                query_text VARCHAR,
                response_type VARCHAR,
                user_feedback INTEGER,
                synced_at TIMESTAMP
            )
        """)

    def get_connection(self):
        """Returns the persistent DuckDB connection."""
        return self.conn

    def execute_query(self, query: str):
        """Executes a SQL query and returns the result."""
        return self.conn.sql(query)
    
    def register_file(self, filename: str, file_hash: str, row_count: int):
        """Registers a new file in the metadata registry."""
        file_id = str(uuid.uuid4())
        upload_timestamp = datetime.now()
        self.conn.execute("""
            INSERT INTO metadata_registry VALUES (?, ?, ?, ?, ?)
        """, (file_id, filename, upload_timestamp, file_hash, row_count))
        return file_id

    def log_interaction(self, user_id: str, query_text: str, response_type: str):
        """Logs user interaction for telemetry."""
        log_id = str(uuid.uuid4())
        timestamp = datetime.now()
        self.conn.execute("""
            INSERT INTO telemetry_log (log_id, user_id, timestamp, query_text, response_type) 
            VALUES (?, ?, ?, ?, ?)
        """, (log_id, user_id, timestamp, query_text, response_type))

