import pandas as pd
import hashlib
from src.core.database import DatabaseManager

class IngestionService:
    def __init__(self):
        self.db = DatabaseManager()
        self.conn = self.db.get_connection()

    def process_file(self, uploaded_file):
        """
        Reads a file (CSV/Excel), normalizes it, and loads it into DuckDB.
        Avoids duplicates using hash checking.
        """
        # 1. Calculate Hash
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        filename = uploaded_file.name
        
        # Check if file already exists
        result = self.conn.execute("SELECT 1 FROM metadata_registry WHERE file_hash = ?", (file_hash,)).fetchone()
        if result:
            return False, f"File '{filename}' already exists."

        # 2. Load into Pandas
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return False, "Unsupported file format."

        # 3. Sanitize Column Names
        df.columns = [self._sanitize_column(col) for col in df.columns]

        # 4. Create Table Name
        table_name = self._sanitize_table_name(filename)

        # 5. Load to DuckDB
        try:
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            # 6. Register Metadata
            self.db.register_file(filename, file_hash, len(df))
            
            return True, f"Successfully loaded '{filename}' into table '{table_name}'."
        except Exception as e:
            return False, f"Error loading file: {str(e)}"

    def _sanitize_column(self, col_name: str) -> str:
        """Converts column names to snake_case."""
        return col_name.strip().lower().replace(' ', '_').replace('-', '_')

    def _sanitize_table_name(self, filename: str) -> str:
        """Converts filename to a valid SQL table name."""
        name = filename.rsplit('.', 1)[0]
        return self._sanitize_column(name)
