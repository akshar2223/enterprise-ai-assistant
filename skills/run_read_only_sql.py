from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from skills.validate_sql_ast import validate_sql_ast

def run_read_only_sql(sql_query: str, db_connection_str: str) -> List[Dict[str, Any]]:
    """Executes a SQL query after confirming read-only AST safety."""
    if not validate_sql_ast(sql_query):
        raise ValueError("Security Violation: SQL statement is not a read-only SELECT query.")
        
    with psycopg2.connect(db_connection_str) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            return [dict(row) for row in cur.fetchall()]
