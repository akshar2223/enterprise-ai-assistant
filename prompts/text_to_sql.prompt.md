# Text-to-SQL Prompt Template

Target Database Dialect: PostgreSQL 16
Schema Context:
{{ database_schema_ddl }}

User Request:
{{ user_request }}

Generate a syntactically correct PostgreSQL `SELECT` query that answers the request.
Respond strictly in JSON:
{
  "sql_query": "<GENERATED_SQL>",
  "explanation": "<brief query logic>"
}
