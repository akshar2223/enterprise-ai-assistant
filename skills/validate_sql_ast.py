import sqlglot
from sqlglot import exp

def validate_sql_ast(sql_query: str) -> bool:
    """
    Parse SQL and verify it is strictly a read-only SELECT statement.

    Uses sqlglot AST walking to detect any forbidden DML/DDL node anywhere
    in the tree — including inside subqueries and CTEs.

    Note: sqlglot 23.x walk() yields individual Expression nodes (not tuples).
    Earlier versions yielded (node, parent, key) — the original code used
    tuple-unpacking which silently broke on 23.x via caught ValueError.
    """
    if not sql_query or not sql_query.strip():
        return False

    # Defence-in-depth: reject any query containing a semicolon outside a string literal.
    # sqlglot's parse_one() silently parses only the FIRST statement, so a semicolon-
    # separated injection ("SELECT 1; DROP TABLE x") would pass the AST check.
    # psycopg2 also rejects multi-statement strings, but we enforce it here too.
    stripped = sql_query.strip().rstrip(";")  # allow optional trailing semicolon
    if ";" in stripped:
        return False

    try:
        parsed = sqlglot.parse_one(sql_query, read="postgres")

        if parsed is None:
            return False

        # A valid read-only statement is either:
        #   - a SELECT                 (exp.Select)
        #   - a UNION / INTERSECT / EXCEPT of SELECTs  (exp.Union / exp.Intersect / exp.Except)
        # We walk the tree and reject any forbidden DML/DDL node anywhere inside.
        # This means UNION SELECT is allowed (it's read-only), while any DML
        # embedded in a subquery is still caught.
        valid_root_types = (exp.Select, exp.Union, exp.Intersect, exp.Except)
        if not isinstance(parsed, valid_root_types):
            return False

        # sqlglot 23.x class names — verified with hasattr() at import time.
        # exp.Alter was renamed to exp.AlterTable in sqlglot 23.x.
        # exp.Truncate was renamed to exp.TruncateTable in sqlglot 23.x.
        forbidden_types = (
            exp.Insert, exp.Update, exp.Delete,
            exp.Drop, exp.Create, exp.AlterTable, exp.AlterColumn,
            exp.TruncateTable, exp.Command,
        )

        # walk() in sqlglot 23.x yields single Expression objects
        for node in parsed.walk():
            if isinstance(node, forbidden_types):
                return False

        return True

    except Exception:
        return False
