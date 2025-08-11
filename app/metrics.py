from fastapi import APIRouter, Query
import sqlite3

metrics_router = APIRouter()
conn = sqlite3.connect("logs/logs.db", check_same_thread=False)
conn.row_factory = sqlite3.Row  # for dict-like access

@metrics_router.get("/metrics")
def get_metrics(
    limit: int = Query(25, ge=1),
    offset: int = Query(0, ge=0),
    status: str = Query(None, description="Filter by 'success' or 'error'"),
    source: str = Query(None, description="Filter by request source (e.g., 'api', 'cli')")
):
    # Total count
    cursor = conn.execute("SELECT COUNT(*) FROM logs")
    total = cursor.fetchone()[0]

    # Success count
    cursor = conn.execute("SELECT COUNT(*) FROM logs WHERE status = 'success'")
    success_count = cursor.fetchone()[0]

    # Error count
    cursor = conn.execute("SELECT COUNT(*) FROM logs WHERE status = 'error'")
    error_count = cursor.fetchone()[0]

    # Build dynamic WHERE clause
    filters = []
    params = []

    if status in ("success", "error"):
        filters.append("status = ?")
        params.append(status)

    if source:
        filters.append("source = ?")
        params.append(source)

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""

    # Paginated logs with filters
    query = f"""
        SELECT * FROM logs
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    cursor = conn.execute(query, params)
    logs = [dict(row) for row in cursor.fetchall()]

    return {
        "total_requests": total,
        "success_count": success_count,
        "error_count": error_count,
        "limit": limit,
        "offset": offset,
        "status_filter": status,
        "source_filter": source,
        "logs": logs
    }
