"""
Graph API routes — serve Neo4j graph data as REST endpoints.

Endpoints:
  GET /api/v1/graph/{repo_id}                  — full graph (nodes + edges)
  GET /api/v1/graph/{repo_id}/stats            — node/edge counts per type
  GET /api/v1/graph/{repo_id}/calls/{func_id}  — call chain for a function
  GET /api/v1/graph/{repo_id}/path             — shortest call path between 2 functions
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from sqlalchemy import select
from db.session import get_db
from db.models import Repository
from api.schemas.graph_schema import (
    GraphResponse, GraphNode, GraphEdge,
    CallChainResponse, GraphStatsResponse,
)
from graphs.traversal import (
    get_repo_graph_for_api,
    get_call_chain,
    get_callers,
    find_path_between_functions,
)
from graphs.neo4j_writer import get_repo_graph_stats
from core.logging import get_logger

router = APIRouter(prefix="/graph", tags=["Graph"])
logger = get_logger(__name__)


async def _get_repo_or_404(repo_id: str, db: AsyncSession) -> Repository:
    """Verify repo exists in PostgreSQL. Raises 404 if not found."""
    result = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = result.scalar_one_or_none()
    if repo is None:
        raise HTTPException(status_code=404, detail=f"Repository {repo_id} not found.")
    return repo


@router.get("/{repo_id}", response_model=GraphResponse)
async def get_repo_graph(
    repo_id: str,
    limit: int = Query(default=500, le=2000, description="Max nodes to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the full graph for a repository as nodes + edges.
    Response is ready for D3.js force-directed visualization.
    """
    await _get_repo_or_404(repo_id, db)

    try:
        graph_data = get_repo_graph_for_api(repo_id, limit_nodes=limit)
    except Exception as e:
        logger.error("graph_api_fetch_failed", repo_id=repo_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch graph from Neo4j.")

    nodes = [
        GraphNode(
            id=n["id"],
            label=n["label"],
            name=n["name"] or "",
            file=n.get("file"),
            start_line=n.get("start_line"),
        )
        for n in graph_data["nodes"] if n.get("id")
    ]
    edges = [
        GraphEdge(source=e["source"], target=e["target"], type=e["type"])
        for e in graph_data["edges"] if e.get("source") and e.get("target")
    ]

    return GraphResponse(
        repo_id=repo_id,
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
    )


@router.get("/{repo_id}/stats", response_model=GraphStatsResponse)
async def get_graph_stats(repo_id: str, db: AsyncSession = Depends(get_db)):
    """Return node and edge counts by type for a repository's graph."""
    await _get_repo_or_404(repo_id, db)
    stats = get_repo_graph_stats(repo_id)
    return GraphStatsResponse(**stats)


@router.get("/{repo_id}/calls/{function_id}", response_model=CallChainResponse)
async def get_function_call_chain(
    repo_id: str,
    function_id: str,
    depth: int = Query(default=3, le=6, description="Max call chain depth"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the full call chain for a function — everything it calls, transitively.
    Also returns callers (who calls this function).
    """
    await _get_repo_or_404(repo_id, db)

    call_chain = get_call_chain(function_id, depth=depth)
    callers = get_callers(function_id, depth=2)
    func_name = call_chain[0]["name"] if call_chain else function_id

    return CallChainResponse(
        function_id=function_id,
        function_name=func_name,
        call_chain=call_chain,
        callers=callers,
    )


@router.get("/{repo_id}/path")
async def get_path_between_functions(
    repo_id: str,
    source_id: str = Query(..., description="UUID of the source function"),
    target_id: str = Query(..., description="UUID of the target function"),
    max_depth: int = Query(default=5, le=8),
    db: AsyncSession = Depends(get_db),
):
    """
    Find the shortest call path between two functions.
    Returns the list of intermediate functions in order.
    """
    await _get_repo_or_404(repo_id, db)
    path = find_path_between_functions(source_id, target_id, max_depth=max_depth)
    return {
        "source_id": source_id,
        "target_id": target_id,
        "path": path,
        "length": len(path),
    }