from fastapi import APIRouter

from app.services.openalex import search_openalex


router = APIRouter(
    prefix="/search",
    tags=["Literature Search"],
)


@router.get("/literature")
async def literature_search(q: str):

    data = await search_openalex(q)

    return {
        "query": q,
        "results": data.get("results", []),
    }