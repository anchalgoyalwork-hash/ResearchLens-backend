import os

import httpx
from dotenv import load_dotenv


load_dotenv()

OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")


async def search_openalex(query: str, per_page: int = 10):

    params = {
        "search": query,
        "per-page": per_page,
        "api_key": OPENALEX_API_KEY,
        "mailto": "anchalgoyalwork@gmail.com",
    }

    headers = {
        "User-Agent": "ResearchLens/0.1 (mailto:anchalgoyalwork@gmail.com)"
    }

    async with httpx.AsyncClient() as client:

        response = await client.get(
            OPENALEX_URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 429:
            return {
                "error": "OpenAlex rate limit reached",
                "message": "Please wait before searching again.",
                "results": [],
            }

        response.raise_for_status()

        return response.json()