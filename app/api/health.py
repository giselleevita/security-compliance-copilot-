from fastapi import APIRouter

from app.core.dependencies import get_health_status

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return get_health_status()
