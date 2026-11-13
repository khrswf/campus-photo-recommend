"""Local HTTP interface for the camera app."""

import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .recommend import recommend


logger = logging.getLogger(__name__)
app = FastAPI(title="校园摄影推荐")
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(default="", max_length=256)
    location_id: Label | None = None
    person_count: int | None = Field(default=None, ge=0)
    framing: Label | None = None
    pose: Label | None = None
    forbidden_poses: list[Label] = Field(default_factory=list, max_length=20)
    required_objects: list[Label] = Field(default_factory=list, max_length=20)
    aesthetic_weight: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    limit: int = Field(default=3, ge=1, le=3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend_photos(request: RecommendRequest):
    try:
        return recommend(**request.model_dump())
    except (OSError, RuntimeError) as exc:
        logger.exception("Recommendation unavailable")
        raise HTTPException(status_code=503, detail="recommendation_unavailable") from exc
