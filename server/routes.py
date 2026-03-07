"""This file should define endpoint behavior.
It exists so route logic is separate from app startup.
Concept:
route = API entry point
Industry term:
handler, endpoint, controller (depending on framework vocabulary)"""

from fastapi import APIRouter, HTTPException, Request

from runtime.model_runner import ModelRunner
from server.schemas import GenerationRequest, GenerationResponse


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate", response_model=GenerationResponse)
def generate(request_data: GenerationRequest, request: Request) -> GenerationResponse:
    runner: ModelRunner | None = getattr(request.app.state, "runner", None)

    if runner is None:
        raise HTTPException(status_code=500, detail="Model runner not initialized")

    try:
        generated_text = runner.generate_text(
            prompt=request_data.prompt,
            max_tokens=request_data.max_tokens,
            temperature=request_data.temperature,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerationResponse(
        generated_text=generated_text,
        model_name=runner.model_name,
    )