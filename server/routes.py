"""This file should define endpoint behavior.
It exists so route logic is separate from app startup.
Concept:
route = API entry point
Industry term:
handler, endpoint, controller (depending on framework vocabulary)"""
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from instrumentation.logging_utils import log_request_metrics
from instrumentation.timers import elapsed_ms, now_s
from runtime.model_runner import ModelRunner
from server.schemas import GenerationRequest


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate")
def generate(request_data: GenerationRequest, request: Request) -> StreamingResponse:
    runner: ModelRunner | None = getattr(request.app.state, "runner", None)

    if runner is None:
        raise HTTPException(status_code=500, detail="Model runner not initialized")

    request_start_s = now_s()

    try:
        inputs, prompt_tokens, tokenization_ms = runner.tokenize_prompt(request_data.prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    def event_stream():
        first_chunk_s = None
        last_chunk_s = None
        chunk_gaps_ms = []
        output_parts = []
        error_message = None

        try:
            for chunk in runner.stream_from_inputs(
                inputs=inputs,
                max_tokens=request_data.max_tokens,
                temperature=request_data.temperature,
            ):
                chunk_now_s = now_s()
                output_parts.append(chunk)

                if first_chunk_s is None:
                    first_chunk_s = chunk_now_s
                else:
                    chunk_gaps_ms.append(elapsed_ms(last_chunk_s, chunk_now_s))

                last_chunk_s = chunk_now_s
                yield f"data: {chunk}\n\n"
        except RuntimeError as exc:
            error_message = str(exc)
            yield f"event: error\ndata: {str(exc)}\n\n"
        finally:
            end_s = now_s()
            generated_text = "".join(output_parts)
            output_tokens_est = runner.estimate_token_count(generated_text)

            total_time_ms = elapsed_ms(request_start_s, end_s)

            ttft_ms = None
            approx_prefill_plus_first_chunk_ms = None
            stream_time_ms = 0.0
            decode_tokens_per_sec_est = None

            if first_chunk_s is not None:
                ttft_ms = elapsed_ms(request_start_s, first_chunk_s)
                approx_prefill_plus_first_chunk_ms = max(ttft_ms - tokenization_ms, 0.0)
                stream_time_ms = elapsed_ms(first_chunk_s, end_s)

                decode_tokens_after_first = max(output_tokens_est - 1, 0)
                if stream_time_ms > 0.0 and decode_tokens_after_first > 0:
                    decode_tokens_per_sec_est = decode_tokens_after_first / (stream_time_ms / 1000.0)

            metrics = {
                "event": "request_metrics",
                "prompt_chars": len(request_data.prompt),
                "prompt_tokens": prompt_tokens,
                "output_tokens_est": output_tokens_est,
                "max_tokens_requested": request_data.max_tokens,
                "temperature": request_data.temperature,
                "tokenization_ms": round(tokenization_ms, 3),
                "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                "approx_prefill_plus_first_chunk_ms": round(approx_prefill_plus_first_chunk_ms, 3) if approx_prefill_plus_first_chunk_ms is not None else None,
                "stream_time_ms": round(stream_time_ms, 3),
                "total_time_ms": round(total_time_ms, 3),
                "chunk_count": len(output_parts),
                "chunk_gap_ms": [round(value, 3) for value in chunk_gaps_ms],
                "decode_tokens_per_sec_est": round(decode_tokens_per_sec_est, 3) if decode_tokens_per_sec_est is not None else None,
                "error": error_message,
            }

            log_request_metrics(metrics)

            client_summary = {
                "prompt_tokens": prompt_tokens,
                "output_tokens_est": output_tokens_est,
                "tokenization_ms": round(tokenization_ms, 3),
                "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                "approx_prefill_plus_first_chunk_ms": round(approx_prefill_plus_first_chunk_ms, 3) if approx_prefill_plus_first_chunk_ms is not None else None,
                "stream_time_ms": round(stream_time_ms, 3),
                "total_time_ms": round(total_time_ms, 3),
                "decode_tokens_per_sec_est": round(decode_tokens_per_sec_est, 3) if decode_tokens_per_sec_est is not None else None,
            }

            yield f"event: done\ndata: {json.dumps(client_summary)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )