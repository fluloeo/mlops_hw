from __future__ import annotations

import asyncio
import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import jwt
import numpy as np
import tritonclient.http.aio as triton_aio
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic_settings import BaseSettings

Image.MAX_IMAGE_PIXELS = None


class Settings(BaseSettings):
    jwt_secret: str
    cv_model: str = "convnext_tiny"
    llm_model: str = "Qwen2.5-1.5B-Instruct"
    triton_url: str = "triton:8000"
    triton_metrics_url: str = "http://triton:8002/metrics"
    vllm_url: str = "http://vllm:8000"
    vllm_metrics_url: str = "http://vllm:8000/metrics"
    node_metrics_url: str = "http://node-exporter:9100/metrics"
    gpu_name: str = ""
    student_name: str = ""
    student_email: str = ""


settings = Settings()

PREPROCESS = {
    "resnet50": (256, Image.BILINEAR),
    "efficientnet_b0": (256, Image.BICUBIC),
    "mobilenet_v3_large": (232, Image.BILINEAR),
    "convnext_tiny": (236, Image.BILINEAR),
}
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SIZE, INTERPOLATION = PREPROCESS[settings.cv_model]

LABELS = Path(__file__).with_name("imagenet_classes.txt").read_text(encoding="utf-8").splitlines()

HEALTH_TIMEOUT = 5.0
CHAT_ROLES = {"system", "user", "assistant"}
TRITON_HTTP_URL = settings.triton_url if settings.triton_url.startswith("http") else f"http://{settings.triton_url}"
TRITON_CLIENT_URL = TRITON_HTTP_URL.split("://", 1)[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=30.0),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )
    app.state.triton = triton_aio.InferenceServerClient(url=TRITON_CLIENT_URL, conn_limit=256)
    try:
        yield
    finally:
        await app.state.triton.close()
        await app.state.http.aclose()


app = FastAPI(title="Real-time Inference Gateway", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

bearer = HTTPBearer(auto_error=False)


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(status_code=401, detail="missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        return jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}", headers={"WWW-Authenticate": "Bearer"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/triton/health")
async def triton_health(request: Request):
    client: httpx.AsyncClient = request.app.state.http
    try:
        server = await client.get(f"{TRITON_HTTP_URL}/v2/health/ready", timeout=HEALTH_TIMEOUT)
        model = await client.get(f"{TRITON_HTTP_URL}/v2/models/{settings.cv_model}/ready", timeout=HEALTH_TIMEOUT)
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=503,
            content={"backend": "triton", "ready": False, "model": settings.cv_model, "detail": f"triton unavailable: {exc!r}"},
        )
    if server.status_code != 200 or model.status_code != 200:
        return JSONResponse(
            status_code=503,
            content={"backend": "triton", "ready": False, "model": settings.cv_model, "detail": "triton not ready"},
        )
    return {"backend": "triton", "ready": True, "model": settings.cv_model}


@app.get("/vllm/health")
async def vllm_health(request: Request):
    client: httpx.AsyncClient = request.app.state.http
    try:
        resp = await client.get(f"{settings.vllm_url}/v1/models", timeout=HEALTH_TIMEOUT)
        if resp.status_code != 200:
            raise ValueError(f"status {resp.status_code}")
        models = [m["id"] for m in resp.json().get("data", [])]
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"backend": "vllm", "ready": False, "models": [], "detail": f"vllm unavailable: {exc!r}"},
        )
    return {"backend": "vllm", "ready": True, "models": models}


async def proxy_metrics(request: Request, url: str, name: str) -> Response:
    client: httpx.AsyncClient = request.app.state.http
    try:
        resp = await client.get(url, timeout=HEALTH_TIMEOUT)
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={"detail": f"{name} unavailable: {exc!r}"})
    if resp.status_code != 200:
        return JSONResponse(status_code=502, content={"detail": f"{name} returned {resp.status_code}"})
    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("text/plain"):
        content_type = "text/plain; version=0.0.4; charset=utf-8"
    return Response(content=resp.content, media_type=content_type)


@app.get("/metrics/triton")
async def metrics_triton(request: Request) -> Response:
    return await proxy_metrics(request, settings.triton_metrics_url, "triton")


@app.get("/metrics/vllm")
async def metrics_vllm(request: Request) -> Response:
    return await proxy_metrics(request, settings.vllm_metrics_url, "vllm")


@app.get("/metrics/node")
async def metrics_node(request: Request) -> Response:
    return await proxy_metrics(request, settings.node_metrics_url, "node-exporter")


@app.get("/info")
def info(_: dict = Depends(require_auth)) -> dict:
    return {
        "student": {"name": settings.student_name, "email": settings.student_email},
        "models": {"vision": settings.cv_model, "llm": settings.llm_model},
        "gpu": settings.gpu_name,
        "version": "1.0",
    }


def preprocess(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as src:
        src.load()
        img = src.convert("RGB")
    w, h = img.size
    if w <= h:
        new_w, new_h = RESIZE_SIZE, int(RESIZE_SIZE * h / w)
    else:
        new_w, new_h = int(RESIZE_SIZE * w / h), RESIZE_SIZE
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), INTERPOLATION)
    left = int(round((new_w - CROP_SIZE) / 2.0))
    top = int(round((new_h - CROP_SIZE) / 2.0))
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def top5(logits: np.ndarray) -> list[dict]:
    x = logits.astype(np.float64).reshape(-1)
    x = np.exp(x - x.max())
    probs = x / x.sum()
    idx = np.argsort(-probs)[:5]
    return [{"class_id": int(i), "label": LABELS[int(i)], "score": float(probs[i])} for i in idx]


@app.post("/vision")
async def vision(request: Request, file: UploadFile = File(...), _: dict = Depends(require_auth)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    start = time.perf_counter()
    try:
        batch = await asyncio.to_thread(preprocess, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc!r}")
    inputs = [triton_aio.InferInput("INPUT__0", list(batch.shape), "FP32")]
    inputs[0].set_data_from_numpy(batch, binary_data=True)
    outputs = [triton_aio.InferRequestedOutput("OUTPUT__0", binary_data=True)]
    try:
        result = await request.app.state.triton.infer(settings.cv_model, inputs, outputs=outputs)
        logits = result.as_numpy("OUTPUT__0")
        if logits is None:
            raise ValueError("empty triton output")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"triton inference failed: {exc!r}")
    predictions = await asyncio.to_thread(top5, logits)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {"model": settings.cv_model, "predictions": predictions, "latency_ms": round(latency_ms, 3)}


def upstream_error(status_code: int, body: str) -> HTTPException:
    code = 400 if 400 <= status_code < 500 else 502
    return HTTPException(status_code=code, detail=f"vllm returned {status_code}: {body[:2000]}")


def stream_error_status(line: str) -> Optional[int]:
    payload = line[len("data:"):].strip()
    if not payload.startswith("{"):
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(obj, dict) or "error" not in obj:
        return None
    err = obj["error"]
    code = err.get("code") if isinstance(err, dict) else None
    return code if isinstance(code, int) else 400


async def relay_stream(resp: httpx.Response, lines: AsyncIterator[str], first: Optional[str]) -> AsyncIterator[str]:
    done = False
    try:
        if first is not None:
            yield f"{first}\n\n"
            done = first.strip() == "data: [DONE]"
        if not done:
            async for line in lines:
                if not line.startswith("data:"):
                    continue
                yield f"{line}\n\n"
                if line.strip() == "data: [DONE]":
                    done = True
                    break
        if not done:
            done = True
            yield "data: [DONE]\n\n"
    except Exception as exc:
        if not done:
            yield f"data: {json.dumps({'error': {'message': f'stream interrupted: {exc!r}'}})}\n\n"
            yield "data: [DONE]\n\n"
    finally:
        await resp.aclose()


@app.post("/chat")
async def chat(request: Request, _: dict = Depends(require_auth)):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="field 'messages' must be a non-empty list")
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail=f"messages[{i}] must be an object")
        if message.get("role") not in CHAT_ROLES:
            raise HTTPException(status_code=400, detail=f"messages[{i}].role must be one of {sorted(CHAT_ROLES)}")
        if not isinstance(message.get("content"), str):
            raise HTTPException(status_code=400, detail=f"messages[{i}].content must be a string")
    stream = body.get("stream", True)
    if not isinstance(stream, bool):
        raise HTTPException(status_code=400, detail="field 'stream' must be boolean")

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 128),
        "temperature": body.get("temperature", 0.0),
        "stream": stream,
    }
    client: httpx.AsyncClient = request.app.state.http
    url = f"{settings.vllm_url}/v1/chat/completions"

    if not stream:
        try:
            resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"vllm unavailable: {exc!r}")
        if resp.status_code != 200:
            raise upstream_error(resp.status_code, resp.text)
        try:
            return JSONResponse(content=resp.json())
        except ValueError:
            raise HTTPException(status_code=502, detail="vllm returned invalid JSON")

    try:
        resp = await client.send(client.build_request("POST", url, json=payload), stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"vllm unavailable: {exc!r}")
    if resp.status_code != 200:
        try:
            text = (await resp.aread()).decode("utf-8", errors="replace")
        finally:
            await resp.aclose()
        raise upstream_error(resp.status_code, text)

    lines = resp.aiter_lines()
    first: Optional[str] = None
    try:
        async for line in lines:
            if line.startswith("data:"):
                first = line
                break
    except httpx.HTTPError as exc:
        await resp.aclose()
        raise HTTPException(status_code=502, detail=f"vllm stream failed: {exc!r}")
    if first is None:
        await resp.aclose()
        raise HTTPException(status_code=502, detail="vllm returned empty stream")
    err_status = stream_error_status(first)
    if err_status is not None:
        await resp.aclose()
        raise upstream_error(err_status, first)

    return StreamingResponse(
        relay_stream(resp, lines, first),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
