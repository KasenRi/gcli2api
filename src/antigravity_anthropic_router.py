from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from log import log

from .antigravity_api import (
    build_antigravity_request_body,
    send_antigravity_request_no_stream,
    send_antigravity_request_stream,
)
from .anthropic_converter import convert_anthropic_request_to_antigravity_components
from .anthropic_streaming import antigravity_sse_to_anthropic_sse, gemini_sse_to_anthropic_sse
from .gcli_chat_api import send_gemini_request
from .models import ChatCompletionRequest
from .openai_transfer import openai_request_to_gemini_payload
from .token_estimator import estimate_input_tokens

router = APIRouter()
security = HTTPBearer(auto_error=False)

_DEBUG_TRUE = {"1", "true", "yes", "on"}
_REDACTED = "<REDACTED>"
_SENSITIVE_KEYS = {
    "authorization",
    "x-api-key",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
}

def _remove_nulls_for_tool_input(value: Any) -> Any:
    """
    Recursively remove null/None entries from dict/list values.
    This avoids null tool inputs in Anthropic tool_use."""
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in value.items():
            if v is None:
                continue
            cleaned[k] = _remove_nulls_for_tool_input(v)
        return cleaned

    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            if item is None:
                continue
            cleaned_list.append(_remove_nulls_for_tool_input(item))
        return cleaned_list

    return value


def _anthropic_debug_max_chars() -> int:
    """
    Max output length for debug log fields (avoid huge base64/schema)."""
    raw = str(os.getenv("ANTHROPIC_DEBUG_MAX_CHARS", "")).strip()
    if not raw:
        return 2000
    try:
        return max(200, int(raw))
    except Exception:
        return 2000


def _anthropic_debug_enabled() -> bool:
    return str(os.getenv("ANTHROPIC_DEBUG", "")).strip().lower() in _DEBUG_TRUE


def _anthropic_debug_body_enabled() -> bool:
    """
    Whether to log request bodies and other large debug payloads.
    Note: set ANTHROPIC_DEBUG_BODY=1 to enable body logging."""
    return str(os.getenv("ANTHROPIC_DEBUG_BODY", "")).strip().lower() in _DEBUG_TRUE


def _redact_for_log(value: Any, *, key_hint: str | None = None, max_chars: int) -> Any:
    """
    Recursively redact or truncate JSON values for logging.
    Goals:
    - Keep request structure visible (system/messages/tools).
    - Avoid leaking secrets or huge base64 blobs."""
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for k, v in value.items():
            k_str = str(k)
            k_lower = k_str.strip().lower()
            if k_lower in _SENSITIVE_KEYS:
                redacted[k_str] = _REDACTED
                continue
            redacted[k_str] = _redact_for_log(v, key_hint=k_lower, max_chars=max_chars)
        return redacted

    if isinstance(value, list):
        return [_redact_for_log(v, key_hint=key_hint, max_chars=max_chars) for v in value]

    if isinstance(value, str):
        if (key_hint or "").lower() == "data" and len(value) > 64:
            return f"<base64 len={len(value)}>"
        if len(value) > max_chars:
            head = value[: max_chars // 2]
            tail = value[-max_chars // 2 :]
            return f"{head}<...omitted {len(value) - len(head) - len(tail)} chars...>{tail}"
        return value

    return value


def _json_dumps_for_log(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(data)


def _debug_log_request_payload(request: Request, payload: Dict[str, Any]) -> None:
    """
    Log inbound request payloads when debug is enabled (redacted/truncated)."""
    if not _anthropic_debug_enabled() or not _anthropic_debug_body_enabled():
        return

    max_chars = _anthropic_debug_max_chars()
    safe_payload = _redact_for_log(payload, max_chars=max_chars)

    headers_of_interest = {
        "content-type": request.headers.get("content-type"),
        "content-length": request.headers.get("content-length"),
        "anthropic-version": request.headers.get("anthropic-version"),
        "user-agent": request.headers.get("user-agent"),
    }
    safe_headers = _redact_for_log(headers_of_interest, max_chars=max_chars)
    log.info(f"[ANTHROPIC][DEBUG] headers={_json_dumps_for_log(safe_headers)}")
    log.info(f"[ANTHROPIC][DEBUG] payload={_json_dumps_for_log(safe_payload)}")


def _debug_log_downstream_request_body(request_body: Dict[str, Any]) -> None:
    """
    Log downstream request body when debug is enabled (redacted/truncated)."""
    if not _anthropic_debug_enabled() or not _anthropic_debug_body_enabled():
        return

    max_chars = _anthropic_debug_max_chars()
    safe_body = _redact_for_log(request_body, max_chars=max_chars)
    log.info(f"[ANTHROPIC][DEBUG] downstream_request_body={_json_dumps_for_log(safe_body)}")


def _anthropic_error(
    *,
    status_code: int,
    message: str,
    error_type: str = "api_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def _extract_api_token(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials]
) -> Optional[str]:
    """
    Anthropic clients use x-api-key or Authorization: Bearer; accept both."""
    if credentials and credentials.credentials:
        return credentials.credentials

    authorization = request.headers.get("authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()

    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key.strip()

    return None


def _infer_project_and_session(credential_data: Dict[str, Any]) -> tuple[str, str]:
    project_id = credential_data.get("project_id")
    session_id = f"session-{uuid.uuid4().hex}"   
    return str(project_id), str(session_id)


def _sse_event(event: str, data: Dict[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _is_search_tool_name(name: str) -> bool:
    return "search" in str(name or "").lower()

def _append_system_instruction(system_instruction: Optional[Dict[str, Any]], extra_text: str) -> Dict[str, Any]:
    if not extra_text:
        return system_instruction or {"role": "user", "parts": []}

    if not system_instruction:
        return {"role": "user", "parts": [{"text": extra_text}]}

    if isinstance(system_instruction, dict):
        parts = system_instruction.get("parts") or []
        if not isinstance(parts, list):
            parts = []
        parts = list(parts)
        parts.append({"text": extra_text})
        return {"role": system_instruction.get("role", "user"), "parts": parts}

    return {"role": "user", "parts": [{"text": extra_text}]}

def _extract_text_from_gemini_response(response_data: Dict[str, Any]) -> str:
    data = response_data
    if not isinstance(data, dict):
        return ""
    if "candidates" not in data and isinstance(data.get("response"), dict):
        data = data.get("response") or {}
    candidate = (data.get("candidates") or [{}])[0] or {}
    parts = candidate.get("content", {}).get("parts", []) or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(str(part.get("text") or ""))
    return "\n".join([t for t in texts if t])

def _extract_grounding_metadata(response_data: Dict[str, Any]) -> Dict[str, Any]:
    data = response_data
    if not isinstance(data, dict):
        return {}
    if "candidates" not in data and isinstance(data.get("response"), dict):
        data = data.get("response") or {}
    candidate = (data.get("candidates") or [{}])[0] or {}
    grounding = candidate.get("groundingMetadata") or {}
    return grounding if isinstance(grounding, dict) else {}

def _grounding_metadata_to_search(
    grounding: Dict[str, Any]
) -> Tuple[str, List[Dict[str, str]]]:
    query = ""
    web_queries = grounding.get("webSearchQueries")
    if isinstance(web_queries, list) and web_queries:
        first = web_queries[0]
        if isinstance(first, str):
            query = first.strip()

    results: List[Dict[str, str]] = []
    chunks = grounding.get("groundingChunks") or []
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web") or {}
            if not isinstance(web, dict):
                continue
            title = str(web.get("title") or "").strip()
            uri = str(web.get("uri") or "").strip()
            snippet = str(web.get("snippet") or "").strip() if isinstance(web.get("snippet"), str) else ""
            if title or uri or snippet:
                results.append({"title": title, "url": uri, "snippet": snippet})
    return query, results

def _parse_search_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}

    cleaned = text.strip()
    if "```" in cleaned:
        fence_start = cleaned.find("```")
        fence_end = cleaned.find("```", fence_start + 3)
        if fence_end != -1:
            cleaned = cleaned[fence_start + 3 : fence_end].strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

    if not cleaned:
        return {}

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(cleaned[brace_start : brace_end + 1])
        except Exception:
            return {}

    return {}

def _infer_search_query(payload: Dict[str, Any]) -> str:
    messages = payload.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        return text
    return ""

def _extract_search_tool_name(payload: Dict[str, Any]) -> str:
    tools = payload.get("tools") or []
    for tool in tools:
        if isinstance(tool, dict) and _is_search_tool_name(tool.get("name", "")):
            return str(tool.get("name"))
    return "web_search"

def _payload_mentions_web_search(payload: Dict[str, Any]) -> bool:
    def _texts_from_blocks(blocks: Any) -> list[str]:
        texts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    texts.append(str(block.get("text") or ""))
        elif isinstance(blocks, str):
            texts.append(blocks)
        return texts

    messages = payload.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for text in _texts_from_blocks(msg.get("content")):
            if "web search" in text.lower():
                return True

    system_blocks = payload.get("system") or []
    for text in _texts_from_blocks(system_blocks):
        if "web search" in text.lower():
            return True

    return False


def _normalize_search_query(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    match = re.search(
        r"web\s*search\s*\(\s*['\"](.+?)['\"]\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    for marker in ("query:", "查询:", "查询："):
        idx = text.lower().find(marker)
        if idx != -1:
            candidate = text[idx + len(marker):].strip()
            if candidate:
                return candidate
    return text


def _anthropic_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()
    return ""


def _anthropic_payload_to_openai_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    openai_messages: List[Dict[str, Any]] = []

    system_blocks = payload.get("system")
    if isinstance(system_blocks, list):
        system_texts: List[str] = []
        for block in system_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "")
                if text:
                    system_texts.append(text)
            elif isinstance(block, str):
                if block.strip():
                    system_texts.append(block)
        if system_texts:
            openai_messages.append({"role": "system", "content": "\n".join(system_texts)})
    elif isinstance(system_blocks, str) and system_blocks.strip():
        openai_messages.append({"role": "system", "content": system_blocks})

    messages = payload.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant", "system"}:
            continue
        text = _anthropic_content_to_text(msg.get("content"))
        if not text:
            continue
        openai_messages.append({"role": role, "content": text})

    return openai_messages


def _is_search_requested(payload: Dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False

    search_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if _is_search_tool_name(tool.get("name", "")) or _is_search_tool_name(tool.get("type", "")):
            search_tools.append(tool)

    if not search_tools:
        return False

    tool_choice = payload.get("tool_choice") or payload.get("toolChoice")
    if isinstance(tool_choice, str):
        return _is_search_tool_name(tool_choice)

    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or tool_choice.get("tool_name") or ""
        if _is_search_tool_name(name):
            return True
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type in {"tool", "required"} and len(search_tools) == 1:
            return True
        return False

    return _payload_mentions_web_search(payload)

def _pick_usage_metadata_from_antigravity_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer usageMetadata from response or candidate if present.
    Sources:
    - response.usageMetadata
    - response.candidates[0].usageMetadata
    Prefer the side with more fields."""
    response = response_data.get("response", {}) or {}
    if not isinstance(response, dict):
        return {}

    response_usage = response.get("usageMetadata", {}) or {}
    if not isinstance(response_usage, dict):
        response_usage = {}

    candidate = (response.get("candidates", []) or [{}])[0] or {}
    if not isinstance(candidate, dict):
        candidate = {}
    candidate_usage = candidate.get("usageMetadata", {}) or {}
    if not isinstance(candidate_usage, dict):
        candidate_usage = {}

    fields = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")

    def score(d: Dict[str, Any]) -> int:
        s = 0
        for f in fields:
            if f in d and d.get(f) is not None:
                s += 1
        return s

    if score(candidate_usage) > score(response_usage):
        return candidate_usage
    return response_usage


def _convert_antigravity_response_to_anthropic_message(
    response_data: Dict[str, Any],
    *,
    model: str,
    message_id: str,
    fallback_input_tokens: int = 0,
) -> Dict[str, Any]:
    candidate = response_data.get("response", {}).get("candidates", [{}])[0] or {}
    parts = candidate.get("content", {}).get("parts", []) or []
    usage_metadata = _pick_usage_metadata_from_antigravity_response(response_data)

    content = []
    has_tool_use = False

    for part in parts:
        if not isinstance(part, dict):
            continue

        if part.get("thought") is True:
            block: Dict[str, Any] = {"type": "thinking", "thinking": part.get("text", "")}
            signature = part.get("thoughtSignature")
            if signature:
                block["signature"] = signature
            content.append(block)
            continue

        if "text" in part:
            content.append({"type": "text", "text": part.get("text", "")})
            continue

        if "functionCall" in part:
            has_tool_use = True
            fc = part.get("functionCall", {}) or {}
            content.append(
                {
                    "type": "tool_use",
                    "id": fc.get("id") or f"toolu_{uuid.uuid4().hex}",
                    "name": fc.get("name") or "",
                    "input": _remove_nulls_for_tool_input(fc.get("args", {}) or {}),
                }
            )
            continue

        if "inlineData" in part:
            inline = part.get("inlineData", {}) or {}
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": inline.get("mimeType", "image/png"),
                        "data": inline.get("data", ""),
                    },
                }
            )
            continue

    finish_reason = candidate.get("finishReason")
    stop_reason = "tool_use" if has_tool_use else "end_turn"
    if finish_reason == "MAX_TOKENS" and not has_tool_use:
        stop_reason = "max_tokens"

    input_tokens_present = isinstance(usage_metadata, dict) and "promptTokenCount" in usage_metadata
    output_tokens_present = isinstance(usage_metadata, dict) and "candidatesTokenCount" in usage_metadata

    input_tokens = usage_metadata.get("promptTokenCount", 0) if isinstance(usage_metadata, dict) else 0
    output_tokens = usage_metadata.get("candidatesTokenCount", 0) if isinstance(usage_metadata, dict) else 0

    if not input_tokens_present:
        input_tokens = max(0, int(fallback_input_tokens or 0))
    if not output_tokens_present:
        output_tokens = 0

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        },
    }


def _pick_usage_metadata_from_gemini_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    response_usage = response_data.get("usageMetadata", {}) or {}
    if not isinstance(response_usage, dict):
        response_usage = {}

    candidate = (response_data.get("candidates", []) or [{}])[0] or {}
    if not isinstance(candidate, dict):
        candidate = {}
    candidate_usage = candidate.get("usageMetadata", {}) or {}
    if not isinstance(candidate_usage, dict):
        candidate_usage = {}

    fields = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")

    def score(d: Dict[str, Any]) -> int:
        s = 0
        for f in fields:
            if f in d and d.get(f) is not None:
                s += 1
        return s

    if score(candidate_usage) > score(response_usage):
        return candidate_usage
    return response_usage


def _convert_gemini_response_to_anthropic_message(
    response_data: Dict[str, Any],
    *,
    model: str,
    message_id: str,
    fallback_input_tokens: int = 0,
) -> Dict[str, Any]:
    candidate = response_data.get("candidates", [{}])[0] or {}
    parts = candidate.get("content", {}).get("parts", []) or []
    usage_metadata = _pick_usage_metadata_from_gemini_response(response_data)

    content = []
    has_tool_use = False

    for part in parts:
        if not isinstance(part, dict):
            continue

        if part.get("thought") is True:
            block: Dict[str, Any] = {"type": "thinking", "thinking": part.get("text", "")}
            signature = part.get("thoughtSignature")
            if signature:
                block["signature"] = signature
            content.append(block)
            continue

        if "text" in part:
            content.append({"type": "text", "text": part.get("text", "")})
            continue

        if "functionCall" in part:
            has_tool_use = True
            fc = part.get("functionCall", {}) or {}
            content.append(
                {
                    "type": "tool_use",
                    "id": fc.get("id") or f"toolu_{uuid.uuid4().hex}",
                    "name": fc.get("name") or "",
                    "input": _remove_nulls_for_tool_input(fc.get("args", {}) or {}),
                }
            )
            continue

        if "inlineData" in part:
            inline = part.get("inlineData", {}) or {}
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": inline.get("mimeType", "image/png"),
                        "data": inline.get("data", ""),
                    },
                }
            )
            continue

    finish_reason = candidate.get("finishReason")
    stop_reason = "tool_use" if has_tool_use else "end_turn"
    if finish_reason == "MAX_TOKENS" and not has_tool_use:
        stop_reason = "max_tokens"

    input_tokens_present = isinstance(usage_metadata, dict) and "promptTokenCount" in usage_metadata
    output_tokens_present = isinstance(usage_metadata, dict) and "candidatesTokenCount" in usage_metadata

    input_tokens = usage_metadata.get("promptTokenCount", 0) if isinstance(usage_metadata, dict) else 0
    output_tokens = usage_metadata.get("candidatesTokenCount", 0) if isinstance(usage_metadata, dict) else 0

    if not input_tokens_present:
        input_tokens = max(0, int(fallback_input_tokens or 0))
    if not output_tokens_present:
        output_tokens = 0

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        },
    }


@router.post("/antigravity/v1/messages")
async def anthropic_messages(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    from config import get_api_password

    password = await get_api_password()
    token = _extract_api_token(request, credentials)
    if token != password:
        return _anthropic_error(status_code=403, message="invalid api key", error_type="authentication_error")

    try:
        payload = await request.json()
    except Exception as e:
        return _anthropic_error(
            status_code=400, message=f"JSON parse failed: {str(e)}", error_type="invalid_request_error"
        )

    if not isinstance(payload, dict):
        return _anthropic_error(
            status_code=400, message="request body must be a JSON object", error_type="invalid_request_error"
        )

    _debug_log_request_payload(request, payload)

    model = payload.get("model")
    max_tokens = payload.get("max_tokens")
    messages = payload.get("messages")
    stream = bool(payload.get("stream", False))
    thinking_present = "thinking" in payload
    thinking_value = payload.get("thinking")
    thinking_summary = None
    if thinking_present:
        if isinstance(thinking_value, dict):
            thinking_summary = {
                "type": thinking_value.get("type"),
                "budget_tokens": thinking_value.get("budget_tokens"),
            }
        else:
            thinking_summary = thinking_value

    if not model or max_tokens is None or not isinstance(messages, list):
        return _anthropic_error(
            status_code=400,
            message="missing required fields: model / max_tokens / messages",
            error_type="invalid_request_error",
        )

    try:
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else "unknown"
    except Exception:
        client_host = "unknown"
        client_port = "unknown"

    user_agent = request.headers.get("user-agent", "")
    log.info(
        f"[ANTHROPIC] /messages received: client={client_host}:{client_port}, model={model}, "
        f"stream={stream}, messages={len(messages)}, thinking_present={thinking_present}, "
        f"thinking={thinking_summary}, ua={user_agent}"
    )

    if len(messages) == 1 and messages[0].get("role") == "user" and messages[0].get("content") == "Hi":
        return JSONResponse(
            content={
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": str(model),
                "content": [
                    {"type": "text", "text": "antigravity Anthropic Messages OK"}
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        )

    from src.credential_manager import get_credential_manager

    if _is_search_requested(payload):
        openai_messages = _anthropic_payload_to_openai_messages(payload)
        if not openai_messages:
            return _anthropic_error(
                status_code=400,
                message="messages cannot be empty; text blocks must be non-empty",
                error_type="invalid_request_error",
            )

        stop_value = None
        stop_sequences = payload.get("stop_sequences")
        if isinstance(stop_sequences, str) and stop_sequences:
            stop_value = stop_sequences
        elif isinstance(stop_sequences, list):
            stop_list = [str(item) for item in stop_sequences if item]
            if stop_list:
                stop_value = stop_list

        request_data = ChatCompletionRequest(
            model="gemini-3-flash-preview-search",
            messages=openai_messages,
            stream=False,
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            stop=stop_value,
            top_k=64,
        )

        log.info(
            f"[ANTHROPIC] search mode: route to /v1/chat/completions (model={request_data.model})"
        )

        estimated_tokens = 0
        try:
            estimated_tokens = estimate_input_tokens(payload)
        except Exception as e:
            log.debug(f"[ANTHROPIC] token estimate failed: {e}")

        api_payload = await openai_request_to_gemini_payload(request_data)
        cred_mgr = await get_credential_manager()

        response = await send_gemini_request(api_payload, False, cred_mgr)
        if getattr(response, "status_code", 200) != 200:
            return _anthropic_error(
                status_code=getattr(response, "status_code", 500),
                message="upstream request failed",
                error_type="api_error",
            )

        request_id = f"msg_{int(time.time() * 1000)}"
        try:
            if hasattr(response, "body"):
                response_data = json.loads(
                    response.body.decode() if isinstance(response.body, bytes) else response.body
                )
            else:
                response_data = json.loads(
                    response.content.decode() if isinstance(response.content, bytes) else response.content
                )
        except Exception as e:
            log.error(f"[ANTHROPIC] failed to parse upstream response: {e}")
            return _anthropic_error(status_code=500, message="failed to parse upstream response", error_type="api_error")

        raw_text = _extract_text_from_gemini_response(response_data)
        parsed = _parse_search_json(raw_text)
        query = str(parsed.get("query") or "")
        results = parsed.get("results")
        if not isinstance(results, list):
            results = []
        summary = str(parsed.get("summary") or raw_text or "")

        grounding = _extract_grounding_metadata(response_data)
        grounding_query, grounding_results = _grounding_metadata_to_search(grounding)
        if not query:
            query = grounding_query
        if not query:
            query = str(_infer_search_query(payload) or "")
        if not results and grounding_results:
            results = grounding_results

        query = _normalize_search_query(query)
        if not query:
            query = "web search"

        tool_name = _extract_search_tool_name(payload)
        tool_use_id = f"toolu_{uuid.uuid4().hex}"
        tool_result_payload = {"query": query, "results": results}
        tool_result_text = json.dumps(tool_result_payload, ensure_ascii=False, separators=(",", ":"))

        anthropic_response = {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": str(model),
            "content": [
                {"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": {"query": query}},
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "name": tool_name,
                    "content": [{"type": "text", "text": tool_result_text}],
                },
                {"type": "text", "text": summary},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": int(estimated_tokens or 0),
                "output_tokens": 0,
            },
        }
        if not stream:
            return JSONResponse(content=anthropic_response)

        async def stream_generator():
            usage = anthropic_response.get("usage", {}) or {}
            yield _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": anthropic_response.get("id", request_id),
                        "type": "message",
                        "role": "assistant",
                        "model": anthropic_response.get("model", str(model)),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": int(usage.get("input_tokens", 0) or 0),
                            "output_tokens": 0,
                        },
                    },
                },
            )

            content_blocks = anthropic_response.get("content", []) or []
            for idx, block in enumerate(content_blocks):
                if not isinstance(block, dict):
                    block = {"type": "text", "text": str(block)}
                block_type = block.get("type", "text")

                if block_type in {"tool_use", "tool_result"}:
                    yield _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": block_type, **{k: v for k, v in block.items() if k != "type"}},
                        },
                    )
                    yield _sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
                    continue

                yield _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": block_type, **{k: v for k, v in block.items() if k != "type"}},
                    },
                )

                if block_type == "thinking":
                    delta = {"type": "thinking_delta", "thinking": block.get("thinking", "")}
                else:
                    delta = {"type": "text_delta", "text": block.get("text", "")}

                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": idx, "delta": delta},
                )
                yield _sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": idx},
                )
            yield _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": anthropic_response.get("stop_reason"), "stop_sequence": None},
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    },
                },
            )
            yield _sse_event("message_stop", {"type": "message_stop"})

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    cred_mgr = await get_credential_manager()
    cred_result = await cred_mgr.get_valid_credential(is_antigravity=True)
    if not cred_result:
        return _anthropic_error(status_code=500, message="no valid antigravity credential")

    _, credential_data = cred_result
    project_id, session_id = _infer_project_and_session(credential_data)

    try:
        components = convert_anthropic_request_to_antigravity_components(payload)
    except Exception as e:
        log.error(f"[ANTHROPIC] request conversion failed: {e}")
        return _anthropic_error(
            status_code=400, message="request conversion failed", error_type="invalid_request_error"
        )

    log.info(f"[ANTHROPIC] /messages model mapping: upstream={model} -> downstream={components['model']}")

    # Downstream requires each text block to be non-empty; ensure contents is not empty.
    if not components.get("contents"):
        return _anthropic_error(
            status_code=400,
            message="messages cannot be empty; text blocks must be non-empty",
            error_type="invalid_request_error",
        )

    # Rough token estimate
    estimated_tokens = 0
    try:
        estimated_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.debug(f"[ANTHROPIC] token estimate failed: {e}")

    request_body = build_antigravity_request_body(
        contents=components["contents"],
        model=components["model"],
        project_id=project_id,
        session_id=session_id,
        system_instruction=components["system_instruction"],
        tools=components["tools"],
        generation_config=components["generation_config"],
    )
    _debug_log_downstream_request_body(request_body)

    if stream:
        message_id = f"msg_{uuid.uuid4().hex}"

        try:
            resources, cred_name, _ = await send_antigravity_request_stream(request_body, cred_mgr)
            response, stream_ctx, client = resources
        except Exception as e:
            log.error(f"[ANTHROPIC] upstream stream request failed: {e}")
            return _anthropic_error(status_code=500, message="upstream request failed", error_type="api_error")

        async def stream_generator():
            try:
                # response is a filtered_lines generator; iterate directly
                async for chunk in antigravity_sse_to_anthropic_sse(
                    response,
                    model=str(model),
                    message_id=message_id,
                    initial_input_tokens=estimated_tokens,
                    credential_manager=cred_mgr,
                    credential_name=cred_name,
                ):
                    yield chunk
            finally:
                try:
                    await stream_ctx.__aexit__(None, None, None)
                except Exception as e:
                    log.debug(f"[ANTHROPIC] failed to close stream_ctx: {e}")
                try:
                    await client.aclose()
                except Exception as e:
                    log.debug(f"[ANTHROPIC] failed to close client: {e}")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    request_id = f"msg_{int(time.time() * 1000)}"
    try:
        response_data, _, _ = await send_antigravity_request_no_stream(request_body, cred_mgr)
    except Exception as e:
        log.error(f"[ANTHROPIC] upstream non-stream request failed: {e}")
        return _anthropic_error(status_code=500, message="upstream request failed", error_type="api_error")

    anthropic_response = _convert_antigravity_response_to_anthropic_message(
        response_data,
        model=str(model),
        message_id=request_id,
        fallback_input_tokens=estimated_tokens,
    )
    return JSONResponse(content=anthropic_response)


@router.post("/antigravity/v1/messages/count_tokens")
async def anthropic_messages_count_tokens(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Compatibility token count endpoint for Anthropic Messages API (for clients like claude-cli).
    Returns {"input_tokens": <int>}."""
    from config import get_api_password

    password = await get_api_password()
    token = _extract_api_token(request, credentials)
    if token != password:
        return _anthropic_error(status_code=403, message="invalid api key", error_type="authentication_error")

    try:
        payload = await request.json()
    except Exception as e:
        return _anthropic_error(
            status_code=400, message=f"JSON parse failed: {str(e)}", error_type="invalid_request_error"
        )

    if not isinstance(payload, dict):
        return _anthropic_error(
            status_code=400, message="request body must be a JSON object", error_type="invalid_request_error"
        )

    _debug_log_request_payload(request, payload)

    if not payload.get("model") or not isinstance(payload.get("messages"), list):
        return _anthropic_error(
            status_code=400,
            message="missing required fields: model / messages",
            error_type="invalid_request_error",
        )

    try:
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else "unknown"
    except Exception:
        client_host = "unknown"
        client_port = "unknown"

    thinking_present = "thinking" in payload
    thinking_value = payload.get("thinking")
    thinking_summary = None
    if thinking_present:
        if isinstance(thinking_value, dict):
            thinking_summary = {
                "type": thinking_value.get("type"),
                "budget_tokens": thinking_value.get("budget_tokens"),
            }
        else:
            thinking_summary = thinking_value

    user_agent = request.headers.get("user-agent", "")
    log.info(
        f"[ANTHROPIC] /messages/count_tokens received: client={client_host}:{client_port}, "
        f"model={payload.get('model')}, messages={len(payload.get('messages') or [])}, "
        f"thinking_present={thinking_present}, thinking={thinking_summary}, ua={user_agent}"
    )

    # Rough token estimate
    try:
        input_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.error(f"[ANTHROPIC] token estimate failed: {e}")

    return JSONResponse(content={"input_tokens": input_tokens})
