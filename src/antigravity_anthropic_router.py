from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional
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
    閫掑綊绉婚櫎 dict/list 涓€间负 null/None 鐨勫瓧娈?鍏冪礌銆?
    鑳屾櫙锛歊oo/Kilo 鍦?Anthropic native tool 璺緞涓嬶紝鑻ユ敹鍒?tool_use.input 涓寘鍚?null锛?    鍙兘浼氭妸 null 褰撲綔鐪熷疄鍏ュ弬鎵ц锛堜緥濡傗€滃湪 null 涓悳绱⑩€濓級銆傚洜姝ゅ湪杩斿洖 tool_use.input 鍓嶅仛鍏滃簳娓呯悊銆?    """
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
    璋冭瘯鏃ュ織涓崟涓瓧绗︿覆瀛楁鐨勬渶澶ц緭鍑洪暱搴︼紙閬垮厤鎶?base64 鍥剧墖/瓒呴暱 schema 鎵撶垎鏃ュ織锛夈€?    """
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
    鏄惁鎵撳嵃璇锋眰浣?涓嬫父璇锋眰浣撶瓑鈥滈珮浣撶Н鈥濊皟璇曟棩蹇椼€?
    璇存槑锛歚ANTHROPIC_DEBUG=1` 浠呭紑鍚?token 瀵规瘮绛夌簿绠€鏃ュ織锛涗负閬垮厤鍒峰睆锛屽叆鍙?涓嬫父 body 蹇呴』鏄惧紡寮€鍚€?    """
    return str(os.getenv("ANTHROPIC_DEBUG_BODY", "")).strip().lower() in _DEBUG_TRUE


def _redact_for_log(value: Any, *, key_hint: str | None = None, max_chars: int) -> Any:
    """
    閫掑綊鑴辨晱/鎴柇鐢ㄤ簬鏃ュ織鎵撳嵃鐨?JSON銆?
    鐩爣锛?    - 璁╃敤鎴疯兘鐪嬪埌鈥滃疄闄呭叆鍙傜粨鏋勨€濓紙system/messages/tools 绛夛級
    - 榛樿閬垮厤娉勯湶鍑瘉/浠ょ墝
    - 閬垮厤鎶婂浘鐗?base64 鎴栬秴闀垮瓧娈电洿鎺ュ啓鍏ユ棩蹇楁枃浠?    """
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
            return f"{head}<...鐪佺暐 {len(value) - len(head) - len(tail)} 瀛楃...>{tail}"
        return value

    return value


def _json_dumps_for_log(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(data)


def _debug_log_request_payload(request: Request, payload: Dict[str, Any]) -> None:
    """
    鍦ㄥ紑鍚?`ANTHROPIC_DEBUG` 鏃舵墦鍗板叆鍙傦紙宸茶劚鏁?鎴柇锛夈€?    """
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
    鍦ㄥ紑鍚?`ANTHROPIC_DEBUG` 鏃舵墦鍗版渶缁堣浆鍙戝埌涓嬫父鐨勮姹備綋锛堝凡鎴柇锛夈€?    """
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
    Anthropic 鐢熸€佸鎴风閫氬父浣跨敤 `x-api-key`锛涚幇鏈夐」鐩叾瀹冭矾鐢变娇鐢?`Authorization: Bearer`銆?    杩欓噷鍚屾椂鍏煎涓ょ鏂瑰紡锛屼究浜庘€滄棤鎰熸帴鍏モ€濄€?    """
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
    candidate = response_data.get("candidates", [{}])[0] or {}
    parts = candidate.get("content", {}).get("parts", []) or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(str(part.get("text") or ""))
    return "\n".join([t for t in texts if t])

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
    鍏煎涓嬫父 usageMetadata 鐨勫绉嶈惤鐐癸細
    - response.usageMetadata
    - response.candidates[0].usageMetadata

    濡備袱鑰呭悓鏃跺瓨鍦紝浼樺厛閫夋嫨鈥滃瓧娈垫洿瀹屾暣鈥濈殑涓€渚с€?    """
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
        return _anthropic_error(status_code=403, message="瀵嗙爜閿欒", error_type="authentication_error")

    try:
        payload = await request.json()
    except Exception as e:
        return _anthropic_error(
            status_code=400, message=f"JSON 瑙ｆ瀽澶辫触: {str(e)}", error_type="invalid_request_error"
        )

    if not isinstance(payload, dict):
        return _anthropic_error(
            status_code=400, message="璇锋眰浣撳繀椤讳负 JSON object", error_type="invalid_request_error"
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
            message="缂哄皯蹇呭～瀛楁锛歮odel / max_tokens / messages",
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
        f"[ANTHROPIC] /messages 鏀跺埌璇锋眰: client={client_host}:{client_port}, model={model}, "
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
        try:
            components = convert_anthropic_request_to_antigravity_components(payload)
        except Exception as e:
            log.error(f"[ANTHROPIC] request conversion failed: {e}")
            return _anthropic_error(
                status_code=400, message="request conversion failed", error_type="invalid_request_error"
            )

        components["model"] = "gemini-3-flash-preview"
        components["tools"] = [{"googleSearch": {}}]
        components["system_instruction"] = _append_system_instruction(
            components.get("system_instruction"),
            "Return ONLY valid JSON with keys: query (string), results (array of {title,url,snippet}), summary (string). "
            "No markdown, no extra text.",
        )
        log.info(
            f"[ANTHROPIC] search mode: route to /v1/chat/completions (model={components['model']})"
        )

        if not (components.get("contents") or []):
            return _anthropic_error(
                status_code=400,
                message="messages cannot be empty; text blocks must be non-empty",
                error_type="invalid_request_error",
            )

        estimated_tokens = 0
        try:
            estimated_tokens = estimate_input_tokens(payload)
        except Exception as e:
            log.debug(f"[ANTHROPIC] token 盲录掳莽庐鈥斆ヂぢ泵绰? {e}")

        request_data: Dict[str, Any] = {
            "contents": components["contents"],
            "generationConfig": components["generation_config"],
        }
        if components.get("system_instruction"):
            request_data["systemInstruction"] = components["system_instruction"]
        if components.get("tools"):
            request_data["tools"] = components["tools"]

        api_payload = {"model": components["model"], "request": request_data}
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
        query = str(parsed.get("query") or _infer_search_query(payload) or "")
        results = parsed.get("results")
        if not isinstance(results, list):
            results = []
        summary = str(parsed.get("summary") or raw_text or "")

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
        return _anthropic_error(status_code=500, message="褰撳墠鏃犲彲鐢?antigravity 鍑瘉")

    _, credential_data = cred_result
    project_id, session_id = _infer_project_and_session(credential_data)

    try:
        components = convert_anthropic_request_to_antigravity_components(payload)
    except Exception as e:
        log.error(f"[ANTHROPIC] 璇锋眰杞崲澶辫触: {e}")
        return _anthropic_error(
            status_code=400, message="璇锋眰杞崲澶辫触", error_type="invalid_request_error"
        )

    log.info(f"[ANTHROPIC] /messages 妯″瀷鏄犲皠: upstream={model} -> downstream={components['model']}")

    # 涓嬫父瑕佹眰姣忔潯 text 鍐呭鍧楀繀椤诲寘鍚€滈潪绌虹櫧鈥濇枃鏈紱涓婃父瀹㈡埛绔伓灏斾細杩藉姞绌虹櫧 text block锛堜緥濡傚浘鐗囧悗璺熶竴涓┖瀛楃涓诧級锛?    # 缁忚繃杞崲杩囨护鍚庡彲鑳藉鑷?contents 涓虹┖锛屾鏃跺簲鍦ㄦ湰鍦扮洿鎺ヨ繑鍥?400锛岄伩鍏嶆妸鏃犳晥璇锋眰鎵撳埌涓嬫父銆?    if not (components.get("contents") or []):
        return _anthropic_error(
            status_code=400,
            message="messages 涓嶈兘涓虹┖锛泃ext 鍐呭鍧楀繀椤诲寘鍚潪绌虹櫧鏂囨湰",
            error_type="invalid_request_error",
        )

    # 绠€鍗曚及绠?token
    estimated_tokens = 0
    try:
        estimated_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.debug(f"[ANTHROPIC] token 浼扮畻澶辫触: {e}")

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
            log.error(f"[ANTHROPIC] 涓嬫父娴佸紡璇锋眰澶辫触: {e}")
            return _anthropic_error(status_code=500, message="涓嬫父璇锋眰澶辫触", error_type="api_error")

        async def stream_generator():
            try:
                # response 鐜板湪鏄?filtered_lines 鐢熸垚鍣紝鐩存帴浣跨敤
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
                    log.debug(f"[ANTHROPIC] 鍏抽棴 stream_ctx 澶辫触: {e}")
                try:
                    await client.aclose()
                except Exception as e:
                    log.debug(f"[ANTHROPIC] 鍏抽棴 client 澶辫触: {e}")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    request_id = f"msg_{int(time.time() * 1000)}"
    try:
        response_data, _, _ = await send_antigravity_request_no_stream(request_body, cred_mgr)
    except Exception as e:
        log.error(f"[ANTHROPIC] 涓嬫父闈炴祦寮忚姹傚け璐? {e}")
        return _anthropic_error(status_code=500, message="涓嬫父璇锋眰澶辫触", error_type="api_error")

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
    Anthropic Messages API 鍏煎鐨?token 璁℃暟绔偣锛堢敤浜?claude-cli 绛夊鎴风棰勬锛夈€?
    杩斿洖缁撴瀯灏介噺璐磋繎 Anthropic锛歚{"input_tokens": <int>}`銆?    """
    from config import get_api_password

    password = await get_api_password()
    token = _extract_api_token(request, credentials)
    if token != password:
        return _anthropic_error(status_code=403, message="瀵嗙爜閿欒", error_type="authentication_error")

    try:
        payload = await request.json()
    except Exception as e:
        return _anthropic_error(
            status_code=400, message=f"JSON 瑙ｆ瀽澶辫触: {str(e)}", error_type="invalid_request_error"
        )

    if not isinstance(payload, dict):
        return _anthropic_error(
            status_code=400, message="璇锋眰浣撳繀椤讳负 JSON object", error_type="invalid_request_error"
        )

    _debug_log_request_payload(request, payload)

    if not payload.get("model") or not isinstance(payload.get("messages"), list):
        return _anthropic_error(
            status_code=400,
            message="缂哄皯蹇呭～瀛楁锛歮odel / messages",
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
        f"[ANTHROPIC] /messages/count_tokens 鏀跺埌璇锋眰: client={client_host}:{client_port}, "
        f"model={payload.get('model')}, messages={len(payload.get('messages') or [])}, "
        f"thinking_present={thinking_present}, thinking={thinking_summary}, ua={user_agent}"
    )

    # 绠€鍗曚及绠?    input_tokens = 0
    try:
        input_tokens = estimate_input_tokens(payload)
    except Exception as e:
        log.error(f"[ANTHROPIC] token 浼扮畻澶辫触: {e}")

    return JSONResponse(content={"input_tokens": input_tokens})
