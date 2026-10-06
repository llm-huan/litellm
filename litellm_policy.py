from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from collections import deque
from contextvars import ContextVar
from datetime import datetime, time as datetime_time
from functools import wraps
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from litellm.integrations.custom_logger import CustomLogger


_MAX_RECOVERED_OUTPUT_INDEX = 1024
# ChatGPT account (auth.json path) of the deployment the router picked for the
# current request; see _patch_chatgpt_account_propagation.
_chatgpt_auth_file_ctx: ContextVar[str | None] = ContextVar(
    "temki_chatgpt_auth_file", default=None
)
_stream_logger = logging.getLogger("temki_litellm_stream")
_chatgpt_token_encoding: Any = None
_SENSITIVE_DIAGNOSTIC_KEYS = {
    "access_token",
    "api-key",
    "api_key",
    "authorization",
    "client_secret",
    "cookie",
    "password",
    "proxy-authorization",
    "refresh_token",
    "secret",
    "set-cookie",
    "token",
    "x-api-key",
}


class _ChatGPTUpstreamResponseError(Exception):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _sensitive_diagnostic_key(key: Any) -> bool:
    key_lc = str(key).lower()
    return (
        key_lc in _SENSITIVE_DIAGNOSTIC_KEYS
        or key_lc.endswith("_api_key")
        or key_lc.endswith("-api-key")
        or key_lc.endswith("_access_token")
        or key_lc.endswith("_refresh_token")
        or key_lc.endswith("_secret")
    )


def _stream_diagnostics_enabled() -> bool:
    return os.getenv("TEMKI_LITELLM_STREAM_DIAGNOSTICS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _raw_stream_diagnostics_enabled() -> bool:
    return _stream_diagnostics_enabled() and os.getenv(
        "TEMKI_LITELLM_RAW_STREAM_DIAGNOSTICS", ""
    ).lower() in {"1", "true", "yes", "on"}


def _diagnostic_jsonable(
    value: Any, *, _seen: set[int] | None = None, _depth: int = 0
) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime, datetime_time, Path)):
        return str(value)
    if _depth > 32:
        return {"python_type": type(value).__name__, "truncated": "max_depth"}

    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return {"python_type": type(value).__name__, "truncated": "cycle"}

    if isinstance(value, dict):
        seen.add(value_id)
        try:
            result = {}
            for key, item in value.items():
                key_text = str(key)
                if _sensitive_diagnostic_key(key_text):
                    result[key_text] = "[REDACTED]"
                elif key_text == "litellm_logging_obj":
                    result[key_text] = {"python_type": type(item).__name__}
                else:
                    result[key_text] = _diagnostic_jsonable(
                        item, _seen=seen, _depth=_depth + 1
                    )
            return result
        finally:
            seen.remove(value_id)

    if isinstance(value, (list, tuple, set)):
        seen.add(value_id)
        try:
            return [
                _diagnostic_jsonable(item, _seen=seen, _depth=_depth + 1)
                for item in value
            ]
        finally:
            seen.remove(value_id)

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            dumped = None
        if dumped is not None:
            return _diagnostic_jsonable(dumped, _seen=seen, _depth=_depth + 1)

    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    return {"python_type": type(value).__name__}


def _write_raw_diagnostic(stage: str, payload: Any, context: Any = None) -> None:
    if not _raw_stream_diagnostics_enabled():
        return
    path = Path(
        os.getenv(
            "TEMKI_LITELLM_RAW_DIAGNOSTIC_PATH",
            "/var/lib/litellm/diagnostics/stream-debug.jsonl",
        )
    )
    try:
        record = {
            "captured_at": datetime.now().astimezone().isoformat(),
            "pid": os.getpid(),
            "stage": stage,
            "context": _diagnostic_jsonable(context),
            "payload": _diagnostic_jsonable(payload),
        }
        encoded = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        max_bytes = int(
            os.getenv("TEMKI_LITELLM_RAW_DIAGNOSTIC_MAX_BYTES", str(512 * 1024 * 1024))
        )
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if path.exists() and path.stat().st_size + len(encoded) > max_bytes:
                rotated = path.with_suffix(path.suffix + ".1")
                rotated.unlink(missing_ok=True)
                path.replace(rotated)
            with path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    except Exception as exc:
        _stream_logger.warning(
            "TEMKI_RAW_DIAGNOSTIC_WRITE_ERROR stage=%s exception=%s",
            stage,
            type(exc).__name__,
        )


def _safe_upstream_error_text(error_text: Any) -> str:
    body_text = str(error_text or "")
    try:
        payload = json.loads(body_text)
    except (TypeError, json.JSONDecodeError):
        return body_text
    try:
        return json.dumps(
            _diagnostic_jsonable(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        return body_text


def _upstream_error_body(raw_response: Any) -> str:
    """Return the complete upstream error body with structured secrets redacted."""
    try:
        payload = raw_response.json()
    except Exception:
        return str(getattr(raw_response, "text", "") or "")
    return _safe_upstream_error_text(json.dumps(payload, ensure_ascii=False))


def _log_upstream_error_text(
    provider: str,
    status_code: Any,
    error_text: Any,
    headers: Any = None,
    model: Any = "",
) -> None:
    """Log complete provider error text before LiteLLM masks the exception."""
    try:
        status_code = int(status_code or 0)
    except (TypeError, ValueError):
        status_code = 0
    if status_code < 400:
        return

    raw_headers = headers or {}
    safe_headers = {}
    for key in (
        "content-type",
        "retry-after",
        "x-request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "cf-ray",
        "date",
    ):
        value = raw_headers.get(key)
        if value is not None:
            safe_headers[key] = str(value)
    record = {
        "provider": provider,
        "model": str(model or ""),
        "status_code": status_code,
        "headers": safe_headers,
        "body": _safe_upstream_error_text(error_text),
    }
    _stream_logger.warning(
        "TEMKI_UPSTREAM_HTTP_ERROR %s",
        json.dumps(record, ensure_ascii=False, sort_keys=True),
    )


def _log_upstream_http_error(provider: str, model: Any, raw_response: Any) -> None:
    """Log an upstream HTTP error before LiteLLM masks it in cooldown state."""
    _log_upstream_error_text(
        provider,
        getattr(raw_response, "status_code", 0),
        _upstream_error_body(raw_response),
        getattr(raw_response, "headers", {}),
        model,
    )


def _diagnostic_model(model: Any) -> bool:
    model_lc = str(model or "").lower()
    return any(name in model_lc for name in ("gpt", "codex", "sol", "terra", "luna"))


def _value_char_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_value_char_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_char_count(item) for item in value)
    return 0


def _stable_fingerprint(value: Any) -> str | None:
    if value is None:
        return None

    def fallback(item: Any) -> dict[str, str]:
        if hasattr(item, "model_dump"):
            try:
                return item.model_dump()
            except Exception:
                pass
        return {"python_type": type(item).__name__}

    try:
        serialized = json.dumps(
            value,
            default=fallback,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def _count_named_key(value: Any, name: str) -> int:
    if isinstance(value, dict):
        return sum(
            (1 if str(key) == name else 0) + _count_named_key(item, name)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_named_key(item, name) for item in value)
    return 0


def _content_shape(value: Any) -> dict[str, Any]:
    block_counts: dict[str, int] = {}
    block_chars: dict[str, int] = {}
    blocks = value if isinstance(value, list) else [value] if value is not None else []
    for block in blocks:
        if isinstance(block, str):
            block_type = "text"
        elif isinstance(block, dict):
            block_type = str(block.get("type", "dict"))
        else:
            block_type = type(block).__name__
        block_counts[block_type] = block_counts.get(block_type, 0) + 1
        block_chars[block_type] = block_chars.get(block_type, 0) + _value_char_count(block)
    return {
        "chars": _value_char_count(value),
        "block_count": len(blocks),
        "block_counts": block_counts,
        "block_chars": block_chars,
        "cache_control_count": _count_named_key(value, "cache_control"),
        "fingerprint": _stable_fingerprint(value),
    }


def _message_shape(messages: Any) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    role_chars: dict[str, int] = {}
    block_counts: dict[str, int] = {}
    block_chars: dict[str, int] = {}
    if not isinstance(messages, list):
        return {
            "message_count": 0,
            "message_chars": _value_char_count(messages),
            "role_counts": role_counts,
            "role_chars": role_chars,
            "block_counts": block_counts,
            "block_chars": block_chars,
            "message_roles_tail": [],
            "message_chars_tail": [],
            "empty_content_count": 0,
            "cache_control_count": _count_named_key(messages, "cache_control"),
            "messages_fingerprint": _stable_fingerprint(messages),
        }

    message_roles: list[str] = []
    message_char_counts: list[int] = []
    empty_content_count = 0
    for message in messages:
        if not isinstance(message, dict):
            block_type = type(message).__name__
            block_counts[block_type] = block_counts.get(block_type, 0) + 1
            block_chars[block_type] = block_chars.get(block_type, 0) + _value_char_count(
                message
            )
            continue
        role = str(message.get("role", "unknown"))
        role_counts[role] = role_counts.get(role, 0) + 1
        content = message.get("content")
        content_chars = _value_char_count(content)
        role_chars[role] = role_chars.get(role, 0) + content_chars
        message_roles.append(role)
        message_char_counts.append(content_chars)
        if content is None or content == "" or content == []:
            empty_content_count += 1
        if isinstance(content, str):
            block_counts["text"] = block_counts.get("text", 0) + 1
            block_chars["text"] = block_chars.get("text", 0) + len(content)
        elif isinstance(content, list):
            for block in content:
                block_type = (
                    str(block.get("type", "unknown"))
                    if isinstance(block, dict)
                    else type(block).__name__
                )
                block_counts[block_type] = block_counts.get(block_type, 0) + 1
                block_chars[block_type] = block_chars.get(
                    block_type, 0
                ) + _value_char_count(block)

    return {
        "message_count": len(messages),
        "message_chars": _value_char_count(messages),
        "role_counts": role_counts,
        "role_chars": role_chars,
        "block_counts": block_counts,
        "block_chars": block_chars,
        "message_roles_tail": message_roles[-12:],
        "message_chars_tail": message_char_counts[-12:],
        "max_message_chars": max(message_char_counts, default=0),
        "empty_content_count": empty_content_count,
        "cache_control_count": _count_named_key(messages, "cache_control"),
        "messages_fingerprint": _stable_fingerprint(messages),
    }


def _tool_shape(tools: Any) -> dict[str, Any]:
    if not isinstance(tools, list):
        return {
            "tool_count": 0,
            "tool_chars": _value_char_count(tools),
            "tool_name_lengths": [],
            "tool_description_chars": 0,
            "tool_schema_chars": 0,
            "duplicate_tool_name_count": 0,
            "tools_fingerprint": _stable_fingerprint(tools),
        }

    names: list[str] = []
    description_chars = 0
    schema_chars = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if isinstance(name, str):
            names.append(name)
        description_chars += _value_char_count(function.get("description"))
        schema_chars += _value_char_count(
            function.get("parameters", function.get("input_schema"))
        )

    return {
        "tool_count": len(tools),
        "tool_chars": _value_char_count(tools),
        "tool_name_lengths": [len(name) for name in names],
        "tool_description_chars": description_chars,
        "tool_schema_chars": schema_chars,
        "duplicate_tool_name_count": len(names) - len(set(names)),
        "cache_control_count": _count_named_key(tools, "cache_control"),
        "tools_fingerprint": _stable_fingerprint(tools),
    }


def _request_shape(values: dict[str, Any]) -> dict[str, Any]:
    messages = values.get("messages", values.get("input"))
    tools = values.get("tools")
    tool_choice = values.get("tool_choice")
    thinking = values.get("thinking")
    extra_kwargs = values.get("extra_kwargs")
    if not isinstance(extra_kwargs, dict):
        extra_kwargs = {}
    system_shape = _content_shape(values.get("system"))
    instructions_shape = _content_shape(values.get("instructions"))
    reasoning = values.get("reasoning", values.get("reasoning_effort"))
    metadata = values.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    shape = {
        "model": values.get("model"),
        "stream": values.get("stream"),
        "store": values.get("store"),
        "system": system_shape,
        "instructions": instructions_shape,
        "input_chars": _value_char_count(values.get("input")),
        "tool_choice_type": tool_choice.get("type")
        if isinstance(tool_choice, dict)
        else tool_choice,
        "tool_choice_has_name": bool(tool_choice.get("name"))
        if isinstance(tool_choice, dict)
        else False,
        "thinking_type": thinking.get("type") if isinstance(thinking, dict) else None,
        "thinking_budget": thinking.get("budget_tokens") if isinstance(thinking, dict) else None,
        "reasoning_effort": reasoning.get("effort")
        if isinstance(reasoning, dict)
        else reasoning,
        "reasoning_summary": reasoning.get("summary")
        if isinstance(reasoning, dict)
        else None,
        "max_tokens": values.get("max_tokens", values.get("max_completion_tokens")),
        "stop_count": len(values.get("stop_sequences", []))
        if isinstance(values.get("stop_sequences"), list)
        else 0,
        "temperature": values.get("temperature"),
        "top_p": values.get("top_p"),
        "top_k": values.get("top_k"),
        "parallel_tool_calls": values.get("parallel_tool_calls"),
        "output_format_present": values.get("output_format") is not None,
        "output_config_keys": sorted(values.get("output_config", {}).keys())
        if isinstance(values.get("output_config"), dict)
        else [],
        "include": values.get("include") if isinstance(values.get("include"), list) else [],
        "truncation": values.get("truncation"),
        "previous_response_id_present": bool(values.get("previous_response_id")),
        "context_management_present": "context_management" in values,
        "top_level_keys": sorted(str(key) for key in values.keys()),
        "extra_keys": sorted(str(key) for key in extra_kwargs.keys()),
        "metadata_keys": sorted(str(key) for key in metadata.keys()),
        "call_id": extra_kwargs.get("litellm_call_id")
        or values.get("litellm_call_id")
        or _find_named_value(metadata, ("litellm_call_id", "call_id")),
        "key_alias": _find_named_value(
            metadata, ("user_api_key_alias", "key_alias")
        ),
        "session_id_present": bool(
            extra_kwargs.get("litellm_session_id")
            or extra_kwargs.get("session_id")
            or _find_named_value(metadata, ("litellm_session_id", "session_id"))
        ),
        "request_fingerprint": _stable_fingerprint(
            {
                key: values.get(key)
                for key in (
                    "model",
                    "messages",
                    "input",
                    "system",
                    "instructions",
                    "tools",
                    "tool_choice",
                    "thinking",
                    "reasoning",
                    "reasoning_effort",
                    "output_format",
                    "output_config",
                    "include",
                    "truncation",
                )
                if key in values
            }
        ),
    }
    shape.update(_message_shape(messages))
    shape.update(_tool_shape(tools))
    return shape


def _output_item_summary(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"python_type": type(item).__name__}
    content = item.get("content")
    content_types = []
    if isinstance(content, list):
        content_types = [
            str(block.get("type", "unknown"))
            for block in content
            if isinstance(block, dict)
        ]
    arguments = item.get("arguments")
    return {
        "type": item.get("type"),
        "role": item.get("role"),
        "status": item.get("status"),
        "content_types": content_types,
        "text_chars": len(_output_item_text(item)),
        "tool_name_present": bool(item.get("name")),
        "arguments_chars": len(arguments) if isinstance(arguments, str) else 0,
        "encrypted_content_present": bool(item.get("encrypted_content")),
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            return {}
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _find_named_value(value: Any, names: tuple[str, ...], depth: int = 0) -> Any:
    mapping = _as_mapping(value)
    if not mapping or depth > 4:
        return None
    for name in names:
        if mapping.get(name) not in (None, ""):
            return mapping[name]
    for key, child in mapping.items():
        if key in {"messages", "input", "tools", "system", "instructions"}:
            continue
        if isinstance(child, dict) or hasattr(child, "model_dump"):
            found = _find_named_value(child, names, depth + 1)
            if found not in (None, ""):
                return found
    return None


_CHATGPT_OPAQUE_TOKEN_KEYS = frozenset({"encrypted_content"})


def _chatgpt_text_token_count(text: str) -> int:
    global _chatgpt_token_encoding
    if _chatgpt_token_encoding is None:
        try:
            import tiktoken

            _chatgpt_token_encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            _chatgpt_token_encoding = False
    if _chatgpt_token_encoding is not False:
        return len(_chatgpt_token_encoding.encode(text))
    return (len(text.encode("utf-8")) + 2) // 3


def _chatgpt_token_estimate(value: Any) -> int:
    if isinstance(value, str):
        return _chatgpt_text_token_count(value)
    if value is None or isinstance(value, (bool, int, float)):
        return 1
    if isinstance(value, dict):
        total = 2
        for key, entry in value.items():
            if str(key) in _CHATGPT_OPAQUE_TOKEN_KEYS and isinstance(entry, str):
                # Opaque payloads (encrypted reasoning) decrypt upstream to roughly
                # base64_len/4 tokens; tokenizing the base64 as text bills them
                # ~2-3x too high.
                total += 1 + len(entry) // 4
                continue
            total += 1 + _chatgpt_token_estimate(entry)
        return total
    if isinstance(value, (list, tuple)):
        return 2 + sum(_chatgpt_token_estimate(entry) for entry in value)
    return _chatgpt_text_token_count(str(_diagnostic_jsonable(value)))


def _chatgpt_serialized_token_count(value: Any) -> tuple[int, str]:
    # Structure-aware estimate. Full-JSON tiktoken counting billed opaque reasoning
    # payloads, JSON escaping, and punctuation at text rate — 417,712 estimated vs
    # ~199,939 actual (incident 2026-07-17) — so the trimmer fired ~150k tokens
    # early and then dropped tool-registration items.
    _chatgpt_text_token_count("")  # resolve the encoder for an honest counter label
    tokens = _chatgpt_token_estimate(value)
    counter = (
        "o200k_structural"
        if _chatgpt_token_encoding is not False
        else "utf8_bytes_div_3_structural"
    )
    return tokens, counter


def _chatgpt_control_input_item(item: Any) -> bool:
    """Structural control items that must survive a context trim. Codex >= 0.144
    registers its whole toolset as {"type": "additional_tools", "role":
    "developer", "tools": [...]} (see _normalize_input_list); any structural
    (non-message) developer item is control-plane, not conversation — dropping
    one silently strips every tool from the session and the agent can only emit
    plain text (incident 2026-07-17)."""
    item_dict = _as_mapping(item)
    item_type = str(item_dict.get("type") or "message").lower()
    if item_type == "additional_tools":
        return True
    role = str(item_dict.get("role") or "").lower()
    return role == "developer" and item_type != "message"


def _trim_chatgpt_subscription_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        token_budget = int(
            os.getenv("TEMKI_CHATGPT_SUBSCRIPTION_INPUT_TOKEN_BUDGET", "300000")
        )
    except ValueError:
        token_budget = 300000

    input_items = request.get("input")
    if token_budget <= 0 or not isinstance(input_items, list):
        return request, {"context_trimmed": False, "trim_reason": "disabled_or_non_list"}

    original_tokens, counter = _chatgpt_serialized_token_count(request)
    base_summary = {
        "context_trimmed": False,
        "token_counter": counter,
        "token_budget": token_budget,
        "original_serialized_tokens": original_tokens,
        "original_input_items": len(input_items),
    }
    if original_tokens <= token_budget:
        return request, base_summary

    prefix: list[Any] = []
    if input_items:
        first = _as_mapping(input_items[0])
        # The instruction preamble (_assistant_input_item) predating the
        # type="message" stamp has no type at all — accept both shapes, or the
        # trim silently drops the whole system preamble (incident 2026-07-17).
        if first.get("role") == "assistant" and first.get("type") in {None, "message"}:
            prefix = [input_items[0]]

    user_boundaries = []
    for index, item in enumerate(input_items):
        item_dict = _as_mapping(item)
        if item_dict.get("role") == "user" and item_dict.get("type") in {
            None,
            "message",
        }:
            user_boundaries.append(index)
    if not user_boundaries:
        base_summary["trim_reason"] = "no_user_boundary"
        return request, base_summary

    def candidate(boundary_index: int) -> tuple[dict[str, Any], int, list[Any]]:
        # Control items (tool registration et al.) survive wherever they sit in
        # the dropped range; conversational items there are discarded.
        controls = [
            item
            for item in input_items[len(prefix) : boundary_index]
            if _chatgpt_control_input_item(item)
        ]
        retained = prefix + controls + list(input_items[boundary_index:])
        candidate_request = dict(request)
        candidate_request["input"] = retained
        tokens, _ = _chatgpt_serialized_token_count(candidate_request)
        return candidate_request, tokens, retained

    selected: tuple[int, dict[str, Any], int, list[Any]] | None = None
    low = 0
    high = len(user_boundaries) - 1
    while low <= high:
        middle = (low + high) // 2
        boundary = user_boundaries[middle]
        candidate_request, tokens, retained = candidate(boundary)
        if tokens <= token_budget:
            selected = (boundary, candidate_request, tokens, retained)
            high = middle - 1
        else:
            low = middle + 1

    over_budget = selected is None
    if selected is None:
        boundary = user_boundaries[-1]
        candidate_request, tokens, retained = candidate(boundary)
        selected = (boundary, candidate_request, tokens, retained)

    boundary, trimmed_request, trimmed_tokens, retained = selected
    summary = {
        **base_summary,
        "context_trimmed": len(retained) < len(input_items),
        "trim_start_index": boundary,
        "retained_input_items": len(retained),
        "retained_control_items": sum(
            1 for item in retained if _chatgpt_control_input_item(item)
        ),
        "dropped_input_items": len(input_items) - len(retained),
        "trimmed_serialized_tokens": trimmed_tokens,
        "over_budget_after_trim": over_budget or trimmed_tokens > token_budget,
    }
    if summary["context_trimmed"]:
        _stream_logger.warning(
            "TEMKI_CHATGPT_CONTEXT_TRIM %s",
            json.dumps(summary, ensure_ascii=True, sort_keys=True),
        )
    return trimmed_request, summary


def _chatgpt_upstream_error(event: Any) -> _ChatGPTUpstreamResponseError | None:
    if not isinstance(event, dict):
        return None
    response = _as_mapping(event.get("response"))
    error = _as_mapping(response.get("error")) or _as_mapping(event.get("error"))
    incomplete = _as_mapping(response.get("incomplete_details"))
    code = error.get("code") or error.get("type") or incomplete.get("reason")
    message = error.get("message")
    event_type = str(event.get("type", "upstream_error"))
    if not isinstance(message, str) or not message:
        if event_type == "error":
            return None
        message = f"ChatGPT upstream ended with {event_type}"
        if code:
            message += f" ({code})"
    return _ChatGPTUpstreamResponseError(message, str(code) if code else None)


def _iterator_correlation(self: Any) -> dict[str, Any]:
    completion_stream = getattr(self, "completion_stream", None)
    logging_obj = getattr(self, "logging_obj", None) or getattr(
        completion_stream, "logging_obj", None
    )
    candidates = [
        getattr(self, "request_data", None),
        getattr(self, "litellm_metadata", None),
        getattr(completion_stream, "request_data", None),
        getattr(completion_stream, "litellm_metadata", None),
        getattr(completion_stream, "_hidden_params", None),
        getattr(logging_obj, "model_call_details", None),
    ]

    def first(names: tuple[str, ...]) -> Any:
        for candidate in candidates:
            value = _find_named_value(candidate, names)
            if value not in (None, ""):
                return value
        return None

    response = getattr(self, "response", None) or getattr(
        completion_stream, "response", None
    )
    headers = getattr(response, "headers", {}) or {}
    started = getattr(self, "_stream_created_time", None)
    return {
        "model": getattr(self, "model", None),
        "provider": str(
            getattr(self, "custom_llm_provider", "")
            or getattr(completion_stream, "custom_llm_provider", "")
            or ""
        ),
        "call_type": getattr(self, "call_type", None)
        or getattr(completion_stream, "call_type", None),
        "call_id": first(("litellm_call_id", "call_id")),
        "key_alias": first(("user_api_key_alias", "key_alias")),
        "session_id_present": first(("litellm_session_id", "session_id")) is not None,
        "http_status": getattr(response, "status_code", None),
        "content_type": headers.get("content-type")
        if hasattr(headers, "get")
        else None,
        "upstream_request_id": (
            headers.get("x-request-id") or headers.get("request-id")
            if hasattr(headers, "get")
            else None
        ),
        "elapsed_ms": round((time.time() - started) * 1000)
        if isinstance(started, (int, float))
        else None,
    }


def _response_payload_summary(value: Any) -> dict[str, Any]:
    response = _as_mapping(value)
    if not response:
        return {"response_present": value is not None}
    output = response.get("output")
    output_items = output if isinstance(output, list) else []
    usage = _as_mapping(response.get("usage"))
    error = _as_mapping(response.get("error"))
    incomplete = _as_mapping(response.get("incomplete_details"))
    message = error.get("message")
    return {
        "response_present": True,
        "response_id": response.get("id"),
        "response_status": response.get("status"),
        "terminal_output_count": len(output_items),
        "terminal_output_items": [_output_item_summary(item) for item in output_items],
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": _find_named_value(usage, ("cached_tokens",)),
        "incomplete_reason": incomplete.get("reason"),
        "error_type": error.get("type"),
        "error_code": error.get("code"),
        "error_message": message[:500] if isinstance(message, str) else None,
    }


def _terminal_event_summary(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {"event_present": False}
    summary = {
        "event_present": True,
        "event_type": event.get("type"),
        "event_keys": sorted(str(key) for key in event.keys()),
    }
    summary.update(_response_payload_summary(event.get("response")))
    error = _as_mapping(event.get("error"))
    if error:
        message = error.get("message")
        summary.update(
            {
                "error_type": error.get("type"),
                "error_code": error.get("code"),
                "error_message": message[:500] if isinstance(message, str) else None,
            }
        )
    return summary


def _stream_event_state(self: Any) -> dict[str, Any]:
    state = {
        "event_counts": getattr(self, "_temki_stream_event_counts", {}),
        "event_sequence": getattr(self, "_temki_stream_event_sequence", []),
        "output_items": getattr(self, "_temki_stream_output_item_summaries", []),
        "delta_chars": getattr(self, "_temki_stream_delta_chars", 0),
        "reasoning_delta_chars": getattr(self, "_temki_stream_reasoning_delta_chars", 0),
        "function_argument_delta_chars": getattr(
            self, "_temki_stream_function_argument_delta_chars", 0
        ),
        "recovered_output_count": len(
            _merge_recovered_output(
                getattr(self, "_temki_streamed_output_items", {}),
                getattr(self, "_temki_streamed_text_only_items", {}),
            )
        ),
        "terminal_seen": bool(getattr(self, "_temki_stream_terminal_seen", False)),
        "done_seen": bool(getattr(self, "_temki_stream_done_seen", False)),
    }
    state.update(_iterator_correlation(self))
    return state


def _capture_raw_stream_chunk(self: Any, namespace: str, chunk: Any) -> None:
    if not _raw_stream_diagnostics_enabled():
        return
    if isinstance(chunk, bytes):
        captured = chunk.decode("utf-8", errors="replace")
        chunk_bytes = len(chunk)
    elif isinstance(chunk, str):
        captured = chunk
        chunk_bytes = len(chunk.encode("utf-8"))
    else:
        captured = _diagnostic_jsonable(chunk)
        chunk_bytes = len(
            json.dumps(captured, ensure_ascii=False, default=str).encode("utf-8")
        )

    size_attr = f"_temki_{namespace}_raw_bytes"
    chunks_attr = f"_temki_{namespace}_raw_chunks"
    truncated_attr = f"_temki_{namespace}_raw_truncated"
    current_bytes = getattr(self, size_attr, 0)
    max_bytes = int(
        os.getenv("TEMKI_LITELLM_RAW_STREAM_MAX_BYTES", str(64 * 1024 * 1024))
    )
    if current_bytes + chunk_bytes > max_bytes:
        setattr(self, truncated_attr, True)
        return
    chunks = getattr(self, chunks_attr, None)
    if not isinstance(chunks, list):
        chunks = []
        setattr(self, chunks_attr, chunks)
    chunks.append(captured)
    setattr(self, size_attr, current_bytes + chunk_bytes)


def _write_raw_stream_snapshot(
    self: Any, namespace: str, stage: str, reason: str
) -> None:
    if not _raw_stream_diagnostics_enabled():
        return
    context = _stream_event_state(self)
    context["snapshot_reason"] = reason
    _write_raw_diagnostic(
        stage,
        {
            "chunks": getattr(self, f"_temki_{namespace}_raw_chunks", []),
            "bytes": getattr(self, f"_temki_{namespace}_raw_bytes", 0),
            "truncated": bool(
                getattr(self, f"_temki_{namespace}_raw_truncated", False)
            ),
        },
        context,
    )


def _parse_sse_json_chunk(chunk: Any) -> dict[str, Any] | None:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    if not isinstance(chunk, str):
        return None

    payload = chunk.strip()
    if payload.startswith("event:"):
        return None
    if payload.startswith("data:"):
        payload = payload[5:].strip()
    if not payload or payload == "[DONE]":
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_sse_done_chunk(chunk: Any) -> bool:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    if not isinstance(chunk, str):
        return False
    payload = chunk.strip()
    if payload.startswith("data:"):
        payload = payload[5:].strip()
    return payload == "[DONE]"


def _safe_output_index(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        index = int(value)
    except (TypeError, ValueError):
        return fallback
    if index < 0 or index > _MAX_RECOVERED_OUTPUT_INDEX:
        return fallback
    return index


def _record_responses_output_event(
    event: dict[str, Any],
    output_items: dict[int, dict[str, Any]],
    text_only_items: dict[int, dict[str, Any]],
) -> None:
    event_type = event.get("type")
    if event_type == "response.output_item.done":
        item = event.get("item")
        if not isinstance(item, dict):
            return
        index = _safe_output_index(event.get("output_index"), len(output_items))
        output_items[index] = dict(item)
        return

    if event_type != "response.output_text.done":
        return
    text = event.get("text")
    if not isinstance(text, str):
        return

    output_index = _safe_output_index(event.get("output_index"), len(text_only_items))
    if output_index in output_items:
        return
    item = text_only_items.setdefault(
        output_index,
        {
            "type": "message",
            "id": event.get("item_id") or f"msg_{output_index}",
            "role": "assistant",
            "status": "completed",
            "content": [],
        },
    )
    content = item.get("content")
    if not isinstance(content, list):
        return
    content_index = _safe_output_index(event.get("content_index"), len(content))
    while len(content) <= content_index:
        content.append({"type": "output_text", "text": "", "annotations": []})
    content[content_index] = {
        "type": "output_text",
        "text": text,
        "annotations": event.get("annotations") or [],
    }


def _merge_recovered_output(
    output_items: dict[int, dict[str, Any]],
    text_only_items: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = dict(text_only_items)
    merged.update(output_items)
    return [item for _, item in sorted(merged.items())]


def _output_item_text(item: Any) -> str:
    if not isinstance(item, dict) or item.get("type") != "message":
        return ""
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") not in {"output_text", "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _missing_stream_text(emitted: str, completed: str) -> str:
    if not completed or completed == emitted:
        return ""
    if completed.startswith(emitted):
        return completed[len(emitted) :]
    # A provider may normalize the final text. Avoid duplicating content when
    # it cannot be reconciled safely with deltas already sent downstream.
    return completed if not emitted else ""


def _recover_responses_output(raw_sse: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sse, (str, bytes)):
        return []
    if isinstance(raw_sse, bytes):
        raw_sse = raw_sse.decode("utf-8", errors="replace")

    output_items: dict[int, dict[str, Any]] = {}
    text_only_items: dict[int, dict[str, Any]] = {}
    for chunk in raw_sse.splitlines():
        event = _parse_sse_json_chunk(chunk)
        if event is None:
            continue
        if event.get("type") == "response.completed":
            response = event.get("response")
            authoritative = response.get("output") if isinstance(response, dict) else None
            if isinstance(authoritative, list) and authoritative:
                return [item for item in authoritative if isinstance(item, dict)]
            continue
        _record_responses_output_event(event, output_items, text_only_items)
    return _merge_recovered_output(output_items, text_only_items)


def _response_output(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("output")
    return getattr(response, "output", None)


def _backfill_response_output(response: Any, raw_sse: Any) -> None:
    if response is None or _response_output(response):
        return
    recovered = _recover_responses_output(raw_sse)
    if not recovered:
        return
    if isinstance(response, dict):
        response["output"] = recovered
    else:
        response.output = recovered


def _logging_original_response(logging_obj: Any) -> Any:
    details = getattr(logging_obj, "model_call_details", None)
    return details.get("original_response") if isinstance(details, dict) else None


def _patch_chatgpt_non_stream_responses() -> None:
    try:
        from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
    except (ImportError, AttributeError):
        return

    original = ChatGPTResponsesAPIConfig.transform_response_api_response
    if getattr(original, "_temki_recovers_empty_output", False):
        return

    @wraps(original)
    def patched(self, model: str, raw_response: Any, logging_obj: Any):
        response = original(self, model=model, raw_response=raw_response, logging_obj=logging_obj)
        _backfill_response_output(response, getattr(raw_response, "text", None))
        return response

    patched._temki_recovers_empty_output = True
    ChatGPTResponsesAPIConfig.transform_response_api_response = patched


def _patch_responses_completion_bridge() -> None:
    try:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            LiteLLMResponsesTransformationHandler,
        )
    except (ImportError, AttributeError):
        return

    original = LiteLLMResponsesTransformationHandler.transform_response
    if getattr(original, "_temki_recovers_empty_output", False):
        return

    @wraps(original)
    def patched(self, *args, **kwargs):
        raw_response = kwargs.get("raw_response")
        logging_obj = kwargs.get("logging_obj")
        if raw_response is None and len(args) > 1:
            raw_response = args[1]
        if logging_obj is None and len(args) > 3:
            logging_obj = args[3]
        _backfill_response_output(raw_response, _logging_original_response(logging_obj))
        return original(self, *args, **kwargs)

    patched._temki_recovers_empty_output = True
    LiteLLMResponsesTransformationHandler.transform_response = patched


def _patch_chatgpt_streaming_responses() -> None:
    try:
        from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
    except (ImportError, AttributeError):
        return

    original = BaseResponsesAPIStreamingIterator._process_chunk
    if getattr(original, "_temki_recovers_empty_output", False):
        return

    @wraps(original)
    def patched(self, chunk):
        event = _parse_sse_json_chunk(chunk)
        is_done = _is_sse_done_chunk(chunk)
        provider = str(getattr(self, "custom_llm_provider", "")).lower()
        if "chatgpt" not in provider:
            return original(self, chunk)
        if _diagnostic_model(getattr(self, "model", None)):
            _capture_raw_stream_chunk(self, "chatgpt_upstream", chunk)

        output_items = getattr(self, "_temki_streamed_output_items", None)
        if not isinstance(output_items, dict):
            output_items = {}
            self._temki_streamed_output_items = output_items
        text_only_items = getattr(self, "_temki_streamed_text_only_items", None)
        if not isinstance(text_only_items, dict):
            text_only_items = {}
            self._temki_streamed_text_only_items = text_only_items

        event_type = event.get("type") if isinstance(event, dict) else None
        terminal_types = {
            "response.completed",
            "response.failed",
            "response.incomplete",
            "error",
        }
        if event is not None:
            event_counts = getattr(self, "_temki_stream_event_counts", None)
            if not isinstance(event_counts, dict):
                event_counts = {}
                self._temki_stream_event_counts = event_counts
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            event_sequence = getattr(self, "_temki_stream_event_sequence", None)
            if not isinstance(event_sequence, list):
                event_sequence = []
                self._temki_stream_event_sequence = event_sequence
            if len(event_sequence) < 256:
                event_sequence.append(str(event_type))

            delta = event.get("delta")
            if event_type == "response.output_text.delta" and isinstance(delta, str):
                self._temki_stream_delta_chars = getattr(self, "_temki_stream_delta_chars", 0) + len(
                    delta
                )
            elif event_type == "response.function_call_arguments.delta" and isinstance(
                delta, str
            ):
                self._temki_stream_function_argument_delta_chars = getattr(
                    self, "_temki_stream_function_argument_delta_chars", 0
                ) + len(delta)
            elif "reasoning" in str(event_type) and str(event_type).endswith(".delta") and isinstance(
                delta, str
            ):
                self._temki_stream_reasoning_delta_chars = getattr(
                    self, "_temki_stream_reasoning_delta_chars", 0
                ) + len(delta)
            elif event_type == "response.output_item.done":
                item_summaries = getattr(self, "_temki_stream_output_item_summaries", None)
                if not isinstance(item_summaries, list):
                    item_summaries = []
                    self._temki_stream_output_item_summaries = item_summaries
                item_summaries.append(_output_item_summary(event.get("item")))

            _record_responses_output_event(event, output_items, text_only_items)
            if event_type in terminal_types:
                self._temki_stream_terminal_seen = True

        if is_done:
            self._temki_stream_done_seen = True

        try:
            transformed = original(self, chunk)
        except BaseException as exc:
            if _stream_diagnostics_enabled():
                summary = _stream_event_state(self)
                summary.update(
                    {
                        "event_type": event_type,
                        "exception_type": type(exc).__name__,
                        **_terminal_event_summary(event),
                    }
                )
                _stream_logger.warning(
                    "TEMKI_STREAM_PROCESSING_ERROR %s",
                    json.dumps(summary, ensure_ascii=True, sort_keys=True),
                )
                _write_raw_stream_snapshot(
                    self,
                    "chatgpt_upstream",
                    "chatgpt_upstream_sse_error",
                    "processing_error",
                )
            raise

        if event is None:
            if is_done and _stream_diagnostics_enabled():
                summary = _stream_event_state(self)
                log_name = (
                    "TEMKI_STREAM_DONE"
                    if summary["terminal_seen"]
                    else "TEMKI_STREAM_DONE_WITHOUT_TERMINAL"
                )
                _stream_logger.warning(
                    "%s %s",
                    log_name,
                    json.dumps(summary, ensure_ascii=True, sort_keys=True),
                )
                _write_raw_stream_snapshot(
                    self,
                    "chatgpt_upstream",
                    "chatgpt_upstream_sse_done",
                    "done_marker",
                )
            return transformed

        if event_type in {"response.failed", "response.incomplete", "error"}:
            if _stream_diagnostics_enabled():
                summary = _stream_event_state(self)
                summary.update(_terminal_event_summary(event))
                _stream_logger.warning(
                    "TEMKI_STREAM_NON_COMPLETED_TERMINAL %s",
                    json.dumps(summary, ensure_ascii=True, sort_keys=True),
                )
                _write_raw_stream_snapshot(
                    self,
                    "chatgpt_upstream",
                    "chatgpt_upstream_sse_terminal",
                    str(event_type),
                )
            upstream_error = _chatgpt_upstream_error(event)
            if upstream_error is not None:
                raise upstream_error

        if event_type == "response.completed":
            transformed_response = (
                getattr(transformed, "response", None) if transformed is not None else None
            )
            raw_response = event.get("response")
            recovered = _merge_recovered_output(output_items, text_only_items)
            if transformed_response is not None and not _response_output(
                transformed_response
            ):
                if isinstance(transformed_response, dict):
                    transformed_response["output"] = recovered
                elif recovered:
                    transformed_response.output = recovered
            if _stream_diagnostics_enabled():
                summary = _stream_event_state(self)
                summary.update(_terminal_event_summary(event))
                summary.update(
                    {
                        "transformed_event_present": transformed is not None,
                        "transformed_response_present": transformed_response is not None,
                        "raw_response_present": raw_response is not None,
                        "recovered_output_count": len(recovered),
                    }
                )
                _stream_logger.warning(
                    "TEMKI_STREAM_TERMINAL %s",
                    json.dumps(summary, ensure_ascii=True, sort_keys=True),
                )
                _write_raw_stream_snapshot(
                    self,
                    "chatgpt_upstream",
                    "chatgpt_upstream_sse_terminal",
                    "response.completed",
                )
        return transformed

    patched._temki_recovers_empty_output = True
    BaseResponsesAPIStreamingIterator._process_chunk = patched


def _patch_chatgpt_stream_lifecycle() -> None:
    try:
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
    except (ImportError, AttributeError):
        return

    original = ResponsesAPIStreamingIterator.__anext__
    if getattr(original, "_temki_logs_stream_exit", False):
        return

    @wraps(original)
    async def patched(self):
        try:
            return await original(self)
        except BaseException as exc:
            provider = str(getattr(self, "custom_llm_provider", "")).lower()
            diagnostic = (
                _stream_diagnostics_enabled()
                and "chatgpt" in provider
                and _diagnostic_model(getattr(self, "model", None))
            )
            if diagnostic and not getattr(self, "_temki_stream_exit_logged", False):
                self._temki_stream_exit_logged = True
                summary = _stream_event_state(self)
                summary.update(
                    {
                        "exit_type": type(exc).__name__,
                        "normal_eof": isinstance(exc, StopAsyncIteration),
                    }
                )
                _stream_logger.warning(
                    "TEMKI_STREAM_EXIT %s",
                    json.dumps(summary, ensure_ascii=True, sort_keys=True),
                )
                _write_raw_stream_snapshot(
                    self,
                    "chatgpt_upstream",
                    "chatgpt_upstream_sse_exit",
                    type(exc).__name__,
                )
            raise

    patched._temki_logs_stream_exit = True
    ResponsesAPIStreamingIterator.__anext__ = patched


def _patch_chatgpt_outgoing_request() -> None:
    try:
        from litellm.llms.chatgpt.responses.transformation import (
            ChatGPTResponsesAPIConfig,
        )
    except (ImportError, AttributeError):
        return

    original = ChatGPTResponsesAPIConfig.transform_responses_api_request
    if getattr(original, "_temki_logs_outgoing_shape", False):
        return

    @wraps(original)
    def patched(self, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        litellm_params = kwargs.get(
            "litellm_params", args[3] if len(args) > 3 else None
        )
        headers = kwargs.get("headers", args[4] if len(args) > 4 else None)
        diagnostic = _stream_diagnostics_enabled() and _diagnostic_model(model)
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            if diagnostic:
                _stream_logger.warning(
                    "TEMKI_CHATGPT_OUTGOING_ERROR model=%s exception=%s",
                    model,
                    type(exc).__name__,
                )
            raise
        trim_summary: dict[str, Any] = {}
        if isinstance(result, dict):
            result, trim_summary = _trim_chatgpt_subscription_request(result)
        if diagnostic and isinstance(result, dict):
            summary = _request_shape(result)
            summary.update(
                {
                    "stage": "chatgpt_http_request",
                    "call_id": summary.get("call_id")
                    or _find_named_value(litellm_params, ("litellm_call_id", "call_id")),
                    "key_alias": _find_named_value(
                        litellm_params, ("user_api_key_alias", "key_alias")
                    ),
                    "session_id_present": summary.get("session_id_present")
                    or _find_named_value(
                        litellm_params, ("litellm_session_id", "session_id")
                    )
                    is not None,
                    "header_names": sorted(str(key).lower() for key in headers.keys())
                    if isinstance(headers, dict)
                    else [],
                    **trim_summary,
                }
            )
            _stream_logger.warning(
                "TEMKI_CHATGPT_OUTGOING %s",
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
            _write_raw_diagnostic("chatgpt_http_request", result, summary)
        return result

    patched._temki_logs_outgoing_shape = True
    ChatGPTResponsesAPIConfig.transform_responses_api_request = patched


def _patch_responses_to_chat_stream() -> None:
    try:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            OpenAiResponsesToChatCompletionStreamIterator,
        )
    except (ImportError, AttributeError):
        return

    original = OpenAiResponsesToChatCompletionStreamIterator.chunk_parser
    if getattr(original, "_temki_recovers_done_text", False):
        return

    @wraps(original)
    def patched(self, chunk: dict[str, Any]):
        event = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
        result = original(self, chunk)
        if not isinstance(event, dict):
            return result

        event_type = event.get("type")
        if hasattr(event_type, "value"):
            event_type = event_type.value
        output_index = _safe_output_index(event.get("output_index"), 0)
        emitted_by_output = getattr(self, "_temki_emitted_text_by_output", None)
        if not isinstance(emitted_by_output, dict):
            emitted_by_output = {}
            self._temki_emitted_text_by_output = emitted_by_output

        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                emitted_by_output[output_index] = emitted_by_output.get(output_index, "") + delta
            return result

        completed_text = ""
        if event_type == "response.output_text.done":
            text = event.get("text")
            completed_text = text if isinstance(text, str) else ""
        elif event_type == "response.output_item.done":
            completed_text = _output_item_text(event.get("item"))
        else:
            return result

        emitted = emitted_by_output.get(output_index, "")
        missing = _missing_stream_text(emitted, completed_text)
        if not missing:
            return result
        emitted_by_output[output_index] = completed_text

        if _stream_diagnostics_enabled():
            _stream_logger.warning(
                "TEMKI_STREAM_DONE_RECOVERY output_index=%d event_type=%s emitted_chars=%d completed_chars=%d missing_chars=%d",
                output_index,
                event_type,
                len(emitted),
                len(completed_text),
                len(missing),
            )

        choices = getattr(result, "choices", None)
        if not choices:
            return result
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
        if isinstance(delta, dict):
            delta["content"] = missing
        elif delta is not None:
            delta.content = missing
        return result

    patched._temki_recovers_done_text = True
    OpenAiResponsesToChatCompletionStreamIterator.chunk_parser = patched


def _patch_mimo_default_max_tokens() -> None:
    # The MiMo upstream (vLLM) rejects max_new_tokens=-1, and the
    # messages-to-completion adapter emits exactly that when a client sends
    # /v1/messages without max_tokens — bypassing the per-deployment
    # max_tokens default, which only merges for absent keys. Lift any
    # non-positive max_tokens to a safe floor for self-hosted MiMo endpoints.
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )
    except (ImportError, AttributeError):
        return

    original = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs
    if getattr(original, "_temki_mimo_default_max_tokens", False):
        return

    @wraps(original)
    def patched(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, tuple) and result and isinstance(result[0], dict):
            mapped = result[0]
            model = mapped.get("model")
            max_tokens = mapped.get("max_tokens")
            if (
                isinstance(model, str)
                and "mimo" in model.lower()
                and (max_tokens is None or (isinstance(max_tokens, int) and max_tokens <= 0))
            ):
                mapped["max_tokens"] = 32768
        return result

    patched._temki_mimo_default_max_tokens = True
    LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs = staticmethod(
        patched
    )


def _patch_anthropic_request_adapter() -> None:
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )
    except (ImportError, AttributeError):
        return

    original = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs
    if getattr(original, "_temki_logs_request_shape", False):
        return

    @wraps(original)
    def patched(*args, **kwargs):
        model = kwargs.get("model")
        diagnostic = _stream_diagnostics_enabled() and _diagnostic_model(model)
        if diagnostic:
            summary = _request_shape(kwargs)
            summary["stage"] = "anthropic_input"
            _stream_logger.warning(
                "TEMKI_ANTHROPIC_INPUT %s",
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
            _write_raw_diagnostic("anthropic_input", kwargs, summary)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            if diagnostic:
                _stream_logger.warning(
                    "TEMKI_ANTHROPIC_MAPPING_ERROR model=%s exception=%s",
                    model,
                    type(exc).__name__,
                )
            raise
        if diagnostic and isinstance(result, tuple) and result and isinstance(result[0], dict):
            mapped = dict(result[0])
            mapped["extra_kwargs"] = kwargs.get("extra_kwargs")
            summary = _request_shape(mapped)
            summary["stage"] = "anthropic_mapped"
            _stream_logger.warning(
                "TEMKI_ANTHROPIC_MAPPED %s",
                json.dumps(summary, ensure_ascii=True, sort_keys=True),
            )
            _write_raw_diagnostic("anthropic_mapped", mapped, summary)
        return result

    patched._temki_logs_request_shape = True
    LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs = staticmethod(
        patched
    )


def _anthropic_sse_event(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if not isinstance(payload, str):
        return None
    for line in payload.splitlines():
        if line.startswith("data:"):
            return _parse_sse_json_chunk(line)
    return None


def _anthropic_error_sse(error: _ChatGPTUpstreamResponseError) -> bytes:
    message = str(error)
    if error.code:
        message = f"[{error.code}] {message}"
    event = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": message,
        },
    }
    return f"event: error\ndata: {json.dumps(event, ensure_ascii=True)}\n\n".encode()


def _patch_anthropic_stream_instance_state() -> None:
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
    except (ImportError, AttributeError):
        return

    original = AnthropicStreamWrapper.__init__
    if getattr(original, "_temki_isolates_stream_state", False):
        return

    @wraps(original)
    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.sent_first_chunk = False
        self.sent_content_block_start = False
        self.sent_content_block_finish = False
        self.current_content_block_type = "text"
        self.sent_last_message = False
        self.holding_chunk = None
        self.holding_stop_reason_chunk = None
        self.queued_usage_chunk = False
        self.current_content_block_index = 0
        self.current_content_block_start = AnthropicStreamWrapper.TextBlock(
            type="text", text=""
        )
        self.chunk_queue = deque()

    patched._temki_isolates_stream_state = True
    AnthropicStreamWrapper.__init__ = patched


def _patch_anthropic_sse_output() -> None:
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
    except (ImportError, AttributeError):
        return

    original = AnthropicStreamWrapper.async_anthropic_sse_wrapper
    if getattr(original, "_temki_logs_sse_summary", False):
        return

    @wraps(original)
    async def patched(self):
        model = getattr(self, "model", None)
        diagnostic = _stream_diagnostics_enabled() and _diagnostic_model(model)
        event_counts: dict[str, int] = {}
        event_sequence: list[str] = []
        content_block_types: list[str] = []
        text_chars = 0
        thinking_chars = 0
        tool_json_chars = 0
        total_bytes = 0
        yield_count = 0
        stop_reason = None
        message_id = None
        initial_usage = None
        usage = None
        ended_normally = False
        delivered_error = False
        exception_type = None
        try:
            async for payload in original(self):
                _capture_raw_stream_chunk(self, "anthropic_client", payload)
                yield_count += 1
                if isinstance(payload, bytes):
                    total_bytes += len(payload)
                elif isinstance(payload, str):
                    total_bytes += len(payload.encode("utf-8"))
                event = _anthropic_sse_event(payload)
                if event is not None:
                    event_type = str(event.get("type", "unknown"))
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                    if len(event_sequence) < 128:
                        event_sequence.append(event_type)
                    if event_type == "message_start":
                        message = event.get("message")
                        if isinstance(message, dict):
                            message_id = message.get("id")
                            if isinstance(message.get("usage"), dict):
                                initial_usage = message["usage"]
                    elif event_type == "content_block_start":
                        block = event.get("content_block")
                        if isinstance(block, dict):
                            content_block_types.append(str(block.get("type", "unknown")))
                    elif event_type == "content_block_delta":
                        delta = event.get("delta")
                        if isinstance(delta, dict):
                            delta_type = delta.get("type")
                            if delta_type == "text_delta" and isinstance(delta.get("text"), str):
                                text_chars += len(delta["text"])
                            elif delta_type == "thinking_delta" and isinstance(
                                delta.get("thinking"), str
                            ):
                                thinking_chars += len(delta["thinking"])
                            elif delta_type == "input_json_delta" and isinstance(
                                delta.get("partial_json"), str
                            ):
                                tool_json_chars += len(delta["partial_json"])
                    elif event_type == "message_delta":
                        delta = event.get("delta")
                        if isinstance(delta, dict):
                            stop_reason = delta.get("stop_reason")
                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]
                yield payload
            ended_normally = True
        except _ChatGPTUpstreamResponseError as exc:
            exception_type = type(exc).__name__
            delivered_error = True
            payload = _anthropic_error_sse(exc)
            _capture_raw_stream_chunk(self, "anthropic_client", payload)
            total_bytes += len(payload)
            yield_count += 1
            event_counts["error"] = event_counts.get("error", 0) + 1
            if len(event_sequence) < 128:
                event_sequence.append("error")
            yield payload
        except BaseException as exc:
            exception_type = type(exc).__name__
            raise
        finally:
            if diagnostic:
                summary = {
                    "model": model,
                    "event_counts": event_counts,
                    "event_sequence": event_sequence,
                    "content_block_types": content_block_types,
                    "text_chars": text_chars,
                    "thinking_chars": thinking_chars,
                    "tool_json_chars": tool_json_chars,
                    "total_bytes": total_bytes,
                    "yield_count": yield_count,
                    "message_id": message_id,
                    "stop_reason": stop_reason,
                    "initial_usage": initial_usage,
                    "usage": usage,
                    "ended_normally": ended_normally,
                    "delivered_error": delivered_error,
                    "exception_type": exception_type,
                }
                summary.update(_iterator_correlation(self))
                _stream_logger.warning(
                    "TEMKI_ANTHROPIC_OUTPUT %s",
                    json.dumps(summary, ensure_ascii=True, sort_keys=True),
                )
                _write_raw_diagnostic(
                    "anthropic_client_sse",
                    {
                        "chunks": getattr(
                            self, "_temki_anthropic_client_raw_chunks", []
                        ),
                        "bytes": getattr(
                            self, "_temki_anthropic_client_raw_bytes", 0
                        ),
                        "truncated": bool(
                            getattr(
                                self,
                                "_temki_anthropic_client_raw_truncated",
                                False,
                            )
                        ),
                    },
                    summary,
                )

    patched._temki_logs_sse_summary = True
    AnthropicStreamWrapper.async_anthropic_sse_wrapper = patched


def _patch_chatgpt_account_propagation() -> None:
    """Carry the selected deployment's ChatGPT account across the Anthropic bridge.

    The chatgpt_auth_file backport resolves credentials from litellm_params,
    which works for /v1/responses. The Anthropic /v1/messages bridge calls
    get_llm_provider without litellm_params, so ChatGPTConfig falls back to the
    process-wide CHATGPT_TOKEN_DIR and every account collapses onto the first
    one — measured 2026-08-12: a request pinned to the second account spent the
    first account's quota and answered with its usage limit. Stashing the
    account on a ContextVar at deployment-selection time gives every later
    resolver the right auth.json, whatever path it takes.
    """
    try:
        from litellm.llms.chatgpt import authenticator as chatgpt_authenticator
        from litellm.llms.chatgpt.chat import transformation as chatgpt_chat
        from litellm.llms.chatgpt.responses import transformation as chatgpt_responses
        from litellm.router import Router
    except (ImportError, AttributeError):
        return

    original_resolver = getattr(chatgpt_authenticator, "get_chatgpt_auth_file", None)
    if original_resolver is None or getattr(original_resolver, "_temki_reads_context", False):
        return

    @wraps(original_resolver)
    def resolver(litellm_params: Any) -> str | None:
        return original_resolver(litellm_params) or _chatgpt_auth_file_ctx.get()

    resolver._temki_reads_context = True
    # The transformation modules imported the symbol by value, so each binding
    # has to be replaced separately.
    for module in (chatgpt_authenticator, chatgpt_chat, chatgpt_responses):
        setattr(module, "get_chatgpt_auth_file", resolver)

    original_select = Router.async_get_available_deployment
    if getattr(original_select, "_temki_sets_chatgpt_account", False):
        return

    @wraps(original_select)
    async def patched(self, *args, **kwargs):
        deployment = await original_select(self, *args, **kwargs)
        auth_file = None
        if isinstance(deployment, dict):
            params = deployment.get("litellm_params")
            if isinstance(params, dict):
                candidate = params.get("chatgpt_auth_file")
                if isinstance(candidate, str) and candidate:
                    auth_file = candidate
        # Reset on every selection, so a later request in the same task never
        # inherits the previous deployment's account.
        _chatgpt_auth_file_ctx.set(auth_file)
        return deployment

    patched._temki_sets_chatgpt_account = True
    Router.async_get_available_deployment = patched


def _patch_chatgpt_upstream_error_logging() -> None:
    try:
        from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
        from litellm.llms.chatgpt.responses.transformation import (
            ChatGPTResponsesAPIConfig,
        )
    except (ImportError, AttributeError):
        return

    for config_class in (ChatGPTResponsesAPIConfig, ChatGPTConfig):
        original_error_class = getattr(config_class, "get_error_class", None)
        if original_error_class is None or getattr(
            original_error_class, "_temki_logs_upstream_errors", False
        ):
            continue

        @wraps(original_error_class)
        def error_class_patched(
            self, error_message, status_code, headers, _original=original_error_class
        ):
            _log_upstream_error_text("chatgpt", status_code, error_message, headers)
            return _original(self, error_message, status_code, headers)

        error_class_patched._temki_logs_upstream_errors = True
        config_class.get_error_class = error_class_patched

    responses_original = getattr(
        ChatGPTResponsesAPIConfig, "transform_response_api_response", None
    )
    if responses_original is not None and not getattr(
        responses_original, "_temki_logs_upstream_errors", False
    ):
        @wraps(responses_original)
        def responses_patched(self, model, raw_response, logging_obj):
            _log_upstream_http_error("chatgpt", model, raw_response)
            return responses_original(self, model, raw_response, logging_obj)

        responses_patched._temki_logs_upstream_errors = True
        ChatGPTResponsesAPIConfig.transform_response_api_response = responses_patched

    chat_original = getattr(ChatGPTConfig, "transform_response", None)
    if chat_original is None or getattr(chat_original, "_temki_logs_upstream_errors", False):
        return

    @wraps(chat_original)
    def chat_patched(self, *args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else "")
        raw_response = kwargs.get("raw_response") or (args[1] if len(args) > 1 else None)
        if raw_response is not None:
            _log_upstream_http_error("chatgpt", model, raw_response)
        return chat_original(self, *args, **kwargs)

    chat_patched._temki_logs_upstream_errors = True
    ChatGPTConfig.transform_response = chat_patched


def _install_chatgpt_responses_recovery() -> None:
    # Backport LiteLLM #26219/#32724. All patches become no-ops when upstream
    # already returns an authoritative non-empty response.output.
    _patch_chatgpt_upstream_error_logging()
    _patch_chatgpt_account_propagation()
    _patch_chatgpt_non_stream_responses()
    _patch_responses_completion_bridge()
    _patch_chatgpt_streaming_responses()
    _patch_chatgpt_stream_lifecycle()
    _patch_chatgpt_outgoing_request()
    _patch_responses_to_chat_stream()
    _patch_mimo_default_max_tokens()
    _patch_anthropic_request_adapter()
    _patch_anthropic_stream_instance_state()
    _patch_anthropic_sse_output()


def _drop_empty_text_blocks(data: dict[str, Any]) -> None:
    """Strip text blocks Anthropic refuses to accept back.

    Switching a Claude Code session from a gpt-* alias to a claude-* one replays
    the whole history, and the assistant turns produced through the ChatGPT
    Responses bridge can carry a text block with an empty string — a
    tool-call-only or reasoning-only turn leaves nothing behind after conversion.
    Anthropic rejects the entire request with "messages: text content blocks must
    be non-empty", so a working session dies the moment the model is switched.

    Empty blocks carry no information, so dropping them is lossless. A message
    left with no blocks at all is dropped too: an empty content array is refused
    just the same.
    """
    messages = data.get("messages")
    if not isinstance(messages, list):
        return

    kept_messages = []
    for message in messages:
        if not isinstance(message, dict):
            kept_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            # A plain string body is the other legal shape; only an empty one is
            # a problem, and there is nothing to salvage inside it.
            if isinstance(content, str) and not content.strip():
                continue
            kept_messages.append(message)
            continue

        kept_blocks = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") == "text"
                and not str(block.get("text") or "").strip()
            )
        ]
        if not kept_blocks:
            continue
        if len(kept_blocks) != len(content):
            message = {**message, "content": kept_blocks}
        kept_messages.append(message)

    if kept_messages != messages:
        data["messages"] = kept_messages


class TemkiLiteLLMPolicy(CustomLogger):
    def __init__(self) -> None:
        _install_chatgpt_responses_recovery()
        self.state_path = Path(
            os.getenv("TEMKI_LITELLM_POLICY_STATE", "/var/lib/litellm/temki-policy-ledger.jsonl")
        )
        self.timezone = ZoneInfo(os.getenv("TEMKI_LITELLM_TIMEZONE", "Europe/Moscow"))
        self.expensive_start = self._parse_clock(
            os.getenv("TEMKI_LITELLM_EXPENSIVE_HOURS_START", "16:00")
        )
        self.expensive_end = self._parse_clock(os.getenv("TEMKI_LITELLM_EXPENSIVE_HOURS_END", "22:00"))

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type: str):
        context = self._request_context(data, user_api_key_dict)
        self._log_stream_request_shape(data, context, call_type, "received")
        if context["is_chatgpt"]:
            self._normalize_chatgpt_instructions(data)
            self._log_stream_request_shape(data, context, call_type, "normalized")

        if not context["is_claude"]:
            return data

        _drop_empty_text_blocks(data)

        blocked = self._blocked_reason(context)
        if blocked is not None:
            from fastapi import HTTPException

            raise HTTPException(status_code=429, detail=blocked)

        metadata = data.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["temki_litellm_policy"] = {
                "device": context["device"],
                "model": context["model"],
                "effective_multiplier": context["multiplier"],
                "expensive_hours": context["expensive_hours"],
                "long_context": context["long_context"],
            }
        return data

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict, response):
        context = self._request_context(data, user_api_key_dict)
        if not context["is_claude"]:
            return response

        warnings = []
        if context["expensive_hours"]:
            warnings.append(
                "Claude expensive-hours window is active; accounting multiplier "
                f"{self._float_env('TEMKI_LITELLM_EXPENSIVE_HOURS_MULTIPLIER', 5.0):g}x"
            )
        if context["long_context"]:
            warnings.append(
                "Claude 1M context detected; accounting multiplier "
                f"{self._float_env('TEMKI_LITELLM_LONG_CONTEXT_MULTIPLIER', 2.0):g}x"
            )

        if warnings:
            self._add_headers(
                response,
                {
                    "x-temki-litellm-warning": " | ".join(warnings),
                    "x-temki-litellm-device": context["device"],
                },
            )
        return response

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        data = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "user": kwargs.get("user"),
            "metadata": self._get(self._get(kwargs, "litellm_params", {}), "metadata", {}),
        }
        context = self._request_context(data, self._get(data, "metadata", {}))
        if not context["is_claude"]:
            return

        usage = self._usage(response_obj)
        total_tokens = int(usage.get("total_tokens") or 0)
        if total_tokens <= 0:
            total_tokens = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)

        now = datetime.now(self.timezone)
        self._append_record(
            {
                "ts": now.isoformat(),
                "device": context["device"],
                "user": context["user"],
                "model": context["model"],
                "model_class": context["model_class"],
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": total_tokens,
                "effective_units": int(round(total_tokens * context["multiplier"])),
                "multiplier": context["multiplier"],
                "expensive_hours": context["expensive_hours"],
                "long_context": context["long_context"],
            }
        )

    def _blocked_reason(self, context: dict[str, Any]) -> dict[str, Any] | None:
        if context["expensive_hours"]:
            peak_policy = os.getenv("TEMKI_LITELLM_EXPENSIVE_HOURS_POLICY", "account").lower()
            if peak_policy == "block":
                return {
                    "error": "temki_policy_expensive_hours",
                    "message": "Claude expensive-hours window is active; request blocked by policy.",
                    "device": context["device"],
                }
            if peak_policy == "block-non-sonnet" and not context["is_sonnet"]:
                return {
                    "error": "temki_policy_expensive_hours_non_sonnet",
                    "message": "Claude expensive-hours window is active; non-Sonnet request blocked.",
                    "device": context["device"],
                }

        if (
            context["long_context"]
            and context["expensive_hours"]
            and self._bool_env("TEMKI_LITELLM_BLOCK_1M_DURING_EXPENSIVE", False)
        ):
            return {
                "error": "temki_policy_1m_expensive_hours",
                "message": "Claude 1M context during expensive hours is blocked by policy.",
                "device": context["device"],
            }

        return None

    def _request_context(self, data: dict[str, Any], user_api_key_dict: Any) -> dict[str, Any]:
        model = self._extract_model(data)
        headers = self._extract_headers(data)
        model_lc = model.lower()
        is_claude = "claude" in model_lc or "anthropic" in model_lc or model_lc in {"sonnet", "opus"}
        is_sonnet = "sonnet" in model_lc or model_lc in {"anthropic-claude", "claude-code"}
        is_chatgpt = self._is_chatgpt_model(model_lc, data)
        long_context = self._detect_long_context(model_lc, headers)
        expensive = self._in_expensive_hours(datetime.now(self.timezone))

        multiplier = 1.0
        peak_policy = os.getenv("TEMKI_LITELLM_EXPENSIVE_HOURS_POLICY", "account").lower()
        if expensive and peak_policy in {"account", "block", "block-non-sonnet"}:
            multiplier *= self._float_env("TEMKI_LITELLM_EXPENSIVE_HOURS_MULTIPLIER", 5.0)
        if long_context:
            multiplier *= self._float_env("TEMKI_LITELLM_LONG_CONTEXT_MULTIPLIER", 2.0)

        return {
            "model": model,
            "model_class": "sonnet" if is_sonnet else "claude-other" if is_claude else "other",
            "is_claude": is_claude,
            "is_sonnet": is_sonnet,
            "is_chatgpt": is_chatgpt,
            "long_context": long_context,
            "expensive_hours": expensive,
            "multiplier": multiplier,
            "device": self._device_name(data, user_api_key_dict),
            "user": str(self._get(data, "user", "") or self._get(user_api_key_dict, "user_id", "") or ""),
        }

    def _extract_model(self, data: dict[str, Any]) -> str:
        for value in (
            self._get(data, "model"),
            self._get(self._get(data, "litellm_params", {}), "model"),
            self._get(self._get(data, "metadata", {}), "model"),
        ):
            if value:
                return str(value)
        return ""

    def _device_name(self, data: dict[str, Any], user_api_key_dict: Any) -> str:
        candidates = [
            self._get(data, "metadata", {}),
            self._get(user_api_key_dict, "metadata", {}),
            user_api_key_dict,
        ]
        for candidate in candidates:
            candidate = self._dict(candidate)
            spend_meta = self._dict(candidate.get("spend_logs_metadata", {}))
            for source in (candidate, spend_meta):
                for key in ("device", "device_id", "key_alias", "team_alias", "user_id"):
                    value = source.get(key)
                    if value:
                        return str(value)
        return "unknown-device"

    def _extract_headers(self, data: dict[str, Any]) -> dict[str, str]:
        headers: dict[str, str] = {}
        containers = [
            data,
            self._get(data, "metadata", {}),
            self._get(data, "litellm_params", {}),
            self._get(self._get(data, "litellm_params", {}), "metadata", {}),
            self._get(data, "optional_params", {}),
        ]
        for container in containers:
            for key in ("headers", "extra_headers", "request_headers"):
                value = self._dict(self._get(container, key, {}))
                headers.update({str(k).lower(): str(v) for k, v in value.items()})
        return headers

    def _detect_long_context(self, model_lc: str, headers: dict[str, str]) -> bool:
        if "1m" in model_lc or "[1m]" in model_lc:
            return True
        return any("context-1m" in value.lower() for value in headers.values())

    def _is_chatgpt_model(self, model_lc: str, data: dict[str, Any]) -> bool:
        litellm_params = self._get(data, "litellm_params", {})
        resolved_model = str(self._get(litellm_params, "model", "") or "").lower()
        custom_provider = str(
            self._get(litellm_params, "custom_llm_provider", "")
            or self._get(data, "custom_llm_provider", "")
            or ""
        ).lower()
        if custom_provider == "chatgpt" or resolved_model.startswith("chatgpt/"):
            return True
        if model_lc.startswith("chatgpt/"):
            return True
        # Every chatgptSpecs deployment carries chatgpt_auth_file, so resolve
        # membership through the live router instead of a hand-maintained
        # list; the literal set below stays as a fallback for tests and for
        # the rare moment the router is not importable.
        if self._has_chatgpt_auth_file(model_lc):
            return True
        return model_lc in {
            "codex",
            "codex-spark",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.6",
            "gpt-6",
            "gpt-6-astra",
            "astra",
            "astra-huanita",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-6-sol",
            "gpt-6-luna",
            "sol",
            "terra",
            "luna",
        }

    def _has_chatgpt_auth_file(self, model_name: str) -> bool:
        try:
            from litellm.proxy.proxy_server import llm_router
        except Exception:
            return False
        if llm_router is None:
            return False
        for deployment in getattr(llm_router, "model_list", None) or []:
            if not isinstance(deployment, dict) or deployment.get("model_name") != model_name:
                continue
            params = deployment.get("litellm_params") or {}
            return bool(params.get("chatgpt_auth_file"))
        return False

    def _normalize_chatgpt_instructions(self, data: dict[str, Any]) -> None:
        top_level = []
        for key in ("system", "developer", "instructions"):
            value = data.pop(key, None)
            text = self._content_to_text(value)
            if text:
                top_level.append(text)

        messages = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = self._normalize_message_list(messages, top_level)
            return

        input_value = data.get("input")
        if isinstance(input_value, list):
            data["input"] = self._normalize_input_list(input_value, top_level)
            return
        if isinstance(input_value, str) and top_level:
            data["input"] = [
                self._assistant_input_item(top_level),
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_value}],
                },
            ]
            return

        if top_level:
            data["messages"] = [{"role": "assistant", "content": self._instruction_preamble(top_level)}]

    def _log_stream_request_shape(
        self,
        data: dict[str, Any],
        context: dict[str, Any],
        call_type: str,
        stage: str,
    ) -> None:
        diagnostic_alias = os.getenv("TEMKI_LITELLM_DIAGNOSTIC_KEY_ALIAS", "")
        if (
            not _stream_diagnostics_enabled()
            or not diagnostic_alias
            or context["device"] != diagnostic_alias
        ):
            return

        summary = _request_shape(data)
        summary.update(
            {
                "alias": diagnostic_alias,
                "model": context["model"],
                "call_type": call_type,
                "stage": stage,
            }
        )
        _stream_logger.warning(
            "TEMKI_STREAM_REQUEST %s",
            json.dumps(summary, ensure_ascii=True, sort_keys=True),
        )
        _write_raw_diagnostic(f"policy_{stage}", data, summary)

    def _normalize_message_list(self, messages: list[Any], top_level: list[str]) -> list[Any]:
        instructions = list(top_level)
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            role = str(message.get("role", "")).lower()
            if role in {"system", "developer"}:
                text = self._content_to_text(message.get("content"))
                if text:
                    instructions.append(text)
                continue
            normalized.append(message)
        if instructions:
            normalized.insert(0, {"role": "assistant", "content": self._instruction_preamble(instructions)})
        return normalized

    def _normalize_input_list(self, input_items: list[Any], top_level: list[str]) -> list[Any]:
        instructions = list(top_level)
        normalized = []
        for item in input_items:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            role = str(item.get("role", "")).lower()
            # Fold ONLY plain instruction messages into the assistant preamble. Codex CLI
            # >= 0.144 (TUI, multi-agent v2) registers its whole toolset as a STRUCTURAL
            # input item {"type": "additional_tools", "role": "developer", "tools": [...]}
            # with no content; swallowing it here strips every tool from the session and
            # the agent reports "no terminal/exec tools" (incident 2026-07-16).
            item_type = str(item.get("type") or "message").lower()
            if role in {"system", "developer"} and item_type == "message":
                text = self._content_to_text(item.get("content"))
                if text:
                    instructions.append(text)
                continue
            normalized.append(item)
        if instructions:
            normalized.insert(0, self._assistant_input_item(instructions))
        return normalized

    def _assistant_input_item(self, instructions: list[str]) -> dict[str, Any]:
        # type="message" is what _trim_chatgpt_subscription_request's prefix guard
        # keys on; without it a context trim dropped the production preamble while
        # the test fixture (which added the type by hand) kept passing
        # (incident 2026-07-17).
        return {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": self._instruction_preamble(instructions),
                }
            ],
        }

    def _instruction_preamble(self, instructions: list[str]) -> str:
        return "\n\n".join(instructions)

    def _content_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"].strip()
            if isinstance(value.get("content"), (str, list, dict)):
                return self._content_to_text(value.get("content"))
            return ""
        if isinstance(value, list):
            parts = []
            for item in value:
                text = self._content_to_text(item)
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        return str(value).strip()

    def _usage(self, response_obj: Any) -> dict[str, Any]:
        usage = self._get(response_obj, "usage", {}) or {}
        return self._dict(usage)

    def _append_record(self, record: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

    def _in_expensive_hours(self, now: datetime) -> bool:
        current = now.time()
        if self.expensive_start <= self.expensive_end:
            return self.expensive_start <= current < self.expensive_end
        return current >= self.expensive_start or current < self.expensive_end

    def _add_headers(self, response: Any, headers: dict[str, str]) -> None:
        hidden = getattr(response, "_hidden_params", None)
        if not isinstance(hidden, dict):
            hidden = {}
            try:
                response._hidden_params = hidden
            except Exception:
                return
        additional = hidden.get("additional_headers") or {}
        additional.update(headers)
        hidden["additional_headers"] = additional

    def _parse_clock(self, value: str) -> datetime_time:
        hour, minute = value.split(":", 1)
        return datetime_time(hour=int(hour), minute=int(minute))

    def _char_count(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        if isinstance(value, dict):
            return sum(self._char_count(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._char_count(v) for v in value)
        return 0

    def _get(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        if value is None:
            return {}
        try:
            return dict(value)
        except Exception:
            return {}

    def _int_env(self, name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    def _float_env(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    def _bool_env(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}


policy_callback = TemkiLiteLLMPolicy()
