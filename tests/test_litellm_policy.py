import importlib.util
import json
import copy
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_policy_module():
    litellm = types.ModuleType("litellm")
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = object
    litellm.integrations = integrations
    integrations.custom_logger = custom_logger

    module_names = ("litellm", "litellm.integrations", "litellm.integrations.custom_logger")
    previous = {name: sys.modules.get(name) for name in module_names}
    sys.modules.update(
        {
            "litellm": litellm,
            "litellm.integrations": integrations,
            "litellm.integrations.custom_logger": custom_logger,
        }
    )
    try:
        path = Path(__file__).with_name("litellm-policy.py")
        spec = importlib.util.spec_from_file_location("temki_litellm_policy_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


policy_module = _load_policy_module()


class InstructionNormalizationTests(unittest.TestCase):
    def test_chat_roles_become_one_assistant_preamble(self):
        data = {
            "system": "top level",
            "messages": [
                {"role": "developer", "content": "developer rule"},
                {"role": "user", "content": "question"},
            ],
        }

        policy_module.policy_callback._normalize_chatgpt_instructions(data)

        self.assertNotIn("system", data)
        self.assertEqual(
            data["messages"],
            [
                {"role": "assistant", "content": "top level\n\ndeveloper rule"},
                {"role": "user", "content": "question"},
            ],
        )

    def test_responses_role_uses_output_text(self):
        data = {
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "system rule"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
            ]
        }

        policy_module.policy_callback._normalize_chatgpt_instructions(data)

        self.assertEqual(data["input"][0]["role"], "assistant")
        self.assertEqual(data["input"][0]["content"][0]["type"], "output_text")
        self.assertEqual(data["input"][0]["content"][0]["text"], "system rule")

    def test_responses_structural_developer_items_pass_through(self):
        # Codex CLI >= 0.144 (TUI, multi-agent v2) registers its toolset as a structural
        # input item {"type": "additional_tools", "role": "developer", "tools": [...]} —
        # not as message content. Folding/dropping it strips every tool from the session
        # (incident 2026-07-16: agent reported "no terminal/exec tools").
        tools_item = {
            "type": "additional_tools",
            "role": "developer",
            "tools": [{"type": "custom", "name": "exec", "description": "run commands"}],
        }
        data = {
            "input": [
                tools_item,
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer rule"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}],
                },
            ]
        }

        policy_module.policy_callback._normalize_chatgpt_instructions(data)

        # preamble first, structural tools item preserved verbatim, user message intact
        self.assertEqual(data["input"][0]["role"], "assistant")
        self.assertEqual(data["input"][0]["content"][0]["text"], "developer rule")
        self.assertEqual(data["input"][1], tools_item)
        self.assertEqual(data["input"][2]["role"], "user")

    def test_string_input_is_expanded_when_instructions_exist(self):
        data = {"instructions": "answer exactly", "input": "question"}

        policy_module.policy_callback._normalize_chatgpt_instructions(data)

        self.assertNotIn("instructions", data)
        self.assertEqual(data["input"][0]["role"], "assistant")
        self.assertEqual(data["input"][1]["role"], "user")
        self.assertEqual(data["input"][1]["content"][0]["text"], "question")


class AstraClaudeCodeNormalizationTests(unittest.IsolatedAsyncioTestCase):
    async def _run_hook(self, model, data):
        payload = {"model": model, **copy.deepcopy(data)}
        return await policy_module.policy_callback.async_pre_call_hook(
            SimpleNamespace(), None, payload, "anthropic_messages"
        )

    async def test_astra_aliases_normalize_anthropic_system_blocks(self):
        request = {
            "system": [
                {"type": "text", "text": "first rule"},
                {"type": "text", "text": "second rule"},
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
            ],
        }

        for model in ("astra", "gpt-6", "gpt-6-astra", "astra-huanita"):
            with self.subTest(model=model):
                normalized = await self._run_hook(model, request)
                self.assertNotIn("system", normalized)
                self.assertEqual(normalized["messages"][0]["role"], "assistant")
                self.assertEqual(normalized["messages"][0]["content"], "first rule\nsecond rule")
                self.assertEqual(normalized["messages"][1], request["messages"][0])

    async def test_astra_normalization_is_idempotent(self):
        normalized = await self._run_hook(
            "astra",
            {
                "system": "system rule",
                "messages": [{"role": "user", "content": "question"}],
            },
        )

        normalized = await policy_module.policy_callback.async_pre_call_hook(
            SimpleNamespace(), None, normalized, "anthropic_messages"
        )

        preambles = [
            message
            for message in normalized["messages"]
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and message.get("content") == "system rule"
        ]
        self.assertEqual(len(preambles), 1)

    async def test_future_alias_uses_resolved_chatgpt_provider(self):
        normalized = await self._run_hook(
            "future-alias",
            {
                "litellm_params": {"model": "chatgpt/future-model"},
                "system": "system rule",
                "messages": [{"role": "user", "content": "question"}],
            },
        )

        self.assertNotIn("system", normalized)
        self.assertEqual(normalized["messages"][0]["role"], "assistant")
        self.assertEqual(normalized["messages"][0]["content"], "system rule")

    async def test_future_alias_uses_custom_chatgpt_provider(self):
        normalized = await self._run_hook(
            "future-alias",
            {
                "litellm_params": {"custom_llm_provider": "chatgpt"},
                "system": "system rule",
                "messages": [{"role": "user", "content": "question"}],
            },
        )

        self.assertNotIn("system", normalized)
        self.assertEqual(normalized["messages"][0]["role"], "assistant")

    async def test_native_claude_keeps_top_level_system(self):
        request = {
            "system": "system rule",
            "messages": [{"role": "user", "content": "question"}],
        }

        unchanged = await self._run_hook("claude-opus-5", request)

        self.assertEqual(unchanged["system"], "system rule")
        self.assertEqual(unchanged["messages"], request["messages"])


class EmptyTextBlockTests(unittest.TestCase):
    def test_empty_text_blocks_are_dropped_from_history(self):
        # Shape a Claude Code session takes after switching away from a gpt-*
        # alias: the converted assistant turn carries an empty text block, and
        # Anthropic refuses the whole request because of it.
        data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "   "}]},
            ]
        }

        policy_module._drop_empty_text_blocks(data)

        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0]["content"][0]["text"], "hi")
        self.assertEqual(
            [block["type"] for block in data["messages"][1]["content"]], ["tool_use"]
        )

    def test_untouched_when_every_block_carries_text(self):
        data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "plain string body"},
            ]
        }
        before = copy.deepcopy(data)

        policy_module._drop_empty_text_blocks(data)

        self.assertEqual(data, before)


class QuotaPolicyTests(unittest.TestCase):
    def test_legacy_weekly_budget_environment_does_not_block_requests(self):
        context = {
            "device": "test",
            "expensive_hours": False,
            "is_sonnet": False,
            "long_context": False,
        }
        environment = {
            "TEMKI_LITELLM_WEEKLY_TOTAL_UNITS": "1",
            "TEMKI_LITELLM_SONNET_RESERVE_UNITS": "1",
            "TEMKI_LITELLM_SONNET_WEEKLY_UNITS": "1",
        }

        with patch.dict(os.environ, environment, clear=False):
            blocked = policy_module.policy_callback._blocked_reason(context)

        self.assertIsNone(blocked)


class ResponsesRecoveryTests(unittest.TestCase):
    def test_request_shape_contains_counts_without_content(self):
        shape = policy_module._request_shape(
            {
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "secret"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "call", "input": {}}],
                    },
                ],
                "tools": [{"name": "tool", "description": "secret tool"}],
                "stream": True,
            }
        )

        self.assertEqual(shape["message_count"], 2)
        self.assertEqual(shape["block_counts"], {"text": 1, "tool_use": 1})
        self.assertEqual(shape["tool_count"], 1)
        self.assertNotIn("secret", json.dumps(shape))

    def test_request_shape_captures_system_cache_and_fingerprints(self):
        shape = policy_module._request_shape(
            {
                "model": "gpt-5.6-sol",
                "system": [
                    {
                        "type": "text",
                        "text": "private system prompt",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": "private question"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "private description",
                        "input_schema": {"type": "object"},
                    }
                ],
            }
        )

        self.assertEqual(shape["system"]["cache_control_count"], 1)
        self.assertEqual(shape["message_roles_tail"], ["user"])
        self.assertEqual(shape["tool_name_lengths"], [6])
        self.assertEqual(len(shape["request_fingerprint"]), 20)
        self.assertNotIn("private", json.dumps(shape))

    def test_raw_diagnostic_keeps_content_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stream-debug.jsonl"
            environment = {
                "TEMKI_LITELLM_STREAM_DIAGNOSTICS": "true",
                "TEMKI_LITELLM_RAW_STREAM_DIAGNOSTICS": "true",
                "TEMKI_LITELLM_RAW_DIAGNOSTIC_PATH": str(output),
            }
            with patch.dict(os.environ, environment, clear=False):
                policy_module._write_raw_diagnostic(
                    "test",
                    {
                        "Authorization": "Bearer credential",
                        "messages": [{"role": "user", "content": "full prompt"}],
                    },
                )

            record = json.loads(output.read_text())
            self.assertEqual(record["payload"]["Authorization"], "[REDACTED]")
            self.assertEqual(
                record["payload"]["messages"][0]["content"], "full prompt"
            )

    def test_context_trim_starts_at_user_boundary_and_keeps_preamble(self):
        preamble = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "system preamble"}],
        }
        current_user = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "current request"}],
        }
        request = {
            "model": "gpt-5.6-sol",
            "input": [
                preamble,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old request" * 20}],
                },
                {
                    "type": "function_call",
                    "call_id": "old_call",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "old_call",
                    "output": "old output" * 20,
                },
                current_user,
                {
                    "type": "function_call",
                    "call_id": "current_call",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "current_call",
                    "output": "current output",
                },
            ],
            "tools": [{"type": "function", "name": "lookup"}],
        }

        def fake_token_count(value):
            return len(json.dumps(value, separators=(",", ":"))), "test_counter"

        expected = dict(request)
        expected["input"] = [preamble, *request["input"][4:]]
        budget = fake_token_count(expected)[0]
        with patch.object(
            policy_module, "_chatgpt_serialized_token_count", fake_token_count
        ):
            with patch.dict(
                os.environ,
                {"TEMKI_CHATGPT_SUBSCRIPTION_INPUT_TOKEN_BUDGET": str(budget)},
            ):
                trimmed, summary = policy_module._trim_chatgpt_subscription_request(
                    request
                )

        self.assertTrue(summary["context_trimmed"])
        self.assertEqual(trimmed["input"], expected["input"])
        self.assertEqual(trimmed["input"][0], preamble)
        self.assertEqual(trimmed["input"][1], current_user)
        self.assertNotIn("old_call", json.dumps(trimmed))
        self.assertIn("current_call", json.dumps(trimmed))

    def test_context_trim_keeps_production_preamble_and_additional_tools(self):
        # REAL production items — not fixture-embellished. The earlier fixture
        # hand-added type="message" that _assistant_input_item did not emit, so
        # tests passed while the production preamble (and the Codex >= 0.144
        # additional_tools registration) was dropped on trim (incident 2026-07-17).
        preamble = policy_module.policy_callback._assistant_input_item(
            ["system rule"]
        )
        tools_item = {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {"type": "custom", "name": "exec", "description": "run commands"}
            ],
        }
        current_user = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "current request"}],
        }
        request = {
            "model": "gpt-5.6-sol",
            "input": [
                preamble,
                tools_item,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old request" * 20}],
                },
                {
                    "type": "function_call",
                    "call_id": "old_call",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "old_call",
                    "output": "old output" * 40,
                },
                current_user,
            ],
        }

        def fake_token_count(value):
            return len(json.dumps(value, separators=(",", ":"))), "test_counter"

        expected_input = [preamble, tools_item, current_user]
        expected = dict(request)
        expected["input"] = expected_input
        budget = fake_token_count(expected)[0]
        with patch.object(
            policy_module, "_chatgpt_serialized_token_count", fake_token_count
        ):
            with patch.dict(
                os.environ,
                {"TEMKI_CHATGPT_SUBSCRIPTION_INPUT_TOKEN_BUDGET": str(budget)},
            ):
                trimmed, summary = policy_module._trim_chatgpt_subscription_request(
                    request
                )

        self.assertEqual(preamble.get("type"), "message")
        self.assertTrue(summary["context_trimmed"])
        self.assertEqual(summary["retained_control_items"], 1)
        self.assertEqual(trimmed["input"], expected_input)
        self.assertNotIn("old_call", json.dumps(trimmed))

    def test_context_trim_keeps_typeless_assistant_preamble(self):
        # Requests already in flight (or rolled-out sessions replaying history)
        # still carry the pre-fix typeless preamble; the prefix guard must accept
        # both shapes.
        preamble = {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "rules"}],
        }
        current_user = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "current request"}],
        }
        request = {
            "model": "gpt-5.6-sol",
            "input": [
                preamble,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old request" * 20}],
                },
                current_user,
            ],
        }

        def fake_token_count(value):
            return len(json.dumps(value, separators=(",", ":"))), "test_counter"

        expected = dict(request)
        expected["input"] = [preamble, current_user]
        budget = fake_token_count(expected)[0]
        with patch.object(
            policy_module, "_chatgpt_serialized_token_count", fake_token_count
        ):
            with patch.dict(
                os.environ,
                {"TEMKI_CHATGPT_SUBSCRIPTION_INPUT_TOKEN_BUDGET": str(budget)},
            ):
                trimmed, summary = policy_module._trim_chatgpt_subscription_request(
                    request
                )

        self.assertTrue(summary["context_trimmed"])
        self.assertEqual(trimmed["input"], [preamble, current_user])

    def test_token_count_discounts_encrypted_reasoning(self):
        # Full-JSON tiktoken counting billed opaque encrypted reasoning at text
        # rate (417,712 estimated vs ~199,939 actual, incident 2026-07-17) and
        # tripped the trimmer ~150k tokens early. The structural counter charges
        # such payloads at len//4.
        base = {
            "model": "gpt-5.6-sol",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello world"}],
                },
            ],
        }
        blob = "A" * 48000
        with_blob = json.loads(json.dumps(base))
        with_blob["input"].append(
            {"type": "reasoning", "encrypted_content": blob, "summary": []}
        )

        base_tokens, counter = policy_module._chatgpt_serialized_token_count(base)
        blob_tokens, _ = policy_module._chatgpt_serialized_token_count(with_blob)

        contribution = blob_tokens - base_tokens
        self.assertLessEqual(contribution, len(blob) // 4 + 32)
        self.assertGreaterEqual(contribution, len(blob) // 8)
        self.assertIn("structural", counter)

    def test_upstream_http_error_logs_full_body_without_credentials(self):
        class FakeResponse:
            status_code = 429
            headers = {
                "content-type": "application/json",
                "retry-after": "30",
                "authorization": "Bearer should-not-be-logged",
            }
            text = '{"error":{"message":"quota window active","code":"rate_limit"}}'

            def json(self):
                return {"error": {"message": "quota window active", "code": "rate_limit"}}

        with self.assertLogs("temki_litellm_stream", level="WARNING") as logs:
            policy_module._log_upstream_http_error("chatgpt", "gpt-5.6-terra", FakeResponse())

        joined = "\\n".join(logs.output)
        self.assertIn("TEMKI_UPSTREAM_HTTP_ERROR", joined)
        self.assertIn("quota window active", joined)
        self.assertIn("rate_limit", joined)
        self.assertIn("retry-after", joined)
        self.assertNotIn("should-not-be-logged", joined)
        self.assertNotIn("authorization", joined.lower())

    def test_upstream_http_error_logs_non_json_body_verbatim(self):
        response = SimpleNamespace(
            status_code=502,
            headers={"content-type": "text/plain"},
            text="upstream gateway said try again",
        )

        with self.assertLogs("temki_litellm_stream", level="WARNING") as logs:
            policy_module._log_upstream_http_error("chatgpt", "gpt-5.6", response)

        self.assertIn("upstream gateway said try again", "\\n".join(logs.output))

    def test_upstream_http_error_ignores_success_response(self):
        response = SimpleNamespace(status_code=200, headers={}, text="ok")

        with patch.object(policy_module._stream_logger, "warning") as warning:
            policy_module._log_upstream_http_error("chatgpt", "gpt-5.6", response)

        warning.assert_not_called()

    def test_chatgpt_transform_hooks_capture_http_error(self):
        class FakeResponseConfig:
            def get_error_class(self, error_message, status_code, headers):
                return (error_message, status_code, headers)

            def transform_response_api_response(self, model, raw_response, logging_obj):
                return "responses-ok"

        class FakeChatConfig:
            def get_error_class(self, error_message, status_code, headers):
                return (error_message, status_code, headers)

            def transform_response(self, *args, **kwargs):
                return "chat-ok"

        chatgpt = types.ModuleType("litellm.llms.chatgpt")
        chat = types.ModuleType("litellm.llms.chatgpt.chat")
        chat_transformation = types.ModuleType("litellm.llms.chatgpt.chat.transformation")
        responses = types.ModuleType("litellm.llms.chatgpt.responses")
        responses_transformation = types.ModuleType(
            "litellm.llms.chatgpt.responses.transformation"
        )
        chat_transformation.ChatGPTConfig = FakeChatConfig
        responses_transformation.ChatGPTResponsesAPIConfig = FakeResponseConfig
        module_names = (
            "litellm.llms.chatgpt",
            "litellm.llms.chatgpt.chat",
            "litellm.llms.chatgpt.chat.transformation",
            "litellm.llms.chatgpt.responses",
            "litellm.llms.chatgpt.responses.transformation",
        )
        previous = {name: sys.modules.get(name) for name in module_names}
        sys.modules.update(
            {
                "litellm.llms.chatgpt": chatgpt,
                "litellm.llms.chatgpt.chat": chat,
                "litellm.llms.chatgpt.chat.transformation": chat_transformation,
                "litellm.llms.chatgpt.responses": responses,
                "litellm.llms.chatgpt.responses.transformation": responses_transformation,
            }
        )
        try:
            response = SimpleNamespace(
                status_code=429,
                headers={"content-type": "application/json"},
                text='{"error":{"message":"upstream rate limit"}}',
            )
            response.json = lambda: {"error": {"message": "upstream rate limit"}}
            policy_module._patch_chatgpt_upstream_error_logging()
            with self.assertLogs("temki_litellm_stream", level="WARNING") as logs:
                self.assertEqual(
                    FakeResponseConfig().get_error_class(
                        response.text, response.status_code, response.headers
                    )[0],
                    response.text,
                )
                self.assertEqual(
                    FakeChatConfig().get_error_class(
                        response.text, response.status_code, response.headers
                    )[0],
                    response.text,
                )
                self.assertEqual(
                    FakeResponseConfig().transform_response_api_response("gpt", response, None),
                    "responses-ok",
                )
                self.assertEqual(
                    FakeChatConfig().transform_response(
                        "gpt", response, None, None, None, None, None, None, None
                    ),
                    "chat-ok",
                )
            self.assertEqual("\\n".join(logs.output).count("TEMKI_UPSTREAM_HTTP_ERROR"), 4)
        finally:
            for name, old_module in previous.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

    def test_context_failure_becomes_anthropic_error_event(self):
        event = {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "code": "context_length_exceeded",
                    "message": "Input exceeds the context window",
                },
            },
        }

        error = policy_module._chatgpt_upstream_error(event)
        payload = policy_module._anthropic_error_sse(error).decode()
        parsed = policy_module._anthropic_sse_event(payload)

        self.assertEqual(parsed["type"], "error")
        self.assertEqual(parsed["error"]["type"], "invalid_request_error")
        self.assertIn("context_length_exceeded", parsed["error"]["message"])

    def test_empty_completed_event_is_logged_when_transform_returns_none(self):
        class FakeBaseResponsesAPIStreamingIterator:
            def _process_chunk(self, chunk):
                return None

        litellm = types.ModuleType("litellm")
        responses = types.ModuleType("litellm.responses")
        streaming_iterator = types.ModuleType("litellm.responses.streaming_iterator")
        streaming_iterator.BaseResponsesAPIStreamingIterator = (
            FakeBaseResponsesAPIStreamingIterator
        )
        litellm.responses = responses
        responses.streaming_iterator = streaming_iterator
        module_names = (
            "litellm",
            "litellm.responses",
            "litellm.responses.streaming_iterator",
        )
        previous = {name: sys.modules.get(name) for name in module_names}
        sys.modules.update(
            {
                "litellm": litellm,
                "litellm.responses": responses,
                "litellm.responses.streaming_iterator": streaming_iterator,
            }
        )
        try:
            policy_module._patch_chatgpt_streaming_responses()
            iterator = FakeBaseResponsesAPIStreamingIterator()
            iterator.custom_llm_provider = "chatgpt"
            iterator.model = "gpt-5.6-sol"
            iterator.request_data = {}
            iterator.litellm_metadata = {}
            iterator.logging_obj = SimpleNamespace(model_call_details={})
            iterator.response = SimpleNamespace(headers={}, status_code=200)
            iterator._stream_created_time = time.time()
            event = {
                "type": "response.completed",
                "response": {
                    "id": "response_1",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            }
            with patch.dict(
                os.environ, {"TEMKI_LITELLM_STREAM_DIAGNOSTICS": "true"}
            ):
                with self.assertLogs("temki_litellm_stream", level="WARNING") as logs:
                    result = iterator._process_chunk("data: " + json.dumps(event))

            self.assertIsNone(result)
            joined = "\n".join(logs.output)
            self.assertIn("TEMKI_STREAM_TERMINAL", joined)
            self.assertIn('"transformed_event_present": false', joined)
            self.assertTrue(iterator._temki_stream_terminal_seen)
        finally:
            for name, old_module in previous.items():
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

    def test_anthropic_sse_event_parser(self):
        event = policy_module._anthropic_sse_event(
            b'event: content_block_delta\ndata: {"type":"content_block_delta",'
            b'"delta":{"type":"text_delta","text":"OK"}}\n\n'
        )

        self.assertEqual(event["type"], "content_block_delta")

    def test_sse_done_marker_detection(self):
        self.assertTrue(policy_module._is_sse_done_chunk("data: [DONE]"))
        self.assertFalse(policy_module._is_sse_done_chunk('data: {"type":"response.completed"}'))

    def test_done_text_only_emits_suffix_missing_from_deltas(self):
        self.assertEqual(policy_module._missing_stream_text("Hello", "Hello world"), " world")
        self.assertEqual(policy_module._missing_stream_text("Hello world", "Hello world"), "")

    def test_output_item_text_joins_message_blocks(self):
        item = {
            "type": "message",
            "content": [
                {"type": "output_text", "text": "Hello"},
                {"type": "output_text", "text": " world"},
            ],
        }

        self.assertEqual(policy_module._output_item_text(item), "Hello world")

    def test_output_item_done_backfills_empty_completed_output(self):
        item = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
        }
        raw_sse = "\n".join(
            [
                "data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    }
                ),
                'data: {"type":"response.completed","response":{"output":[]}}',
                "data: [DONE]",
            ]
        )
        response = SimpleNamespace(output=[])

        policy_module._backfill_response_output(response, raw_sse)

        self.assertEqual(response.output, [item])

    def test_text_done_builds_assistant_message(self):
        raw_sse = "\n".join(
            [
                'data: {"type":"response.output_text.done","output_index":0,'
                '"content_index":0,"item_id":"msg_2","text":"Recovered"}',
                'data: {"type":"response.completed","response":{"output":[]}}',
            ]
        )

        output = policy_module._recover_responses_output(raw_sse)

        self.assertEqual(output[0]["role"], "assistant")
        self.assertEqual(output[0]["content"][0]["text"], "Recovered")


if __name__ == "__main__":
    unittest.main()
