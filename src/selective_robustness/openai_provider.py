"""Fail-closed OpenAI Responses API boundary for native SiliSocS runs."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from silisocs.runtime.language_models.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TERMINATORS,
    DEFAULT_TIMEOUT_SECONDS,
    LanguageModel,
)

from .parsing import parse_model_response
from .probabilities import extract_first_branch_probability
from .sampling import LogprobMode, SamplingSpec, TopPMode


OFFICIAL_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_TRANSPORT_MAX_RETRIES = 0

_ALLOWED_REQUEST_OPTIONS = frozenset(
    {
        "include",
        "max_output_tokens",
        "metadata",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "temperature",
        "text",
        "top_logprobs",
        "top_p",
        "truncation",
    }
)
_STATEFUL_OR_TOOL_OPTIONS = frozenset(
    {
        "background",
        "conversation",
        "input",
        "instructions",
        "model",
        "previous_response_id",
        "prompt",
        "store",
        "stream",
        "stream_options",
        "tool_choice",
        "tools",
    }
)
_SPECIAL_EXTRA_OPTIONS = frozenset(
    {
        "call_id",
        "expected_prompt_file",
        "expected_prompt_sha256",
        "run_manifest_sha256",
        "sampling_spec",
    }
)
_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LOG_LOCK = threading.Lock()


def create_openai_client(
    *,
    api_key: str | None = None,
    api_base: str | None = OFFICIAL_OPENAI_API_BASE,
    max_retries: int = DEFAULT_TRANSPORT_MAX_RETRIES,
) -> Any:
    """Construct an SDK client pinned to the official OpenAI endpoint.

    Custom endpoints intentionally have no configuration path here. Supporting
    one later requires a distinct, explicitly authorized boundary.
    """

    retry_count = _validate_retry_count(max_retries)
    if retry_count != 0:
        raise ValueError("the hospital live boundary requires zero transport retries")
    endpoint = _normalize_official_endpoint(api_base)
    forbidden_ambient = sorted(
        name
        for name in os.environ
        if (
            (name.upper().startswith("OPENAI_") and name.upper() != "OPENAI_API_KEY")
            or name.upper()
            in {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "NO_PROXY",
                "GRPC_PROXY",
                "WS_PROXY",
                "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
            }
        )
    )
    if forbidden_ambient:
        raise ValueError(
            "ambient provider routing is forbidden at the OpenAI boundary: "
            + ", ".join(forbidden_ambient)
        )
    authorized_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not authorized_key:
        raise ValueError("OPENAI_API_KEY is required at the OpenAI boundary")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "The OpenAI live extra is required to construct an OpenAI client."
        ) from exc
    return OpenAI(
        api_key=authorized_key,
        base_url=endpoint,
        max_retries=retry_count,
    )


class OpenAIResponsesModel(LanguageModel):
    """One fresh provider request with durable, append-only attempt records."""

    def __init__(
        self,
        model_name: str,
        *,
        log_file: str = "prompts_and_outputs.jsonl",
        debug: bool | None = True,
        api_base: str | None = OFFICIAL_OPENAI_API_BASE,
        api_key: str | None = None,
        temperature: float | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        client: Any | None = None,
        max_retries: int | None = None,
    ) -> None:
        normalized_model = str(model_name or "").strip()
        if not normalized_model:
            raise ValueError("model_name is required and has no default")

        extras = dict(extra_kwargs or {})
        special = {
            key: extras.pop(key)
            for key in tuple(extras)
            if key in _SPECIAL_EXTRA_OPTIONS
        }
        configured_retries = extras.pop("max_retries", DEFAULT_TRANSPORT_MAX_RETRIES)
        if max_retries is not None:
            configured_retries = max_retries
        self._transport_max_retries = _validate_retry_count(configured_retries)
        if self._transport_max_retries != 0:
            raise ValueError(
                "the hospital live boundary requires zero transport retries"
            )

        raw_sampling = special.get("sampling_spec")
        self._sampling_spec = (
            SamplingSpec.from_mapping(raw_sampling)
            if isinstance(raw_sampling, Mapping)
            else None
        )
        if raw_sampling is not None and self._sampling_spec is None:
            raise TypeError("sampling_spec must be a mapping")
        if self._sampling_spec is not None:
            if temperature is None or float(temperature) != float(
                self._sampling_spec.temperature
            ):
                raise ValueError(
                    "SiliSocS provider temperature disagrees with canonical sampling_spec"
                )
            if self._transport_max_retries != self._sampling_spec.transport_retries:
                raise ValueError(
                    "provider retry policy disagrees with canonical sampling_spec"
                )

        self._request_defaults = _validated_request_options(extras)
        self._model_name = normalized_model
        self._temperature = temperature
        self._log_file = str(log_file)
        self.debug = bool(debug)
        self._local = threading.local()
        self._endpoint = _normalize_official_endpoint(api_base)
        self._credential_for_redaction = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._call_id = _optional_call_id(special.get("call_id"))
        self._run_manifest_sha256 = _optional_sha256(
            special.get("run_manifest_sha256"), "run_manifest_sha256"
        )
        self._expected_prompt_sha256 = _optional_sha256(
            special.get("expected_prompt_sha256"), "expected_prompt_sha256"
        )
        self._expected_prompt = _load_expected_prompt(
            special.get("expected_prompt_file"), self._expected_prompt_sha256
        )
        is_guarded_live = self._expected_prompt is not None
        if is_guarded_live and (
            self._sampling_spec is None
            or self._call_id is None
            or self._run_manifest_sha256 is None
            or not self.debug
        ):
            raise ValueError(
                "guarded live calls require sampling_spec, call_id, an immutable "
                "run manifest binding, and durable logging"
            )
        if client is None and not is_guarded_live:
            raise ValueError(
                "real provider construction is available only for a manifest-bound "
                "reviewed pilot call"
            )

        self._client = (
            client
            if client is not None
            else create_openai_client(
                api_key=api_key,
                api_base=self._endpoint,
                max_retries=self._transport_max_retries,
            )
        )
        responses = getattr(self._client, "responses", None)
        if responses is None or not callable(getattr(responses, "create", None)):
            raise TypeError("client must expose responses.create")

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
        terminators: Collection[str] | None = DEFAULT_TERMINATORS,
        temperature: float | None = None,
        timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
        media: Sequence[str] | None = None,
        seed: int | None = None,
        **request_options: Any,
    ) -> str:
        """Return text from exactly one stateless Responses API request."""

        if media:
            raise NotImplementedError("this text-only provider does not accept media")
        if terminators:
            raise NotImplementedError(
                "the Responses API boundary does not support terminators"
            )
        if seed is not None:
            raise ValueError("the Responses API boundary does not support a seed")
        prompt_text = str(prompt)
        if (
            self._credential_for_redaction
            and self._credential_for_redaction in prompt_text
        ):
            raise ValueError("reviewed prompt unexpectedly contains the API credential")
        self._verify_expected_prompt(prompt_text)
        overrides = _validated_request_options(request_options)
        self._verify_canonical_sampling(
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            overrides=overrides,
        )

        request: dict[str, Any] = dict(self._request_defaults)
        request.update(overrides)
        if self._sampling_spec is not None:
            request["reasoning"] = {
                "effort": self._sampling_spec.reasoning_effort.value
            }
        effective_temperature = (
            self._temperature if temperature is None else temperature
        )
        if effective_temperature is not None:
            request["temperature"] = float(effective_temperature)
        if max_tokens is not None:
            request["max_output_tokens"] = int(max_tokens)
        if timeout is not None:
            request["timeout"] = float(timeout)
        request["model"] = self._model_name
        request["input"] = prompt_text
        create = self._client.responses.create
        if _supports_keyword(create, "store"):
            request["store"] = False
        self._verify_guarded_request_parameters(request, prompt_text)

        self._local.last_call_metadata = {}
        call_id = self._call_id or _standalone_call_id()
        requested_at = _utc_timestamp()
        request_for_log = _json_clone(request)
        start_record = {
            "event": "openai_responses_attempt_started",
            "status": "started",
            "call_id": call_id,
            "run_manifest_sha256": self._run_manifest_sha256,
            "requested_at_utc": requested_at,
            "prompt": prompt_text,
            "prompt_sha256": _text_sha256(prompt_text),
            "request_parameters": request_for_log,
            "routing": self._routing_metadata(),
        }
        self._write_attempt_start(start_record)

        phase = "provider_dispatch"
        dispatch_invoked = False
        response_received = False
        try:
            dispatch_invoked = True
            response = create(**request)
            response_received = True
            phase = "response_serialization"
            raw_payload = _response_payload(response)
            phase = "metadata_extraction"
            returned_model = _response_field(response, raw_payload, "model")
            metadata = {
                "provider": "openai",
                "api": "responses",
                "provider_attempts": 1,
                "transport_max_retries": self._transport_max_retries,
                "requested_model_identifier": self._model_name,
                "returned_model_identifier": returned_model,
                # Compatibility aliases retained in raw provider metadata; the
                # explicit identifier fields above are authoritative.
                "requested_model": self._model_name,
                "returned_model": returned_model,
                "response_id": _response_field(response, raw_payload, "id"),
                "request_id": _json_value(getattr(response, "_request_id", None)),
                "model": returned_model,
                "model_id": returned_model,
                "status": _response_field(response, raw_payload, "status"),
                "service_tier": _response_field(response, raw_payload, "service_tier"),
                "created_at": _response_field(response, raw_payload, "created_at"),
                "system_fingerprint": _response_field(
                    response, raw_payload, "system_fingerprint"
                ),
                "usage": _response_field(response, raw_payload, "usage"),
                "request_parameters": request_for_log,
                "sampling_spec": (
                    self._sampling_spec.as_dict()
                    if self._sampling_spec is not None
                    else None
                ),
                "reasoning_effort": (
                    self._sampling_spec.reasoning_effort.value
                    if self._sampling_spec is not None
                    else None
                ),
                "routing": self._routing_metadata(),
                "call_id": call_id,
                "run_manifest_sha256": self._run_manifest_sha256,
                "raw_provider_response": raw_payload,
            }
            output_text = getattr(
                response, "output_text", raw_payload.get("output_text")
            )
            if output_text is None:
                output_text = ""
            if not isinstance(output_text, str):
                raise TypeError("Responses API output_text must be a string or null")
            phase = "canonical_response_parsing"
            parsed = parse_model_response(output_text)
            metadata["canonical_parse"] = {
                "valid": parsed.valid,
                "method": parsed.method.value,
                "action": parsed.action.value if parsed.action is not None else None,
            }
            if (
                self._sampling_spec is not None
                and self._sampling_spec.logprob_mode
                is LogprobMode.FIRST_ACTION_BRANCH
            ):
                metadata["first_action_branch_probability"] = asdict(
                    extract_first_branch_probability(
                        raw_payload, output_text=output_text
                    )
                )
            serialized_metadata = _json_clone(metadata)
            phase = "credential_leak_validation"
            if (
                self._credential_for_redaction
                and self._credential_for_redaction
                in json.dumps(
                    {"metadata": serialized_metadata, "output_text": output_text},
                    ensure_ascii=True,
                )
            ):
                raise ValueError(
                    "provider response unexpectedly contains the API credential"
                )
            phase = "success_finalization"
            completed_at = _utc_timestamp()
            self._write_attempt_final(
                {
                    "event": "openai_responses_attempt_finalized",
                    "status": "success",
                    "call_id": call_id,
                    "run_manifest_sha256": self._run_manifest_sha256,
                    "requested_at_utc": requested_at,
                    "completed_at_utc": completed_at,
                    "prompt": prompt_text,
                    "prompt_sha256": _text_sha256(prompt_text),
                    "output_text": output_text,
                    "request_parameters": request_for_log,
                    "provider_metadata": serialized_metadata,
                }
            )
            self._local.last_call_metadata = serialized_metadata
            return output_text
        except Exception as exc:
            completed_at = _utc_timestamp()
            failure = {
                "event": "openai_responses_attempt_finalized",
                "status": "failure",
                "call_id": call_id,
                "run_manifest_sha256": self._run_manifest_sha256,
                "requested_at_utc": requested_at,
                "completed_at_utc": completed_at,
                "failure_phase": phase,
                "provider_response_received": response_received,
                "possibly_paid_attempt": dispatch_invoked,
                "prompt": prompt_text,
                "prompt_sha256": _text_sha256(prompt_text),
                "request_parameters": request_for_log,
                "exception_type": type(exc).__name__,
                "exception_message": _redact_secret(
                    str(exc), self._credential_for_redaction
                ),
            }
            try:
                self._write_attempt_final(failure)
            except Exception as log_error:
                raise RuntimeError(
                    "provider attempt finalization failed; the durable start record "
                    "marks an uncertain possibly-paid attempt"
                ) from log_error
            raise

    def get_last_call_metadata(self) -> dict[str, Any]:
        value = getattr(self._local, "last_call_metadata", {})
        return deepcopy(value) if isinstance(value, dict) else {}

    @property
    def last_call_metadata(self) -> dict[str, Any]:
        return self.get_last_call_metadata()

    def _routing_metadata(self) -> dict[str, str]:
        return {
            "endpoint": self._endpoint,
            "organization": "not_configured",
            "project": "not_configured",
            "ambient_proxy_routing": "disabled",
        }

    def _verify_expected_prompt(self, prompt: str) -> None:
        if self._expected_prompt is None:
            return
        if prompt != self._expected_prompt:
            raise ValueError(
                "actual provider input differs from the reviewed fixture text"
            )
        if _text_sha256(prompt) != self._expected_prompt_sha256:
            raise ValueError(
                "actual provider input differs from the reviewed fixture hash"
            )

    def _verify_canonical_sampling(
        self,
        *,
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        overrides: Mapping[str, Any],
    ) -> None:
        spec = self._sampling_spec
        if spec is None:
            return
        actual_temperature = self._temperature if temperature is None else temperature
        if actual_temperature is None or float(actual_temperature) != float(
            spec.temperature
        ):
            raise ValueError(
                "executed temperature disagrees with canonical sampling_spec"
            )
        if max_tokens != spec.max_output_tokens:
            raise ValueError(
                "executed token limit disagrees with canonical sampling_spec"
            )
        if timeout is None or float(timeout) != float(spec.timeout_seconds):
            raise ValueError("executed timeout disagrees with canonical sampling_spec")
        if spec.top_p_mode is TopPMode.OMITTED_PROVIDER_DEFAULT:
            if "top_p" in overrides or "top_p" in self._request_defaults:
                raise ValueError("top_p must be omitted for omitted_provider_default")
        else:
            actual = overrides.get("top_p", self._request_defaults.get("top_p"))
            if actual is None or float(actual) != float(spec.top_p):
                raise ValueError(
                    "executed top_p disagrees with canonical sampling_spec"
                )
        expected_reasoning = {"effort": spec.reasoning_effort.value}
        configured_reasoning = overrides.get(
            "reasoning", self._request_defaults.get("reasoning")
        )
        if (
            configured_reasoning is not None
            and configured_reasoning != expected_reasoning
        ):
            raise ValueError(
                "executed reasoning effort disagrees with canonical sampling_spec"
            )
        expected_logprobs = spec.provider_request_options()
        for name in ("include", "top_logprobs"):
            expected = expected_logprobs.get(name)
            actual = overrides.get(name, self._request_defaults.get(name))
            if actual != expected:
                raise ValueError(
                    f"executed {name} disagrees with canonical sampling_spec"
                )

    def _verify_guarded_request_parameters(
        self, request: Mapping[str, Any], prompt: str
    ) -> None:
        if self._expected_prompt is None:
            return
        assert self._sampling_spec is not None
        expected: dict[str, Any] = {
            "model": self._model_name,
            "input": prompt,
            "store": False,
            "temperature": float(self._sampling_spec.temperature),
            "reasoning": {"effort": self._sampling_spec.reasoning_effort.value},
            "max_output_tokens": self._sampling_spec.max_output_tokens,
            "timeout": float(self._sampling_spec.timeout_seconds),
        }
        if self._sampling_spec.top_p is not None:
            expected["top_p"] = float(self._sampling_spec.top_p)
        expected.update(
            {
                key: value
                for key, value in self._sampling_spec.provider_request_options().items()
                if key in {"include", "top_logprobs"}
            }
        )
        if dict(request) != expected:
            raise ValueError(
                "provider request parameters differ from the manifest-bound "
                "canonical sampling specification"
            )

    def _write_attempt_start(self, payload: Mapping[str, Any]) -> None:
        if not self.debug or not self._log_file:
            return
        path = Path(self._log_file)
        with _LOG_LOCK:
            if _call_id_has_start(path, str(payload["call_id"])):
                raise ValueError(f"duplicate provider call_id: {payload['call_id']}")
            _append_durable_jsonl(path, payload)

    def _write_attempt_final(self, payload: Mapping[str, Any]) -> None:
        if not self.debug or not self._log_file:
            return
        with _LOG_LOCK:
            _append_durable_jsonl(Path(self._log_file), payload)


def _normalize_official_endpoint(value: str | None) -> str:
    endpoint = str(value or OFFICIAL_OPENAI_API_BASE).strip().rstrip("/")
    if endpoint != OFFICIAL_OPENAI_API_BASE:
        raise ValueError(
            "custom OpenAI endpoints require a separate explicitly authorized boundary"
        )
    return OFFICIAL_OPENAI_API_BASE


def _validate_retry_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_retries must be a non-negative integer")
    return value


def _validated_request_options(options: Mapping[str, Any]) -> dict[str, Any]:
    non_null = {key: value for key, value in options.items() if value is not None}
    forbidden = sorted(set(non_null).intersection(_STATEFUL_OR_TOOL_OPTIONS))
    if forbidden:
        raise ValueError(
            "state, storage, streaming, and tools are disabled: " + ", ".join(forbidden)
        )
    unknown = sorted(set(non_null).difference(_ALLOWED_REQUEST_OPTIONS))
    if unknown:
        raise TypeError("unsupported Responses API options: " + ", ".join(unknown))
    return {key: deepcopy(value) for key, value in non_null.items()}


def _supports_keyword(callable_object: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        payload: Any = dict(response)
    elif callable(getattr(response, "model_dump", None)):
        payload = response.model_dump(mode="json")
    elif callable(getattr(response, "to_dict", None)):
        payload = response.to_dict()
    elif callable(getattr(response, "model_dump_json", None)):
        payload = json.loads(response.model_dump_json())
    elif is_dataclass(response) and not isinstance(response, type):
        payload = asdict(response)
    else:
        raise TypeError(
            "provider response must expose a JSON-serializable object payload"
        )
    if not isinstance(payload, Mapping):
        raise TypeError("provider response payload must be a JSON object")
    return _json_clone(dict(payload))


def _response_field(response: Any, payload: Mapping[str, Any], name: str) -> Any:
    if name in payload:
        return deepcopy(payload[name])
    return _json_value(getattr(response, name, None))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return _json_clone(value)
    if callable(getattr(value, "model_dump", None)):
        return _json_clone(value.model_dump(mode="json"))
    if callable(getattr(value, "to_dict", None)):
        return _json_clone(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_clone(asdict(value))
    raise TypeError(f"metadata value is not JSON-serializable: {type(value).__name__}")


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("provider metadata must be JSON-serializable") from exc


def _optional_call_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not _CALL_ID_PATTERN.fullmatch(normalized):
        raise ValueError("call_id must be a safe non-empty identifier")
    return normalized


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _load_expected_prompt(value: Any, expected_hash: str | None) -> str | None:
    if value is None and expected_hash is None:
        return None
    if value is None or expected_hash is None:
        raise ValueError("expected prompt file and hash must be configured together")
    path = Path(str(value)).resolve()
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("reviewed prompt fixture text is unavailable") from exc
    if _text_sha256(prompt) != expected_hash:
        raise ValueError(
            "reviewed prompt fixture file does not match its expected hash"
        )
    return prompt


def _call_id_has_start(path: Path, call_id: str) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"provider call ledger is invalid: {path}") from exc
        if (
            isinstance(item, dict)
            and item.get("event") == "openai_responses_attempt_started"
            and item.get("call_id") == call_id
        ):
            return True
    return False


def _append_durable_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_clone(dict(payload)), ensure_ascii=True, allow_nan=False, sort_keys=True
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _standalone_call_id() -> str:
    # Tests and non-live injected clients may omit a scheduled identifier. This
    # value is still collision-resistant within a process without implying that
    # an unscheduled call belongs to a pilot manifest.
    return f"standalone-{os.getpid()}-{threading.get_ident()}-{datetime.now(timezone.utc).timestamp():.6f}".replace(
        ".", "-"
    )


def _text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _redact_secret(value: str, secret: str) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
