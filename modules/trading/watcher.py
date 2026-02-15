from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import random
import re
import time
from collections import deque
from itertools import combinations
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import aiohttp

from modules.common import log_event

from .types import (
    FAIL_REASON_BELOW_STAGEA_REQUIRED,
    FAIL_REASON_BELOW_STAGEB_REQUIRED,
    FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED,
    PairConfig,
    SpreadObservation,
    now_iso,
)

DEFAULT_JUPITER_QUOTE_ENDPOINT = "https://api.jup.ag/swap/v1/quote"
_URL_QUERY_REDACTION_PATTERN = re.compile(r"(?i)(api[-_]?key=)[^&\s\"']+")
_HEADER_REDACTION_PATTERN = re.compile(r"(?i)(x-api-key\s*[:=]\s*)([^\s,;\"']+)")


class _TokenBucket:
    def __init__(self, *, rate_per_second: float, capacity: float) -> None:
        self._rate_per_second = max(0.01, float(rate_per_second))
        self._capacity = max(1.0, float(capacity))
        self._tokens = self._capacity
        self._updated_at = 0.0

    def _refill(self, *, now: float) -> None:
        if self._updated_at <= 0:
            self._updated_at = now
            return
        elapsed = max(0.0, now - self._updated_at)
        if elapsed > 0:
            self._tokens = min(
                self._capacity,
                self._tokens + (elapsed * self._rate_per_second),
            )
            self._updated_at = now

    def set_rate(self, *, rate_per_second: float, now: float) -> None:
        self._refill(now=now)
        self._rate_per_second = max(0.01, float(rate_per_second))

    def available_tokens(self, *, now: float) -> float:
        self._refill(now=now)
        return max(0.0, float(self._tokens))

    def wait_time_for(self, *, tokens: float, now: float) -> float:
        self._refill(now=now)
        required = max(0.0, float(tokens))
        if self._tokens >= required:
            return 0.0
        deficit = required - self._tokens
        return deficit / self._rate_per_second

    def consume(self, *, tokens: float, now: float) -> bool:
        self._refill(now=now)
        required = max(0.0, float(tokens))
        if self._tokens < required:
            return False
        self._tokens -= required
        return True


class HeliusRateLimitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        provider: str = "quote",
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider


class QuoteNoRoutesError(RuntimeError):
    pass


def _parse_retry_after_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None

    value = raw.strip()
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        return None

    return seconds if seconds > 0 else None


def _extract_api_key_from_url(url: str) -> str:
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    candidates = query.get("api-key") or query.get("api_key") or query.get("apikey")
    if not candidates:
        return ""
    return str(candidates[0]).strip()


def _is_helius_rpc_netloc(netloc: str) -> bool:
    host = netloc.lower()
    return host.endswith("helius-rpc.com") or ".helius-rpc.com" in host


def _is_jupiter_netloc(netloc: str) -> bool:
    host = netloc.lower()
    return host.endswith("jup.ag") or ".jup.ag" in host


def _is_helius_jup_proxy_path(path: str) -> bool:
    return "/jup-proxy/" in path


def _is_method_not_found_error(message: str) -> bool:
    return "method not found" in message.lower()


def _canonical_helius_jup_proxy_quote_path(path: str) -> str | None:
    normalized_path = (path or "/").rstrip("/")
    if not normalized_path:
        return None

    marker = "/jup-proxy"
    index = normalized_path.find(marker)
    if index < 0:
        return None
    prefix = normalized_path[: index + len(marker)]
    return f"{prefix}/quote"


def _append_unique(endpoints: list[str], endpoint: str) -> None:
    if endpoint and endpoint not in endpoints:
        endpoints.append(endpoint)


def _build_quote_endpoints(api_base_url: str, *, enable_helius_jup_proxy: bool) -> list[str]:
    parsed = urlsplit(api_base_url)
    endpoints: list[str] = []

    if _is_helius_rpc_netloc(parsed.netloc):
        # Helius provides Jupiter proxy under /v0/jup-proxy/.
        if _is_helius_jup_proxy_path(parsed.path):
            if enable_helius_jup_proxy:
                canonical_path = _canonical_helius_jup_proxy_quote_path(parsed.path)
                if canonical_path:
                    _append_unique(
                        endpoints,
                        urlunsplit((parsed.scheme, parsed.netloc, canonical_path, parsed.query, parsed.fragment)),
                    )
                _append_unique(
                    endpoints,
                    DEFAULT_JUPITER_QUOTE_ENDPOINT,
                )
                return endpoints

            _append_unique(endpoints, DEFAULT_JUPITER_QUOTE_ENDPOINT)
            return endpoints

        # Raw Helius RPC endpoint itself does not provide Jupiter `/quote`.
        _append_unique(endpoints, DEFAULT_JUPITER_QUOTE_ENDPOINT)
        return endpoints

    path = parsed.path or "/"

    if path.endswith("/quote"):
        quote_path = path
    else:
        normalized_path = path.rstrip("/")
        quote_path = f"{normalized_path}/quote" if normalized_path else "/quote"

    _append_unique(
        endpoints,
        urlunsplit((parsed.scheme, parsed.netloc, quote_path, parsed.query, parsed.fragment)),
    )
    return endpoints


def _build_helius_probe_endpoint(api_base_url: str) -> str | None:
    parsed = urlsplit(api_base_url)
    if not _is_helius_rpc_netloc(parsed.netloc):
        return None

    if _is_helius_jup_proxy_path(parsed.path):
        # Probe the raw RPC endpoint with the same key.
        return urlunsplit((parsed.scheme, parsed.netloc, "/", parsed.query, ""))

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, parsed.fragment))


def _error_message_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if message:
            return str(message)
        details = payload.get("details")
        if details:
            return str(details)
    return str(payload)


def _quote_provider_from_endpoint(endpoint: str) -> str:
    netloc = urlsplit(endpoint).netloc
    if _is_helius_rpc_netloc(netloc):
        return "helius"
    if _is_jupiter_netloc(netloc):
        return "jupiter"
    return "quote"


def _sanitize_endpoint_for_log(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


def _sanitize_preview_for_log(text: str, *, limit: int = 240) -> str:
    if not text:
        return ""
    sanitized = _URL_QUERY_REDACTION_PATTERN.sub(r"\1***", text)
    sanitized = _HEADER_REDACTION_PATTERN.sub(r"\1***", sanitized)
    return sanitized[:limit]


def _is_no_routes_error_text(text: str) -> bool:
    normalized = (text or "").lower()
    return (
        "no_routes_found" in normalized
        or "could not find any route" in normalized
        or "no route" in normalized
    )


def _safe_forward_output_amount(quote: dict[str, Any]) -> int:
    out_amount = int(quote.get("outAmount") or 0)
    min_out_amount = int(quote.get("otherAmountThreshold") or 0)
    if out_amount <= 0:
        return 0
    if min_out_amount <= 0:
        return out_amount
    return max(1, min(out_amount, min_out_amount))


def _normalize_quote_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in params.items():
        key = str(raw_key).strip()
        if not key:
            continue
        if isinstance(raw_value, bool):
            normalized[key] = raw_value
            continue
        if isinstance(raw_value, int):
            normalized[key] = raw_value
            continue
        if isinstance(raw_value, float):
            normalized[key] = int(raw_value) if raw_value.is_integer() else raw_value
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            normalized[key] = lowered == "true"
            continue
        try:
            normalized[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            parsed_float = float(value)
            normalized[key] = int(parsed_float) if parsed_float.is_integer() else parsed_float
            continue
        except ValueError:
            pass
        normalized[key] = value
    return normalized


_DEX_LABEL_ALIASES: dict[str, str] = {
    "orca whirlpool": "Whirlpool",
    "whirlpool": "Whirlpool",
    "raydium clmm": "Raydium CLMM",
    "raydium concentrated liquidity": "Raydium CLMM",
    "raydium cpmm": "Raydium CPMM",
    "meteora dlmm": "Meteora DLMM",
}


def _normalize_dex_label(label: str) -> str:
    text = str(label or "").strip().replace(",", " ")
    if not text:
        return ""
    key = " ".join(text.lower().split())
    return _DEX_LABEL_ALIASES.get(key, text)


def _normalize_dex_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for label in labels:
        canonical = _normalize_dex_label(label)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def _extract_probe_routes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        routes = payload.get("routes")
        if isinstance(routes, list):
            return [item for item in routes if isinstance(item, dict)]
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(payload.get("routePlan"), list):
            return [payload]
    elif isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _route_out_amount(route: dict[str, Any]) -> int:
    try:
        return int(route.get("outAmount") or 0)
    except (TypeError, ValueError):
        return 0


def _extract_route_dexes(quote: dict[str, Any]) -> tuple[str, ...]:
    route_plan = quote.get("routePlan")
    if not isinstance(route_plan, list):
        market_infos = quote.get("marketInfos")
        if not isinstance(market_infos, list):
            return ()
        dexes: list[str] = []
        for market in market_infos:
            if not isinstance(market, dict):
                continue
            label = str(market.get("label") or market.get("ammLabel") or market.get("ammKey") or "").strip()
            normalized_label = _normalize_dex_label(label)
            if normalized_label and normalized_label not in dexes:
                dexes.append(normalized_label)
        return tuple(dexes)

    dexes: list[str] = []
    for hop in route_plan:
        if not isinstance(hop, dict):
            continue
        swap_info = hop.get("swapInfo")
        if not isinstance(swap_info, dict):
            continue
        label = str(swap_info.get("label") or "").strip()
        if not label:
            label = str(swap_info.get("ammKey") or "").strip()
        normalized_label = _normalize_dex_label(label)
        if normalized_label and normalized_label not in dexes:
            dexes.append(normalized_label)

    return tuple(dexes)


def _quote_route_fingerprint(quote: dict[str, Any]) -> str:
    route_dexes = _extract_route_dexes(quote)
    if not route_dexes:
        return "unknown"
    return ">".join(route_dexes)


def _route_hash_from_fingerprint(route_fingerprint: str) -> str:
    normalized = str(route_fingerprint or "unknown").strip().lower()
    if not normalized:
        normalized = "unknown"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


class HeliusQuoteWatcher:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        api_base_url: str,
        api_key: str | None = None,
        jupiter_api_key: str | None = None,
        enable_helius_jup_proxy: bool = False,
        timeout_seconds: float = 8.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        rate_limit_backoff_seconds: float = 10.0,
        send_api_key_header: bool = True,
        default_quote_params: dict[str, Any] | None = None,
        exploration_top_n: int = 3,
        exploration_log_interval_seconds: float = 5.0,
    ) -> None:
        self._logger = logger
        self._api_base_url = api_base_url.rstrip("/")
        self._enable_helius_jup_proxy = enable_helius_jup_proxy
        parsed_base_url = urlsplit(api_base_url)
        self._configured_helius_jup_proxy = _is_helius_rpc_netloc(parsed_base_url.netloc) and _is_helius_jup_proxy_path(
            parsed_base_url.path
        )
        self._quote_endpoints = _build_quote_endpoints(
            api_base_url,
            enable_helius_jup_proxy=enable_helius_jup_proxy,
        )
        self._quote_endpoint_index = 0
        self._quote_endpoint = self._quote_endpoints[self._quote_endpoint_index]
        self._helius_probe_endpoint = _build_helius_probe_endpoint(api_base_url)
        self._api_key = api_key.strip() if api_key else _extract_api_key_from_url(api_base_url)
        self._jupiter_api_key = jupiter_api_key.strip() if jupiter_api_key else ""
        self._quote_endpoint_is_helius = _is_helius_rpc_netloc(urlsplit(self._quote_endpoint).netloc)
        self._fallback_quote_logged = False
        self._migration_plan_logged = False
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.05, retry_backoff_seconds)
        self._rate_limit_backoff_seconds = max(1.0, rate_limit_backoff_seconds)
        self._send_api_key_header = send_api_key_header
        self._default_quote_params = _normalize_quote_params(default_quote_params)
        self._exploration_top_n = max(1, int(exploration_top_n))
        self._exploration_log_interval_seconds = max(0.0, float(exploration_log_interval_seconds))
        self._last_exploration_log_at = 0.0
        self._quote_request_semaphore = asyncio.Semaphore(1)
        self._quote_max_rps = 0.4
        self._quote_global_max_rps = 0.4
        self._quote_exploration_max_rps = 0.3
        self._quote_execution_max_rps = 0.15
        self._quote_min_interval_seconds = 1.0 / self._quote_max_rps
        self._quote_global_bucket_capacity = 3.0
        self._quote_exploration_bucket_capacity = 3.0
        self._quote_execution_bucket_capacity = 2.0
        self._global_quote_bucket = _TokenBucket(
            rate_per_second=self._quote_global_max_rps,
            capacity=self._quote_global_bucket_capacity,
        )
        self._exploration_quote_bucket = _TokenBucket(
            rate_per_second=self._quote_exploration_max_rps,
            capacity=self._quote_exploration_bucket_capacity,
        )
        self._execution_quote_bucket = _TokenBucket(
            rate_per_second=self._quote_execution_max_rps,
            capacity=self._quote_execution_bucket_capacity,
        )
        self._rate_limiter_state_log_interval_seconds = 60.0
        self._last_rate_limiter_state_log_at = 0.0
        self._rate_limit_pause_until_epoch = 0.0
        self._rate_limit_backoff_level = 0
        self._pause_execution_retry_budget_max = 1
        self._pause_execution_retry_budget = self._pause_execution_retry_budget_max
        self._default_rate_limit_pause_seconds = 60.0
        self._max_rate_limit_pause_seconds = 300.0
        self._quote_cache_ttl_seconds = 0.5
        self._quote_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._quote_call_count = 0
        self._quote_cache_hits = 0
        self._quote_network_request_count = 0
        self._quote_network_timestamps: deque[float] = deque()
        self._rate_limited_count = 0
        self._no_routes_cache_ttl_seconds = 120.0
        self._no_routes_failure_cache: dict[str, float] = {}
        self._probe_max_routes = 50
        self._probe_base_amounts_raw: tuple[int, ...] = (10_000_000, 20_000_000, 40_000_000)
        self._sweep_top_k_max = 8
        self._near_miss_expand_bps = 0.5
        self._median_requote_max_range_bps = 0.6
        self._min_improvement_bps = 0.2
        self._dynamic_allowlist_topk = 10
        self._dynamic_allowlist_good_candidate_alpha = 2.0
        self._dynamic_allowlist_ttl_seconds = 300.0
        self._dynamic_allowlist_refresh_seconds = 5.0
        self._last_dynamic_allowlist_refresh_at = 0.0
        self._dynamic_dex_state: dict[str, dict[str, float]] = {}
        self._route_instability_cooldowns: dict[str, float] = {}
        self._enable_probe_unconstrained = True
        self._enable_probe_multi_amount = True
        self._enable_stagea_relaxed_gate = True
        self._enable_route_instability_cooldown = True
        self._route_instability_cooldown_requote_seconds = 60.0
        self._route_instability_cooldown_decay_requote_bps = 2.0
        self._negative_fallback_streak_threshold = 10
        self._negative_best_spread_streak = 0
        self._missing_api_key_logged = False
        self._missing_jupiter_api_key_logged = False
        self._connectivity_checked = False
        self._session: aiohttp.ClientSession | None = None

    def _build_helius_rpc_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "crypto-trader-bot/1.0 helius",
        }
        if (
            self._send_api_key_header
            and self._api_key
            and _is_helius_rpc_netloc(urlsplit(self._api_base_url).netloc)
        ):
            headers["x-api-key"] = self._api_key
        return headers

    def _active_quote_endpoint_for_log(self) -> str:
        return _sanitize_endpoint_for_log(self._quote_endpoint)

    def _configured_api_base_for_log(self) -> str:
        return _sanitize_endpoint_for_log(self._api_base_url)

    def _quote_endpoints_for_log(self) -> list[str]:
        return [_sanitize_endpoint_for_log(endpoint) for endpoint in self._quote_endpoints]

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "crypto-trader-bot/1.0 helius",
        }
        provider = _quote_provider_from_endpoint(self._quote_endpoint)
        if provider == "helius":
            if self._send_api_key_header and self._api_key:
                headers["x-api-key"] = self._api_key
            elif not self._api_key and not self._missing_api_key_logged:
                self._missing_api_key_logged = True
                log_event(
                    self._logger,
                    level="warning",
                    event="quote_api_key_missing",
                    message="HELIUS_API_KEY is not set; quote requests may be rate-limited",
                    quote_endpoint=self._active_quote_endpoint_for_log(),
                )
        elif provider == "jupiter":
            if self._send_api_key_header and self._jupiter_api_key:
                headers["x-api-key"] = self._jupiter_api_key
            elif not self._jupiter_api_key and not self._missing_jupiter_api_key_logged:
                self._missing_jupiter_api_key_logged = True
                log_event(
                    self._logger,
                    level="warning",
                    event="quote_api_key_missing",
                    message=(
                        "JUPITER_API_KEY is not set; fallback Jupiter endpoint may be strongly rate-limited."
                    ),
                    quote_endpoint=self._active_quote_endpoint_for_log(),
                )
        return headers

    async def _probe_helius_connectivity(self) -> None:
        if self._session is None:
            raise RuntimeError("Helius HTTP session is not initialized.")

        if self._helius_probe_endpoint is None:
            return

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getHealth",
            "params": [],
        }

        try:
            async with self._session.post(
                self._helius_probe_endpoint,
                json=payload,
                headers=self._build_helius_rpc_headers(),
            ) as response:
                status = response.status
                body = await response.text()
        except aiohttp.ClientError as error:
            log_event(
                self._logger,
                level="critical",
                event="helius_connectivity_error",
                message="Failed to reach Helius endpoint during connectivity probe",
                error=str(error),
            )
            raise RuntimeError(f"Helius connectivity check failed: {error}") from error

        parsed: Any = None
        with_error_payload = False
        try:
            parsed = json.loads(body)
            with_error_payload = isinstance(parsed, dict) and "error" in parsed
        except json.JSONDecodeError:
            parsed = None

        if with_error_payload:
            error_payload = parsed.get("error")
            message = _error_message_from_payload(error_payload)
            log_event(
                self._logger,
                level="critical",
                event="helius_rpc_error",
                message="Helius returned RPC error during connectivity probe",
                status=status,
                error=message,
                body_preview=_sanitize_preview_for_log(body),
            )
            raise RuntimeError(f"Helius connectivity check failed: {message}")

        if status in {401, 403}:
            log_event(
                self._logger,
                level="critical",
                event="helius_connectivity_auth_failed",
                message="Helius API key is invalid or unauthorized",
                status=status,
                body_preview=_sanitize_preview_for_log(body),
            )
            raise RuntimeError(
                "Helius connectivity check failed: invalid API key or unauthorized access."
            )

        if status >= 400:
            raise RuntimeError(
                "Helius connectivity check failed: "
                f"status={status} body={_sanitize_preview_for_log(body)!r}"
            )

    async def connect(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        if not self._migration_plan_logged and len(self._quote_endpoints) > 1:
            self._migration_plan_logged = True
            log_event(
                self._logger,
                level="info",
                event="quote_endpoint_failover_enabled",
                message="Quote endpoint failover is enabled",
                endpoint_total=len(self._quote_endpoints),
                active_quote_endpoint=self._active_quote_endpoint_for_log(),
                quote_endpoints=self._quote_endpoints_for_log(),
            )

        if (
            not self._fallback_quote_logged
            and self._configured_helius_jup_proxy
            and not self._enable_helius_jup_proxy
        ):
            self._fallback_quote_logged = True
            log_event(
                self._logger,
                level="warning",
                event="helius_jup_proxy_disabled",
                message=(
                    "Helius jup-proxy endpoint is disabled by configuration; quote requests start with Jupiter API."
                ),
                configured_url=self._configured_api_base_for_log(),
                quote_endpoint=self._active_quote_endpoint_for_log(),
            )
        elif (
            not self._fallback_quote_logged
            and _is_helius_rpc_netloc(urlsplit(self._api_base_url).netloc)
            and not self._quote_endpoint_is_helius
        ):
            self._fallback_quote_logged = True
            log_event(
                self._logger,
                level="warning",
                event="helius_quote_endpoint_unsupported",
                message=(
                    "HELIUS_QUOTE_API points to an RPC endpoint; quote requests will use Jupiter Quote API."
                ),
                configured_url=self._configured_api_base_for_log(),
                quote_endpoint=self._active_quote_endpoint_for_log(),
            )

        if not self._connectivity_checked:
            if self._configured_helius_jup_proxy and not self._enable_helius_jup_proxy:
                self._connectivity_checked = True
            else:
                await self._probe_helius_connectivity()
                self._connectivity_checked = True

    def _switch_quote_endpoint(self, *, reason: str, event: str, body_preview: str = "") -> bool:
        next_index = self._quote_endpoint_index + 1
        if next_index >= len(self._quote_endpoints):
            return False

        previous_endpoint = self._quote_endpoint
        self._quote_endpoint_index = next_index
        self._quote_endpoint = self._quote_endpoints[self._quote_endpoint_index]
        self._quote_endpoint_is_helius = _is_helius_rpc_netloc(urlsplit(self._quote_endpoint).netloc)

        log_event(
            self._logger,
            level="warning",
            event=event,
            message="Switching quote endpoint after runtime incompatibility",
            reason=reason,
            previous_quote_endpoint=_sanitize_endpoint_for_log(previous_endpoint),
            next_quote_endpoint=self._active_quote_endpoint_for_log(),
            endpoint_index=self._quote_endpoint_index,
            endpoint_total=len(self._quote_endpoints),
            body_preview=_sanitize_preview_for_log(body_preview),
        )
        log_event(
            self._logger,
            level="info",
            event="quote_endpoint_migrated",
            message="Quote endpoint migration completed",
            reason=reason,
            previous_quote_endpoint=_sanitize_endpoint_for_log(previous_endpoint),
            next_quote_endpoint=self._active_quote_endpoint_for_log(),
            previous_provider=_quote_provider_from_endpoint(previous_endpoint),
            next_provider=_quote_provider_from_endpoint(self._quote_endpoint),
            endpoint_index=self._quote_endpoint_index,
            endpoint_total=len(self._quote_endpoints),
        )
        return True

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._connectivity_checked = False

    async def healthcheck(self) -> None:
        await self.connect()

    def _apply_runtime_limits(
        self,
        *,
        quote_max_rps: float | None = None,
        quote_exploration_max_rps: float | None = None,
        quote_execution_max_rps: float | None = None,
        quote_cache_ttl_ms: int | None = None,
        no_routes_cache_ttl_seconds: float | None = None,
        probe_max_routes: int | None = None,
        probe_base_amounts_raw: tuple[int, ...] | None = None,
        sweep_top_k_max: int | None = None,
        near_miss_expand_bps: float | None = None,
        median_requote_max_range_bps: float | None = None,
        min_improvement_bps: float | None = None,
        dynamic_allowlist_topk: int | None = None,
        dynamic_allowlist_good_candidate_alpha: float | None = None,
        dynamic_allowlist_ttl_seconds: float | None = None,
        dynamic_allowlist_refresh_seconds: float | None = None,
        negative_fallback_streak_threshold: int | None = None,
        enable_probe_unconstrained: bool | None = None,
        enable_probe_multi_amount: bool | None = None,
        enable_stagea_relaxed_gate: bool | None = None,
        enable_route_instability_cooldown: bool | None = None,
        route_instability_cooldown_requote_seconds: float | None = None,
        route_instability_cooldown_decay_requote_bps: float | None = None,
    ) -> None:
        if quote_max_rps is not None:
            self._quote_global_max_rps = max(0.05, float(quote_max_rps))
        if quote_exploration_max_rps is not None:
            self._quote_exploration_max_rps = max(0.05, float(quote_exploration_max_rps))
        if quote_execution_max_rps is not None:
            self._quote_execution_max_rps = max(0.05, float(quote_execution_max_rps))
        self._quote_exploration_max_rps = min(
            self._quote_exploration_max_rps,
            self._quote_global_max_rps,
        )
        self._quote_execution_max_rps = min(
            self._quote_execution_max_rps,
            self._quote_global_max_rps,
        )
        self._quote_max_rps = self._quote_global_max_rps
        self._quote_min_interval_seconds = 1.0 / self._quote_global_max_rps
        self._sync_quote_bucket_rates()
        if quote_cache_ttl_ms is not None:
            self._quote_cache_ttl_seconds = max(0.0, float(quote_cache_ttl_ms) / 1000.0)
        if no_routes_cache_ttl_seconds is not None:
            self._no_routes_cache_ttl_seconds = max(1.0, float(no_routes_cache_ttl_seconds))
        if probe_max_routes is not None:
            self._probe_max_routes = max(1, int(probe_max_routes))
        if probe_base_amounts_raw is not None:
            parsed_amounts = tuple(
                amount
                for amount in (int(value) for value in probe_base_amounts_raw)
                if amount > 0
            )
            if parsed_amounts:
                self._probe_base_amounts_raw = parsed_amounts
        if sweep_top_k_max is not None:
            self._sweep_top_k_max = max(1, int(sweep_top_k_max))
        if near_miss_expand_bps is not None:
            self._near_miss_expand_bps = max(0.0, float(near_miss_expand_bps))
        if median_requote_max_range_bps is not None:
            self._median_requote_max_range_bps = max(0.0, float(median_requote_max_range_bps))
        if min_improvement_bps is not None:
            self._min_improvement_bps = max(0.0, float(min_improvement_bps))
        if dynamic_allowlist_topk is not None:
            self._dynamic_allowlist_topk = max(1, int(dynamic_allowlist_topk))
        if dynamic_allowlist_good_candidate_alpha is not None:
            self._dynamic_allowlist_good_candidate_alpha = max(
                0.0, float(dynamic_allowlist_good_candidate_alpha)
            )
        if dynamic_allowlist_ttl_seconds is not None:
            self._dynamic_allowlist_ttl_seconds = max(1.0, float(dynamic_allowlist_ttl_seconds))
        if dynamic_allowlist_refresh_seconds is not None:
            self._dynamic_allowlist_refresh_seconds = max(0.5, float(dynamic_allowlist_refresh_seconds))
        if negative_fallback_streak_threshold is not None:
            self._negative_fallback_streak_threshold = max(1, int(negative_fallback_streak_threshold))
        if enable_probe_unconstrained is not None:
            self._enable_probe_unconstrained = bool(enable_probe_unconstrained)
        if enable_probe_multi_amount is not None:
            self._enable_probe_multi_amount = bool(enable_probe_multi_amount)
        if enable_stagea_relaxed_gate is not None:
            self._enable_stagea_relaxed_gate = bool(enable_stagea_relaxed_gate)
        if enable_route_instability_cooldown is not None:
            self._enable_route_instability_cooldown = bool(enable_route_instability_cooldown)
        if route_instability_cooldown_requote_seconds is not None:
            self._route_instability_cooldown_requote_seconds = max(
                0.0, float(route_instability_cooldown_requote_seconds)
            )
        if route_instability_cooldown_decay_requote_bps is not None:
            self._route_instability_cooldown_decay_requote_bps = max(
                0.0, float(route_instability_cooldown_decay_requote_bps)
            )

    @staticmethod
    def _quote_cache_key(endpoint: str, params: dict[str, str]) -> str:
        parts = [f"{key}={params[key]}" for key in sorted(params)]
        return f"{endpoint}?{'&'.join(parts)}"

    @staticmethod
    def _amount_bucket(amount: int) -> int:
        if amount <= 0:
            return 0
        if amount < 1_000_000:
            return amount
        return int(amount / 1_000_000) * 1_000_000

    @staticmethod
    def _to_dex_csv(dexes: tuple[str, ...]) -> str:
        cleaned: list[str] = []
        for dex in dexes:
            normalized = _normalize_dex_label(dex)
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return ",".join(cleaned)

    @staticmethod
    def _group_dex_labels(labels: tuple[str, ...], *, group_sizes: tuple[int, ...], limit: int) -> list[tuple[str, ...]]:
        groups: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        unique_labels: list[str] = []
        for label in labels:
            normalized = _normalize_dex_label(label)
            if normalized and normalized not in unique_labels:
                unique_labels.append(normalized)
        if not unique_labels:
            return groups

        for size in group_sizes:
            if size <= 1 or len(unique_labels) < size:
                continue
            for group in combinations(unique_labels, size):
                if group in seen:
                    continue
                seen.add(group)
                groups.append(group)
                if len(groups) >= limit:
                    return groups
        return groups

    def _prune_quote_cache(self, now: float) -> None:
        if not self._quote_cache:
            return
        expired = [key for key, (expires_at, _) in self._quote_cache.items() if expires_at <= now]
        for key in expired:
            self._quote_cache.pop(key, None)

    def _prune_no_routes_cache(self, now: float) -> None:
        if not self._no_routes_failure_cache:
            return
        expired = [key for key, expires_at in self._no_routes_failure_cache.items() if expires_at <= now]
        for key in expired:
            self._no_routes_failure_cache.pop(key, None)

    def _prune_dynamic_dex_state(self, now: float) -> None:
        if not self._dynamic_dex_state:
            return
        expired = [label for label, state in self._dynamic_dex_state.items() if state.get("expires_at", 0.0) <= now]
        for label in expired:
            self._dynamic_dex_state.pop(label, None)

    def _mark_no_routes_failure(self, *, key: str, now: float) -> None:
        self._no_routes_failure_cache[key] = now + self._no_routes_cache_ttl_seconds

    def _is_no_routes_cached(self, *, key: str, now: float) -> bool:
        expires_at = self._no_routes_failure_cache.get(key)
        return bool(expires_at and expires_at > now)

    def _record_dynamic_dex_labels(self, labels: tuple[str, ...], *, now: float) -> None:
        normalized_labels = _normalize_dex_labels(labels)
        if not normalized_labels:
            return
        self._prune_dynamic_dex_state(now)
        for label in normalized_labels:
            current = self._dynamic_dex_state.get(
                label,
                {"hit_count": 0.0, "good_candidate_count": 0.0, "last_seen": 0.0, "expires_at": 0.0},
            )
            current["hit_count"] = float(current.get("hit_count", 0.0)) + 1.0
            current["last_seen"] = now
            current["expires_at"] = now + self._dynamic_allowlist_ttl_seconds
            self._dynamic_dex_state[label] = current

    def _record_dynamic_good_candidate_labels(self, labels: tuple[str, ...], *, now: float) -> None:
        normalized_labels = _normalize_dex_labels(labels)
        if not normalized_labels:
            return
        self._prune_dynamic_dex_state(now)
        for label in normalized_labels:
            current = self._dynamic_dex_state.get(
                label,
                {"hit_count": 0.0, "good_candidate_count": 0.0, "last_seen": 0.0, "expires_at": 0.0},
            )
            current["good_candidate_count"] = float(current.get("good_candidate_count", 0.0)) + 1.0
            current["last_seen"] = now
            current["expires_at"] = now + self._dynamic_allowlist_ttl_seconds
            self._dynamic_dex_state[label] = current

    def _dynamic_dex_labels_top(self, *, now: float, topk: int | None = None) -> tuple[str, ...]:
        self._prune_dynamic_dex_state(now)
        if not self._dynamic_dex_state:
            return ()
        limit = max(1, int(topk or self._dynamic_allowlist_topk))

        alpha = max(0.0, float(self._dynamic_allowlist_good_candidate_alpha))

        def rank_score(state: dict[str, float]) -> tuple[float, float]:
            hit_count = float(state.get("hit_count", state.get("score", 0.0)))
            good_candidate_count = float(state.get("good_candidate_count", 0.0))
            score = hit_count + (alpha * good_candidate_count)
            return score, float(state.get("last_seen", 0.0))

        ranked = sorted(
            self._dynamic_dex_state.items(),
            key=lambda item: rank_score(item[1]),
            reverse=True,
        )
        return tuple(label for label, _state in ranked[:limit])

    @staticmethod
    def _route_pair_key(forward_route_hash: str, reverse_route_hash: str) -> str:
        return f"{forward_route_hash}|{reverse_route_hash}"

    def _prune_route_instability_cooldowns(self, now: float) -> None:
        if not self._route_instability_cooldowns:
            return
        expired = [key for key, expires_at in self._route_instability_cooldowns.items() if expires_at <= now]
        for key in expired:
            self._route_instability_cooldowns.pop(key, None)

    def _is_route_pair_cooling(self, *, forward_route_hash: str, reverse_route_hash: str, now: float) -> bool:
        self._prune_route_instability_cooldowns(now)
        key = self._route_pair_key(forward_route_hash, reverse_route_hash)
        expires_at = self._route_instability_cooldowns.get(key, 0.0)
        return expires_at > now

    def _mark_route_pair_cooldown(
        self,
        *,
        forward_route_hash: str,
        reverse_route_hash: str,
        cooldown_seconds: float,
    ) -> None:
        if cooldown_seconds <= 0:
            return
        now = asyncio.get_running_loop().time()
        key = self._route_pair_key(forward_route_hash, reverse_route_hash)
        self._route_instability_cooldowns[key] = max(
            self._route_instability_cooldowns.get(key, 0.0),
            now + cooldown_seconds,
        )

    def register_route_instability_cooldown(
        self,
        *,
        forward_route_hash: str,
        reverse_route_hash: str,
        decay_bps: float,
        cooldown_seconds: float,
        source: str,
        plan_id: str | None = None,
    ) -> None:
        if not self._enable_route_instability_cooldown:
            return
        normalized_forward_hash = str(forward_route_hash or "").strip()
        normalized_reverse_hash = str(reverse_route_hash or "").strip()
        if not normalized_forward_hash or not normalized_reverse_hash:
            return
        self._mark_route_pair_cooldown(
            forward_route_hash=normalized_forward_hash,
            reverse_route_hash=normalized_reverse_hash,
            cooldown_seconds=cooldown_seconds,
        )
        log_event(
            self._logger,
            level="warning",
            event="route_instability_cooldown_armed",
            message="Route pair cooldown was armed due to severe spread decay",
            source=source,
            plan_id=plan_id,
            forward_route_hash=normalized_forward_hash,
            reverse_route_hash=normalized_reverse_hash,
            decay_bps=round(float(decay_bps), 6),
            cooldown_seconds=round(float(cooldown_seconds), 3),
        )

    def _effective_allowlist(
        self,
        *,
        static_labels: tuple[str, ...],
        dynamic_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        result: list[str] = []
        for label in static_labels:
            normalized = _normalize_dex_label(label)
            if normalized and normalized not in result:
                result.append(normalized)
        for label in dynamic_labels:
            normalized = _normalize_dex_label(label)
            if normalized and normalized not in result:
                result.append(normalized)
        return tuple(result)

    async def _refresh_dynamic_dex_labels(
        self,
        *,
        pair: PairConfig,
        base_quote_params: dict[str, Any],
        quote_params_source: str,
    ) -> tuple[str, ...]:
        now = asyncio.get_running_loop().time()
        if (now - self._last_dynamic_allowlist_refresh_at) < self._dynamic_allowlist_refresh_seconds:
            return self._dynamic_dex_labels_top(now=now)
        if not self._enable_probe_unconstrained:
            return self._dynamic_dex_labels_top(now=now)

        probe_params = dict(base_quote_params)
        probe_params.pop("dexes", None)
        probe_params.pop("excludeDexes", None)
        probe_params.pop("exclude_dexes", None)
        probe_params["maxRoutes"] = max(10, int(self._probe_max_routes))
        probe_amounts: list[int] = []
        if self._enable_probe_multi_amount:
            for amount in self._probe_base_amounts_raw:
                normalized_amount = int(amount)
                if normalized_amount > 0 and normalized_amount not in probe_amounts:
                    probe_amounts.append(normalized_amount)
        if not probe_amounts:
            probe_amounts = [int(pair.base_amount)]
        elif int(pair.base_amount) not in probe_amounts:
            probe_amounts.append(int(pair.base_amount))

        probe_started_at = asyncio.get_running_loop().time()
        probe_routes: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for probe_amount in probe_amounts:
            try:
                probe_quote = await self.quote(
                    input_mint=pair.base_mint,
                    output_mint=pair.quote_mint,
                    amount=probe_amount,
                    slippage_bps=pair.slippage_bps,
                    extra_params=probe_params,
                )
            except HeliusRateLimitError:
                raise
            except Exception as error:
                last_error = error
                continue

            extracted = _extract_probe_routes(probe_quote)
            if extracted:
                probe_routes.extend(extracted)
            else:
                probe_routes.append(probe_quote)

        probe_latency_ms = (asyncio.get_running_loop().time() - probe_started_at) * 1000.0
        if not probe_routes:
            log_event(
                self._logger,
                level="info",
                event="observed_dex_labels_probe_failed",
                message="Observed DEX label probe failed; keeping existing dynamic allowlist",
                error=str(last_error) if last_error is not None else "probe returned no routes",
                probe_pair=pair.symbol,
                probe_base_amounts=probe_amounts,
                probe_quote_params=probe_params,
                quote_params_source=quote_params_source or "watcher_default",
                probe_latency_ms=round(probe_latency_ms, 3),
            )
            return self._dynamic_dex_labels_top(now=now)

        self._last_dynamic_allowlist_refresh_at = now
        probe_routes.sort(key=_route_out_amount, reverse=True)
        top_routes = probe_routes[: max(1, int(self._dynamic_allowlist_topk))]
        labels_list: list[str] = []
        unique_probe_labels: set[str] = set()
        for route in top_routes:
            for label in _extract_route_dexes(route):
                normalized = _normalize_dex_label(label)
                if normalized and normalized not in labels_list:
                    labels_list.append(normalized)
                if normalized:
                    unique_probe_labels.add(normalized)
        labels = tuple(labels_list)
        self._record_dynamic_dex_labels(labels, now=now)
        top_labels = self._dynamic_dex_labels_top(now=now)
        event_level = "warning" if len(probe_routes) <= 1 else "info"
        if len(probe_routes) <= 1:
            log_event(
                self._logger,
                level="warning",
                event="observed_dex_labels_probe_thin",
                message="Observed DEX probe returned too few route candidates",
                probe_pair=pair.symbol,
                probe_base_amounts=probe_amounts,
                probe_quote_params=probe_params,
                probe_route_candidate_count=len(probe_routes),
                probe_route_sampled_count=len(top_routes),
                route_candidate_count=len(probe_routes),
                route_sampled_count=len(top_routes),
                probe_unique_dex_count=len(unique_probe_labels),
                quote_params_source=quote_params_source or "watcher_default",
                probe_latency_ms=round(probe_latency_ms, 3),
            )
        log_event(
            self._logger,
            level=event_level,
            event="observed_dex_labels_top",
            message="Observed DEX labels refreshed from unconstrained quote probe",
            observed_dex_labels_top=list(top_labels),
            probe_pair=pair.symbol,
            probe_base_amounts=probe_amounts,
            probe_quote_params=probe_params,
            probe_route_candidate_count=len(probe_routes),
            probe_route_sampled_count=len(top_routes),
            route_candidate_count=len(probe_routes),
            route_sampled_count=len(top_routes),
            probe_unique_dex_count=len(unique_probe_labels),
            probe_max_routes=self._probe_max_routes,
            quote_params_source=quote_params_source or "watcher_default",
            probe_latency_ms=round(probe_latency_ms, 3),
        )
        return top_labels

    def _note_quote_network_request(self, now: float) -> None:
        self._quote_network_request_count += 1
        self._quote_network_timestamps.append(now)
        cutoff = now - 1.0
        while self._quote_network_timestamps and self._quote_network_timestamps[0] < cutoff:
            self._quote_network_timestamps.popleft()

    def _quote_rps_effective(self) -> float:
        now = asyncio.get_running_loop().time()
        cutoff = now - 1.0
        while self._quote_network_timestamps and self._quote_network_timestamps[0] < cutoff:
            self._quote_network_timestamps.popleft()
        return float(len(self._quote_network_timestamps))

    def _quote_cache_hit_rate(self) -> float:
        if self._quote_call_count <= 0:
            return 0.0
        return self._quote_cache_hits / self._quote_call_count

    def _sync_quote_bucket_rates(self, *, now: float | None = None) -> None:
        if now is None:
            with contextlib.suppress(RuntimeError):
                now = asyncio.get_running_loop().time()
        normalized_now = float(now if now is not None else time.monotonic())
        self._global_quote_bucket.set_rate(
            rate_per_second=self._quote_global_max_rps,
            now=normalized_now,
        )
        self._exploration_quote_bucket.set_rate(
            rate_per_second=self._quote_exploration_max_rps,
            now=normalized_now,
        )
        self._execution_quote_bucket.set_rate(
            rate_per_second=self._quote_execution_max_rps,
            now=normalized_now,
        )

    async def _consume_quote_tokens(self, *, is_execution_purpose: bool) -> None:
        scope_bucket = (
            self._execution_quote_bucket
            if is_execution_purpose
            else self._exploration_quote_bucket
        )
        while True:
            now = asyncio.get_running_loop().time()
            wait_global = self._global_quote_bucket.wait_time_for(tokens=1.0, now=now)
            wait_scope = scope_bucket.wait_time_for(tokens=1.0, now=now)
            wait_seconds = max(wait_global, wait_scope)
            if wait_seconds <= 0:
                self._global_quote_bucket.consume(tokens=1.0, now=now)
                scope_bucket.consume(tokens=1.0, now=now)
                self._maybe_log_rate_limiter_state(now=now)
                return
            await asyncio.sleep(wait_seconds)

    def _maybe_log_rate_limiter_state(self, *, now: float) -> None:
        if (now - self._last_rate_limiter_state_log_at) < self._rate_limiter_state_log_interval_seconds:
            return
        self._last_rate_limiter_state_log_at = now
        log_event(
            self._logger,
            level="info",
            event="rate_limiter_state",
            message="Quote rate limiter token state snapshot",
            global_tokens=round(self._global_quote_bucket.available_tokens(now=now), 3),
            exploration_tokens=round(self._exploration_quote_bucket.available_tokens(now=now), 3),
            execution_tokens=round(self._execution_quote_bucket.available_tokens(now=now), 3),
            global_rate_rps=round(self._quote_global_max_rps, 6),
            exploration_rate_rps=round(self._quote_exploration_max_rps, 6),
            execution_rate_rps=round(self._quote_execution_max_rps, 6),
            quote_rps_effective=round(self._quote_rps_effective(), 3),
            rate_limited_count=self._rate_limited_count,
            pause_until_epoch=(
                round(self._rate_limit_pause_until_epoch, 3)
                if self._rate_limit_pause_until_epoch > time.time()
                else None
            ),
        )

    def rate_limit_pause_until_epoch(self) -> float:
        return max(0.0, float(self._rate_limit_pause_until_epoch))

    def rate_limit_pause_remaining_seconds(self) -> float:
        remaining = max(0.0, self.rate_limit_pause_until_epoch() - time.time())
        if remaining <= 0:
            self._pause_execution_retry_budget = self._pause_execution_retry_budget_max
        return remaining

    def set_external_rate_limit_pause_until(
        self,
        *,
        pause_until_epoch: float,
        source: str = "external",
    ) -> None:
        normalized_pause_until = max(0.0, float(pause_until_epoch))
        if normalized_pause_until <= self._rate_limit_pause_until_epoch:
            return
        self._rate_limit_pause_until_epoch = normalized_pause_until
        self._pause_execution_retry_budget = self._pause_execution_retry_budget_max
        remaining = max(0.0, normalized_pause_until - time.time())
        if remaining > 0:
            log_event(
                self._logger,
                level="info",
                event="quote_rate_limit_pause_synced",
                message="Quote rate-limit pause was synchronized from shared state",
                source=source,
                pause_until_epoch=round(normalized_pause_until, 3),
                remaining_seconds=round(remaining, 3),
            )

    def _apply_rate_limit_pause(
        self,
        *,
        provider: str,
        retry_after_seconds: float | None,
        status: int | None = None,
    ) -> float:
        sanitized_retry_after = (
            float(retry_after_seconds)
            if retry_after_seconds is not None and retry_after_seconds > 0
            else 0.0
        )
        if sanitized_retry_after > 0:
            pause_seconds = min(
                self._max_rate_limit_pause_seconds,
                sanitized_retry_after + random.uniform(0.0, 0.5),
            )
            self._rate_limit_backoff_level = min(3, self._rate_limit_backoff_level + 1)
        else:
            backoff_level = min(3, self._rate_limit_backoff_level)
            base_pause_seconds = self._default_rate_limit_pause_seconds * (2**backoff_level)
            jitter_seconds = random.uniform(0.0, max(0.5, base_pause_seconds * 0.15))
            pause_seconds = min(
                self._max_rate_limit_pause_seconds,
                base_pause_seconds + jitter_seconds,
            )
            self._rate_limit_backoff_level = min(3, self._rate_limit_backoff_level + 1)

        pause_until_epoch = time.time() + pause_seconds
        self._rate_limit_pause_until_epoch = max(
            self._rate_limit_pause_until_epoch,
            pause_until_epoch,
        )
        self._pause_execution_retry_budget = self._pause_execution_retry_budget_max
        log_event(
            self._logger,
            level="warning",
            event="quote_rate_limit_pause_armed",
            message="Quote rate limit pause was armed after API throttling",
            provider=provider,
            status=status,
            retry_after_seconds=(
                round(sanitized_retry_after, 3) if sanitized_retry_after > 0 else None
            ),
            pause_seconds=round(pause_seconds, 3),
            pause_until_epoch=round(self._rate_limit_pause_until_epoch, 3),
            rate_limited_count=self._rate_limited_count,
            rate_limit_backoff_level=self._rate_limit_backoff_level,
        )
        log_event(
            self._logger,
            level="warning",
            event="rate_limit_pause_set",
            message="Quote provider pause window has been set due to rate limiting",
            provider=provider,
            status=status,
            retry_after_seconds=(
                round(sanitized_retry_after, 3) if sanitized_retry_after > 0 else None
            ),
            backoff_seconds=round(pause_seconds, 3),
            pause_until=round(self._rate_limit_pause_until_epoch, 3),
        )
        return pause_seconds

    def _reset_rate_limit_pause_backoff(self) -> None:
        self._rate_limit_backoff_level = 0

    async def quote(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
        extra_params: dict[str, str | int | float | bool] | None = None,
        request_purpose: str = "exploration",
        allow_during_pause: bool = False,
    ) -> dict[str, Any]:
        if self._session is None:
            await self.connect()
        if self._session is None:
            raise RuntimeError("Helius quote HTTP session is not initialized.")

        self._quote_call_count += 1

        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
        }
        if extra_params:
            for key, value in extra_params.items():
                normalized_key = str(key).strip()
                if not normalized_key:
                    continue
                if isinstance(value, bool):
                    params[normalized_key] = "true" if value else "false"
                    continue
                if isinstance(value, int):
                    params[normalized_key] = str(value)
                    continue
                if isinstance(value, float):
                    params[normalized_key] = str(int(value) if value.is_integer() else value)
                    continue
                normalized_value = str(value).strip()
                if not normalized_value:
                    continue
                lowered = normalized_value.lower()
                params[normalized_key] = lowered if lowered in {"true", "false"} else normalized_value

        loop = asyncio.get_running_loop()
        now = loop.time()
        cache_key = self._quote_cache_key(self._quote_endpoint, params)
        self._prune_quote_cache(now)
        cached_quote = self._quote_cache.get(cache_key)
        if cached_quote and cached_quote[0] > now:
            self._quote_cache_hits += 1
            return dict(cached_quote[1])

        last_error: Exception | None = None
        max_attempts = self._max_retries + 1
        normalized_purpose = str(request_purpose or "exploration").strip().lower()
        is_execution_purpose = normalized_purpose == "execution"

        for attempt in range(1, max_attempts + 1):
            headers = self._build_headers()
            provider = _quote_provider_from_endpoint(self._quote_endpoint)
            try:
                async with self._quote_request_semaphore:
                    now = loop.time()
                    self._prune_quote_cache(now)
                    cached_quote = self._quote_cache.get(cache_key)
                    if cached_quote and cached_quote[0] > now:
                        self._quote_cache_hits += 1
                        return dict(cached_quote[1])
                    pause_remaining_seconds = self.rate_limit_pause_remaining_seconds()
                    if pause_remaining_seconds > 0:
                        if (
                            is_execution_purpose
                            and allow_during_pause
                            and self._pause_execution_retry_budget > 0
                        ):
                            self._pause_execution_retry_budget -= 1
                            log_event(
                                self._logger,
                                level="warning",
                                event="quote_rate_limit_execution_retry_allowed",
                                message=(
                                    "Executing a single quote retry during rate-limit pause "
                                    "because execution was already in-flight."
                                ),
                                provider=provider,
                                pause_remaining_seconds=round(pause_remaining_seconds, 3),
                                remaining_retry_budget=self._pause_execution_retry_budget,
                            )
                        else:
                            raise HeliusRateLimitError(
                                "Quote requests are paused due to provider rate limiting",
                                retry_after_seconds=pause_remaining_seconds,
                                provider=provider,
                            )
                    await self._consume_quote_tokens(
                        is_execution_purpose=is_execution_purpose,
                    )

                    async with self._session.get(self._quote_endpoint, params=params, headers=headers) as response:
                        status = response.status
                        retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
                        body = await response.text()
                    self._note_quote_network_request(loop.time())
            except aiohttp.ClientError as error:
                last_error = error
                if attempt < max_attempts:
                    log_event(
                        self._logger,
                        level="warning",
                        event="helius_quote_network_retry",
                        message="Quote request failed; retrying",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=str(error),
                    )
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(f"Quote request failed: {error}") from error

            data: Any = None
            parse_error: Exception | None = None
            try:
                data = json.loads(body)
            except json.JSONDecodeError as error:
                parse_error = error

            if isinstance(data, dict) and "error" in data:
                helius_error = _error_message_from_payload(data.get("error"))
                is_proxy_method_error = _is_helius_jup_proxy_path(
                    urlsplit(self._quote_endpoint).path
                ) and _is_method_not_found_error(helius_error)
                error_event = "helius_rpc_error" if provider == "helius" else "quote_api_error_payload"
                error_message = (
                    "Helius response includes RPC error payload"
                    if provider == "helius"
                    else "Quote API response includes error payload"
                )

                log_event(
                    self._logger,
                    level="warning",
                    event=error_event,
                    message=error_message,
                    status=status,
                    error=helius_error,
                    provider=provider,
                    body_preview=_sanitize_preview_for_log(body),
                )
                if is_proxy_method_error and self._switch_quote_endpoint(
                    reason=helius_error,
                    event="helius_jup_proxy_unavailable",
                    body_preview=body,
                ):
                    continue
                if _is_no_routes_error_text(helius_error):
                    raise QuoteNoRoutesError(f"NO_ROUTES_FOUND: {helius_error}")
                if "rate limit" in helius_error.lower():
                    self._rate_limited_count += 1
                    pause_seconds = self._apply_rate_limit_pause(
                        provider=provider,
                        retry_after_seconds=retry_after_seconds,
                        status=status,
                    )
                    raise HeliusRateLimitError(
                        f"Quote failed with {provider} rate limit: {helius_error}",
                        retry_after_seconds=pause_seconds,
                        provider=provider,
                    )
                raise RuntimeError(f"Quote API error ({provider}): {helius_error}")

            if status == 429:
                body_preview = _sanitize_preview_for_log(body, limit=300)
                self._rate_limited_count += 1
                pause_seconds = self._apply_rate_limit_pause(
                    provider=provider,
                    retry_after_seconds=retry_after_seconds,
                    status=status,
                )
                raise HeliusRateLimitError(
                    f"Quote failed: status={status} non_json_body={body_preview!r}",
                    retry_after_seconds=pause_seconds,
                    provider=provider,
                )

            retryable_statuses = {500, 502, 503, 504}
            if status in retryable_statuses and attempt < max_attempts:
                sleep_seconds = (
                    retry_after_seconds
                    if retry_after_seconds is not None
                    else self._retry_backoff_seconds * attempt
                )
                log_event(
                    self._logger,
                    level="warning",
                    event="helius_quote_retry",
                    message="Quote endpoint returned retryable status",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status=status,
                    retry_after_seconds=sleep_seconds,
                    body_preview=_sanitize_preview_for_log(body, limit=200),
                )
                await asyncio.sleep(sleep_seconds)
                continue

            if status >= 400:
                body_preview = _sanitize_preview_for_log(body, limit=300)
                if status == 404 and _is_helius_jup_proxy_path(urlsplit(self._quote_endpoint).path):
                    if self._switch_quote_endpoint(
                        reason="status=404",
                        event="helius_quote_endpoint_not_found",
                        body_preview=body_preview,
                    ):
                        continue
                if status in {401, 403}:
                    raise RuntimeError(
                        "Quote API authentication failed. Set HELIUS_API_KEY for the quote provider."
                    )
                if isinstance(data, dict) and data:
                    if _is_no_routes_error_text(str(data)):
                        raise QuoteNoRoutesError(f"NO_ROUTES_FOUND: {data}")
                    raise RuntimeError(f"Quote failed: status={status} body={data}")
                if _is_no_routes_error_text(body_preview):
                    raise QuoteNoRoutesError(f"NO_ROUTES_FOUND: {body_preview}")
                raise RuntimeError(f"Quote failed: status={status} non_json_body={body_preview!r}")

            if not isinstance(data, dict):
                if parse_error is None:
                    parse_error = RuntimeError("response is not a JSON object")
                last_error = parse_error
                if attempt < max_attempts:
                    log_event(
                        self._logger,
                        level="warning",
                        event="helius_quote_parse_retry",
                        message="Quote response was not valid JSON; retrying",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        status=status,
                        error=str(parse_error),
                        body_preview=_sanitize_preview_for_log(body, limit=200),
                    )
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)
                    continue
                raise RuntimeError(
                    "Quote endpoint returned non-JSON response: "
                    f"status={status} body={_sanitize_preview_for_log(body, limit=300)!r}"
                ) from parse_error

            if "outAmount" not in data:
                raise RuntimeError(f"Unexpected quote response: {data}")

            self._reset_rate_limit_pause_backoff()
            if self._quote_cache_ttl_seconds > 0:
                expires_at = loop.time() + self._quote_cache_ttl_seconds
                self._quote_cache[cache_key] = (expires_at, dict(data))
            return data

        raise RuntimeError(f"Quote request failed after retries: {last_error}")

    def _initial_exploration_strategies(
        self,
        *,
        base_quote_params: dict[str, Any],
        dex_allowlist: tuple[str, ...],
        dex_excludelist: tuple[str, ...],
        strategy_limit: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        strategies: list[tuple[str, dict[str, Any]]] = []

        excluded_set = {
            _normalize_dex_label(dex)
            for dex in dex_excludelist
            if _normalize_dex_label(dex)
        }
        filtered_allowlist = tuple(
            _normalize_dex_label(dex)
            for dex in dex_allowlist
            if _normalize_dex_label(dex) and _normalize_dex_label(dex) not in excluded_set
        )
        exclude_dexes = self._to_dex_csv(tuple(excluded_set))

        grouped_dexes = self._group_dex_labels(
            filtered_allowlist,
            group_sizes=(3, 2),
            limit=max(1, strategy_limit),
        )
        for dex_group in grouped_dexes:
            dexes_csv = self._to_dex_csv(dex_group)
            if not dexes_csv:
                continue
            overrides: dict[str, Any] = {"dexes": dexes_csv}
            if exclude_dexes:
                overrides["excludeDexes"] = exclude_dexes
            strategies.append((f"dex_group:{dexes_csv}", overrides))

        if not strategies:
            overrides: dict[str, Any] = {}
            if exclude_dexes:
                overrides["excludeDexes"] = exclude_dexes
            strategies.append(("base", overrides))
        return strategies

    async def _collect_quote_candidates(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
        base_quote_params: dict[str, Any],
        direction: str,
        dex_allowlist: tuple[str, ...],
        dex_excludelist: tuple[str, ...],
        top_k: int,
        request_purpose: str = "exploration",
        allow_during_pause: bool = False,
        deadline_epoch: float | None = None,
        exploration_call_budget: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_candidate_keys: set[tuple[str, int]] = set()
        last_error: Exception | None = None

        for strategy_name, overrides in self._initial_exploration_strategies(
            base_quote_params=base_quote_params,
            dex_allowlist=dex_allowlist,
            dex_excludelist=dex_excludelist,
            strategy_limit=max(6, top_k * 3),
        ):
            params = dict(base_quote_params)
            params.update(overrides)
            amount_bucket = self._amount_bucket(amount)
            strategy_cache_key = f"{direction}:{strategy_name}:{amount_bucket}"
            now = asyncio.get_running_loop().time()
            self._prune_no_routes_cache(now)
            if self._is_no_routes_cached(key=strategy_cache_key, now=now):
                log_event(
                    self._logger,
                    level="info",
                    event="spread_exploration_no_routes_cached_skip",
                    message="Skipping quote strategy because NO_ROUTES result is cached",
                    direction=direction,
                    strategy=strategy_name,
                    amount_bucket=amount_bucket,
                )
                continue
            try:
                if deadline_epoch is not None and asyncio.get_running_loop().time() >= float(deadline_epoch):
                    break
                if exploration_call_budget is not None:
                    remaining = int(exploration_call_budget.get("remaining", 0))
                    if remaining <= 0:
                        break
                    exploration_call_budget["remaining"] = remaining - 1
                quote = await self.quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=amount,
                    slippage_bps=slippage_bps,
                    extra_params=params,
                    request_purpose=request_purpose,
                    allow_during_pause=allow_during_pause,
                )
            except QuoteNoRoutesError as error:
                last_error = error
                self._mark_no_routes_failure(key=strategy_cache_key, now=now)
                log_event(
                    self._logger,
                    level="info",
                    event="spread_exploration_no_routes",
                    message="Spread exploration strategy returned NO_ROUTES and was cached",
                    direction=direction,
                    strategy=strategy_name,
                    amount_bucket=amount_bucket,
                    ttl_seconds=round(self._no_routes_cache_ttl_seconds, 3),
                    error=str(error),
                )
            except HeliusRateLimitError:
                raise
            except Exception as error:
                last_error = error
                log_event(
                    self._logger,
                    level="warning",
                    event="spread_exploration_quote_failed",
                    message="Spread exploration quote candidate failed",
                    direction=direction,
                    strategy=strategy_name,
                    amount=amount,
                    error=str(error),
                )
                continue
            else:
                out_amount = int(quote.get("outAmount") or 0)
                if out_amount <= 0:
                    continue

                route_fingerprint = _quote_route_fingerprint(quote)
                candidate_key = (route_fingerprint, out_amount)
                if candidate_key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(candidate_key)

                candidates.append(
                    {
                        "strategy": strategy_name,
                        "params": params,
                        "quote": quote,
                        "out_amount": out_amount,
                        "route_fingerprint": route_fingerprint,
                        "route_dexes": _extract_route_dexes(quote),
                    }
                )
                if len(candidates) >= top_k:
                    break

        if len(candidates) < top_k:
            fallback_base_params = dict(base_quote_params)
            fallback_base_params.pop("dexes", None)
            fallback_base_params.pop("excludeDexes", None)
            fallback_base_params.pop("exclude_dexes", None)
            fallback_max_accounts = []
            for candidate in [fallback_base_params.get("maxAccounts"), 64, 48, 32]:
                try:
                    parsed = int(candidate)
                except (TypeError, ValueError):
                    continue
                if parsed > 0 and parsed not in fallback_max_accounts:
                    fallback_max_accounts.append(parsed)
            if not fallback_max_accounts:
                fallback_max_accounts = [64]

            amount_bucket = self._amount_bucket(amount)
            for max_accounts in fallback_max_accounts:
                if len(candidates) >= top_k:
                    break
                fallback_params = dict(fallback_base_params)
                fallback_params["maxAccounts"] = max_accounts
                strategy_name = f"fallback_unconstrained_max_accounts_{max_accounts}"
                strategy_cache_key = f"{direction}:{strategy_name}:{amount_bucket}"
                now = asyncio.get_running_loop().time()
                if self._is_no_routes_cached(key=strategy_cache_key, now=now):
                    continue
                try:
                    if deadline_epoch is not None and asyncio.get_running_loop().time() >= float(deadline_epoch):
                        break
                    if exploration_call_budget is not None:
                        remaining = int(exploration_call_budget.get("remaining", 0))
                        if remaining <= 0:
                            break
                        exploration_call_budget["remaining"] = remaining - 1
                    quote = await self.quote(
                        input_mint=input_mint,
                        output_mint=output_mint,
                        amount=amount,
                        slippage_bps=slippage_bps,
                        extra_params=fallback_params,
                        request_purpose=request_purpose,
                        allow_during_pause=allow_during_pause,
                    )
                    out_amount = int(quote.get("outAmount") or 0)
                    if out_amount <= 0:
                        continue
                    route_fingerprint = _quote_route_fingerprint(quote)
                    candidate_key = (route_fingerprint, out_amount)
                    if candidate_key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(candidate_key)
                    candidates.append(
                        {
                            "strategy": strategy_name,
                            "params": fallback_params,
                            "quote": quote,
                            "out_amount": out_amount,
                            "route_fingerprint": route_fingerprint,
                            "route_dexes": _extract_route_dexes(quote),
                        }
                    )
                except QuoteNoRoutesError as error:
                    last_error = error
                    self._mark_no_routes_failure(key=strategy_cache_key, now=now)
                    log_event(
                        self._logger,
                        level="info",
                        event="spread_exploration_no_routes",
                        message="Fallback unconstrained quote returned NO_ROUTES and was cached",
                        direction=direction,
                        strategy=strategy_name,
                        amount_bucket=amount_bucket,
                        ttl_seconds=round(self._no_routes_cache_ttl_seconds, 3),
                        error=str(error),
                    )
                except HeliusRateLimitError:
                    raise
                except Exception as error:
                    last_error = error

        if not candidates and last_error is not None:
            raise RuntimeError(f"Spread exploration produced no {direction} candidates: {last_error}") from last_error

        candidates.sort(key=lambda item: int(item.get("out_amount") or 0), reverse=True)
        return candidates[: max(1, int(top_k))]

    async def _find_best_combo(
        self,
        *,
        pair: PairConfig,
        base_quote_params: dict[str, Any],
        dex_allowlist_forward: tuple[str, ...],
        dex_allowlist_reverse: tuple[str, ...],
        dex_excludelist: tuple[str, ...],
        resolved_top_k: int,
        resolved_combo_limit: int,
        request_purpose: str = "exploration",
        allow_during_pause: bool = False,
        deadline_epoch: float | None = None,
        exploration_call_budget: dict[str, int] | None = None,
    ) -> tuple[dict[str, Any] | None, int, int, list[dict[str, Any]], list[dict[str, Any]], int]:
        forward_candidates = await self._collect_quote_candidates(
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount=pair.base_amount,
            slippage_bps=pair.slippage_bps,
            base_quote_params=base_quote_params,
            direction="forward",
            dex_allowlist=dex_allowlist_forward,
            dex_excludelist=dex_excludelist,
            top_k=resolved_top_k,
            request_purpose=request_purpose,
            allow_during_pause=allow_during_pause,
            deadline_epoch=deadline_epoch,
            exploration_call_budget=exploration_call_budget,
        )
        if not forward_candidates:
            return None, 0, 0, [], [], 0

        best_combo: dict[str, Any] | None = None
        reverse_candidate_count = 0
        evaluated_combo_count = 0
        all_reverse_candidates: list[dict[str, Any]] = []
        route_cooldown_skipped_count = 0
        now = asyncio.get_running_loop().time()

        for forward_candidate in forward_candidates:
            if deadline_epoch is not None and asyncio.get_running_loop().time() >= float(deadline_epoch):
                break
            reverse_candidates = await self._collect_quote_candidates(
                input_mint=pair.quote_mint,
                output_mint=pair.base_mint,
                amount=int(forward_candidate["out_amount"]),
                slippage_bps=pair.slippage_bps,
                base_quote_params=base_quote_params,
                direction="reverse",
                dex_allowlist=dex_allowlist_reverse,
                dex_excludelist=dex_excludelist,
                top_k=resolved_top_k,
                request_purpose=request_purpose,
                allow_during_pause=allow_during_pause,
                deadline_epoch=deadline_epoch,
                exploration_call_budget=exploration_call_budget,
            )
            reverse_candidate_count += len(reverse_candidates)
            all_reverse_candidates.extend(reverse_candidates)

            for reverse_candidate in reverse_candidates:
                if evaluated_combo_count >= resolved_combo_limit:
                    break
                forward_route_hash = _route_hash_from_fingerprint(
                    str(forward_candidate.get("route_fingerprint") or "unknown")
                )
                reverse_route_hash = _route_hash_from_fingerprint(
                    str(reverse_candidate.get("route_fingerprint") or "unknown")
                )
                if self._is_route_pair_cooling(
                    forward_route_hash=forward_route_hash,
                    reverse_route_hash=reverse_route_hash,
                    now=now,
                ):
                    route_cooldown_skipped_count += 1
                    continue
                reverse_out = int(reverse_candidate["out_amount"])
                spread_bps = ((reverse_out - pair.base_amount) / pair.base_amount) * 10_000
                evaluated_combo_count += 1
                if best_combo is None or spread_bps > float(best_combo["spread_bps"]):
                    best_combo = {
                        "spread_bps": spread_bps,
                        "forward_route_hash": forward_route_hash,
                        "reverse_route_hash": reverse_route_hash,
                        "forward": forward_candidate,
                        "reverse": reverse_candidate,
                    }

            if evaluated_combo_count >= resolved_combo_limit:
                break

        return (
            best_combo,
            reverse_candidate_count,
            evaluated_combo_count,
            forward_candidates,
            all_reverse_candidates,
            route_cooldown_skipped_count,
        )

    async def _requote_best_combo_median(
        self,
        *,
        pair: PairConfig,
        best_combo: dict[str, Any],
        sample_count: int = 3,
        request_purpose: str = "exploration",
        allow_during_pause: bool = False,
        deadline_epoch: float | None = None,
        exploration_call_budget: dict[str, int] | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        forward_params = dict(best_combo["forward"]["params"])
        reverse_params = dict(best_combo["reverse"]["params"])
        samples: list[dict[str, Any]] = []
        sample_entries: list[dict[str, Any]] = []

        for _attempt in range(max(1, int(sample_count))):
            try:
                if deadline_epoch is not None and asyncio.get_running_loop().time() >= float(deadline_epoch):
                    break
                if exploration_call_budget is not None:
                    remaining = int(exploration_call_budget.get("remaining", 0))
                    if remaining <= 0:
                        break
                    exploration_call_budget["remaining"] = remaining - 1
                forward_quote = await self.quote(
                    input_mint=pair.base_mint,
                    output_mint=pair.quote_mint,
                    amount=pair.base_amount,
                    slippage_bps=pair.slippage_bps,
                    extra_params=forward_params,
                    request_purpose=request_purpose,
                    allow_during_pause=allow_during_pause,
                )
                forward_out = int(forward_quote.get("outAmount") or 0)
                if forward_out <= 0:
                    continue

                if deadline_epoch is not None and asyncio.get_running_loop().time() >= float(deadline_epoch):
                    break
                if exploration_call_budget is not None:
                    remaining = int(exploration_call_budget.get("remaining", 0))
                    if remaining <= 0:
                        break
                    exploration_call_budget["remaining"] = remaining - 1
                reverse_quote = await self.quote(
                    input_mint=pair.quote_mint,
                    output_mint=pair.base_mint,
                    amount=forward_out,
                    slippage_bps=pair.slippage_bps,
                    extra_params=reverse_params,
                    request_purpose=request_purpose,
                    allow_during_pause=allow_during_pause,
                )
                reverse_out = int(reverse_quote.get("outAmount") or 0)
                if reverse_out <= 0:
                    continue

                spread_bps = ((reverse_out - pair.base_amount) / pair.base_amount) * 10_000
                forward_route_fingerprint = _quote_route_fingerprint(forward_quote)
                reverse_route_fingerprint = _quote_route_fingerprint(reverse_quote)
                sample_entries.append(
                    {
                        "spread_bps": spread_bps,
                        "forward_route_fingerprint": forward_route_fingerprint,
                        "reverse_route_fingerprint": reverse_route_fingerprint,
                        "forward_route_hash": _route_hash_from_fingerprint(forward_route_fingerprint),
                        "reverse_route_hash": _route_hash_from_fingerprint(reverse_route_fingerprint),
                    }
                )
                samples.append(
                    {
                        "spread_bps": spread_bps,
                        "forward": {
                            "strategy": "median_requote",
                            "params": forward_params,
                            "quote": forward_quote,
                            "out_amount": forward_out,
                            "route_fingerprint": forward_route_fingerprint,
                            "route_dexes": _extract_route_dexes(forward_quote),
                        },
                        "reverse": {
                            "strategy": "median_requote",
                            "params": reverse_params,
                            "quote": reverse_quote,
                            "out_amount": reverse_out,
                            "route_fingerprint": reverse_route_fingerprint,
                            "route_dexes": _extract_route_dexes(reverse_quote),
                        },
                    }
                )
            except HeliusRateLimitError:
                raise
            except Exception:
                continue

        if not samples:
            return None, sample_entries
        samples.sort(key=lambda item: float(item["spread_bps"]))
        return samples[len(samples) // 2], sample_entries

    async def fetch_spread(
        self,
        pair: PairConfig,
        *,
        quote_params: dict[str, Any] | None = None,
        quote_params_source: str = "",
        dex_allowlist_forward: tuple[str, ...] = (),
        dex_allowlist_reverse: tuple[str, ...] = (),
        dex_excludelist: tuple[str, ...] = (),
        exploration_mode: str = "MIXED",
        sweep_top_k: int | None = None,
        sweep_top_k_max: int | None = None,
        sweep_combo_limit: int | None = None,
        near_miss_expand_bps: float | None = None,
        median_requote_max_range_bps: float | None = None,
        min_improvement_bps: float | None = None,
        quote_max_rps: float | None = None,
        quote_exploration_max_rps: float | None = None,
        quote_execution_max_rps: float | None = None,
        quote_cache_ttl_ms: int | None = None,
        no_routes_cache_ttl_seconds: float | None = None,
        probe_max_routes: int | None = None,
        probe_base_amounts_raw: tuple[int, ...] | None = None,
        dynamic_allowlist_topk: int | None = None,
        dynamic_allowlist_good_candidate_alpha: float | None = None,
        dynamic_allowlist_ttl_seconds: float | None = None,
        dynamic_allowlist_refresh_seconds: float | None = None,
        negative_fallback_streak_threshold: int | None = None,
        enable_probe_unconstrained: bool | None = None,
        enable_probe_multi_amount: bool | None = None,
        enable_stagea_relaxed_gate: bool | None = None,
        enable_route_instability_cooldown: bool | None = None,
        route_instability_cooldown_requote_seconds: float | None = None,
        route_instability_cooldown_decay_requote_bps: float | None = None,
        stage_a_required_bps: float | None = None,
        stage_a_min_margin_bps: float | None = None,
        stage_a_required_components: dict[str, float] | None = None,
        stage_b_required_bps_without_tip: float | None = None,
        stage_b_tip_share: float | None = None,
        stage_b_tip_lamports_max: int | None = None,
        stage_b_required_components: dict[str, float] | None = None,
    ) -> SpreadObservation:
        # Keep defaults initialized so diagnostics can safely reference them
        # even when exploration exits early or raises.
        stage_a_pass: bool | None = None
        stage_b_pass: bool | None = None
        stage_a_margin_bps: float | None = None
        stage_b_margin_bps: float | None = None
        fail_reason = ""
        median_requote_applied = False
        median_requote_sample_count = 0
        median_requote_range_bps: float | None = None
        requote_spread_samples_bps: list[float] = []
        requote_samples_route_hashes_forward: list[str] = []
        requote_samples_route_hashes_reverse: list[str] = []
        requote_median_spread_bps: float | None = None
        best_spread_pre_requote: float | None = None
        best_spread_post_requote: float | None = None

        self._apply_runtime_limits(
            quote_max_rps=quote_max_rps,
            quote_exploration_max_rps=quote_exploration_max_rps,
            quote_execution_max_rps=quote_execution_max_rps,
            quote_cache_ttl_ms=quote_cache_ttl_ms,
            no_routes_cache_ttl_seconds=no_routes_cache_ttl_seconds,
            probe_max_routes=probe_max_routes,
            probe_base_amounts_raw=probe_base_amounts_raw,
            sweep_top_k_max=sweep_top_k_max,
            near_miss_expand_bps=near_miss_expand_bps,
            median_requote_max_range_bps=median_requote_max_range_bps,
            min_improvement_bps=min_improvement_bps,
            dynamic_allowlist_topk=dynamic_allowlist_topk,
            dynamic_allowlist_good_candidate_alpha=dynamic_allowlist_good_candidate_alpha,
            dynamic_allowlist_ttl_seconds=dynamic_allowlist_ttl_seconds,
            dynamic_allowlist_refresh_seconds=dynamic_allowlist_refresh_seconds,
            negative_fallback_streak_threshold=negative_fallback_streak_threshold,
            enable_probe_unconstrained=enable_probe_unconstrained,
            enable_probe_multi_amount=enable_probe_multi_amount,
            enable_stagea_relaxed_gate=enable_stagea_relaxed_gate,
            enable_route_instability_cooldown=enable_route_instability_cooldown,
            route_instability_cooldown_requote_seconds=route_instability_cooldown_requote_seconds,
            route_instability_cooldown_decay_requote_bps=route_instability_cooldown_decay_requote_bps,
        )
        base_quote_params: dict[str, Any] = _normalize_quote_params(
            quote_params if quote_params is not None else self._default_quote_params
        )
        resolved_top_k = max(1, int(sweep_top_k or self._exploration_top_n))
        resolved_top_k_max = max(resolved_top_k, int(self._sweep_top_k_max))
        resolved_combo_limit = max(
            1,
            int(sweep_combo_limit or (resolved_top_k * resolved_top_k)),
        )
        resolved_near_miss_expand_bps = max(0.0, float(self._near_miss_expand_bps))
        near_miss_expand_enabled = resolved_near_miss_expand_bps < 100.0
        resolved_median_requote_max_range_bps = max(0.0, float(self._median_requote_max_range_bps))
        if stage_a_required_bps is not None:
            resolved_median_requote_max_range_bps = max(
                resolved_median_requote_max_range_bps,
                0.25 * max(0.0, float(stage_a_required_bps)),
            )
        resolved_min_improvement_bps = max(0.0, float(self._min_improvement_bps))
        resolved_stage_a_min_margin_bps = max(0.0, float(stage_a_min_margin_bps or 0.0))
        normalized_mode = "TIER1_ONLY" if str(exploration_mode or "").strip().upper() == "TIER1_ONLY" else "MIXED"
        dynamic_dex_labels = await self._refresh_dynamic_dex_labels(
            pair=pair,
            base_quote_params=base_quote_params,
            quote_params_source=quote_params_source,
        )
        dynamic_allowlist_for_mode = dynamic_dex_labels if normalized_mode == "MIXED" else ()
        effective_allowlist_forward = self._effective_allowlist(
            static_labels=dex_allowlist_forward,
            dynamic_labels=dynamic_allowlist_for_mode,
        )
        effective_allowlist_reverse = self._effective_allowlist(
            static_labels=dex_allowlist_reverse,
            dynamic_labels=dynamic_allowlist_for_mode,
        )
        if not effective_allowlist_reverse:
            effective_allowlist_reverse = effective_allowlist_forward

        route_cooldown_skipped_count = 0
        try:
            (
                best_combo,
                reverse_candidate_count,
                evaluated_combo_count,
                forward_candidates,
                _reverse_candidates,
                route_cooldown_skipped_count,
            ) = await self._find_best_combo(
                pair=pair,
                base_quote_params=base_quote_params,
                dex_allowlist_forward=effective_allowlist_forward,
                dex_allowlist_reverse=effective_allowlist_reverse,
                dex_excludelist=dex_excludelist,
                resolved_top_k=resolved_top_k,
                resolved_combo_limit=resolved_combo_limit,
            )
        except Exception as error:
            log_event(
                self._logger,
                level="warning",
                event="spread_exploration_failed",
                message="Spread exploration failed before best-combo selection",
                mode=normalized_mode,
                fail_reason=fail_reason or "NONE",
                stageA_pass=stage_a_pass,
                stageB_pass=stage_b_pass,
                error=str(error),
            )
            raise

        if best_combo is None:
            log_event(
                self._logger,
                level="warning",
                event="spread_exploration_no_combo",
                message="Spread exploration did not produce any forward/reverse combination",
                mode=normalized_mode,
                fail_reason=fail_reason or "NONE",
                stageA_pass=stage_a_pass,
                stageB_pass=stage_b_pass,
                route_candidate_count=len(forward_candidates) + reverse_candidate_count,
                route_sampled_count=evaluated_combo_count,
            )
            raise RuntimeError("Spread exploration did not produce any forward/reverse combination.")

        if len(forward_candidates) <= 1 or reverse_candidate_count <= 1:
            log_event(
                self._logger,
                level="info",
                event="spread_exploration_candidate_thin",
                message="Spread exploration produced thin candidate depth",
                mode=normalized_mode,
                route_candidate_count=len(forward_candidates) + reverse_candidate_count,
                route_sampled_count=evaluated_combo_count,
                candidate_count_forward=len(forward_candidates),
                candidate_count_reverse=reverse_candidate_count,
                sweep_top_k=resolved_top_k,
                sweep_top_k_max=resolved_top_k_max,
                sweep_combo_limit=resolved_combo_limit,
                probe_max_routes=self._probe_max_routes,
                allowlist_forward_size=len(effective_allowlist_forward),
                allowlist_reverse_size=len(effective_allowlist_reverse),
            )

        spread_bps = float(best_combo["spread_bps"])
        if (
            near_miss_expand_enabled
            and
            stage_a_required_bps is not None
            and spread_bps < float(stage_a_required_bps)
            and (float(stage_a_required_bps) - spread_bps) <= resolved_near_miss_expand_bps
            and resolved_top_k < resolved_top_k_max
        ):
            expanded_top_k = min(resolved_top_k_max, max(resolved_top_k + 2, resolved_top_k + 1))
            expanded_combo_limit = max(resolved_combo_limit, expanded_top_k * expanded_top_k)
            (
                expanded_combo,
                expanded_reverse_count,
                expanded_eval_count,
                expanded_forward_candidates,
                _,
                expanded_route_cooldown_skipped_count,
            ) = await self._find_best_combo(
                pair=pair,
                base_quote_params=base_quote_params,
                dex_allowlist_forward=effective_allowlist_forward,
                dex_allowlist_reverse=effective_allowlist_reverse,
                dex_excludelist=dex_excludelist,
                resolved_top_k=expanded_top_k,
                resolved_combo_limit=expanded_combo_limit,
            )
            expanded_spread_bps = (
                float(expanded_combo["spread_bps"]) if expanded_combo is not None else spread_bps
            )
            improved = bool(
                expanded_combo is not None
                and expanded_spread_bps >= (spread_bps + resolved_min_improvement_bps)
            )
            log_event(
                self._logger,
                level="info",
                event="spread_exploration_topk_expanded",
                message="Expanded spread exploration top-k for near-threshold candidate",
                mode=normalized_mode,
                improved=improved,
                previous_best_spread_bps=round(spread_bps, 6),
                expanded_best_spread_bps=round(expanded_spread_bps, 6),
                min_improvement_bps=round(resolved_min_improvement_bps, 6),
                stageA_required_spread_bps=round(float(stage_a_required_bps), 6),
                near_miss_expand_bps=round(resolved_near_miss_expand_bps, 6),
                sweep_top_k=resolved_top_k,
                sweep_top_k_expanded=expanded_top_k,
                sweep_combo_limit=resolved_combo_limit,
                sweep_combo_limit_expanded=expanded_combo_limit,
            )
            if improved and expanded_combo is not None:
                best_combo = expanded_combo
                spread_bps = expanded_spread_bps
                resolved_top_k = expanded_top_k
                resolved_combo_limit = expanded_combo_limit
                forward_candidates = expanded_forward_candidates
                reverse_candidate_count = expanded_reverse_count
                evaluated_combo_count = expanded_eval_count
                route_cooldown_skipped_count = expanded_route_cooldown_skipped_count

        if spread_bps < 0:
            self._negative_best_spread_streak += 1
        else:
            self._negative_best_spread_streak = 0

        should_probe_unconstrained = (
            self._enable_probe_unconstrained
            and
            self._negative_best_spread_streak >= self._negative_fallback_streak_threshold
            and self._negative_best_spread_streak % self._negative_fallback_streak_threshold == 0
            and (effective_allowlist_forward or effective_allowlist_reverse)
        )
        if should_probe_unconstrained:
            unconstrained_combo, _, unconstrained_eval_count, _, _, _ = await self._find_best_combo(
                pair=pair,
                base_quote_params=base_quote_params,
                dex_allowlist_forward=(),
                dex_allowlist_reverse=(),
                dex_excludelist=dex_excludelist,
                resolved_top_k=resolved_top_k,
                resolved_combo_limit=resolved_combo_limit,
            )
            unconstrained_spread_bps = (
                float(unconstrained_combo["spread_bps"]) if unconstrained_combo is not None else None
            )
            improved = bool(
                unconstrained_combo is not None
                and unconstrained_spread_bps is not None
                and unconstrained_spread_bps > spread_bps
            )
            log_event(
                self._logger,
                level="info",
                event="spread_negative_fallback_probe",
                message="Negative spread streak triggered unconstrained fallback probe",
                constrained_best_spread_bps=round(spread_bps, 6),
                unconstrained_best_spread_bps=(
                    round(unconstrained_spread_bps, 6) if unconstrained_spread_bps is not None else None
                ),
                improved=improved,
                negative_streak_count=self._negative_best_spread_streak,
                streak_threshold=self._negative_fallback_streak_threshold,
                unconstrained_evaluated_combo_count=unconstrained_eval_count,
            )
            if improved and unconstrained_combo is not None:
                best_combo = unconstrained_combo
                spread_bps = float(best_combo["spread_bps"])
                self._negative_best_spread_streak = 0

        best_spread_pre_requote = spread_bps
        expected_net_bps_stage_a_pre = (
            spread_bps - float(stage_a_required_bps)
            if stage_a_required_bps is not None
            else None
        )
        log_event(
            self._logger,
            level="info",
            event="spread_scan_result",
            message="Spread scan result was captured before burst refinement",
            mode="scan",
            pre_requote_spread_bps=round(spread_bps, 6),
            stageA_expected_net_bps_pre=(
                round(float(expected_net_bps_stage_a_pre), 6)
                if expected_net_bps_stage_a_pre is not None
                else None
            ),
            route_label=(
                f"{best_combo['forward'].get('route_fingerprint', 'unknown')} -> "
                f"{best_combo['reverse'].get('route_fingerprint', 'unknown')}"
            ),
            amount_raw=pair.base_amount,
        )

        burst_trigger_reason = ""
        if best_spread_pre_requote is not None and best_spread_pre_requote >= 1.5:
            burst_trigger_reason = "best_spread_pre_requote>=1.5bps"
        elif spread_bps >= 1.2:
            burst_trigger_reason = "pre_requote_spread>=1.2bps"
        elif (
            expected_net_bps_stage_a_pre is not None
            and expected_net_bps_stage_a_pre >= 0.2
        ):
            burst_trigger_reason = "stageA_expected_net_pre>=0.2bps"

        if burst_trigger_reason:
            burst_duration_seconds = 2.0
            burst_deadline = asyncio.get_running_loop().time() + min(3.0, burst_duration_seconds)
            burst_budget: dict[str, int] = {"remaining": 3}
            burst_top_k = min(2, max(1, resolved_top_k_max))
            burst_combo_limit = min(2, max(1, burst_top_k * burst_top_k))
            log_event(
                self._logger,
                level="info",
                event="spread_burst_triggered",
                message="Burst refinement was triggered by pre-requote spread signal",
                mode="burst",
                trigger_reason=burst_trigger_reason,
                pre_requote_spread_bps=round(spread_bps, 6),
                stageA_expected_net_bps_pre=(
                    round(float(expected_net_bps_stage_a_pre), 6)
                    if expected_net_bps_stage_a_pre is not None
                    else None
                ),
                burst_duration_seconds=burst_duration_seconds,
                burst_quote_call_budget=burst_budget["remaining"],
                burst_top_k=burst_top_k,
                burst_combo_limit=burst_combo_limit,
            )

            (
                burst_combo,
                burst_reverse_candidate_count,
                burst_evaluated_combo_count,
                burst_forward_candidates,
                _burst_reverse_candidates,
                burst_route_cooldown_skipped_count,
            ) = await self._find_best_combo(
                pair=pair,
                base_quote_params=base_quote_params,
                dex_allowlist_forward=effective_allowlist_forward,
                dex_allowlist_reverse=effective_allowlist_reverse,
                dex_excludelist=dex_excludelist,
                resolved_top_k=burst_top_k,
                resolved_combo_limit=burst_combo_limit,
                request_purpose="exploration",
                allow_during_pause=False,
                deadline_epoch=burst_deadline,
                exploration_call_budget=burst_budget,
            )
            if burst_combo is not None:
                best_combo = burst_combo
                spread_bps = float(best_combo["spread_bps"])
                resolved_top_k = burst_top_k
                resolved_combo_limit = burst_combo_limit
                forward_candidates = burst_forward_candidates
                reverse_candidate_count = burst_reverse_candidate_count
                evaluated_combo_count = burst_evaluated_combo_count
                route_cooldown_skipped_count = burst_route_cooldown_skipped_count

            median_combo, requote_samples = await self._requote_best_combo_median(
                pair=pair,
                best_combo=best_combo,
                sample_count=2,
                request_purpose="exploration",
                allow_during_pause=False,
                deadline_epoch=burst_deadline,
                exploration_call_budget=burst_budget,
            )
            if median_combo is not None:
                median_requote_applied = True
                median_requote_sample_count = len(requote_samples)
                requote_spread_samples_bps = [
                    float(sample.get("spread_bps", 0.0))
                    for sample in requote_samples
                ]
                requote_samples_route_hashes_forward = [
                    str(sample.get("forward_route_hash") or "")
                    for sample in requote_samples
                ]
                requote_samples_route_hashes_reverse = [
                    str(sample.get("reverse_route_hash") or "")
                    for sample in requote_samples
                ]
                if requote_spread_samples_bps:
                    median_requote_range_bps = max(requote_spread_samples_bps) - min(requote_spread_samples_bps)
                best_combo = median_combo
                spread_bps = float(best_combo["spread_bps"])
                best_spread_post_requote = spread_bps
                requote_median_spread_bps = spread_bps
                decay_initial_to_requote_bps = (
                    (best_spread_post_requote - best_spread_pre_requote)
                    if best_spread_pre_requote is not None
                    else None
                )
                if (
                    self._enable_route_instability_cooldown
                    and decay_initial_to_requote_bps is not None
                    and self._route_instability_cooldown_decay_requote_bps > 0
                    and decay_initial_to_requote_bps <= -self._route_instability_cooldown_decay_requote_bps
                ):
                    self.register_route_instability_cooldown(
                        forward_route_hash=str(best_combo.get("forward_route_hash") or ""),
                        reverse_route_hash=str(best_combo.get("reverse_route_hash") or ""),
                        decay_bps=float(decay_initial_to_requote_bps),
                        cooldown_seconds=self._route_instability_cooldown_requote_seconds,
                        source="requote_decay",
                    )

            log_event(
                self._logger,
                level="info",
                event="spread_burst_result",
                message="Burst refinement result was recorded",
                mode="burst",
                median_spread_bps=(
                    round(float(spread_bps), 6)
                    if median_requote_applied
                    else None
                ),
                requote_samples=[round(value, 6) for value in requote_spread_samples_bps],
                requote_sample_count=median_requote_sample_count,
                requote_range_bps=(
                    round(float(median_requote_range_bps), 6)
                    if median_requote_range_bps is not None
                    else None
                ),
                topk=burst_top_k,
                combo=burst_combo_limit,
                burst_quote_call_budget_remaining=int(burst_budget.get("remaining", 0)),
            )

            if (
                median_requote_range_bps is not None
                and resolved_median_requote_max_range_bps > 0
                and median_requote_range_bps > resolved_median_requote_max_range_bps
            ):
                fail_reason = FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED
                if self._enable_route_instability_cooldown:
                    self._mark_route_pair_cooldown(
                        forward_route_hash=str(best_combo.get("forward_route_hash") or ""),
                        reverse_route_hash=str(best_combo.get("reverse_route_hash") or ""),
                        cooldown_seconds=self._route_instability_cooldown_requote_seconds,
                    )

        best_forward_route_hash = str(
            best_combo.get("forward_route_hash")
            or _route_hash_from_fingerprint(str(best_combo["forward"].get("route_fingerprint") or "unknown"))
        )
        best_reverse_route_hash = str(
            best_combo.get("reverse_route_hash")
            or _route_hash_from_fingerprint(str(best_combo["reverse"].get("route_fingerprint") or "unknown"))
        )
        best_combo["forward_route_hash"] = best_forward_route_hash
        best_combo["reverse_route_hash"] = best_reverse_route_hash

        now = asyncio.get_running_loop().time()
        self._record_dynamic_good_candidate_labels(
            tuple(
                list(best_combo["forward"].get("route_dexes", ()))
                + list(best_combo["reverse"].get("route_dexes", ()))
            ),
            now=now,
        )

        forward_quote = best_combo["forward"]["quote"]
        reverse_quote = best_combo["reverse"]["quote"]
        forward_out = int(best_combo["forward"]["out_amount"])
        reverse_out = int(best_combo["reverse"]["out_amount"])

        base_units = pair.base_amount / (10**pair.base_decimals)
        quote_units = forward_out / (10**pair.quote_decimals)
        forward_price = quote_units / base_units if base_units else 0.0

        expected_net_bps_stage_a: float | None = None
        expected_net_bps_stage_b: float | None = None
        stage_b_tip_lamports_effective = 0
        stage_b_tip_fee_bps_effective = 0.0
        stage_b_required_bps_effective: float | None = None
        if stage_a_required_bps is not None:
            expected_net_bps_stage_a = spread_bps - float(stage_a_required_bps)
        if stage_b_required_bps_without_tip is not None:
            stage_b_required_bps_effective = float(stage_b_required_bps_without_tip)
            tip_share = max(0.0, min(1.0, float(stage_b_tip_share or 0.0)))
            if expected_net_bps_stage_a is not None and pair.base_amount > 0:
                expected_profit_stage_a_lamports = max(0, int((pair.base_amount * expected_net_bps_stage_a) / 10_000))
                stage_b_tip_lamports_effective = int(expected_profit_stage_a_lamports * tip_share)
                if expected_profit_stage_a_lamports > 0 and tip_share > 0 and stage_b_tip_lamports_effective <= 0:
                    stage_b_tip_lamports_effective = 1
                max_tip = max(0, int(stage_b_tip_lamports_max or 0))
                if max_tip > 0:
                    stage_b_tip_lamports_effective = min(stage_b_tip_lamports_effective, max_tip)
                stage_b_tip_fee_bps_effective = (
                    (stage_b_tip_lamports_effective / pair.base_amount) * 10_000
                    if pair.base_amount > 0
                    else 0.0
                )
            stage_b_required_bps_effective += stage_b_tip_fee_bps_effective
            expected_net_bps_stage_b = spread_bps - stage_b_required_bps_effective

        stage_a_margin_bps = expected_net_bps_stage_a
        stage_b_margin_bps = expected_net_bps_stage_b
        stage_a_gate_mode = "strict_margin"
        stage_a_pass_threshold_bps: float | None = None
        if stage_a_required_bps is not None:
            stage_a_pass_threshold_bps = float(stage_a_required_bps) + resolved_stage_a_min_margin_bps

        if stage_a_margin_bps is None:
            stage_a_pass = None
        elif self._enable_stagea_relaxed_gate and stage_a_required_bps is not None:
            stage_a_gate_mode = "relaxed_non_negative"
            stage_a_pass_threshold_bps = float(stage_a_required_bps)
            stage_a_pass = bool(
                spread_bps >= float(stage_a_required_bps)
                and stage_a_margin_bps >= 0.0
            )
        else:
            stage_a_pass = stage_a_margin_bps >= resolved_stage_a_min_margin_bps
        stage_b_pass = None if stage_b_margin_bps is None else stage_b_margin_bps >= 0

        if fail_reason == "":
            if stage_a_pass is False:
                fail_reason = FAIL_REASON_BELOW_STAGEA_REQUIRED
            elif stage_b_pass is False:
                fail_reason = FAIL_REASON_BELOW_STAGEB_REQUIRED

        stage_a_components_for_log: dict[str, float] = {
            key: float(value) for key, value in (stage_a_required_components or {}).items()
        }
        if stage_a_required_bps is not None:
            stage_a_components_for_log.setdefault("required_spread_bps", float(stage_a_required_bps))
        stage_a_components_for_log.setdefault("min_stagea_margin_bps", float(resolved_stage_a_min_margin_bps))

        stage_b_components_for_log: dict[str, float] = {
            key: float(value) for key, value in (stage_b_required_components or {}).items()
        }
        if stage_b_required_bps_effective is not None:
            stage_b_components_for_log.setdefault(
                "required_spread_bps",
                float(stage_b_required_bps_effective),
            )
        stage_b_components_for_log["tip_fee_bps"] = float(stage_b_tip_fee_bps_effective)

        should_log_best_combo = (
            self._exploration_log_interval_seconds <= 0
            or (now - self._last_exploration_log_at) >= self._exploration_log_interval_seconds
        )
        if should_log_best_combo:
            self._last_exploration_log_at = now
            log_event(
                self._logger,
                level="info",
                event="dex_allowlist_effective",
                message="Effective allowlist assembled from static + dynamic observed labels",
                mode=normalized_mode,
                observed_dex_labels_top=list(dynamic_dex_labels),
                dex_allowlist_effective={
                    "forward": list(effective_allowlist_forward),
                    "reverse": list(effective_allowlist_reverse),
                },
            )
            log_event(
                self._logger,
                level="info",
                event="spread_exploration_best_combo",
                message="Best route combination selected from initial spread exploration",
                mode=normalized_mode,
                best_spread_bps=round(spread_bps, 6),
                best_forward_fingerprint=str(best_combo["forward"]["route_fingerprint"]),
                best_reverse_fingerprint=str(best_combo["reverse"]["route_fingerprint"]),
                best_forward_route_hash=best_forward_route_hash,
                best_reverse_route_hash=best_reverse_route_hash,
                base_amount_raw=pair.base_amount,
                forward_out_raw=forward_out,
                reverse_out_raw=reverse_out,
                route_candidate_count=len(forward_candidates) + reverse_candidate_count,
                route_sampled_count=evaluated_combo_count,
                route_cooldown_skipped_count=route_cooldown_skipped_count,
                candidate_count_forward=len(forward_candidates),
                candidate_count_reverse=reverse_candidate_count,
                evaluated_combo_count=evaluated_combo_count,
                sweep_top_k=resolved_top_k,
                sweep_top_k_max=resolved_top_k_max,
                sweep_combo_limit=resolved_combo_limit,
                quote_params_source=quote_params_source or "watcher_default",
                quote_params=base_quote_params,
                forward_quote_params=best_combo["forward"]["params"],
                reverse_quote_params=best_combo["reverse"]["params"],
                observed_dex_labels_top=list(dynamic_dex_labels),
                dex_allowlist_effective={
                    "forward": list(effective_allowlist_forward),
                    "reverse": list(effective_allowlist_reverse),
                },
                dex_excludelist=list(dex_excludelist),
                quote_rps_effective=round(self._quote_rps_effective(), 3),
                cache_hit_rate=round(self._quote_cache_hit_rate(), 6),
                rate_limited_count=self._rate_limited_count,
                stageA_required_spread_bps=(
                    round(float(stage_a_required_bps), 6)
                    if stage_a_required_bps is not None
                    else None
                ),
                stageA_min_margin_bps=round(resolved_stage_a_min_margin_bps, 6),
                stageA_pass_threshold_bps=(
                    round(float(stage_a_pass_threshold_bps), 6)
                    if stage_a_pass_threshold_bps is not None
                    else None
                ),
                stageA_gate_mode=stage_a_gate_mode,
                stageA_required_components={
                    key: round(float(value), 6)
                    for key, value in stage_a_components_for_log.items()
                },
                stageA_pass=stage_a_pass,
                stageA_margin_bps=(
                    round(stage_a_margin_bps, 6)
                    if stage_a_margin_bps is not None
                    else None
                ),
                expected_net_bps_stageA=(
                    round(expected_net_bps_stage_a, 6)
                    if expected_net_bps_stage_a is not None
                    else None
                ),
                stageB_required_spread_bps=(
                    round(stage_b_required_bps_effective, 6)
                    if stage_b_required_bps_effective is not None
                    else None
                ),
                stageB_required_components={
                    key: round(float(value), 6)
                    for key, value in stage_b_components_for_log.items()
                },
                stageB_pass=stage_b_pass,
                stageB_margin_bps=(
                    round(stage_b_margin_bps, 6)
                    if stage_b_margin_bps is not None
                    else None
                ),
                expected_net_bps_stageB=(
                    round(expected_net_bps_stage_b, 6)
                    if expected_net_bps_stage_b is not None
                    else None
                ),
                stageB_tip_lamports_effective=stage_b_tip_lamports_effective,
                stageB_tip_fee_bps_effective=round(stage_b_tip_fee_bps_effective, 6),
                median_requote_applied=median_requote_applied,
                median_requote_sample_count=median_requote_sample_count,
                median_requote_range_bps=(
                    round(median_requote_range_bps, 6)
                    if median_requote_range_bps is not None
                    else None
                ),
                requote_spread_samples_bps=[round(value, 6) for value in requote_spread_samples_bps],
                requote_samples_route_hashes_forward=requote_samples_route_hashes_forward,
                requote_samples_route_hashes_reverse=requote_samples_route_hashes_reverse,
                requote_median_spread_bps=(
                    round(requote_median_spread_bps, 6)
                    if requote_median_spread_bps is not None
                    else None
                ),
                unstable_drop_threshold_bps=(
                    round(resolved_median_requote_max_range_bps, 6)
                    if resolved_median_requote_max_range_bps > 0
                    else 0.0
                ),
                best_spread_pre_requote=(
                    round(float(best_spread_pre_requote), 6)
                    if best_spread_pre_requote is not None
                    else None
                ),
                best_spread_post_requote=(
                    round(float(best_spread_post_requote), 6)
                    if best_spread_post_requote is not None
                    else None
                ),
                fail_reason=fail_reason or "NONE",
            )

        return SpreadObservation(
            pair=pair.symbol,
            timestamp=now_iso(),
            forward_out_amount=forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
            quote_params=base_quote_params,
            forward_route_dexes=tuple(best_combo["forward"]["route_dexes"]),
            reverse_route_dexes=tuple(best_combo["reverse"]["route_dexes"]),
            forward_route_hash=best_forward_route_hash,
            reverse_route_hash=best_reverse_route_hash,
            stage_a_pass=stage_a_pass,
            stage_a_margin_bps=stage_a_margin_bps,
            stage_b_pass=stage_b_pass,
            stage_b_margin_bps=stage_b_margin_bps,
            fail_reason=fail_reason,
            route_candidate_count=len(forward_candidates) + reverse_candidate_count,
            route_sampled_count=evaluated_combo_count,
            route_cooldown_skipped_count=route_cooldown_skipped_count,
            exploration_mode=normalized_mode,
            median_requote_applied=median_requote_applied,
            median_requote_sample_count=median_requote_sample_count,
            median_requote_range_bps=median_requote_range_bps,
            requote_spread_samples_bps=tuple(requote_spread_samples_bps),
            requote_samples_route_hashes_forward=tuple(requote_samples_route_hashes_forward),
            requote_samples_route_hashes_reverse=tuple(requote_samples_route_hashes_reverse),
            requote_median_spread_bps=requote_median_spread_bps,
            best_spread_pre_requote=best_spread_pre_requote,
            best_spread_post_requote=best_spread_post_requote,
            unstable_drop_threshold_bps=(
                resolved_median_requote_max_range_bps
                if resolved_median_requote_max_range_bps > 0
                else None
            ),
        )


__all__ = [
    "HeliusQuoteWatcher",
    "HeliusRateLimitError",
]
