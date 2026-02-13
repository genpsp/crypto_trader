from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import aiohttp

from modules.common import log_event

from .types import PairConfig, SpreadObservation, now_iso

DEFAULT_JUPITER_QUOTE_ENDPOINT = "https://api.jup.ag/swap/v1/quote"
_URL_QUERY_REDACTION_PATTERN = re.compile(r"(?i)(api[-_]?key=)[^&\s\"']+")
_HEADER_REDACTION_PATTERN = re.compile(r"(?i)(x-api-key\s*[:=]\s*)([^\s,;\"']+)")


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


def _safe_forward_output_amount(quote: dict[str, Any]) -> int:
    out_amount = int(quote.get("outAmount") or 0)
    min_out_amount = int(quote.get("otherAmountThreshold") or 0)
    if out_amount <= 0:
        return 0
    if min_out_amount <= 0:
        return out_amount
    return max(1, min(out_amount, min_out_amount))


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

    async def quote(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
        extra_params: dict[str, str | int | bool] | None = None,
    ) -> dict[str, Any]:
        if self._session is None:
            await self.connect()
        if self._session is None:
            raise RuntimeError("Helius quote HTTP session is not initialized.")

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
                params[normalized_key] = str(value)
        last_error: Exception | None = None
        max_attempts = self._max_retries + 1

        for attempt in range(1, max_attempts + 1):
            headers = self._build_headers()
            provider = _quote_provider_from_endpoint(self._quote_endpoint)
            try:
                async with self._session.get(self._quote_endpoint, params=params, headers=headers) as response:
                    status = response.status
                    retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
                    body = await response.text()
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
                if "rate limit" in helius_error.lower():
                    raise HeliusRateLimitError(
                        f"Quote failed with {provider} rate limit: {helius_error}",
                        retry_after_seconds=retry_after_seconds or self._rate_limit_backoff_seconds,
                        provider=provider,
                    )
                raise RuntimeError(f"Quote API error ({provider}): {helius_error}")

            if status == 429:
                body_preview = _sanitize_preview_for_log(body, limit=300)
                rate_limit_backoff = self._rate_limit_backoff_seconds
                if provider == "jupiter" and not self._jupiter_api_key:
                    rate_limit_backoff = max(rate_limit_backoff, 30.0)
                raise HeliusRateLimitError(
                    f"Quote failed: status={status} non_json_body={body_preview!r}",
                    retry_after_seconds=retry_after_seconds or rate_limit_backoff,
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
                    raise RuntimeError(f"Quote failed: status={status} body={data}")
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

            return data

        raise RuntimeError(f"Quote request failed after retries: {last_error}")

    async def fetch_spread(self, pair: PairConfig) -> SpreadObservation:
        forward_quote = await self.quote(
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount=pair.base_amount,
            slippage_bps=pair.slippage_bps,
        )
        forward_out = int(forward_quote["outAmount"])
        reusable_forward_out = _safe_forward_output_amount(forward_quote)
        if reusable_forward_out <= 0:
            raise RuntimeError("Forward quote did not provide a valid reusable output amount.")

        reverse_quote = await self.quote(
            input_mint=pair.quote_mint,
            output_mint=pair.base_mint,
            amount=reusable_forward_out,
            slippage_bps=pair.slippage_bps,
        )
        reverse_out = int(reverse_quote["outAmount"])

        base_units = pair.base_amount / (10**pair.base_decimals)
        quote_units = forward_out / (10**pair.quote_decimals)
        forward_price = quote_units / base_units if base_units else 0.0

        spread_bps = ((reverse_out - pair.base_amount) / pair.base_amount) * 10_000

        return SpreadObservation(
            pair=pair.symbol,
            timestamp=now_iso(),
            forward_out_amount=reusable_forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
        )


__all__ = [
    "HeliusQuoteWatcher",
    "HeliusRateLimitError",
]
