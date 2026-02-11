from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any

from google.cloud import firestore
from redis import asyncio as redis
from redis.asyncio.client import Redis

from modules.common import log_event

from .firestore_ops import FirestoreStorageOps
from .redis_ops import RedisStorageOps
from .settings import StorageSettings


class StorageGateway(FirestoreStorageOps, RedisStorageOps):
    def __init__(self, settings: StorageSettings, logger: logging.Logger) -> None:
        self.settings = settings
        self._logger = logger
        self._redis: Redis | None = None
        self._firestore: firestore.Client | None = None
        self._bot_doc_ref: Any | None = None
        self._run_doc_ref: Any | None = None
        self._events_collection_ref: Any | None = None
        self._trades_collection_ref: Any | None = None
        self._pnl_daily_collection_ref: Any | None = None
        self._metrics_doc_ref: Any | None = None
        self._config_doc_ref: Any | None = None
        self._watch: Any | None = None
        self._resolved_firestore_config_doc = self.settings.firestore_config_doc

    @property
    def bot_id(self) -> str:
        return self.settings.bot_id

    @property
    def run_id(self) -> str:
        return self.settings.bot_run_id

    async def connect(self) -> None:
        firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
        if firebase_credentials and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = firebase_credentials

        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)
        await self._redis.ping()
        log_event(
            self._logger,
            level="info",
            event="redis_connected",
            message="Connected to Redis",
        )

        self._firestore = firestore.Client(project=self.settings.firestore_project_id)

        (
            self._resolved_firestore_config_doc,
            was_collection_path,
        ) = self._normalize_doc_path(
            self.settings.firestore_config_doc,
            self.settings.firestore_config_leaf_doc_id,
        )
        if was_collection_path:
            log_event(
                self._logger,
                level="warning",
                event="config_doc_path_normalized",
                message="FIRESTORE_CONFIG_DOC was a collection path and has been normalized",
                doc_path=self._resolved_firestore_config_doc,
            )

        self._initialize_namespace_refs()
        await self._ensure_bot_namespace()

        self._config_doc_ref = self._firestore.document(self._resolved_firestore_config_doc)

        startup_snapshot = await asyncio.to_thread(self._config_doc_ref.get)
        if startup_snapshot.exists:
            await self.sync_config_to_redis(startup_snapshot.to_dict() or {}, source="startup")
        else:
            await self.sync_config_to_redis({}, source="startup_missing")
            log_event(
                self._logger,
                level="warning",
                event="config_missing",
                message="Runtime config document does not exist",
                doc_path=self.settings.firestore_config_doc,
            )

        log_event(
            self._logger,
            level="info",
            event="firestore_connected",
            message="Connected to Firestore",
            doc_path=self._resolved_firestore_config_doc,
        )

    async def healthcheck(self) -> None:
        redis_client = self._require_redis()
        await redis_client.ping()

        if self._config_doc_ref is None:
            raise RuntimeError("Firestore config document reference is not initialized.")

        await asyncio.to_thread(self._config_doc_ref.get)

    async def close(self) -> None:
        if self._watch is not None:
            with contextlib.suppress(Exception):  # watcher uses sync callback threads; ignore close race
                self._watch.unsubscribe()
            self._watch = None

        if self._redis is not None:
            close = getattr(self._redis, "aclose", None)
            if close:
                await close()
            else:
                await self._redis.close()
            self._redis = None

        self._bot_doc_ref = None
        self._run_doc_ref = None
        self._events_collection_ref = None
        self._trades_collection_ref = None
        self._pnl_daily_collection_ref = None
        self._metrics_doc_ref = None
        self._config_doc_ref = None
        self._firestore = None
