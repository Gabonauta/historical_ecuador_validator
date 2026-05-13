from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import streamlit as st
from sqlalchemy import DateTime, Double, ForeignKey, LargeBinary, SmallInteger, Text, Uuid, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, selectinload, sessionmaker


MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
HISTORY_LIMIT = 20


class Base(DeclarativeBase):
    pass


class TextEvaluation(Base):
    __tablename__ = "text_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    candidates: Mapped[list["TextCandidateResult"]] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
        order_by="TextCandidateResult.slot",
    )


class TextCandidateResult(Base):
    __tablename__ = "text_candidate_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text_evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("text_evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    slot: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_text: Mapped[str] = mapped_column(Text, nullable=False)
    bleu_score: Mapped[float] = mapped_column(Double, nullable=False)
    bert_precision: Mapped[float] = mapped_column(Double, nullable=False)
    bert_recall: Mapped[float] = mapped_column(Double, nullable=False)
    bert_f1: Mapped[float] = mapped_column(Double, nullable=False)
    evaluation: Mapped[TextEvaluation] = relationship(back_populates="candidates")


class ImageEvaluation(Base):
    __tablename__ = "image_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    fid_1_vs_23: Mapped[float] = mapped_column(Double, nullable=False)
    fid_2_vs_13: Mapped[float] = mapped_column(Double, nullable=False)
    fid_3_vs_12: Mapped[float] = mapped_column(Double, nullable=False)
    clip_1: Mapped[float] = mapped_column(Double, nullable=False)
    clip_2: Mapped[float] = mapped_column(Double, nullable=False)
    clip_3: Mapped[float] = mapped_column(Double, nullable=False)
    assets: Mapped[list["ImageAsset"]] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
        order_by="ImageAsset.slot",
    )


class ImageAsset(Base):
    __tablename__ = "image_assets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("image_evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    slot: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    image_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    evaluation: Mapped[ImageEvaluation] = relationship(back_populates="assets")


@dataclass(frozen=True)
class StorageStatus:
    configured: bool
    available: bool
    message: str


def _get_secret_value(key: str) -> str | None:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        return None
    return None


def _append_sslmode(database_url: str, sslmode: str | None) -> str:
    if not sslmode or "sslmode=" in database_url:
        return database_url

    parsed = urlparse(database_url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params["sslmode"] = sslmode
    return urlunparse(parsed._replace(query=urlencode(query_params)))


def resolve_database_url() -> str | None:
    database_url = _get_secret_value("DATABASE_URL") or os.getenv("DATABASE_URL")
    sslmode = _get_secret_value("DB_SSLMODE") or os.getenv("DB_SSLMODE")
    if not database_url:
        return None
    return _append_sslmode(database_url, sslmode)


def resolve_write_password() -> str | None:
    return _get_secret_value("STORAGE_WRITE_PASSWORD") or os.getenv("STORAGE_WRITE_PASSWORD")


def serialize_image_prompt_texts(prompt_texts: dict[int, str]) -> str:
    return json.dumps(
        {
            "version": 2,
            "clip_texts": {str(slot): text for slot, text in sorted(prompt_texts.items())},
        },
        ensure_ascii=False,
    )


def deserialize_image_prompt_texts(raw_prompt_text: str) -> dict[int, str]:
    try:
        payload = json.loads(raw_prompt_text)
    except json.JSONDecodeError:
        return {1: raw_prompt_text}

    clip_texts = payload.get("clip_texts")
    if not isinstance(clip_texts, dict):
        return {1: raw_prompt_text}

    parsed: dict[int, str] = {}
    for slot, value in clip_texts.items():
        try:
            parsed[int(slot)] = str(value)
        except (TypeError, ValueError):
            continue
    return parsed or {1: raw_prompt_text}


@lru_cache(maxsize=4)
def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


@lru_cache(maxsize=4)
def get_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


@lru_cache(maxsize=4)
def ensure_database_ready(database_url: str) -> bool:
    Base.metadata.create_all(get_engine(database_url))
    return True


def get_storage_status() -> StorageStatus:
    database_url = resolve_database_url()
    if not database_url:
        return StorageStatus(
            configured=False,
            available=False,
            message="Persistencia desactivada: configura DATABASE_URL para guardar historial en PostgreSQL.",
        )

    try:
        ensure_database_ready(database_url)
    except Exception as exc:
        return StorageStatus(
            configured=True,
            available=False,
            message=f"Persistencia no disponible: {exc}",
        )

    return StorageStatus(
        configured=True,
        available=True,
        message="Persistencia activa en PostgreSQL.",
    )


def _get_required_database_url() -> str:
    database_url = resolve_database_url()
    if not database_url:
        raise RuntimeError("No se encontró DATABASE_URL para persistencia.")
    ensure_database_ready(database_url)
    return database_url


def _prepare_image_payloads(uploaded_files: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    for slot, uploaded_file in enumerate(uploaded_files, start=1):
        if uploaded_file is None:
            continue
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            raise ValueError(f"La Imagen {slot} no contiene datos.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(
                f"La Imagen {slot} supera el límite de {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
            )

        filename = uploaded_file.name or f"imagen_{slot}"
        mime_type = uploaded_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        payloads.append(
            {
                "slot": slot,
                "filename": filename,
                "mime_type": mime_type,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_bytes": image_bytes,
            }
        )

    return payloads


def save_text_evaluation(source_text: str, text_results: list[dict[str, Any]]) -> uuid.UUID:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)

    with session_factory.begin() as session:
        evaluation = TextEvaluation(source_text=source_text)
        session.add(evaluation)
        session.flush()

        for slot, result in enumerate(text_results, start=1):
            bert_results = result["bert_results"]
            session.add(
                TextCandidateResult(
                    text_evaluation_id=evaluation.id,
                    slot=slot,
                    label=str(result["label"]),
                    candidate_text=str(result["candidate_text"]),
                    bleu_score=float(result["bleu_score"]),
                    bert_precision=float(bert_results["precision"]),
                    bert_recall=float(bert_results["recall"]),
                    bert_f1=float(bert_results["f1"]),
                )
            )

    return evaluation.id


def save_image_evaluation(
    prompt_texts: dict[int, str],
    uploaded_files: list[Any],
    fid_results: dict[str, float],
    clip_results: dict[str, float],
) -> uuid.UUID:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)
    image_payloads = _prepare_image_payloads(uploaded_files)

    with session_factory.begin() as session:
        evaluation = ImageEvaluation(
            prompt_text=serialize_image_prompt_texts(prompt_texts),
            fid_1_vs_23=float(fid_results["fid_1_vs_23"]),
            fid_2_vs_13=float(fid_results["fid_2_vs_13"]),
            fid_3_vs_12=float(fid_results["fid_3_vs_12"]),
            clip_1=float(clip_results["clip_1"]),
            clip_2=float(clip_results["clip_2"]),
            clip_3=float(clip_results["clip_3"]),
        )
        session.add(evaluation)
        session.flush()

        for payload in image_payloads:
            session.add(
                ImageAsset(
                    image_evaluation_id=evaluation.id,
                    slot=payload["slot"],
                    filename=payload["filename"],
                    mime_type=payload["mime_type"],
                    sha256=payload["sha256"],
                    image_bytes=payload["image_bytes"],
                )
            )

    return evaluation.id


def list_recent_text_evaluations(limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)

    with session_factory() as session:
        stmt = (
            select(TextEvaluation)
            .options(selectinload(TextEvaluation.candidates))
            .order_by(TextEvaluation.created_at.desc())
            .limit(limit)
        )
        evaluations = session.scalars(stmt).all()

    return [
        {
            "id": str(evaluation.id),
            "created_at": evaluation.created_at,
            "source_text": evaluation.source_text,
            "candidates": [
                {
                    "slot": candidate.slot,
                    "label": candidate.label,
                    "candidate_text": candidate.candidate_text,
                    "bleu_score": candidate.bleu_score,
                    "bert_precision": candidate.bert_precision,
                    "bert_recall": candidate.bert_recall,
                    "bert_f1": candidate.bert_f1,
                }
                for candidate in evaluation.candidates
            ],
        }
        for evaluation in evaluations
    ]


def list_recent_image_evaluations(limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)

    with session_factory() as session:
        stmt = (
            select(ImageEvaluation)
            .options(selectinload(ImageEvaluation.assets))
            .order_by(ImageEvaluation.created_at.desc())
            .limit(limit)
        )
        evaluations = session.scalars(stmt).all()

    return [
        {
            "id": str(evaluation.id),
            "created_at": evaluation.created_at,
            "prompt_text": evaluation.prompt_text,
            "prompt_texts": deserialize_image_prompt_texts(evaluation.prompt_text),
            "fid_1_vs_23": evaluation.fid_1_vs_23,
            "fid_2_vs_13": evaluation.fid_2_vs_13,
            "fid_3_vs_12": evaluation.fid_3_vs_12,
            "clip_1": evaluation.clip_1,
            "clip_2": evaluation.clip_2,
            "clip_3": evaluation.clip_3,
            "assets": [
                {
                    "slot": asset.slot,
                    "filename": asset.filename,
                    "mime_type": asset.mime_type,
                    "sha256": asset.sha256,
                    "image_bytes": asset.image_bytes,
                }
                for asset in evaluation.assets
            ],
        }
        for evaluation in evaluations
    ]
