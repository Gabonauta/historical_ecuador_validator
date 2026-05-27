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
from sqlalchemy import DateTime, Double, ForeignKey, LargeBinary, SmallInteger, Text, Uuid, create_engine, func, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, load_only, mapped_column, relationship, selectinload, sessionmaker


MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
HISTORY_LIMIT = 20
SUPABASE_PUBLIC_TABLES = (
    "text_evaluations",
    "text_candidate_results",
    "image_evaluations",
    "image_assets",
    "text_expert_reviews",
    "image_expert_reviews",
)
REQUIRED_PUBLIC_TABLES = SUPABASE_PUBLIC_TABLES


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
    expert_reviews: Mapped[list["TextExpertReview"]] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
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


class TextExpertReview(Base):
    __tablename__ = "text_expert_reviews"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text_evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("text_evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    evaluator_name: Mapped[str] = mapped_column(Text, nullable=False)
    evaluator_specialty: Mapped[str] = mapped_column(Text, nullable=False)
    evaluator_institution: Mapped[str] = mapped_column(Text, nullable=False)
    observations: Mapped[str | None] = mapped_column(Text, nullable=True)
    responses_json: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation: Mapped[TextEvaluation] = relationship(back_populates="expert_reviews")


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
    expert_reviews: Mapped[list["ImageExpertReview"]] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
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


class ImageExpertReview(Base):
    __tablename__ = "image_expert_reviews"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_evaluation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("image_evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    evaluator_name: Mapped[str] = mapped_column(Text, nullable=False)
    evaluator_specialty: Mapped[str] = mapped_column(Text, nullable=False)
    evaluator_institution: Mapped[str] = mapped_column(Text, nullable=False)
    observations: Mapped[str | None] = mapped_column(Text, nullable=True)
    responses_json: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation: Mapped[ImageEvaluation] = relationship(back_populates="expert_reviews")


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


def resolve_db_auto_migrate() -> bool:
    raw_value = _get_secret_value("DB_AUTO_MIGRATE") or os.getenv("DB_AUTO_MIGRATE")
    if raw_value is None:
        return False

    normalized = str(raw_value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "on"}


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


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(value: str) -> Any:
    return json.loads(value)


def _serialize_expert_review(
    review: TextExpertReview | ImageExpertReview,
) -> dict[str, Any]:
    return {
        "id": str(review.id),
        "created_at": review.created_at,
        "evaluator_name": review.evaluator_name,
        "evaluator_specialty": review.evaluator_specialty,
        "evaluator_institution": review.evaluator_institution,
        "display_name": f"{review.evaluator_name} - {review.evaluator_institution}",
        "observations": review.observations,
        "responses": _json_load(review.responses_json),
    }


@lru_cache(maxsize=4)
def get_engine(database_url: str) -> Engine:
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"application_name": "historical_evaluator_streamlit"},
    )


@lru_cache(maxsize=4)
def get_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


@lru_cache(maxsize=8)
def ensure_database_ready(database_url: str, auto_migrate: bool) -> bool:
    engine = get_engine(database_url)
    if auto_migrate:
        Base.metadata.create_all(engine)
        _ensure_supabase_rls(engine)

    _validate_required_tables(engine)
    return True


def _validate_required_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names(schema="public"))
    missing_tables = [table for table in REQUIRED_PUBLIC_TABLES if table not in existing_tables]

    if missing_tables:
        missing = ", ".join(missing_tables)
        raise RuntimeError(
            "Faltan tablas requeridas en PostgreSQL. "
            "Ejecuta el SQL de bootstrap/migración y vuelve a intentar. "
            f"Tablas faltantes: {missing}."
        )


def _ensure_supabase_rls(engine: Engine) -> None:
    """Enable RLS on exposed public tables so Supabase's Data API doesn't flag them."""
    if engine.dialect.name != "postgresql":
        return

    with engine.begin() as connection:
        for table_name in SUPABASE_PUBLIC_TABLES:
            connection.exec_driver_sql(f"ALTER TABLE public.{table_name} ENABLE ROW LEVEL SECURITY")


def get_storage_status() -> StorageStatus:
    database_url = resolve_database_url()
    auto_migrate = resolve_db_auto_migrate()
    if not database_url:
        return StorageStatus(
            configured=False,
            available=False,
            message="Persistencia desactivada: configura DATABASE_URL para guardar historial en PostgreSQL.",
        )

    try:
        ensure_database_ready(database_url, auto_migrate)
    except Exception as exc:
        return StorageStatus(
            configured=True,
            available=False,
            message=(
                "Persistencia no disponible: error de conexión o inicialización "
                f"({type(exc).__name__})."
            ),
        )

    return StorageStatus(
        configured=True,
        available=True,
        message=(
            "Persistencia activa en PostgreSQL. "
            f"DB_AUTO_MIGRATE={'on' if auto_migrate else 'off'}."
        ),
    )


def _get_required_database_url() -> str:
    database_url = resolve_database_url()
    auto_migrate = resolve_db_auto_migrate()
    if not database_url:
        raise RuntimeError("No se encontró DATABASE_URL para persistencia.")
    ensure_database_ready(database_url, auto_migrate)
    return database_url


def _coerce_uuid(value: str | uuid.UUID, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except ValueError as exc:
        raise ValueError(f"El identificador '{field_name}' no tiene un formato UUID válido.") from exc


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


def save_text_expert_review(
    text_evaluation_id: str | uuid.UUID,
    evaluator_name: str,
    evaluator_specialty: str,
    evaluator_institution: str,
    responses: dict[str, Any],
    observations: str | None = None,
) -> uuid.UUID:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)
    evaluation_uuid = _coerce_uuid(text_evaluation_id, "text_evaluation_id")

    with session_factory.begin() as session:
        review = TextExpertReview(
            text_evaluation_id=evaluation_uuid,
            evaluator_name=evaluator_name,
            evaluator_specialty=evaluator_specialty,
            evaluator_institution=evaluator_institution,
            observations=observations,
            responses_json=_json_dump(responses),
        )
        session.add(review)
        session.flush()
        return review.id


def save_image_expert_review(
    image_evaluation_id: str | uuid.UUID,
    evaluator_name: str,
    evaluator_specialty: str,
    evaluator_institution: str,
    responses: dict[str, Any],
    observations: str | None = None,
) -> uuid.UUID:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)
    evaluation_uuid = _coerce_uuid(image_evaluation_id, "image_evaluation_id")

    with session_factory.begin() as session:
        review = ImageExpertReview(
            image_evaluation_id=evaluation_uuid,
            evaluator_name=evaluator_name,
            evaluator_specialty=evaluator_specialty,
            evaluator_institution=evaluator_institution,
            observations=observations,
            responses_json=_json_dump(responses),
        )
        session.add(review)
        session.flush()
        return review.id


def get_image_assets_for_evaluation(evaluation_id: str | uuid.UUID, include_bytes: bool = True) -> list[dict[str, Any]]:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)
    evaluation_uuid = _coerce_uuid(evaluation_id, "image_evaluation_id")

    with session_factory() as session:
        loader = selectinload(ImageEvaluation.assets)
        if not include_bytes:
            loader = loader.load_only(
                ImageAsset.id,
                ImageAsset.slot,
                ImageAsset.filename,
                ImageAsset.mime_type,
                ImageAsset.sha256,
            )

        stmt = (
            select(ImageEvaluation)
            .options(loader)
            .where(ImageEvaluation.id == evaluation_uuid)
        )
        evaluation = session.scalars(stmt).first()

    if evaluation is None:
        return []

    return [
        {
            "id": str(asset.id),
            "slot": asset.slot,
            "filename": asset.filename,
            "mime_type": asset.mime_type,
            "sha256": asset.sha256,
            **({"image_bytes": asset.image_bytes} if include_bytes else {}),
        }
        for asset in evaluation.assets
    ]


def list_recent_text_evaluations(limit: int = HISTORY_LIMIT, offset: int = 0) -> list[dict[str, Any]]:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)

    safe_limit = max(1, int(limit))
    safe_offset = max(0, int(offset))

    with session_factory() as session:
        stmt = (
            select(TextEvaluation)
            .options(
                load_only(
                    TextEvaluation.id,
                    TextEvaluation.created_at,
                    TextEvaluation.source_text,
                ),
                selectinload(TextEvaluation.candidates).load_only(
                    TextCandidateResult.slot,
                    TextCandidateResult.label,
                    TextCandidateResult.candidate_text,
                    TextCandidateResult.bleu_score,
                    TextCandidateResult.bert_precision,
                    TextCandidateResult.bert_recall,
                    TextCandidateResult.bert_f1,
                ),
                selectinload(TextEvaluation.expert_reviews).load_only(
                    TextExpertReview.id,
                    TextExpertReview.created_at,
                    TextExpertReview.evaluator_name,
                    TextExpertReview.evaluator_specialty,
                    TextExpertReview.evaluator_institution,
                    TextExpertReview.observations,
                    TextExpertReview.responses_json,
                ),
            )
            .order_by(TextEvaluation.created_at.desc())
            .offset(safe_offset)
            .limit(safe_limit)
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
            "expert_reviews": [
                _serialize_expert_review(review)
                for review in sorted(
                    evaluation.expert_reviews,
                    key=lambda current: current.created_at.timestamp() if current.created_at else 0.0,
                    reverse=True,
                )
            ],
        }
        for evaluation in evaluations
    ]


def list_recent_image_evaluations(limit: int = HISTORY_LIMIT, offset: int = 0) -> list[dict[str, Any]]:
    database_url = _get_required_database_url()
    session_factory = get_session_factory(database_url)

    safe_limit = max(1, int(limit))
    safe_offset = max(0, int(offset))

    with session_factory() as session:
        stmt = (
            select(ImageEvaluation)
            .options(
                load_only(
                    ImageEvaluation.id,
                    ImageEvaluation.created_at,
                    ImageEvaluation.prompt_text,
                    ImageEvaluation.fid_1_vs_23,
                    ImageEvaluation.fid_2_vs_13,
                    ImageEvaluation.fid_3_vs_12,
                    ImageEvaluation.clip_1,
                    ImageEvaluation.clip_2,
                    ImageEvaluation.clip_3,
                ),
                selectinload(ImageEvaluation.assets).load_only(
                    ImageAsset.id,
                    ImageAsset.slot,
                    ImageAsset.filename,
                    ImageAsset.mime_type,
                    ImageAsset.sha256,
                ),
                selectinload(ImageEvaluation.expert_reviews).load_only(
                    ImageExpertReview.id,
                    ImageExpertReview.created_at,
                    ImageExpertReview.evaluator_name,
                    ImageExpertReview.evaluator_specialty,
                    ImageExpertReview.evaluator_institution,
                    ImageExpertReview.observations,
                    ImageExpertReview.responses_json,
                ),
            )
            .order_by(ImageEvaluation.created_at.desc())
            .offset(safe_offset)
            .limit(safe_limit)
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
                    "id": str(asset.id),
                    "slot": asset.slot,
                    "filename": asset.filename,
                    "mime_type": asset.mime_type,
                    "sha256": asset.sha256,
                }
                for asset in evaluation.assets
            ],
            "expert_reviews": [
                _serialize_expert_review(review)
                for review in sorted(
                    evaluation.expert_reviews,
                    key=lambda current: current.created_at.timestamp() if current.created_at else 0.0,
                    reverse=True,
                )
            ],
        }
        for evaluation in evaluations
    ]
