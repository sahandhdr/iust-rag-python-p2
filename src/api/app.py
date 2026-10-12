# app.py
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from auth.rbac import UserContext, qdrant_sync
from config.settings import get_settings
from utils.dependencies import get_current_user
from ingest.documents import document_ingestor
from api.chat import router as chat_router
from api.sync import router as sync_router
from utils.api_responser import ApiResponser
from api.embed import router as embed_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("iust_rag")
settings = get_settings()

if settings.debug:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _warn_insecure_config() -> None:
    """Production hardening: surface weak secrets without crashing."""
    if settings.debug:
        return
    key = (settings.laravel.internal_api_key or "").strip()
    weak = {"", "change-me", "change-me-shared-secret", "secret", "password"}
    if key.lower() in weak or len(key) < 16:
        logger.warning(
            "SECURITY: LARAVEL__INTERNAL_API_KEY is empty or weak while DEBUG=false. "
            "Set a strong shared secret matching Laravel PYTHON_INTERNAL_API_KEY."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s | debug=%s | api_prefix=%s",
        settings.app_name,
        settings.debug,
        settings.api_prefix,
    )
    _warn_insecure_config()
    try:
        _ = settings.db.qdrant_collection
        yield
    except Exception:
        logger.exception("Fatal error during application lifespan")
        raise
    finally:
        logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="IUST RAG API",
    version="1.0.0",
    description="سیستم RAG — Phase 1 hardened",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

_cors_origins = settings.resolved_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-Key", "Accept"],
)

app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(sync_router, prefix=settings.api_prefix)
app.include_router(embed_router, prefix=settings.api_prefix)


def _client_safe_error(exc: Exception) -> str:
    if settings.debug:
        return str(exc) or exc.__class__.__name__
    return "internal-error"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error path=%s detail=%s", request.url.path, exc.errors())
    return ApiResponser.error_response(
        message="درخواست نامعتبر است.",
        errors=exc.errors(),
        status_code=422,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, (dict, list)):
        message = "خطای درخواست"
        errors = detail
    else:
        message = str(detail) if detail else "خطای درخواست"
        errors = None
    return ApiResponser.error_response(
        message=message,
        errors=errors,
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error path=%s method=%s", request.url.path, request.method)
    return ApiResponser.error_response(
        message="خطای داخلی سرور رخ داد.",
        errors=_client_safe_error(exc),
        status_code=500,
    )


def _parse_json_list(raw: Optional[str], field_name: str) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid JSON array") from exc
    if not isinstance(data, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return [str(item).strip() for item in data if str(item).strip()]


def _parse_overwrite(raw: Optional[str], default: bool = True) -> bool:
    if raw is None or str(raw).strip() == "":
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _can_manage_documents(user: UserContext) -> bool:
    roles = set(user.roles or [])
    return bool(roles & {"admin", "developer", "superadmin"})


def _safe_temp_name(filename: Optional[str]) -> str:
    base = os.path.basename(filename or "upload.bin")
    base = re.sub(r"[^a-zA-Z0-9._\-]", "_", base)
    return f"temp_ingest_{base}"


def _qdrant_chunk_count(doc_uuid: str) -> int:
    info = qdrant_sync.search_by_doc_uuid(doc_uuid, limit=10_000)
    if isinstance(info, list):
        return len(info)
    if isinstance(info, dict):
        if info.get("result") == "doc-notExists":
            return 0
        try:
            return int(info.get("total_chunks") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


@app.get("/check")
@app.get("/health")
async def check_health():
    data = {
        "status": "healthy",
        "version": "1.0.0",
        "app": settings.app_name,
    }
    if settings.debug:
        data["debug"] = True
    try:
        return ApiResponser.success_response(
            message="Host is up and running",
            data=data,
        )
    except Exception as exc:
        logger.exception("Health check failed")
        return ApiResponser.error_response(
            message="خطا در بررسی سلامت سرویس",
            errors=_client_safe_error(exc),
            status_code=500,
        )


@app.post(f"{settings.api_prefix}/files/ingest")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    department: str = Form("public"),
    doc_uuid: Optional[str] = Form(None),
    roles: Optional[str] = Form(None),
    departments: Optional[str] = Form(None),
    permissions: Optional[str] = Form(None),
    status: str = Form("published"),
    version: int = Form(1),
    overwrite: Optional[str] = Form("1"),
    current_user: UserContext = Depends(get_current_user),
):
    if not _can_manage_documents(current_user):
        return ApiResponser.error_response(
            message="شما مجوز آپلود سند مرجع را ندارید.",
            status_code=403,
        )

    try:
        roles_list = _parse_json_list(roles, "roles")
        depts_list = _parse_json_list(departments, "departments")
        perms_list = _parse_json_list(permissions, "permissions")
    except ValueError as exc:
        return ApiResponser.error_response(
            message="پارامتر roles/departments/permissions نامعتبر است.",
            errors=str(exc),
            status_code=422,
        )

    status_norm = (status or "published").strip().lower()
    if status_norm not in {"draft", "published", "archived"}:
        return ApiResponser.error_response(
            message="status باید یکی از draft|published|archived باشد.",
            status_code=422,
        )

    overwrite_flag = _parse_overwrite(overwrite, default=True)
    version_norm = version if isinstance(version, int) and version >= 1 else 1
    temp_file_path = _safe_temp_name(file.filename)

    try:
        content = await file.read()
        if not content:
            return ApiResponser.error_response(message="فایل خالی است.", status_code=422)

        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        result = document_ingestor.process_single_file(
            file_path=temp_file_path,
            department=department or "public",
            doc_uuid=doc_uuid,
            roles=roles_list,
            departments=depts_list,
            permissions=perms_list,
            status=status_norm,
            version=version_norm,
            overwrite=overwrite_flag,
        )

        if not result.get("success"):
            return ApiResponser.error_response(
                message="پردازش فایل با شکست مواجه شد. لطفاً محتوای فایل را بررسی کنید.",
                errors=result.get("error") if settings.debug else "ingest-failed",
                status_code=422,
            )

        return ApiResponser.success_response(
            message="فایل با موفقیت پردازش و به پایگاه دانش اضافه شد.",
            data={
                "filename": file.filename,
                "department": department or "public",
                "doc_uuid": result.get("doc_uuid") or doc_uuid or "auto-generated",
                "roles": result.get("roles") or roles_list,
                "departments": result.get("departments") or depts_list,
                "permissions": result.get("permissions") or perms_list,
                "status": result.get("status") or status_norm,
                "version": result.get("version") or version_norm,
                "chunks": result.get("chunks"),
                "overwrite": overwrite_flag,
                "status_ingest": "ingested",
            },
        )
    except Exception as exc:
        logger.exception(
            "Ingestion error for file %s (doc_uuid=%s)", file.filename, doc_uuid
        )
        return ApiResponser.error_response(
            message="خطا در پردازش فایل",
            errors=_client_safe_error(exc),
            status_code=500,
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as cleanup_err:
                logger.warning("Failed to cleanup temp file: %s", cleanup_err)


@app.delete(f"{settings.api_prefix}/files/{{doc_uuid}}")
async def delete_document_endpoint(
    doc_uuid: str = Path(..., description="شناسه یکتای سند (doc_uuid)"),
    current_user: UserContext = Depends(get_current_user),
):
    if not _can_manage_documents(current_user):
        return ApiResponser.error_response("شما مجوز حذف سند را ندارید.", 403)

    try:
        existing = _qdrant_chunk_count(doc_uuid)
        if existing <= 0:
            return ApiResponser.error_response(
                message=f"سند {doc_uuid} در پایگاه دانش موجود نیست.",
                status_code=404,
                errors={"doc_uuid": doc_uuid, "result": "doc-notExists"},
            )

        success = qdrant_sync.delete_document_by_uuid(doc_uuid)
        if not success:
            return ApiResponser.error_response("حذف سند ناموفق بود.", 500)

        leftover = _qdrant_chunk_count(doc_uuid)
        if leftover > 0:
            logger.error(
                "Delete reported success but chunks remain | doc_uuid=%s leftover=%s",
                doc_uuid,
                leftover,
            )
            return ApiResponser.error_response(
                message="حذف ناقص بود؛ هنوز چانک باقی است.",
                status_code=500,
                errors={"leftover": leftover} if settings.debug else "delete-incomplete",
            )

        return ApiResponser.success_response(
            message=f"سند {doc_uuid} با موفقیت حذف شد.",
            data={
                "doc_uuid": doc_uuid,
                "status": "deleted",
                "removed_chunks": existing,
            },
        )
    except Exception as exc:
        logger.exception("Delete error doc_uuid=%s", doc_uuid)
        return ApiResponser.error_response(
            message="خطا در حذف سند",
            errors=_client_safe_error(exc),
            status_code=500,
        )


def create_app() -> FastAPI:
    return app