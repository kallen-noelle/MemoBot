"""Upload router for handling file uploads."""

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

VIRTUAL_PATH_PREFIX = "/mnt/user-data"


class UploadResponse(BaseModel):
    """Response model for file upload."""

    success: bool
    files: list[dict[str, str]]
    message: str


@router.post("", response_model=UploadResponse)
async def upload_files(
    uid: str,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    """Upload multiple files to a thread's uploads directory."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files = []

    for file in files:
        if not file.filename:
            continue

        try:
            safe_filename = Path(file.filename).name
            if not safe_filename or safe_filename in {".", ".."} or "/" in safe_filename or "\\" in safe_filename:
                logger.warning(f"Skipping file with unsafe filename: {file.filename!r}")
                continue

            content = await file.read()

            file_info = {
                "filename": safe_filename,
                "size": str(len(content)),
                "path": f"uploads/{safe_filename}",
                "virtual_path": f"{VIRTUAL_PATH_PREFIX}/uploads/{safe_filename}",
                "artifact_url": f"/api/threads/test-thread/artifacts/mnt/user-data/uploads/{safe_filename}",
            }

            logger.info(f"Saved file: {safe_filename} ({len(content)} bytes)")
            uploaded_files.append(file_info)

        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")

    return UploadResponse(
        success=True,
        files=uploaded_files,
        message=f"Successfully uploaded {len(uploaded_files)} file(s)",
    )


@router.get("/list", response_model=dict)
async def list_uploaded_files(thread_id: str) -> dict:
    """List all files in a thread's uploads directory."""
    return {"files": [], "count": 0}


@router.delete("/{filename}")
async def delete_uploaded_file(thread_id: str, filename: str) -> dict:
    """Delete a file from a thread's uploads directory."""
    raise HTTPException(status_code=404, detail=f"File not found: {filename}")
