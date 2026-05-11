"""Upload router for handling file uploads."""
from sympy import threaded
import logging
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path
import tempfile
from typing import List
from pydantic import BaseModel
from backend.utils import RagBuilder
_rag_builder = RagBuilder()    

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

    
class UploadResponse(BaseModel):
    """Response model for file upload."""

    success: bool
    files: list[dict[str, str]]
    message: str

@router.post("/upload", response_model=List[UploadResponse])
async def upload_and_process(
    uid: str,
    files: List[UploadFile] = File(...),
) -> List[UploadResponse]:
    """
    上传文件并处理内容（不保存到磁盘）
    
    流程：
    1. 在内存/临时目录接收文件
    2. 转换为 Markdown
    3. 读取内容
    4. 调用 process_file 处理
    5. 清理临时文件
    """
    from markitdown import MarkItDown
    CONVERTIBLE_EXTENSIONS = {".pdf", ".ppt", ".pptx", ".xls", ".xlsx", ".doc", ".docx"}
    results = []
    
    for file in files:
        if not file.filename:
             continue
        safe_filename = Path(file.filename).name
        
        try:
            if not safe_filename or safe_filename in {".", ".."} or "/" in safe_filename or "\\" in safe_filename:
                logger.warning(f"Skipping file with unsafe filename: {file.filename!r}")
                continue
            # 使用临时目录存储

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / safe_filename
                
                # 保存文件到临时目录
                content = await file.read()
                tmp_path.write_bytes(content)
                
                # 转换为 Markdown（如需）
                file_ext = tmp_path.suffix.lower()
                target_path = tmp_path
                if file_ext in CONVERTIBLE_EXTENSIONS:
                    try:
                        md = MarkItDown()
                        result = md.convert(str(tmp_path))
                        md_path = tmp_path.with_suffix(".md")
                        md_path.write_text(result.text_content, encoding="utf-8")
                        target_path = md_path
                    except Exception as e:
                        results.append(UploadResponse(
                            success=False,
                            files=[{"filename": safe_filename}],message=f"Conversion failed: {e}"
                            )
                        )
                        continue
                
                # 读取文件内容
                try:
                    file_content = target_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    file_content = target_path.read_bytes().decode("utf-8", errors="replace")
                _rag_builder.build_from_file(content=file_content, source=safe_filename, uid=uid)    
                results.append(UploadResponse(
                    success=True,
                    files=[{"filename": safe_filename}],
                    message="Processed successfully"
                ))
                
        except Exception as e:
            results.append(UploadResponse(
                success=False,
                files=[{"filename": safe_filename}],
                message=str(e)
            ))
    
    return results


@router.get("/list", response_model=dict)
async def list_uploaded_files(uid: str) -> dict:
    """List all files in a thread's uploads directory."""

    return {"files": [], "count": 0}


@router.delete("/{filename}")
async def delete_uploaded_file(uid: str, filename: str) -> dict:
    """Delete a file from a thread's uploads directory."""
    raise HTTPException(status_code=404, detail=f"File not found: {filename}")
