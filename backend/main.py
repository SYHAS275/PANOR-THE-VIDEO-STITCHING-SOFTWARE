from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Query,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import shutil
import os
import uuid
import subprocess
import mimetypes
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import time

# Import authentication module
from auth import (
    UserCreate,
    UserLogin,
    User,
    Token,
    AuthResponse,
    create_user,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_optional_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# Configure logging
# Force handler addition in case basicConfig was already called
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("debug.log", mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)
root_logger.addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Stitcher API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants - Use project root paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory job store
jobs = {}


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Authentication Endpoints
# =============================================================================


@app.post("/api/auth/signup", response_model=AuthResponse)
async def signup(user_data: UserCreate):
    """Register a new user and return access token with user info."""
    user = create_user(user_data)
    access_token = create_access_token(data={"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "name": user.name, "email": user.email},
    }


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(credentials: UserLogin):
    """Authenticate user and return access token with user info."""
    user = authenticate_user(credentials.email, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    access_token = create_access_token(data={"sub": user["email"]})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user["id"], "name": user["name"], "email": user["email"]},
    }


@app.get("/api/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information."""
    return current_user


# =============================================================================
# Video Stitching Endpoints
# =============================================================================


async def run_stitching_job(
    job_id: str,
    file_paths: List[str],
    mode: str,
    moving_camera: bool = True,
    enable_detection: bool = True,
    use_timestamps: bool = False,
):
    """
    Background task to run the stitching process.
    """
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["message"] = "Initializing stitcher..."

        import sys

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from backend.processor import process_video_job

        def update_progress(progress: float, message: str):
            jobs[job_id]["progress"] = int(progress * 100)
            jobs[job_id]["message"] = message

        output_filename = await process_video_job(
            job_id,
            file_paths,
            mode,
            update_progress,
            moving_camera=moving_camera,
            enable_detection=enable_detection,
            use_timestamps=use_timestamps,
        )

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Stitching complete!"
        jobs[job_id]["output_file"] = output_filename
        _ensure_source_previews(job_id)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = "Processing failed"


def parse_bool(value) -> bool:
    """Parse boolean from string or bool value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _guess_media_type(file_path: str) -> str:
    media_type, _ = mimetypes.guess_type(file_path)
    return media_type or "application/octet-stream"


def _preview_path_for(job_id: str, index: int) -> str:
    preview_dir = os.path.join(OUTPUT_DIR, "source_previews", job_id)
    os.makedirs(preview_dir, exist_ok=True)
    return os.path.join(preview_dir, f"cam_{index}_preview.mp4")


def _create_preview(input_path: str, output_path: str) -> bool:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        "0",
        "-t",
        "4",
        "-i",
        input_path,
        "-an",
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False


def _ensure_source_previews(job_id: str):
    entry = jobs.get(job_id)
    if not entry:
        return
    previews = []
    for s in entry.get("source_files", []):
        index = s["index"]
        src_path = s["path"]
        preview_path = _preview_path_for(job_id, index)
        ok = _create_preview(src_path, preview_path)
        if ok:
            previews.append({"index": index, "path": preview_path, "filename": os.path.basename(preview_path)})
    entry["source_previews"] = previews


@app.post("/api/upload")
async def upload_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    mode: str = Query("2cam"),
    moving_camera: str = Query("true"),
    enable_detection: str = Query("true"),
    use_timestamps: str = Query("false"),
):
    # Parse string booleans from query params
    moving_camera_bool = parse_bool(moving_camera)
    enable_detection_bool = parse_bool(enable_detection)
    use_timestamps_bool = parse_bool(use_timestamps)

    logger.info(
        f"Stitching options: moving_camera={moving_camera_bool}, enable_detection={enable_detection_bool}, use_timestamps={use_timestamps_bool}"
    )
    if len(files) not in [2, 3]:
        raise HTTPException(
            status_code=400, detail="Please upload exactly 2 or 3 videos"
        )

    if mode == "2cam" and len(files) != 2:
        raise HTTPException(
            status_code=400, detail="2-camera mode requires exactly 2 videos"
        )

    if mode == "3cam" and len(files) != 3:
        raise HTTPException(
            status_code=400, detail="3-camera mode requires exactly 3 videos"
        )

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    file_paths = []
    source_files = []

    for i, file in enumerate(files):
        ext = os.path.splitext(file.filename)[1]
        save_name = f"cam_{i}{ext}"
        path = os.path.join(job_dir, save_name)

        # Read file content using async await (proper FastAPI UploadFile handling)
        content = await file.read()

        # Write to disk synchronously to ensure complete write
        with open(path, "wb") as buffer:
            buffer.write(content)
            buffer.flush()  # Flush to disk

        # Verify file was written correctly
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {file.filename}"
            )

        logger.info(f"Saved {file.filename} ({len(content)} bytes) to {path}")
        file_paths.append(path)
        source_files.append({"index": i, "filename": save_name, "path": path})

    # Store options in job
    jobs[job_id] = {
        "status": JobStatus.QUEUED,
        "progress": 0,
        "message": "Queued for processing",
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "job_dir": job_dir,
        "source_files": source_files,
        "options": {
            "moving_camera": moving_camera_bool,
            "enable_detection": enable_detection_bool,
            "use_timestamps": use_timestamps_bool,
        },
    }

    background_tasks.add_task(
        run_stitching_job,
        job_id,
        file_paths,
        mode,
        moving_camera_bool,
        enable_detection_bool,
        use_timestamps_bool,
    )

    return {"job_id": job_id, "message": "Upload successful, processing started"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/sources/{job_id}")
async def list_sources(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    if not entry.get("source_previews"):
        _ensure_source_previews(job_id)
    sources = [
        {
            "index": s["index"],
            "filename": s["filename"],
            "url": f"/api/source/{job_id}/{s['index']}",
            "preview_url": f"/api/source/{job_id}/{s['index']}?preview=true",
        }
        for s in entry.get("source_files", [])
    ]
    return {"mode": entry.get("mode"), "sources": sources}


@app.get("/api/source/{job_id}/{index}")
async def get_source(job_id: str, index: int, download: bool = False, preview: bool = False):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    entry = jobs[job_id]
    file_path = None
    out_name = None
    if preview:
        if not entry.get("source_previews"):
            _ensure_source_previews(job_id)
        matches = [p for p in entry.get("source_previews", []) if p["index"] == index]
        if matches:
            file_path = matches[0]["path"]
            out_name = matches[0]["filename"]
    if not file_path:
        matches = [s for s in entry.get("source_files", []) if s["index"] == index]
        if not matches:
            raise HTTPException(status_code=404, detail="Source not found")
        file_path = matches[0]["path"]
        out_name = matches[0]["filename"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = _guess_media_type(file_path)
    disposition_type = "attachment" if download else "inline"
    return FileResponse(
        file_path,
        filename=out_name,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"{disposition_type}; filename={out_name}",
        },
    )

@app.get("/api/download/{filename}")
async def download_file(filename: str, download: bool = False):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type based on file extension
    media_type = "video/mp4"
    if filename.endswith(".avi"):
        media_type = "video/x-msvideo"

    disposition_type = "attachment" if download else "inline"

    return FileResponse(
        file_path,
        filename=filename,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"{disposition_type}; filename={filename}",
        },
    )


@app.get("/")
def read_root():
    return {"message": "Video Stitching API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
