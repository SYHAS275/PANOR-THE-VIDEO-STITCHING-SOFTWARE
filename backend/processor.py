import os
import sys
import logging
import asyncio
from typing import List, Callable, Optional

# Add parent directory to path to import pano
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


import cv2
import numpy as np


async def process_video_job(
    job_id: str,
    file_paths: List[str],
    mode: str,
    progress_callback: Callable[[float, str], None],
    moving_camera: bool = True,
    enable_detection: bool = True,
    use_timestamps: bool = False,
):
    """
    Wraps the stitching logic from pano.py.
    Run in a threadpool because the original code is blocking/CPU dense.
    """

    # Define output path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"panorama_{job_id}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    logger.info(f"Starting job {job_id} in mode {mode}")
    logger.info(f"Files: {file_paths}")
    logger.info(
        f"Options: moving_camera={moving_camera}, enable_detection={enable_detection}, use_timestamps={use_timestamps}"
    )
    logger.info(f"Output: {output_path}")

    try:
        # Import panor module (GPU-Accelerated Video Panorama Stitcher)
        import importlib

        pano_module = importlib.import_module("panor")

        # Run the blocking stitching function in a separate thread
        loop = asyncio.get_event_loop()

        def run_stitch():
            # Log the actual options being used - use print for immediate visibility
            print(f"\n{'=' * 60}")
            print(f"STITCHING OPTIONS RECEIVED:")
            print(f"  moving_camera   = {moving_camera}")
            print(f"  enable_detection = {enable_detection}")
            print(f"  use_timestamps  = {use_timestamps}")
            print(f"{'=' * 60}\n")
            logger.info(
                f"Stitching with options: moving_camera={moving_camera}, enable_detection={enable_detection}, use_timestamps={use_timestamps}"
            )

            # Common params - pass through the actual option values
            params = {
                "output_name": output_path,
                "show_preview": False,  # HEADLESS MODE (GUI crashes in threads)
                "moving_camera": moving_camera,
                "enable_detection": enable_detection,
                "detection_model": "n",  # Nano is fast
                "enable_tracking": enable_detection,  # Enable tracking when detection is enabled
                "sync_method": "nearest",
                "timestamp_files": None,  # Timestamp files not supported via web UI yet
                "target_fps": None,  # Use source FPS
            }

            # Inject progress and preview callbacks
            if hasattr(pano_module, "set_progress_callback"):
                pano_module.set_progress_callback(progress_callback)

            if mode == "2cam":
                return pano_module.stitch_videos_2cam(
                    left_video_path=file_paths[0],
                    right_video_path=file_paths[1],
                    **params,
                )
            elif mode == "3cam":
                return pano_module.stitch_videos_3cam(
                    left_video_path=file_paths[0],
                    center_video_path=file_paths[1],
                    right_video_path=file_paths[2],
                    **params,
                )
            elif mode == "cylindrical":
                # Import cylindrical module dynamically and reload to pick up changes
                cylindrical_module = importlib.import_module("cylindrical")
                importlib.reload(cylindrical_module)

                # Set progress callback if available
                if hasattr(cylindrical_module, "set_progress_callback"):
                    cylindrical_module.set_progress_callback(progress_callback)

                return cylindrical_module.stitch_video_cylindrical(
                    left_video_path=file_paths[0],
                    right_video_path=file_paths[1],
                    **params,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Execute in thread pool
        output_file_path = await asyncio.to_thread(run_stitch)

        # The panor.py returns the full path of the output file
        # Extract just the filename for the API response
        if output_file_path and os.path.exists(output_file_path):
            actual_filename = os.path.basename(output_file_path)
            logger.info(f"Stitching completed successfully: {actual_filename}")
            return actual_filename
        else:
            # Fallback: check if our expected output exists
            if os.path.exists(output_path):
                logger.info(f"Using expected output file: {output_filename}")
                return output_filename
            # Check for temp avi file (ffmpeg encoding may have failed)
            temp_avi = output_path.replace(".mp4", "_temp.avi")
            if os.path.exists(temp_avi):
                logger.warning(f"MP4 encoding failed, using temp AVI: {temp_avi}")
                return os.path.basename(temp_avi)
            raise FileNotFoundError(f"Output file not found: {output_path}")

    except Exception as e:
        logger.error(f"Stitching failed: {e}")
        raise e
