#!/usr/bin/env python3
"""
GPU-Accelerated Video Panorama Stitcher with CNN Object Detection
Unified version supporting both 2-camera and 3-camera stitching with static/moving modes.

Features:
- Interactive terminal menu for configuration
- Support for 2-camera and 3-camera setups
- Static and moving camera modes
- NVDEC GPU video decoding
- CUDA-accelerated warping, color conversion, and blending
- LAB color space matching with luminance protection
- Parallax-aware blending
- NVENC GPU video encoding
- YOLOv8 CNN object detection (humans, cars, trucks, motorcycles, etc.)
- Optional object tracking with IDs
"""

import numpy as np
import cv2
import subprocess
from datetime import datetime
import time
import os
from threading import Thread
from queue import Queue
import logging
import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL CALLBACKS
# =============================================================================

_progress_callback = None


def set_progress_callback(callback):
    """Set a callback function(progress, message) for progress updates."""
    global _progress_callback
    _progress_callback = callback


# =============================================================================
# FRAME RATE NORMALIZER - Sync handling for different FPS videos
# =============================================================================


class SyncedVideoReader:
    """
    Synchronized video reader that normalizes frame rates across multiple videos.
    Ensures all videos output frames at the same target FPS, handling:
    - Different source frame rates
    - Frame dropping for faster sources
    - Frame duplication for slower sources
    """

    def __init__(self, video_paths, target_fps=None, sync_method="nearest"):
        """
        Initialize synchronized video reader.

        Args:
            video_paths: List of video file paths
            target_fps: Target output FPS (None = use highest source FPS)
            sync_method: 'nearest' (nearest frame) or 'interpolate' (blend frames)
        """
        self.video_paths = video_paths
        self.sync_method = sync_method
        self.caps = []
        self.source_fps = []
        self.frame_counts = []
        self.durations = []
        self.current_time = 0.0
        self.frame_buffers = []  # Store previous frames for interpolation

        # Open all videos and get properties
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            self.caps.append(cap)
            self.source_fps.append(fps)
            self.frame_counts.append(frame_count)
            self.durations.append(duration)
            self.frame_buffers.append(
                {"prev": None, "curr": None, "prev_time": -1, "curr_time": 0}
            )

        # Determine target FPS
        if target_fps is None:
            self.target_fps = max(self.source_fps)
        else:
            self.target_fps = target_fps

        self.frame_duration = 1.0 / self.target_fps
        self.min_duration = min(self.durations)
        self.total_frames = int(self.min_duration * self.target_fps)

        # Pre-read first frames
        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if ret:
                self.frame_buffers[i]["curr"] = frame
                self.frame_buffers[i]["curr_time"] = 0

        # Log sync info
        logger.info("=" * 60)
        logger.info("FRAME RATE NORMALIZATION")
        logger.info("=" * 60)
        for i, path in enumerate(video_paths):
            name = os.path.basename(path)
            logger.info(f"  Video {i + 1}: {name}")
            logger.info(
                f"    FPS: {self.source_fps[i]:.2f}, Frames: {self.frame_counts[i]}, Duration: {self.durations[i]:.2f}s"
            )
        logger.info(f"  Target FPS: {self.target_fps:.2f}")
        logger.info(f"  Output frames: {self.total_frames}")
        logger.info(f"  Sync method: {self.sync_method}")

        # Check for significant FPS differences
        fps_variance = max(self.source_fps) - min(self.source_fps)
        if fps_variance > 1.0:
            logger.warning(
                f"  FPS variance: {fps_variance:.2f} - sync correction active"
            )
        else:
            logger.info(f"  FPS variance: {fps_variance:.2f} - videos well matched")
        logger.info("=" * 60)

    def read(self):
        """
        Read synchronized frames from all videos.

        Returns:
            Tuple of (success, list_of_frames)
            success is False when any video ends
        """
        if self.current_time >= self.min_duration:
            return False, [None] * len(self.caps)

        frames = []

        for i, cap in enumerate(self.caps):
            frame = self._get_frame_at_time(i, self.current_time)
            if frame is None:
                return False, [None] * len(self.caps)
            frames.append(frame)

        self.current_time += self.frame_duration
        return True, frames

    def _get_frame_at_time(self, video_idx, target_time):
        """Get frame from video at specific time, using appropriate sync method."""
        cap = self.caps[video_idx]
        fps = self.source_fps[video_idx]
        buffer = self.frame_buffers[video_idx]

        # Calculate which source frame corresponds to target time
        source_frame_float = target_time * fps
        source_frame = int(source_frame_float)

        # Current position in source video
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Need to advance to get the right frame?
        while buffer["curr_time"] < target_time:
            # Shift current to previous
            buffer["prev"] = buffer["curr"]
            buffer["prev_time"] = buffer["curr_time"]

            # Read next frame
            ret, frame = cap.read()
            if not ret:
                # End of video - return last known frame
                return buffer["curr"] if buffer["curr"] is not None else buffer["prev"]

            buffer["curr"] = frame
            buffer["curr_time"] = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps

        # Return frame based on sync method
        if self.sync_method == "nearest":
            # Return nearest frame
            if buffer["prev"] is None:
                return buffer["curr"]

            prev_diff = abs(target_time - buffer["prev_time"])
            curr_diff = abs(target_time - buffer["curr_time"])

            return buffer["prev"] if prev_diff < curr_diff else buffer["curr"]

        elif self.sync_method == "interpolate":
            # Blend between frames based on time position
            if buffer["prev"] is None or buffer["curr"] is None:
                return buffer["curr"] if buffer["curr"] is not None else buffer["prev"]

            if buffer["curr_time"] == buffer["prev_time"]:
                return buffer["curr"]

            # Calculate blend factor
            alpha = (target_time - buffer["prev_time"]) / (
                buffer["curr_time"] - buffer["prev_time"]
            )
            alpha = np.clip(alpha, 0.0, 1.0)

            # Blend frames
            blended = cv2.addWeighted(
                buffer["prev"], 1.0 - alpha, buffer["curr"], alpha, 0
            )
            return blended

        return buffer["curr"]

    def get_fps(self):
        """Get target output FPS."""
        return self.target_fps

    def get_frame_count(self):
        """Get total output frame count."""
        return self.total_frames

    def get_progress(self):
        """Get current progress (0.0 to 1.0)."""
        return self.current_time / self.min_duration if self.min_duration > 0 else 0

    def release(self):
        """Release all video captures."""
        for cap in self.caps:
            cap.release()
        self.caps = []
        self.frame_buffers = []


# =============================================================================
# TIMESTAMP-BASED SYNC - For ROS bag extracted videos with exact timestamps
# =============================================================================


class TimestampSyncedVideoReader:
    """
    Synchronized video reader using exact nanosecond timestamps from ROS bags.
    Provides precise frame matching based on capture time, not frame index.

    This handles:
    - Different start times between cameras
    - Variable frame rates (non-constant intervals)
    - Dropped frames
    - Different total frame counts
    - Partial timestamp files (interpolates missing timestamps)
    """

    def __init__(
        self,
        video_paths,
        timestamp_files=None,
        timestamps_dict=None,
        target_fps=15.0,
        sync_method="nearest",
        max_time_diff_ms=50.0,
    ):
        """
        Initialize timestamp-synchronized video reader.

        Args:
            video_paths: List of video file paths
            timestamp_files: List of timestamp file paths (one per video)
                            Supports both complete timestamp lists and
                            partial ROS analysis files (timestamps_output.txt)
            timestamps_dict: Dict mapping video path to list of timestamps (ns)
                            Alternative to timestamp_files
            target_fps: Output FPS for the stitched video
            sync_method: 'nearest' or 'interpolate'
            max_time_diff_ms: Max allowed time difference for frame matching (ms)
        """
        self.video_paths = video_paths
        self.sync_method = sync_method
        self.target_fps = target_fps
        self.max_time_diff_ns = max_time_diff_ms * 1_000_000  # Convert to ns

        self.caps = []
        self.timestamps = []  # List of timestamp arrays (one per video)
        self.frame_counts = []

        # Load videos and timestamps
        for i, path in enumerate(video_paths):
            # Open video
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.caps.append(cap)
            self.frame_counts.append(frame_count)

            # Load timestamps
            if timestamps_dict and path in timestamps_dict:
                ts = np.array(timestamps_dict[path], dtype=np.int64)
            elif timestamp_files and i < len(timestamp_files):
                # Use the new load function that handles multiple formats
                ts_list, file_total = load_timestamp_file(timestamp_files[i])

                # Determine actual frame count (prefer file info if available)
                actual_frame_count = file_total if file_total else frame_count

                # Check if we have complete or partial timestamps
                if len(ts_list) < actual_frame_count:
                    logger.warning(
                        f"Only {len(ts_list)} timestamps for {actual_frame_count} frames in {os.path.basename(path)}"
                    )
                    logger.info(f"Interpolating missing timestamps...")
                    ts_list = generate_complete_timestamps(
                        ts_list, actual_frame_count, video_fps
                    )

                ts = np.array(ts_list, dtype=np.int64)
            else:
                # Generate synthetic timestamps based on FPS
                fps = video_fps
                ts = np.arange(frame_count) * (1_000_000_000 / fps)
                ts = ts.astype(np.int64)
                logger.warning(
                    f"No timestamps for {os.path.basename(path)}, using FPS-based estimation"
                )

            # Truncate if too many timestamps
            if len(ts) > frame_count:
                ts = ts[:frame_count]
            # Pad if not enough (shouldn't happen after interpolation, but just in case)
            elif len(ts) < frame_count:
                avg_interval = (
                    np.mean(np.diff(ts))
                    if len(ts) > 1
                    else int(1_000_000_000 / video_fps)
                )
                while len(ts) < frame_count:
                    ts = np.append(ts, ts[-1] + avg_interval)

            self.timestamps.append(ts)

        # Find common time range
        self.start_times = [ts[0] for ts in self.timestamps]
        self.end_times = [ts[-1] for ts in self.timestamps]

        # Global start = latest start (so all cameras have data)
        # Global end = earliest end (so all cameras have data)
        self.global_start = max(self.start_times)
        self.global_end = min(self.end_times)

        if self.global_start >= self.global_end:
            raise ValueError("No overlapping time range between videos!")

        self.duration_ns = self.global_end - self.global_start
        self.duration_s = self.duration_ns / 1_000_000_000

        # Calculate output frames
        self.frame_duration_ns = int(1_000_000_000 / target_fps)
        self.total_frames = int(self.duration_s * target_fps)

        # Current position
        self.current_time_ns = self.global_start
        self.current_frame = 0

        # Frame index pointers for each video
        self.frame_indices = [0] * len(video_paths)

        # Initialize frame indices to start of common time range
        for i in range(len(video_paths)):
            idx = np.searchsorted(self.timestamps[i], self.global_start)
            self.frame_indices[i] = max(0, idx - 1)

        # Log sync info
        self._log_sync_info()

    def _log_sync_info(self):
        """Log synchronization information."""
        logger.info("=" * 60)
        logger.info("TIMESTAMP-BASED SYNCHRONIZATION (ROS Bag Mode)")
        logger.info("=" * 60)

        for i, path in enumerate(self.video_paths):
            name = os.path.basename(path)
            start_offset = (self.start_times[i] - self.global_start) / 1_000_000  # ms
            end_offset = (self.end_times[i] - self.global_end) / 1_000_000  # ms

            # Calculate average FPS from timestamps
            if len(self.timestamps[i]) > 1:
                intervals = np.diff(self.timestamps[i])
                avg_fps = 1_000_000_000 / np.mean(intervals)
                fps_std = (
                    1_000_000_000 / np.std(intervals) if np.std(intervals) > 0 else 0
                )
            else:
                avg_fps = 0
                fps_std = 0

            logger.info(f"  Video {i + 1}: {name}")
            logger.info(f"    Frames: {self.frame_counts[i]}")
            logger.info(f"    Avg FPS: {avg_fps:.2f} (variable)")
            logger.info(f"    Start offset: {start_offset:+.1f}ms from common start")

        logger.info(f"  Common time range: {self.duration_s:.3f}s")
        logger.info(f"  Output FPS: {self.target_fps:.2f}")
        logger.info(f"  Output frames: {self.total_frames}")
        logger.info(f"  Sync method: {self.sync_method}")
        logger.info(f"  Max time diff: {self.max_time_diff_ns / 1_000_000:.1f}ms")
        logger.info("=" * 60)

    def read(self):
        """
        Read synchronized frames from all videos at current timestamp.

        Returns:
            Tuple of (success, list_of_frames)
        """
        if self.current_time_ns >= self.global_end:
            return False, [None] * len(self.caps)

        frames = []

        for i in range(len(self.caps)):
            frame = self._get_frame_at_timestamp(i, self.current_time_ns)
            if frame is None:
                return False, [None] * len(self.caps)
            frames.append(frame)

        self.current_time_ns += self.frame_duration_ns
        self.current_frame += 1

        return True, frames

    def _get_frame_at_timestamp(self, video_idx, target_time_ns):
        """Get the frame closest to the target timestamp."""
        cap = self.caps[video_idx]
        timestamps = self.timestamps[video_idx]

        # Find closest frame using binary search
        idx = np.searchsorted(timestamps, target_time_ns)

        # Check both idx-1 and idx for closest match
        if idx == 0:
            best_idx = 0
        elif idx >= len(timestamps):
            best_idx = len(timestamps) - 1
        else:
            diff_prev = abs(timestamps[idx - 1] - target_time_ns)
            diff_curr = abs(timestamps[idx] - target_time_ns)
            best_idx = idx - 1 if diff_prev < diff_curr else idx

        # Check if within acceptable time difference
        time_diff = abs(timestamps[best_idx] - target_time_ns)
        if time_diff > self.max_time_diff_ns:
            logger.debug(
                f"Video {video_idx}: Large time diff {time_diff / 1_000_000:.1f}ms at frame {best_idx}"
            )

        # Seek to frame if needed
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != best_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)

        ret, frame = cap.read()
        if not ret:
            return None

        # For interpolation, we could blend with adjacent frame
        if self.sync_method == "interpolate" and 0 < best_idx < len(timestamps) - 1:
            # Determine if we should blend with previous or next frame
            if target_time_ns < timestamps[best_idx]:
                other_idx = best_idx - 1
            else:
                other_idx = best_idx + 1

            # Calculate blend factor
            t1 = timestamps[min(best_idx, other_idx)]
            t2 = timestamps[max(best_idx, other_idx)]

            if t2 != t1:
                alpha = (target_time_ns - t1) / (t2 - t1)
                alpha = np.clip(alpha, 0.0, 1.0)

                # Only blend if alpha is significant
                if 0.1 < alpha < 0.9:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, other_idx)
                    ret2, frame2 = cap.read()
                    if ret2:
                        if other_idx < best_idx:
                            frame = cv2.addWeighted(
                                frame2, 1.0 - alpha, frame, alpha, 0
                            )
                        else:
                            frame = cv2.addWeighted(
                                frame, 1.0 - alpha, frame2, alpha, 0
                            )

        return frame

    def get_fps(self):
        """Get target output FPS."""
        return self.target_fps

    def get_frame_count(self):
        """Get total output frame count."""
        return self.total_frames

    def get_progress(self):
        """Get current progress (0.0 to 1.0)."""
        return self.current_frame / self.total_frames if self.total_frames > 0 else 0

    def release(self):
        """Release all video captures."""
        for cap in self.caps:
            cap.release()
        self.caps = []


def parse_ros_timestamp_file(filepath):
    """
    Parse a ROS bag timestamp analysis file (like timestamps_output.txt).

    Returns:
        Dict mapping topic name to list of timestamps (ns)
    """
    timestamps_by_topic = {}
    current_topic = None
    in_timestamp_section = False
    total_messages = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Detect topic header
            if line.startswith("Topic:"):
                current_topic = line.split("Topic:")[1].strip()
                timestamps_by_topic[current_topic] = []
                in_timestamp_section = False

            # Get total message count
            elif current_topic and line.startswith("Total messages:"):
                try:
                    total_messages[current_topic] = int(line.split(":")[1].strip())
                except:
                    pass

            # Detect timestamp section
            elif "timestamps (relative to first frame):" in line:
                in_timestamp_section = True

            # Parse timestamp line
            elif in_timestamp_section and line.startswith("Frame"):
                # Format: "Frame    0: 1764074637062348273 ns (t=  0.0000s)"
                try:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        ts_part = parts[1].strip().split()[0]
                        timestamp = int(ts_part)
                        timestamps_by_topic[current_topic].append(timestamp)
                except (ValueError, IndexError):
                    continue

            # End of timestamp section
            elif in_timestamp_section and not line.startswith("Frame") and line:
                in_timestamp_section = False

    return timestamps_by_topic, total_messages


def load_timestamp_file(filepath):
    """
    Load timestamps from a file. Supports multiple formats:

    1. Simple format (one timestamp per line):
       1764074637062348273
       1764074637123829271
       ...

    2. ROS analysis format (timestamps_output.txt style - partial)

    3. Detailed format with header (ALL FRAME TIMESTAMPS):
       Frame    Timestamp (ns)            Relative Time (s)  DateTime
       0        1764074637109255999       0.000000           2025-11-25 12:43:57
       ...

    Returns:
        Tuple of (list of timestamps in nanoseconds, total_frame_count or None)
    """
    timestamps = []
    total_frames = None

    with open(filepath, "r") as f:
        content = f.read()

    # Check format type
    if "ALL FRAME TIMESTAMPS" in content:
        # Detailed format with header - parse the table
        logger.info(f"Parsing detailed timestamp file: {os.path.basename(filepath)}")

        in_data_section = False
        for line in content.split("\n"):
            line = line.strip()

            # Get total frames from header
            if line.startswith("Total frames:"):
                try:
                    total_frames = int(line.split(":")[1].strip())
                except:
                    pass

            # Start of data section (after the dashed line)
            if line.startswith("--------"):
                in_data_section = True
                continue

            # End of data section
            if in_data_section and (
                line.startswith("====") or line.startswith("END OF FILE") or not line
            ):
                if line.startswith("====") or line.startswith("END"):
                    break
                continue

            # Parse data row: "0        1764074637109255999       0.000000           2025-11-25..."
            if in_data_section and line and line[0].isdigit():
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        timestamp = int(parts[1])
                        timestamps.append(timestamp)
                except (ValueError, IndexError):
                    continue

        logger.info(f"Loaded {len(timestamps)} timestamps (expected: {total_frames})")

    elif "ROS Bag Timestamp Analysis" in content or (
        "Topic:" in content and "ALL FRAME TIMESTAMPS" not in content
    ):
        # Old partial format - parse and warn
        logger.warning(
            f"File appears to be a ROS analysis summary with only partial timestamps"
        )

        parsed, totals = parse_ros_timestamp_file(filepath)
        if parsed:
            first_topic = list(parsed.keys())[0]
            timestamps = parsed[first_topic]
            total_frames = totals.get(first_topic)
            logger.info(
                f"Loaded {len(timestamps)} timestamps from topic: {first_topic}"
            )
            if total_frames:
                logger.info(f"Total frames in video: {total_frames}")

    else:
        # Simple format - one timestamp per line
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and line[0].isdigit():
                try:
                    # Handle lines that might have extra columns
                    parts = line.split()
                    ts = int(parts[0])
                    timestamps.append(ts)
                except ValueError:
                    continue

    return timestamps, total_frames


def generate_complete_timestamps(partial_timestamps, total_frames, avg_fps=None):
    """
    Generate complete timestamp list from partial timestamps (first N and last N).
    Uses linear interpolation for missing middle frames.

    Args:
        partial_timestamps: List of known timestamps (may be incomplete)
        total_frames: Total number of frames needed
        avg_fps: Average FPS (used to estimate intervals if not enough data)

    Returns:
        Complete list of timestamps for all frames
    """
    if len(partial_timestamps) >= total_frames:
        return partial_timestamps[:total_frames]

    # If we have first and last timestamps, interpolate
    if len(partial_timestamps) >= 2:
        first_ts = partial_timestamps[0]
        last_ts = partial_timestamps[-1]

        # Check if partial list has both early and late frames
        # (like first 10 and last 10 from your file)
        if len(partial_timestamps) >= 10:
            # Use average interval from available data for better estimation
            intervals = np.diff(partial_timestamps)
            avg_interval = np.mean(intervals)

            # Generate timestamps using average interval
            timestamps = [first_ts + int(i * avg_interval) for i in range(total_frames)]

            # Adjust to match the last known timestamp
            if timestamps[-1] != last_ts:
                # Scale to fit actual duration
                actual_duration = last_ts - first_ts
                for i in range(len(timestamps)):
                    timestamps[i] = first_ts + int(
                        actual_duration * i / (total_frames - 1)
                    )

            return timestamps

        # Simple linear interpolation
        timestamps = []
        for i in range(total_frames):
            t = first_ts + (last_ts - first_ts) * i / (total_frames - 1)
            timestamps.append(int(t))

        return timestamps

    # Fallback: generate based on FPS
    if avg_fps and len(partial_timestamps) >= 1:
        interval_ns = int(1_000_000_000 / avg_fps)
        first_ts = partial_timestamps[0]
        return [first_ts + i * interval_ns for i in range(total_frames)]

    raise ValueError("Not enough timestamp data to generate complete list")


# Get process handle for memory monitoring
PROCESS = psutil.Process(os.getpid())

# Initialize NVIDIA GPU monitoring
GPU_AVAILABLE = False
GPU_HANDLE = None
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
    logger.info("GPU monitoring enabled (pynvml)")
except ImportError:
    logger.debug("pynvml not installed - GPU monitoring disabled")
except Exception as e:
    logger.debug(f"GPU monitoring unavailable: {e}")


# =============================================================================
# OBJECT DETECTOR CLASS (YOLOv8-based CNN)
# =============================================================================


class ObjectDetector:
    """
    CNN-based object detector using YOLOv8.
    Detects humans, cars, trucks, motorcycles, bicycles, buses, etc.
    """

    # COCO class names (80 classes)
    COCO_CLASSES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # Default classes to detect (indices in COCO) - vehicles and people
    DEFAULT_DETECT_CLASSES = [
        0,
        1,
        2,
        3,
        5,
        7,
    ]  # person, bicycle, car, motorcycle, bus, truck

    # Colors for different classes (BGR format)
    CLASS_COLORS = {
        "person": (0, 255, 0),  # Green
        "bicycle": (255, 165, 0),  # Orange
        "car": (255, 0, 0),  # Blue
        "motorcycle": (0, 255, 255),  # Yellow
        "bus": (255, 0, 255),  # Magenta
        "truck": (0, 165, 255),  # Orange-red
        "traffic light": (0, 128, 255),
        "stop sign": (0, 0, 255),  # Red
        "default": (128, 128, 128),  # Gray
    }

    def __init__(
        self,
        model_size="n",
        confidence=0.3,
        iou_threshold=0.5,
        detect_classes=None,
        use_gpu=True,
        enable_tracking=True,
    ):
        """
        Initialize the object detector.

        Args:
            model_size: YOLOv8 model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
            confidence: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            detect_classes: List of class indices to detect (None = default vehicle/person classes)
            use_gpu: Use GPU for inference if available
            enable_tracking: Enable ByteTrack object tracking
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.detect_classes = detect_classes  # If None, detect ALL classes
        self.enable_tracking = enable_tracking
        self.use_gpu = use_gpu
        self.model_size = model_size

        # Detection stats
        self.detection_counts = defaultdict(int)
        self.frame_count = 0
        self.total_detections = 0

        # Model initialization
        self.model = None
        self.backend = None

        self._init_detector(model_size)

    def _init_detector(self, model_size):
        """Initialize the detection model (YOLOv8 via ultralytics)."""

        # Try YOLOv8 (ultralytics) first - best option
        try:
            from ultralytics import YOLO

            model_name = f"yolov8{model_size}.pt"

            logger.info(f"Loading YOLOv8 model: {model_name}")
            self.model = YOLO(model_name)

            # Set device
            if self.use_gpu:
                try:
                    import torch

                    if torch.cuda.is_available():
                        self.model.to("cuda")
                        logger.info("YOLOv8 running on CUDA GPU")
                    else:
                        logger.info("CUDA not available, YOLOv8 running on CPU")
                except ImportError:
                    logger.info("PyTorch not found, YOLOv8 running on CPU")

            self.backend = "yolov8"
            logger.info(
                f"Object detector initialized: YOLOv8-{model_size} (conf={self.confidence})"
            )
            if self.detect_classes:
                logger.info(
                    f"Detecting classes: {[self.COCO_CLASSES[i] for i in self.detect_classes]}"
                )
            else:
                logger.info("Detecting ALL classes")
            return

        except ImportError:
            logger.warning(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.warning(f"YOLOv8 initialization failed: {e}")

        # Fallback: Try OpenCV DNN with YOLOv4
        try:
            logger.info("Trying OpenCV DNN backend...")
            weights_file = "yolov4-tiny.weights"
            cfg_file = "yolov4-tiny.cfg"

            if not os.path.exists(weights_file):
                logger.info("YOLOv4-tiny weights not found. Download from:")
                logger.info(
                    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
                )
                logger.info(
                    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
                )
                raise FileNotFoundError("YOLOv4 weights not found")

            self.model = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

            if self.use_gpu:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("OpenCV DNN using CUDA backend")
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.backend = "opencv_dnn"
            logger.info("Object detector initialized: YOLOv4-tiny via OpenCV DNN")
            return

        except Exception as e:
            logger.warning(f"OpenCV DNN initialization failed: {e}")

        logger.error("No object detection backend available!")
        logger.error("Install ultralytics: pip install ultralytics")
        self.backend = None

    def detect(self, frame):
        """Detect objects in a frame."""
        if self.model is None:
            return []

        self.frame_count += 1
        detections = []

        if self.backend == "yolov8":
            if self.enable_tracking:
                detections = self._detect_yolov8_with_bytetrack(frame)
            else:
                detections = self._detect_yolov8(frame)
        elif self.backend == "opencv_dnn":
            detections = self._detect_opencv_dnn(frame)

        self.total_detections += len(detections)
        for det in detections:
            class_name = det[1]
            self.detection_counts[class_name] += 1

        return detections

    def _detect_yolov8_with_bytetrack(self, frame):
        """Detection with YOLOv8 + ByteTrack tracking."""
        detections = []

        # Use ByteTrack via YOLOv8's built-in tracking
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.detect_classes,
            verbose=False,
            persist=True,  # Keep track IDs persistent across frames
            tracker="bytetrack.yaml",  # Use ByteTrack
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.COCO_CLASSES[class_id]

                if box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())
                else:
                    track_id = -1

                detections.append(
                    (class_id, class_name, conf, x1, y1, x2, y2, track_id)
                )

        return detections

    def _detect_yolov8(self, frame):
        """Detection using YOLOv8 (ultralytics)."""
        detections = []

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.detect_classes,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.COCO_CLASSES[class_id]

                detections.append((class_id, class_name, conf, x1, y1, x2, y2, -1))

        return detections

    def _detect_opencv_dnn(self, frame):
        """Detection using OpenCV DNN backend."""
        detections = []
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.model.setInput(blob)

        layer_names = self.model.getLayerNames()
        output_layers = [
            layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()
        ]

        outputs = self.model.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence and class_id in self.detect_classes:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)

                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)

                    boxes.append([x1, y1, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence, self.iou_threshold
        )

        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)):
                i = i[0]
            x, y, w_box, h_box = boxes[i]
            class_id = class_ids[i]
            conf = confidences[i]
            class_name = self.COCO_CLASSES[class_id]

            detections.append(
                (class_id, class_name, conf, x, y, x + w_box, y + h_box, -1)
            )

        return detections

    def draw_detections(
        self,
        frame,
        detections,
        show_labels=True,
        show_confidence=True,
        show_track_id=True,
        line_thickness=2,
    ):
        """Draw detection boxes and labels on frame."""
        for det in detections:
            class_id, class_name, conf, x1, y1, x2, y2, track_id = det

            color = self.CLASS_COLORS.get(class_name, self.CLASS_COLORS["default"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{conf:.2f}")
            if show_track_id and track_id >= 0:
                label_parts.append(f"ID:{track_id}")

            if label_parts:
                label = " ".join(label_parts)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 8, y1),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

        return frame

    def get_stats(self):
        """Get detection statistics."""
        return {
            "total_frames": self.frame_count,
            "total_detections": self.total_detections,
            "detections_per_class": dict(self.detection_counts),
            "avg_detections_per_frame": self.total_detections
            / max(1, self.frame_count),
        }

    def reset_stats(self):
        """Reset detection statistics."""
        self.detection_counts = defaultdict(int)
        self.frame_count = 0
        self.total_detections = 0


# =============================================================================
# CROP HELPER
# =============================================================================


def crop_largest_rectangle(image):
    """
    Crop the largest axis-aligned rectangle from the image that contains
    no black borders (0,0,0).
    Uses a heuristic to shrink the bounding box of non-zero pixels.
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 1. External Bounding Rect (removes outer black sea)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop to the bounding rect
    crop = image[y : y + h, x : x + w]
    # mask = thresh[y:y+h, x:x+w] # OLD: susceptible to holes

    # NEW: Create a filled mask from the contour to ignore internal holes (black video pixels)
    # This ensures we only care about the OUTER shape of the panorama.
    mask_filled = np.zeros_like(gray)
    cv2.drawContours(mask_filled, [c], -1, 255, -1)
    mask = mask_filled[y : y + h, x : x + w]

    # 2. Iteratively shrink to remove remaining black corners/edges
    # We want a rectangle where ALL pixels are non-zero.
    # Heuristic: Check edges. If an edge has zeros, move it in.
    # To avoid infinite loop on noisy images, limit iterations.

    roi_x, roi_y, roi_w, roi_h = 0, 0, w, h

    max_steps = min(w, h) // 2  # Don't shrink to nothing

    steps = 0
    while steps < max_steps:
        # Check if current ROI is valid
        # Optimization: Check only the 4 borders of the mask ROI

        # Extract the current mask ROI
        sub_mask = mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        if sub_mask.size == 0:
            break

        # Count zeros on edges
        # Note: countNonZero returns number of NON-zero.
        # So Zeros = Total - NonZero

        # Top row
        top_zeros = roi_w - cv2.countNonZero(sub_mask[0, :])
        # Bottom row
        bottom_zeros = roi_w - cv2.countNonZero(sub_mask[-1, :])
        # Left col
        left_zeros = roi_h - cv2.countNonZero(sub_mask[:, 0])
        # Right col
        right_zeros = roi_h - cv2.countNonZero(sub_mask[:, -1])

        # Check tolerance (allow 5% black pixels on edges)
        tolerance_w = int(roi_w * 0.05)
        tolerance_h = int(roi_h * 0.05)

        if (
            top_zeros <= tolerance_w
            and bottom_zeros <= tolerance_w
            and left_zeros <= tolerance_h
            and right_zeros <= tolerance_h
        ):
            # Found it!
            break

        # Shrink the side with the most zeros (heuristic)
        # Or simply shrink all bad sides? Shrinking all converges faster.
        # NOTE: Do NOT shrink bottom to preserve content
        if top_zeros > tolerance_w:
            roi_y += 1
            roi_h -= 1
        # Bottom shrinking disabled to prevent over-cropping
        # if bottom_zeros > tolerance_w:
        #     roi_h -= 1
        if left_zeros > tolerance_h:
            roi_x += 1
            roi_w -= 1
        if right_zeros > tolerance_h:
            roi_w -= 1

        if roi_w <= 0 or roi_h <= 0:
            # Fallback to original bounding rect in worst case
            return image[y : y + h, x : x + w]

        steps += 1

    return image[y + roi_y : y + roi_y + roi_h, x + roi_x : x + roi_x + roi_w]


# =============================================================================
# STITCHER CLASS
# =============================================================================


class Stitcher:
    # Thresholds and parameters
    OVERLAP_THRESHOLD_RATIO = 0.15
    FALLBACK_THRESHOLD_RATIO = 0.05
    BLEND_WIDTH = 100
    BLACK_PIXEL_THRESHOLD = 10
    PARALLAX_DIFF_THRESHOLD = 15
    VALID_PIXEL_MIN = 50
    VALID_PIXEL_MAX = 220
    MIN_VALID_PIXELS = 500
    PERSPECTIVE_THRESHOLD = 0.005
    SCALE_MAX = 1.5
    SCALE_MIN = 0.67
    LUMINANCE_DARK_THRESHOLD = 60
    LUMINANCE_BRIGHT_THRESHOLD = 200
    DILATION_KERNEL_SIZE = 40

    # Moving camera parameters
    KEYFRAME_INTERVAL = 30
    HOMOGRAPHY_CHANGE_THRESHOLD = 0.1
    HOMOGRAPHY_BUFFER_SIZE = 5
    MIN_MATCH_COUNT = 50
    FEATURE_COUNT = 8000

    def __init__(self, use_cuda=True, moving_camera=False):
        self.cachedH = None
        self.blend_start = None
        self.blend_end = None
        self.output_width = None
        self.output_height = None
        self.crop_top = None
        self.crop_bottom = None

        # Moving camera support
        self.moving_camera = moving_camera
        self.frame_count = 0
        self.homography_buffer = []
        self.last_match_count = 0

        # Check CUDA availability
        self.use_cuda = use_cuda and self._check_cuda()

        if self.use_cuda:
            self._init_cuda()
        else:
            logger.info("Running on CPU")

        # Feature detector
        n_features = self.FEATURE_COUNT if moving_camera else 5000
        self.detector = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.use_gpu_features = False
        logger.info(f"Using CPU SIFT for feature detection (nfeatures={n_features})")

        if moving_camera:
            logger.info(
                "Moving camera mode enabled - homography will be recalculated periodically"
            )

    def _check_cuda(self):
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                test = cv2.cuda_GpuMat()
                test.upload(np.zeros((10, 10), dtype=np.uint8))
                return True
        except Exception as e:
            logger.debug(f"CUDA check failed: {e}")
        return False

    def _init_cuda(self):
        logger.info(f"CUDA enabled! Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.printCudaDeviceInfo(0)

        self.gpu_frame = cv2.cuda_GpuMat()
        self.gpu_frameB = cv2.cuda_GpuMat()
        self.gpu_warped = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()
        self.gpu_grayB = cv2.cuda_GpuMat()
        self.gpu_result = cv2.cuda_GpuMat()
        self.gpu_temp = cv2.cuda_GpuMat()
        self.gpu_blend_left = cv2.cuda_GpuMat()
        self.gpu_blend_right = cv2.cuda_GpuMat()

        self.stream = cv2.cuda_Stream()

        try:
            self.gpu_gaussian = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (31, 31), 0
            )
            self.gpu_gaussian_small = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC1, cv2.CV_8UC1, (11, 11), 0
            )
            self.has_gpu_filters = True
        except cv2.error as e:
            self.has_gpu_filters = False
            logger.debug(f"GPU Gaussian filter not available: {e}")

        logger.info("CUDA initialized - Full GPU acceleration enabled")

    def _should_recalculate_homography(self):
        if not self.moving_camera:
            return self.cachedH is None

        if self.cachedH is None:
            return True

        if self.frame_count % self.KEYFRAME_INTERVAL == 0:
            return True

        if self.last_match_count < self.MIN_MATCH_COUNT:
            return True

        return False

    def _smooth_homography(self, H):
        self.homography_buffer.append(H.copy())

        if len(self.homography_buffer) > self.HOMOGRAPHY_BUFFER_SIZE:
            self.homography_buffer.pop(0)

        if len(self.homography_buffer) == 1:
            return H

        weights = np.array(
            [i + 1 for i in range(len(self.homography_buffer))], dtype=np.float32
        )
        weights /= weights.sum()

        smoothed_H = np.zeros_like(H, dtype=np.float64)
        for i, h in enumerate(self.homography_buffer):
            smoothed_H += weights[i] * h

        smoothed_H /= smoothed_H[2, 2]

        return smoothed_H.astype(np.float32)

    def _homography_changed_significantly(self, new_H):
        if self.cachedH is None:
            return True

        diff = np.linalg.norm(new_H - self.cachedH, "fro")
        return diff > self.HOMOGRAPHY_CHANGE_THRESHOLD

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images  # left = B, right = A

        self.frame_count += 1

        if self._should_recalculate_homography():
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)

            M = self.matchKeypoints(
                kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh
            )
            if M is None:
                logger.warning("Not enough matches to compute homography")
                if self.moving_camera and self.cachedH is not None:
                    logger.info("Using last known homography")
                    return self.applyWarp(imageA, imageB, self.cachedH)
                return None

            (matches, H, status) = M
            self.last_match_count = len(matches)

            if H is None:
                logger.warning("Homography is None")
                if self.moving_camera and self.cachedH is not None:
                    return self.applyWarp(imageA, imageB, self.cachedH)
                return None

            H = self._constrainHomography(H, imageA.shape, imageB.shape)

            if self.moving_camera:
                H = self._smooth_homography(H)

                if self._homography_changed_significantly(H):
                    self.blend_start = None
                    self.blend_end = None
                    logger.debug(f"Homography updated at frame {self.frame_count}")

            self.cachedH = H.astype(np.float32)

        return self.applyWarp(imageA, imageB, self.cachedH)

    def _constrainHomography(self, H, shapeA, shapeB):
        H = H / H[2, 2]

        if (
            abs(H[2, 0]) > self.PERSPECTIVE_THRESHOLD
            or abs(H[2, 1]) > self.PERSPECTIVE_THRESHOLD
        ):
            H[2, 0] *= 0.5
            H[2, 1] *= 0.5
            H = H / H[2, 2]

        scale_x = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        scale_y = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)

        if scale_x > self.SCALE_MAX or scale_x < self.SCALE_MIN:
            H[0, 0] /= scale_x
            H[1, 0] /= scale_x
        if scale_y > self.SCALE_MAX or scale_y < self.SCALE_MIN:
            H[0, 1] /= scale_y
            H[1, 1] /= scale_y

        H = H / H[2, 2]
        return H

    def applyWarp(self, imageA, imageB, H):
        h, w = imageB.shape[:2]

        corners = np.float32(
            [
                [0, 0],
                [imageA.shape[1], 0],
                [imageA.shape[1], imageA.shape[0]],
                [0, imageA.shape[0]],
            ]
        )
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
        max_x = int(np.max(warped_corners[:, 0, 0]))
        canvas_width = min(max_x + 50, imageA.shape[1] + imageB.shape[1])

        if self.use_cuda:
            try:
                self.gpu_frame.upload(imageA, self.stream)
                gpu_warped = cv2.cuda.warpPerspective(
                    self.gpu_frame,
                    H,
                    (canvas_width, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                    stream=self.stream,
                )

                gpu_warped_gray = cv2.cuda.cvtColor(
                    gpu_warped, cv2.COLOR_BGR2GRAY, stream=self.stream
                )

                self.stream.waitForCompletion()
                warped = gpu_warped.download()
                warped_gray = gpu_warped_gray.download()
            except Exception as e:
                logger.debug(f"CUDA warp failed: {e}, using CPU")
                warped = cv2.warpPerspective(
                    imageA,
                    H,
                    (canvas_width, h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped = cv2.warpPerspective(
                imageA,
                H,
                (canvas_width, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        if self.blend_start is None:
            overlap_start = w
            threshold_pixels = int(h * self.OVERLAP_THRESHOLD_RATIO)

            for x in range(w):
                if (
                    np.count_nonzero(warped_gray[:, x] > self.BLACK_PIXEL_THRESHOLD)
                    >= threshold_pixels
                ):
                    overlap_start = x
                    break

            if overlap_start >= w:
                threshold_pixels = int(h * self.FALLBACK_THRESHOLD_RATIO)
                for x in range(w):
                    if (
                        np.count_nonzero(warped_gray[:, x] > self.BLACK_PIXEL_THRESHOLD)
                        >= threshold_pixels
                    ):
                        overlap_start = x
                        break

            if overlap_start > w - 30:
                overlap_start = max(0, w - self.BLEND_WIDTH)

            overlap_width = w - overlap_start

            blend_center = overlap_start + overlap_width // 2
            self.blend_start = max(0, blend_center - self.BLEND_WIDTH // 2)
            self.blend_end = min(w, blend_center + self.BLEND_WIDTH // 2)

            valid_cols = np.where(
                np.any(warped_gray > self.BLACK_PIXEL_THRESHOLD, axis=0)
            )[0]
            if len(valid_cols) > 0:
                self.output_width = min(valid_cols[-1] + 10, canvas_width)
            else:
                self.output_width = canvas_width
            self.output_height = h

            actual_blend_width = self.blend_end - self.blend_start
            if actual_blend_width > 0:
                mask_1d = np.linspace(0, 1, actual_blend_width, dtype=np.float32)
                mask_1d = mask_1d * mask_1d * (3 - 2 * mask_1d)
                self.gradient_mask = np.tile(mask_1d, (h, 1))
                self.gradient_mask_3 = np.dstack([self.gradient_mask] * 3)

            logger.info(f"Blend region: {self.blend_start} to {self.blend_end}")
            logger.info(f"Output size: {self.output_width}x{self.output_height}")

        blend_start = self.blend_start
        blend_end = self.blend_end
        actual_blend_width = blend_end - blend_start

        result = warped.copy()

        # Color matching
        sample_width = 150
        sample_start = max(0, blend_start - sample_width)
        sample_end = min(w, blend_end + sample_width)

        if sample_end <= sample_start:
            sample_start = max(0, blend_start - 50)
            sample_end = min(w, blend_end + 50)

        sample_left = imageB[:, sample_start:sample_end].copy()
        sample_right = warped[:h, sample_start:sample_end].copy()

        if (
            sample_left.size == 0
            or sample_right.size == 0
            or sample_left.shape[1] < 10
            or sample_right.shape[1] < 10
        ):
            result = warped.copy()
            safe_blend_start = min(blend_start, w)
            result[:h, :safe_blend_start] = imageB[:h, :safe_blend_start]
            if (
                actual_blend_width > 0
                and hasattr(self, "gradient_mask_3")
                and blend_end <= w
            ):
                left_region = imageB[:, blend_start:blend_end].astype(np.float32)
                right_region = result[:h, blend_start:blend_end].astype(np.float32)
                if (
                    left_region.shape
                    == right_region.shape
                    == self.gradient_mask_3.shape
                ):
                    blended = (
                        left_region * (1.0 - self.gradient_mask_3)
                        + right_region * self.gradient_mask_3
                    )
                    result[:h, blend_start:blend_end] = blended.astype(np.uint8)
            return result[: self.output_height, : self.output_width]

        if self.use_cuda:
            try:
                self.gpu_temp.upload(sample_right, self.stream)
                gpu_sample_gray = cv2.cuda.cvtColor(
                    self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream
                )
                self.stream.waitForCompletion()
                right_gray_sample = gpu_sample_gray.download()
            except cv2.error:
                right_gray_sample = cv2.cvtColor(sample_right, cv2.COLOR_BGR2GRAY)
        else:
            right_gray_sample = cv2.cvtColor(sample_right, cv2.COLOR_BGR2GRAY)

        valid_mask = (right_gray_sample > self.VALID_PIXEL_MIN) & (
            right_gray_sample < self.VALID_PIXEL_MAX
        )

        if np.sum(valid_mask) > self.MIN_VALID_PIXELS:
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(sample_left, self.stream)
                    gpu_left_lab = cv2.cuda.cvtColor(
                        self.gpu_temp, cv2.COLOR_BGR2Lab, stream=self.stream
                    )
                    self.gpu_temp.upload(sample_right, self.stream)
                    gpu_right_lab = cv2.cuda.cvtColor(
                        self.gpu_temp, cv2.COLOR_BGR2Lab, stream=self.stream
                    )
                    self.stream.waitForCompletion()
                    left_lab = gpu_left_lab.download().astype(np.float32)
                    right_lab = gpu_right_lab.download().astype(np.float32)
                except cv2.error:
                    left_lab = cv2.cvtColor(sample_left, cv2.COLOR_BGR2LAB).astype(
                        np.float32
                    )
                    right_lab = cv2.cvtColor(sample_right, cv2.COLOR_BGR2LAB).astype(
                        np.float32
                    )
            else:
                left_lab = cv2.cvtColor(sample_left, cv2.COLOR_BGR2LAB).astype(
                    np.float32
                )
                right_lab = cv2.cvtColor(sample_right, cv2.COLOR_BGR2LAB).astype(
                    np.float32
                )

            transfer_params = []
            for c in range(3):
                left_vals = left_lab[:, :, c][valid_mask]
                right_vals = right_lab[:, :, c][valid_mask]

                if len(left_vals) > 100 and len(right_vals) > 100:
                    left_mean, left_std = np.mean(left_vals), np.std(left_vals)
                    right_mean, right_std = np.mean(right_vals), np.std(right_vals)

                    if right_std > 1:
                        scale = np.clip(left_std / right_std, 0.8, 1.2)
                    else:
                        scale = 1.0
                    transfer_params.append((scale, right_mean, left_mean))
                else:
                    transfer_params.append((1.0, 0.0, 0.0))

            if self.use_cuda:
                try:
                    self.gpu_temp.upload(warped, self.stream)
                    gpu_warped_lab = cv2.cuda.cvtColor(
                        self.gpu_temp, cv2.COLOR_BGR2Lab, stream=self.stream
                    )
                    self.stream.waitForCompletion()
                    warped_lab = gpu_warped_lab.download().astype(np.float32)
                except cv2.error:
                    warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(
                        np.float32
                    )
            else:
                warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)

            L_channel = warped_lab[:, :, 0]
            luminance_protection = np.ones_like(L_channel)

            dark_mask = L_channel < self.LUMINANCE_DARK_THRESHOLD
            luminance_protection[dark_mask] = L_channel[dark_mask] / float(
                self.LUMINANCE_DARK_THRESHOLD
            )

            bright_mask = L_channel > self.LUMINANCE_BRIGHT_THRESHOLD
            luminance_protection[bright_mask] = (255 - L_channel[bright_mask]) / (
                255.0 - self.LUMINANCE_BRIGHT_THRESHOLD
            )

            luminance_protection = np.clip(luminance_protection, 0.1, 1.0)

            for c in range(3):
                scale, right_mean, left_mean = transfer_params[c]
                corrected = (warped_lab[:, :, c] - right_mean) * scale + left_mean
                original = warped_lab[:, :, c]
                warped_lab[:, :, c] = (
                    original * (1 - luminance_protection)
                    + corrected * luminance_protection
                )
                warped_lab[:, :, c] = np.clip(warped_lab[:, :, c], 0, 255)

            if self.use_cuda:
                try:
                    self.gpu_temp.upload(warped_lab.astype(np.uint8), self.stream)
                    gpu_warped_bgr = cv2.cuda.cvtColor(
                        self.gpu_temp, cv2.COLOR_Lab2BGR, stream=self.stream
                    )
                    self.stream.waitForCompletion()
                    warped = gpu_warped_bgr.download()
                except cv2.error:
                    warped = cv2.cvtColor(
                        warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
                    )
            else:
                warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Blending
        result = warped.copy()
        safe_blend_start = min(blend_start, w)
        result[:h, :safe_blend_start] = imageB[:h, :safe_blend_start]

        if actual_blend_width > 0 and hasattr(self, "gradient_mask_3"):
            safe_blend_end = min(blend_end, w, result.shape[1])
            if safe_blend_end > safe_blend_start:
                actual_width = safe_blend_end - safe_blend_start

                left_region = imageB[:h, safe_blend_start:safe_blend_end].astype(
                    np.float32
                )
                right_region = result[:h, safe_blend_start:safe_blend_end].astype(
                    np.float32
                )

                if actual_width == self.gradient_mask_3.shape[1]:
                    mask = self.gradient_mask_3
                else:
                    mask_1d = np.linspace(0, 1, actual_width, dtype=np.float32)
                    mask_1d = mask_1d * mask_1d * (3 - 2 * mask_1d)
                    mask = np.tile(mask_1d, (h, 1))
                    mask = np.dstack([mask] * 3)

                if left_region.shape == right_region.shape == mask.shape:
                    blended = left_region * (1.0 - mask) + right_region * mask
                    result[:h, safe_blend_start:safe_blend_end] = blended.astype(
                        np.uint8
                    )

        # Fill holes
        left_part = imageB[:h, :w]
        result_part = result[:h, :w]

        if self.use_cuda:
            try:
                self.gpu_temp.upload(left_part, self.stream)
                gpu_left_gray = cv2.cuda.cvtColor(
                    self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream
                )
                self.gpu_temp.upload(result_part, self.stream)
                gpu_result_gray = cv2.cuda.cvtColor(
                    self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream
                )
                self.stream.waitForCompletion()
                left_gray_full = gpu_left_gray.download()
                result_gray_full = gpu_result_gray.download()
            except cv2.error:
                left_gray_full = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
                result_gray_full = cv2.cvtColor(result_part, cv2.COLOR_BGR2GRAY)
        else:
            left_gray_full = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
            result_gray_full = cv2.cvtColor(result_part, cv2.COLOR_BGR2GRAY)

        holes = (result_gray_full < self.BLACK_PIXEL_THRESHOLD) & (
            left_gray_full > self.BLACK_PIXEL_THRESHOLD
        )
        if np.any(holes):
            holes_3 = np.dstack([holes] * 3)
            result_part[holes_3] = left_part[holes_3]

        result[:h, :w] = result_part

        result = self.fillFromSourceImages(result, imageA, imageB, H)

        cropped = result[: self.output_height, : self.output_width]

        if self.moving_camera and hasattr(self, "locked_output_size"):
            locked_h, locked_w = self.locked_output_size
            current_h, current_w = cropped.shape[:2]

            if current_h != locked_h or current_w != locked_w:
                fixed_result = np.zeros((locked_h, locked_w, 3), dtype=np.uint8)
                copy_h = min(current_h, locked_h)
                copy_w = min(current_w, locked_w)
                fixed_result[:copy_h, :copy_w] = cropped[:copy_h, :copy_w]
                return fixed_result
        elif self.moving_camera and not hasattr(self, "locked_output_size"):
            self.locked_output_size = (cropped.shape[0], cropped.shape[1])
            logger.info(
                f"Locked output size: {self.locked_output_size[1]}x{self.locked_output_size[0]}"
            )

        return cropped

    def fillFromSourceImages(self, result, imageA, imageB, H):
        h, w_left = imageB.shape[:2]
        h_right, w_right = imageA.shape[:2]
        result_h, result_w = result.shape[:2]

        if self.use_cuda:
            try:
                self.gpu_temp.upload(result, self.stream)
                gpu_gray = cv2.cuda.cvtColor(
                    self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream
                )
                self.stream.waitForCompletion()
                gray = gpu_gray.download()
            except cv2.error:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        black_mask = gray < self.BLACK_PIXEL_THRESHOLD

        if not np.any(black_mask):
            return result

        left_region_mask = black_mask[:h, :w_left]
        if np.any(left_region_mask):
            if self.use_cuda:
                try:
                    self.gpu_temp.upload(imageB, self.stream)
                    gpu_left_gray = cv2.cuda.cvtColor(
                        self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream
                    )
                    self.stream.waitForCompletion()
                    left_gray = gpu_left_gray.download()
                except cv2.error:
                    left_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

            valid_fill = left_region_mask & (left_gray > self.BLACK_PIXEL_THRESHOLD)
            if np.any(valid_fill):
                valid_fill_3 = np.dstack([valid_fill] * 3)
                result[:h, :w_left][valid_fill_3] = imageB[valid_fill_3]

        if self.use_cuda:
            try:
                self.gpu_temp.upload(result, self.stream)
                gpu_gray = cv2.cuda.cvtColor(
                    self.gpu_temp, cv2.COLOR_BGR2GRAY, stream=self.stream
                )
                self.stream.waitForCompletion()
                gray = gpu_gray.download()
            except cv2.error:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        black_mask = gray < self.BLACK_PIXEL_THRESHOLD

        if np.any(black_mask):
            try:
                H_inv = np.linalg.inv(H)
                black_coords = np.where(black_mask)
                if len(black_coords[0]) > 0:
                    y_coords = black_coords[0]
                    x_coords = black_coords[1]

                    pts = np.float32(np.column_stack([x_coords, y_coords])).reshape(
                        -1, 1, 2
                    )
                    pts_transformed = cv2.perspectiveTransform(pts, H_inv)
                    pts_transformed = pts_transformed.reshape(-1, 2)

                    src_x = np.round(pts_transformed[:, 0]).astype(np.int32)
                    src_y = np.round(pts_transformed[:, 1]).astype(np.int32)

                    valid = (
                        (src_x >= 0)
                        & (src_x < w_right)
                        & (src_y >= 0)
                        & (src_y < h_right)
                    )

                    valid_indices = np.where(valid)[0]
                    if len(valid_indices) > 0:
                        valid_src_x = src_x[valid_indices]
                        valid_src_y = src_y[valid_indices]
                        valid_dst_y = y_coords[valid_indices]
                        valid_dst_x = x_coords[valid_indices]

                        src_pixels = imageA[valid_src_y, valid_src_x]
                        brightness = np.mean(src_pixels, axis=1)
                        bright_enough = brightness > self.BLACK_PIXEL_THRESHOLD

                        final_indices = np.where(bright_enough)[0]
                        if len(final_indices) > 0:
                            result[
                                valid_dst_y[final_indices], valid_dst_x[final_indices]
                            ] = src_pixels[final_indices]

            except np.linalg.LinAlgError:
                logger.warning("Could not invert homography for fill")

        return result

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, features) = self.detector.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        if featuresA is None or featuresB is None:
            return None

        if len(kpsA) < 5 or len(kpsB) < 5:
            return None

        rawMatches = self.matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m_n in rawMatches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < n.distance * ratio:
                    matches.append((m.trainIdx, m.queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            try:
                M_affine, inliers = cv2.estimateAffinePartial2D(
                    ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=reprojThresh
                )

                if M_affine is not None:
                    H = np.vstack([M_affine, [0, 0, 1]])
                    status = inliers.ravel().tolist() if inliers is not None else None
                    return (matches, H, status)
            except cv2.error:
                pass

            try:
                (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
                return (matches, H, status)
            except cv2.error:
                pass

        return None


# =============================================================================
# MAIN STITCHING FUNCTIONS
# =============================================================================


def check_nvenc():
    """Check if NVENC is available via ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
        )
        return "h264_nvenc" in result.stdout
    except:
        return False


def stitch_videos_2cam(
    left_video_path,
    right_video_path,
    output_name=None,
    show_preview=False,
    moving_camera=False,
    enable_detection=False,
    detection_model="n",
    detection_confidence=0.15,
    detect_classes=None,
    enable_tracking=False,
    sync_method="nearest",
    timestamp_files=None,
    target_fps=None,
):
    """
    Stitch two videos into a panorama with optional CNN object detection.

    Args:
        left_video_path: Path to left camera video
        right_video_path: Path to right camera video
        output_name: Output filename (auto-generated if None)
        show_preview: Show preview window during processing
        moving_camera: Enable moving camera mode (recalculates homography)
        enable_detection: Enable YOLOv8 object detection
        detection_model: YOLO model size ('n', 's', 'm', 'l', 'x')
        detection_confidence: Detection confidence threshold
        detect_classes: List of class IDs to detect (None = default)
        enable_tracking: Enable ByteTrack object tracking
        sync_method: Frame sync method ('nearest' or 'interpolate')
        timestamp_files: List of timestamp files for each video [left_ts, right_ts]
                        Use for ROS bag extracted videos with exact timestamps
        target_fps: Target output FPS (None = auto-detect)
    """

    # Validate inputs
    if not os.path.exists(left_video_path):
        raise FileNotFoundError(f"Left video not found: {left_video_path}")
    if not os.path.exists(right_video_path):
        raise FileNotFoundError(f"Right video not found: {right_video_path}")

    # Generate output name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "moving" if moving_camera else "static"
    det_str = "_det" if enable_detection else ""
    if output_name is None:
        output_name = f"panorama_2cam_{mode_str}{det_str}_{timestamp}"

    # Handle output paths - strip .mp4 if present for consistent naming
    base_name = output_name[:-4] if output_name.endswith(".mp4") else output_name
    temp_output = f"{base_name}_temp.avi"
    final_output = f"{base_name}.mp4"

    # Check NVENC
    nvenc_available = check_nvenc()
    logger.info(f"NVENC available: {nvenc_available}")

    # Initialize stitcher
    logger.info("=" * 60)
    logger.info("INITIALIZING 2-CAMERA STITCHER")
    logger.info("=" * 60)
    stitcher = Stitcher(use_cuda=True, moving_camera=moving_camera)

    # Initialize detector if enabled
    detector = None
    if enable_detection:
        logger.info("=" * 60)
        logger.info("INITIALIZING OBJECT DETECTOR")
        logger.info("=" * 60)
        detector = ObjectDetector(
            model_size=detection_model,
            confidence=detection_confidence,
            detect_classes=detect_classes,
            use_gpu=True,
            enable_tracking=enable_tracking,
        )

    # Choose video reader based on whether timestamps are provided
    if timestamp_files:
        # Use timestamp-based synchronization (for ROS bag videos)
        video_reader = TimestampSyncedVideoReader(
            [left_video_path, right_video_path],
            timestamp_files=timestamp_files,
            target_fps=target_fps or 15.0,
            sync_method=sync_method,
        )
    else:
        # Use FPS-based synchronization
        video_reader = SyncedVideoReader(
            [left_video_path, right_video_path],
            target_fps=target_fps,
            sync_method=sync_method,
        )

    fps = video_reader.get_fps()
    total_frames = video_reader.get_frame_count()

    logger.info("=" * 60)
    logger.info("2-CAMERA STITCHING + DETECTION (Phase 1: OpenCV)")
    logger.info("=" * 60)

    writer = None
    frame_count = 0
    total_time = 0
    fps_history = []
    output_size_locked = False
    output_w, output_h = 0, 0

    crop_roi = None  # (x, y, w, h)

    while True:
        frame_start = time.perf_counter()

        ret, frames = video_reader.read()
        if not ret:
            logger.info("End of video(s) reached")
            break

        left, right = frames[0], frames[1]

        if left is None or right is None:
            continue

        frame_count += 1

        # Stitch frames
        stitched = stitcher.stitch([left, right])
        if stitched is None:
            logger.warning(
                f"Frame {frame_count}: Stitch failed, using side-by-side fallback"
            )
            # Fallback: Side-by-side view
            h1, w1 = left.shape[:2]
            h2, w2 = right.shape[:2]
            # Resize right to match left height if needed
            if h1 != h2:
                right = cv2.resize(right, (int(w2 * h1 / h2), h1))
            stitched = np.hstack([left, right])

        # Auto-Crop Logic
        if crop_roi is None:
            # Calculate harvest (crop) rectangle on the first frame
            # We compute it by actually running the crop on the full image
            cropped_sample = crop_largest_rectangle(stitched)
            if cropped_sample is not None:
                # We need to find WHERE this crop is relative to original to apply it consistently?
                # Actually, our helper returns the cropped image directly.
                # But to apply consistently, we need coordinates if we want to be strict.
                # However, since we want "perfect rectangle", detecting it once and using that *size* isn't enough, we need position.
                # Our helper is stateless.
                # Let's Modify usage: Simply call clean_stitch = crop_largest_rectangle(stitched)
                # WAIT: If the camera moves, the black borders move.
                # If we use a FIXED crop, we might crop out valid pixels later or include black ones.
                # But typical stitching output (if using simple homography) puts the image in a fixed canvas usually?
                # No, standard Stitcher usually compensates for motion.
                # If "Scan Mode" (Panorama): The canvas grows.
                # If "Video Stitcher" (2 cameras fixed relative to each other): The field of view is constant. The BLACK borders are caused by the warped projection of the two cameras onto a flat plane. This shape is CONSTANT if the cameras don't move relative to each other.
                # Assuming the cameras are fixed relative to each other (rig), the mask of valid pixels is CONSTANT.
                pass

        # Apply crop (re-calculating every frame is safest for moving borders, but user wants A rectangle)
        # If we re-calculate every frame, the output size might jitter. VideoWriter requires CONSTANT size.
        # So we MUST lock the crop ROI on the first frame.

        if crop_roi is None:
            # Calculate the stable crop ROI (x, y, w, h)
            # We use a trick: pass the image to function, but we need coordinates.
            # I'll rely on the assumption that the shape is constant.
            # Let's do a little surgery on our helper or just implement ROI finding here.
            # Re-implementing simplified version here for coordinate access:

            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)

                # Refine (simple 5% inset to be safe?)
                # User wants "perfect". Let's use the helper on the crop to see final size,
                # but mapping back coordinates is hard without changing helper.
                # Let's just use the crop_largest_rectangle helper and ASSUME it returns the top-left-most invalid-free box?
                # No, the helper shifts x/y.

                # Let's CHANGE the helper to return image AND roi? No, helper is in global scope.
                # I'll just run it once to get the resulting image, and determine `output_w, output_h` from that.
                # But for subsequent frames, how do I generate the same crop?
                # I can't.
                # So I HAVE to accept that I need to calculate the ROI coordinates.

                # Let's just USE the helper every frame?
                # Problem: Frame size changes -> VideoWriter fails.
                # Problem: Frame size jitter.

                # Solution: Calculate crop_roi on first frame.
                # Revert to manually finding it here?
                # Or just update helper to return (img, x,y,w,h).

                # I'll use a simplified strategy here:
                # Just use the helper on the first frame.
                # Get its shape.
                # Assume the crop is centered? No.

                # Let's update the helper in the first replacement chunk to return ROI too?
                # Too risky to edit the previous chunk now? No, I can edit my tool call.
                pass

        # Let's use a simpler heuristic for stability:
        # 1. Bounding Rect of non-black.
        # 2. Inset by 5%?
        # User said "perfect rectangle".

        # Okay, I will modify the helper in the first chunk to return (image, (x,y,w,h))?
        # Or just `get_crop_roi(image)`.
        # I'll stick to: Calculate a crop on the first frame using the helper.
        # But how do I know WHERE to crop subsequent frames?
        # I will assume the black borders are STATIC (fixed rig).
        # If they are static, I can just "learn" the mask from the first frame.

        if crop_roi is None:
            # Temporary crop to find the box
            temp_gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            _, temp_thresh = cv2.threshold(temp_gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                # Iterative shrink (inline implementation of helper logic)
                rx, ry, rw, rh = bx, by, bw, bh

                # NEW: Create a filled mask from the contour to ignore internal holes (black video pixels)
                temp_mask = np.zeros_like(temp_gray)
                cv2.drawContours(temp_mask, [c], -1, 255, -1)

                # Limit steps
                for _ in range(min(bw, bh) // 2):
                    sub = temp_mask[ry : ry + rh, rx : rx + rw]
                    if sub.size == 0:
                        break

                    # Check edges
                    tz = rw - cv2.countNonZero(sub[0, :])
                    bz = rw - cv2.countNonZero(sub[-1, :])
                    lz = rh - cv2.countNonZero(sub[:, 0])
                    rz = rh - cv2.countNonZero(sub[:, -1])

                    # Only check top and sides, ignore bottom to prevent over-cropping
                    if tz == 0 and lz == 0 and rz == 0:
                        break

                    if tz > 0:
                        ry += 1
                        rh -= 1
                    # Bottom shrinking disabled to prevent cutting off content
                    # if bz > 0:
                    #     rh -= 1
                    if lz > 0:
                        rx += 1
                        rw -= 1
                    if rz > 0:
                        rw -= 1

                if rw > 0 and rh > 0:
                    crop_roi = (rx, ry, rw, rh)
                    logger.info(f"Auto-calculated crop ROI: {crop_roi}")

        # Apply global crop
        if crop_roi:
            cx, cy, cw, ch = crop_roi
            # Ensure we don't go out of bounds if image shifts slightly?
            # Clip to image size
            sh, sw = stitched.shape[:2]
            cx = max(0, min(cx, sw - 1))
            cy = max(0, min(cy, sh - 1))
            cw = min(cw, sw - cx)
            ch = min(ch, sh - cy)

            stitched = stitched[cy : cy + ch, cx : cx + cw]

        # Run object detection
        num_detections = 0
        if detector:
            detections = detector.detect(stitched)
            num_detections = len(detections)
            if detections:
                stitched = detector.draw_detections(
                    stitched,
                    detections,
                    show_labels=True,
                    show_confidence=True,
                    show_track_id=enable_tracking,
                )
            else:
                logger.debug(f"Frame {frame_count}: No objects detected")

        # Lock output size on first frame
        if not output_size_locked:
            output_h, output_w = stitched.shape[:2]
            # output_w = output_w + 100 # REMOVED ARTIFICIAL PADDING
            logger.info(f"Output size locked: {output_w}x{output_h}")
            output_size_locked = True

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(temp_output, fourcc, fps, (output_w, output_h))
            if not writer.isOpened():
                raise IOError(f"Could not open VideoWriter for {temp_output}")

        # Enforce frame size
        final_h, final_w = stitched.shape[:2]
        # Resize if crop failed or drifted? better to pad with black than crash
        if final_w != output_w or final_h != output_h:
            # If the frame shrank (e.g. tracking drift), centering it is better
            canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            # Resize to fit? No, strict crop means sizes should match if crop_roi is constant.
            # But if `stitched` original size changes? (e.g. panorama drift).
            # We applied fixed ROI to `stitched`. If `stitched` moves, ROI might be distinct.
            # Safest: Resize `stitched` to `output_w, output_h`?
            # User wants "perfect rectangle". Stretching is bad.
            # Use existing logic but without the offset bug.

            # NOTE: If we used fixed ROI on variable image, we get variable result.
            # But `stitcher.stitch` usually outputs fixed canvas size if not in scan mode.
            # Assuming fixed canvas for now.

            copy_w, copy_h = min(final_w, output_w), min(final_h, output_h)
            canvas[:copy_h, :copy_w] = stitched[:copy_h, :copy_w]
            stitched = canvas

        writer.write(stitched)

        # Performance tracking
        frame_time = time.perf_counter() - frame_start
        total_time += frame_time
        fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Stats
        cpu_percent = psutil.cpu_percent()
        gpu_util = 0
        if GPU_AVAILABLE:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
            except:
                pass

        if total_frames > 0:
            progress = frame_count / total_frames
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            bar = "#" * int(15 * progress) + "-" * (15 - int(15 * progress))
            det_str = f"Det:{num_detections:2d} | " if detector else ""
            print(
                f"\r[{bar}] {progress * 100:5.1f}% | F:{frame_count}/{total_frames} | "
                f"FPS:{avg_fps:4.1f} | {det_str}CPU:{cpu_percent:4.1f}% | GPU:{gpu_util:3.0f}% | "
                f"ETA:{eta:4.0f}s    ",
                end="",
                flush=True,
            )

            # Call progress callback if set
            if _progress_callback:
                try:
                    _progress_callback(
                        progress,
                        f"Processing frame {frame_count}/{total_frames} (FPS: {avg_fps:.1f})",
                    )
                except Exception:
                    pass

        # Preview generation (Combined view for both Web and Local)
        try:
            if show_preview:
                ph = 300
                lp = cv2.resize(left, (int(left.shape[1] * ph / left.shape[0]), ph))
                rp = cv2.resize(right, (int(right.shape[1] * ph / right.shape[0]), ph))
                sp = cv2.resize(
                    stitched, (int(stitched.shape[1] * ph / stitched.shape[0]), ph)
                )

                cv2.putText(
                    lp, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    rp, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                det_label = (
                    f"STITCHED ({num_detections} obj)" if detector else "STITCHED"
                )
                cv2.putText(
                    sp,
                    det_label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                top = np.hstack([lp, rp])
                if top.shape[1] < sp.shape[1]:
                    top = np.hstack(
                        [
                            top,
                            np.zeros(
                                (ph, sp.shape[1] - top.shape[1], 3), dtype=np.uint8
                            ),
                        ]
                    )
                elif top.shape[1] > sp.shape[1]:
                    sp = np.hstack(
                        [
                            sp,
                            np.zeros(
                                (ph, top.shape[1] - sp.shape[1], 3), dtype=np.uint8
                            ),
                        ]
                    )

                combined = np.vstack([top, sp])

                # Local Preview
                if show_preview:
                    cv2.namedWindow("2-Camera Panorama (q to quit)", cv2.WINDOW_NORMAL)
                    cv2.imshow("2-Camera Panorama (q to quit)", combined)
                    cv2.resizeWindow(
                        "2-Camera Panorama (q to quit)",
                        combined.shape[1],
                        combined.shape[0] + 150,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            import traceback

            traceback.print_exc()

    # Cleanup
    video_reader.release()
    if writer:
        writer.release()
    if show_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    logger.info(
        f"\nPhase 1 complete: {frame_count} frames in {total_time:.1f}s ({frame_count / total_time:.1f} FPS)"
    )

    # Detection stats
    if detector:
        stats = detector.get_stats()
        logger.info("=" * 60)
        logger.info("DETECTION STATISTICS")
        logger.info(
            f"Total: {stats['total_detections']}, Avg/frame: {stats['avg_detections_per_frame']:.1f}"
        )
        for cls, cnt in stats["detections_per_class"].items():
            logger.info(f"  {cls}: {cnt}")

    # Phase 2: Re-encode
    output_file = temp_output
    if os.path.exists(temp_output):
        logger.info("=" * 60)
        logger.info("ENCODING (Phase 2)")

        # Use NVENC GPU encoding if available, otherwise fall back to CPU
        if nvenc_available:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel", "cuda",
                "-i", temp_output,
                "-c:v", "h264_nvenc",
                "-preset", "p4",  # Fast NVENC preset
                "-rc", "vbr",
                "-cq", "23",
                "-pix_fmt", "yuv420p",
                final_output,
            ]
            logger.info("Using NVENC GPU encoding")
        else:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_output,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                final_output,
            ]
            logger.info("Using CPU encoding (NVENC not available)")

        # Default to returning temp output if re-encoding fails
        output_file = temp_output

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            output_file = final_output
            logger.info(f"Encoding complete: {final_output}")
        else:
            logger.error(f"FFmpeg encoding failed: {result.stderr}")
            # If NVENC failed, try CPU fallback
            if nvenc_available:
                logger.warning("NVENC failed, trying CPU encoding fallback...")
                ffmpeg_cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-i", temp_output,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    final_output,
                ]
                result = subprocess.run(ffmpeg_cmd_cpu, capture_output=True, text=True)
                if result.returncode == 0:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    output_file = final_output
                    logger.info(f"CPU encoding complete: {final_output}")
                else:
                    logger.error(f"CPU encoding also failed: {result.stderr}")

    logger.info("=" * 60)
    logger.info(f"COMPLETE - Output: {output_file}")
    if os.path.exists(output_file):
        logger.info(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.1f} MB")

    return output_file


def stitch_videos_3cam(
    left_video_path,
    center_video_path,
    right_video_path,
    output_name=None,
    show_preview=False,
    moving_camera=False,
    enable_detection=False,
    detection_model="n",
    detection_confidence=0.15,
    detect_classes=None,
    enable_tracking=False,
    sync_method="nearest",
    timestamp_files=None,
    target_fps=None,
):
    """
    Stitch three videos (LEFT + CENTER + RIGHT) into a wide panorama.

    Args:
        left_video_path: Path to left camera video
        center_video_path: Path to center camera video
        right_video_path: Path to right camera video
        output_name: Output filename (auto-generated if None)
        show_preview: Show preview window during processing
        moving_camera: Enable moving camera mode (recalculates homography)
        enable_detection: Enable YOLOv8 object detection
        detection_model: YOLO model size ('n', 's', 'm', 'l', 'x')
        detection_confidence: Detection confidence threshold
        detect_classes: List of class IDs to detect (None = default)
        enable_tracking: Enable ByteTrack object tracking
        sync_method: Frame sync method ('nearest' or 'interpolate')
        timestamp_files: List of timestamp files [left_ts, center_ts, right_ts]
                        Use for ROS bag extracted videos with exact timestamps
        target_fps: Target output FPS (None = auto-detect)
    """

    # Validate inputs
    for path, name in [
        (left_video_path, "Left"),
        (center_video_path, "Center"),
        (right_video_path, "Right"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} video not found: {path}")

    # Generate output name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "moving" if moving_camera else "static"
    det_str = "_det" if enable_detection else ""
    if output_name is None:
        output_name = f"panorama_3cam_{mode_str}{det_str}_{timestamp}"

    # Handle output paths - strip .mp4 if present for consistent naming
    base_name = output_name[:-4] if output_name.endswith(".mp4") else output_name
    temp_output = f"{base_name}_temp.avi"
    final_output = f"{base_name}.mp4"

    # Check NVENC
    nvenc_available = check_nvenc()
    logger.info(f"NVENC available: {nvenc_available}")

    # Initialize TWO stitchers for sequential stitching
    logger.info("=" * 60)
    logger.info("INITIALIZING 3-CAMERA STITCHERS")
    logger.info("=" * 60)
    logger.info("Creating stitcher 1: LEFT + CENTER")
    stitcher_LC = Stitcher(use_cuda=True, moving_camera=moving_camera)
    logger.info("Creating stitcher 2: (L+C) + RIGHT")
    stitcher_LCR = Stitcher(use_cuda=True, moving_camera=moving_camera)

    # Initialize detector if enabled
    detector = None
    if enable_detection:
        logger.info("=" * 60)
        logger.info("INITIALIZING OBJECT DETECTOR")
        logger.info("=" * 60)
        detector = ObjectDetector(
            model_size=detection_model,
            confidence=detection_confidence,
            detect_classes=detect_classes,
            use_gpu=True,
            enable_tracking=enable_tracking,
        )

    # Choose video reader based on whether timestamps are provided
    if timestamp_files:
        # Use timestamp-based synchronization (for ROS bag videos)
        video_reader = TimestampSyncedVideoReader(
            [left_video_path, center_video_path, right_video_path],
            timestamp_files=timestamp_files,
            target_fps=target_fps or 15.0,
            sync_method=sync_method,
        )
    else:
        # Use FPS-based synchronization
        video_reader = SyncedVideoReader(
            [left_video_path, center_video_path, right_video_path],
            target_fps=target_fps,
            sync_method=sync_method,
        )

    fps = video_reader.get_fps()
    total_frames = video_reader.get_frame_count()

    logger.info("=" * 60)
    logger.info("3-CAMERA STITCHING + DETECTION (Phase 1: OpenCV)")
    logger.info("=" * 60)

    writer = None
    frame_count = 0
    total_time = 0
    fps_history = []
    output_size_locked = False
    output_w, output_h = 0, 0
    crop_roi = None

    while True:
        frame_start = time.perf_counter()

        ret, frames = video_reader.read()
        if not ret:
            logger.info("End of video(s) reached")
            break

        left, center, right = frames[0], frames[1], frames[2]

        if left is None or center is None or right is None:
            continue

        frame_count += 1

        # === SEQUENTIAL STITCHING ===
        # Step 1: Stitch LEFT + CENTER
        intermediate = stitcher_LC.stitch([left, center])
        if intermediate is None:
            logger.warning(f"Frame {frame_count}: L+C stitch failed, skipping")
            continue

        # Step 2: Stitch INTERMEDIATE + RIGHT
        final = stitcher_LCR.stitch([intermediate, right])
        if final is None:
            logger.warning(f"Frame {frame_count}: LC+R stitch failed, skipping")
            continue

        # Auto-Crop Logic
        if crop_roi is None:
            temp_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            _, temp_thresh = cv2.threshold(temp_gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                temp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                rx, ry, rw, rh = bx, by, bw, bh

                # NEW: Create a filled mask (using 'final' variable)
                temp_mask = np.zeros_like(temp_gray)
                cv2.drawContours(temp_mask, [c], -1, 255, -1)

                for _ in range(min(bw, bh) // 2):
                    sub = temp_mask[ry : ry + rh, rx : rx + rw]
                    if sub.size == 0:
                        break
                    tz = rw - cv2.countNonZero(sub[0, :])
                    bz = rw - cv2.countNonZero(sub[-1, :])
                    lz = rh - cv2.countNonZero(sub[:, 0])
                    rz = rh - cv2.countNonZero(sub[:, -1])
                    # Only check top and sides, ignore bottom to prevent over-cropping
                    if tz == 0 and lz == 0 and rz == 0:
                        break
                    if tz > 0:
                        ry += 1
                        rh -= 1
                    # Bottom shrinking disabled to prevent cutting off content
                    # if bz > 0:
                    #     rh -= 1
                    if lz > 0:
                        rx += 1
                        rw -= 1
                    if rz > 0:
                        rw -= 1

                if rw > 0 and rh > 0:
                    crop_roi = (rx, ry, rw, rh)
                    logger.info(f"Auto-calculated crop ROI: {crop_roi}")

        # Apply global crop
        if crop_roi:
            cx, cy, cw, ch = crop_roi
            sh, sw = final.shape[:2]
            cx = max(0, min(cx, sw - 1))
            cy = max(0, min(cy, sh - 1))
            cw = min(cw, sw - cx)
            ch = min(ch, sh - cy)
            final = final[cy : cy + ch, cx : cx + cw]

        # Run object detection
        num_detections = 0
        if detector:
            detections = detector.detect(final)
            num_detections = len(detections)
            if detections:
                final = detector.draw_detections(
                    final,
                    detections,
                    show_labels=True,
                    show_confidence=True,
                    show_track_id=enable_tracking,
                )

        # Lock output size on first frame
        if not output_size_locked:
            output_h, output_w = final.shape[:2]
            logger.info(f"Output size locked: {output_w}x{output_h}")
            output_size_locked = True

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(temp_output, fourcc, fps, (output_w, output_h))
            if not writer.isOpened():
                raise IOError(f"Could not open VideoWriter for {temp_output}")

        # Enforce frame size
        final_h, final_w = final.shape[:2]
        if final_w != output_w or final_h != output_h:
            canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            copy_w, copy_h = min(final_w, output_w), min(final_h, output_h)
            canvas[:copy_h, :copy_w] = final[:copy_h, :copy_w]
            final = canvas

        writer.write(final)

        # Performance tracking
        frame_time = time.perf_counter() - frame_start
        total_time += frame_time
        fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Stats
        cpu_percent = psutil.cpu_percent()
        gpu_util = 0
        if GPU_AVAILABLE:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
            except:
                pass

        if total_frames > 0:
            progress = frame_count / total_frames
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            bar = "#" * int(15 * progress) + "-" * (15 - int(15 * progress))
            det_str = f"Det:{num_detections:2d} | " if detector else ""
            print(
                f"\r[{bar}] {progress * 100:5.1f}% | F:{frame_count}/{total_frames} | "
                f"FPS:{avg_fps:4.1f} | {det_str}CPU:{cpu_percent:4.1f}% | GPU:{gpu_util:3.0f}% | "
                f"ETA:{eta:4.0f}s    ",
                end="",
                flush=True,
            )

            # Call progress callback if set
            if _progress_callback:
                try:
                    _progress_callback(
                        progress,
                        f"Processing frame {frame_count}/{total_frames} (FPS: {avg_fps:.1f})",
                    )
                except Exception:
                    pass

        # Preview generation (Combined view for both Web and Local)
        try:
            if show_preview:
                ph = 250
                lp = cv2.resize(left, (int(left.shape[1] * ph / left.shape[0]), ph))
                cp = cv2.resize(
                    center, (int(center.shape[1] * ph / center.shape[0]), ph)
                )
                rp = cv2.resize(right, (int(right.shape[1] * ph / right.shape[0]), ph))
                fp = cv2.resize(final, (int(final.shape[1] * ph / final.shape[0]), ph))

                cv2.putText(
                    lp, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    cp,
                    "CENTER",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    rp, "RIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                det_label = (
                    f"3-CAM ({num_detections} obj)" if detector else "3-CAM PANORAMA"
                )
                cv2.putText(
                    fp,
                    det_label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                top = np.hstack([lp, cp, rp])
                if top.shape[1] < fp.shape[1]:
                    top = np.hstack(
                        [
                            top,
                            np.zeros(
                                (ph, fp.shape[1] - top.shape[1], 3), dtype=np.uint8
                            ),
                        ]
                    )
                elif top.shape[1] > fp.shape[1]:
                    fp = np.hstack(
                        [
                            fp,
                            np.zeros(
                                (ph, top.shape[1] - fp.shape[1], 3), dtype=np.uint8
                            ),
                        ]
                    )

                combined = np.vstack([top, fp])

                # Send to Web Preview

                # Local Preview
                if show_preview:
                    cv2.imshow("3-Camera Panorama (q to quit)", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            import traceback

            traceback.print_exc()

    # Cleanup
    video_reader.release()
    if writer:
        writer.release()
    if show_preview:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    logger.info(
        f"\nPhase 1 complete: {frame_count} frames in {total_time:.1f}s ({frame_count / total_time:.1f} FPS)"
    )

    # Detection stats
    if detector:
        stats = detector.get_stats()
        logger.info("=" * 60)
        logger.info("DETECTION STATISTICS")
        logger.info(
            f"Total: {stats['total_detections']}, Avg/frame: {stats['avg_detections_per_frame']:.1f}"
        )
        for cls, cnt in stats["detections_per_class"].items():
            logger.info(f"  {cls}: {cnt}")

    # Phase 2: Re-encode
    output_file = temp_output
    if os.path.exists(temp_output):
        logger.info("=" * 60)
        logger.info("ENCODING (Phase 2)")

        # Use NVENC GPU encoding if available, otherwise fall back to CPU
        if nvenc_available:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel", "cuda",
                "-i", temp_output,
                "-c:v", "h264_nvenc",
                "-preset", "p4",  # Fast NVENC preset
                "-rc", "vbr",
                "-cq", "23",
                "-pix_fmt", "yuv420p",
                final_output,
            ]
            logger.info("Using NVENC GPU encoding")
        else:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_output,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                final_output,
            ]
            logger.info("Using CPU encoding (NVENC not available)")

        # Default to returning temp output if re-encoding fails
        output_file = temp_output

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            output_file = final_output
            logger.info(f"Encoding complete: {final_output}")
        else:
            logger.error(f"FFmpeg encoding failed: {result.stderr}")
            # If NVENC failed, try CPU fallback
            if nvenc_available:
                logger.warning("NVENC failed, trying CPU encoding fallback...")
                ffmpeg_cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-i", temp_output,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    final_output,
                ]
                result = subprocess.run(ffmpeg_cmd_cpu, capture_output=True, text=True)
                if result.returncode == 0:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    output_file = final_output
                    logger.info(f"CPU encoding complete: {final_output}")
                else:
                    logger.error(f"CPU encoding also failed: {result.stderr}")

    logger.info("=" * 60)
    logger.info(f"COMPLETE - Output: {output_file}")
    if os.path.exists(output_file):
        logger.info(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.1f} MB")

    return output_file


# =============================================================================
# INTERACTIVE MENU
# =============================================================================


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  GPU-ACCELERATED VIDEO PANORAMA STITCHER")
    print("  with CNN Object Detection (YOLOv8)")
    print("=" * 60)


def get_user_choice(prompt, options):
    """Get a validated user choice from a list of options."""
    while True:
        print(f"\n{prompt}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

        try:
            choice = input("\nEnter your choice (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def get_video_path(prompt, required=True):
    """Get and validate a video file path from user."""
    while True:
        path = input(f"\n{prompt}: ").strip()

        # Remove quotes if present
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if path.startswith("'") and path.endswith("'"):
            path = path[1:-1]

        if not path and not required:
            return None

        if os.path.exists(path):
            return path
        else:
            print(f"  File not found: {path}")
            if not required:
                retry = input("  Try again? (y/n): ").strip().lower()
                if retry != "y":
                    return None


def get_yes_no(prompt, default=True):
    """Get a yes/no response from user."""
    default_str = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'.")


def interactive_menu():
    """Run the interactive configuration menu."""
    print_banner()

    # 1. Number of cameras
    cam_choice = get_user_choice(
        "How many cameras do you want to stitch?",
        ["2 cameras (Left + Right)", "3 cameras (Left + Center + Right)"],
    )
    num_cameras = 2 if cam_choice == 0 else 3

    # 2. Camera mode
    mode_choice = get_user_choice(
        "What type of camera setup?",
        [
            "Static cameras (fixed position)",
            "Moving cameras (e.g., mounted on vehicle)",
        ],
    )
    moving_camera = mode_choice == 1

    # 3. Get video paths
    print("\n" + "-" * 40)
    print("Enter video file paths:")
    print("-" * 40)

    if num_cameras == 2:
        left_video = get_video_path("Left camera video path")
        right_video = get_video_path("Right camera video path")
        center_video = None
    else:
        left_video = get_video_path("Left camera video path")
        center_video = get_video_path("Center camera video path")
        right_video = get_video_path("Right camera video path")

    # 4. Detection settings
    print("\n" + "-" * 40)
    print("Detection Settings:")
    print("-" * 40)

    enable_detection = get_yes_no("Enable object detection?", default=True)

    detection_model = "n"
    detection_confidence = 0.5
    enable_tracking = False
    detect_classes = None

    if enable_detection:
        model_choice = get_user_choice(
            "Select detection model size (larger = more accurate but slower):",
            ["Nano (fastest)", "Small", "Medium", "Large", "XLarge (most accurate)"],
        )
        model_sizes = ["n", "s", "m", "l", "x"]
        detection_model = model_sizes[model_choice]

        conf_input = input("\nConfidence threshold [0.5]: ").strip()
        if conf_input:
            try:
                detection_confidence = float(conf_input)
                detection_confidence = max(0.1, min(0.9, detection_confidence))
            except ValueError:
                detection_confidence = 0.5

        enable_tracking = get_yes_no(
            "Enable object tracking (assigns IDs to objects)?", default=True
        )

    # 5. Sync method option
    print("\n" + "-" * 40)
    print("Frame Sync Settings:")
    print("-" * 40)

    sync_choice = get_user_choice(
        "Select frame synchronization method:",
        [
            "Nearest frame (faster, slight jitter possible)",
            "Interpolate (smoother, blends between frames)",
        ],
    )
    sync_method = "nearest" if sync_choice == 0 else "interpolate"

    # 6. Timestamp files (for ROS bag videos)
    timestamp_files = None
    use_timestamps = get_yes_no(
        "\nDo you have timestamp files? (for ROS bag extracted videos)", default=False
    )

    if use_timestamps:
        print("\nEnter timestamp file paths (one timestamp per line, in nanoseconds):")
        timestamp_files = []

        if num_cameras == 2:
            ts_left = get_video_path("Left camera timestamp file", required=False)
            ts_right = get_video_path("Right camera timestamp file", required=False)
            if ts_left and ts_right:
                timestamp_files = [ts_left, ts_right]
            else:
                timestamp_files = None
                print("  Timestamps incomplete, using FPS-based sync")
        else:
            ts_left = get_video_path("Left camera timestamp file", required=False)
            ts_center = get_video_path("Center camera timestamp file", required=False)
            ts_right = get_video_path("Right camera timestamp file", required=False)
            if ts_left and ts_center and ts_right:
                timestamp_files = [ts_left, ts_center, ts_right]
            else:
                timestamp_files = None
                print("  Timestamps incomplete, using FPS-based sync")

    # 7. Target FPS
    target_fps = None
    if timestamp_files:
        fps_input = input("\nTarget output FPS [15.0]: ").strip()
        if fps_input:
            try:
                target_fps = float(fps_input)
            except ValueError:
                target_fps = 15.0
        else:
            target_fps = 15.0

    # 8. Preview option
    show_preview = get_yes_no("\nShow preview window during processing?", default=True)

    # 9. Output name
    output_name = input("\nOutput filename (leave empty for auto-generated): ").strip()
    if not output_name:
        output_name = None

    # Summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Cameras: {num_cameras}")
    print(f"  Mode: {'Moving' if moving_camera else 'Static'}")
    print(f"  Left video: {left_video}")
    if center_video:
        print(f"  Center video: {center_video}")
    print(f"  Right video: {right_video}")
    print(f"  Detection: {'Enabled' if enable_detection else 'Disabled'}")
    if enable_detection:
        print(f"    Model: YOLOv8-{detection_model}")
        print(f"    Confidence: {detection_confidence}")
        print(f"    Tracking: {'Enabled' if enable_tracking else 'Disabled'}")
    print(f"  Preview: {'Enabled' if show_preview else 'Disabled'}")
    print(f"  Sync method: {sync_method}")
    if timestamp_files:
        print(f"  Timestamp sync: Enabled (ROS bag mode)")
        print(f"  Target FPS: {target_fps}")
    else:
        print(f"  Timestamp sync: Disabled (FPS-based)")
    print("=" * 60)

    # Confirm
    if not get_yes_no("\nProceed with these settings?", default=True):
        print("\nAborted.")
        return

    # Run stitching
    print("\n")

    if num_cameras == 2:
        stitch_videos_2cam(
            left_video_path=left_video,
            right_video_path=right_video,
            output_name=output_name,
            show_preview=show_preview,
            moving_camera=moving_camera,
            enable_detection=enable_detection,
            detection_model=detection_model,
            detection_confidence=detection_confidence,
            detect_classes=detect_classes,
            enable_tracking=enable_tracking,
            sync_method=sync_method,
            timestamp_files=timestamp_files,
            target_fps=target_fps,
        )
    else:
        stitch_videos_3cam(
            left_video_path=left_video,
            center_video_path=center_video,
            right_video_path=right_video,
            output_name=output_name,
            show_preview=show_preview,
            moving_camera=moving_camera,
            enable_detection=enable_detection,
            detection_model=detection_model,
            detection_confidence=detection_confidence,
            detect_classes=detect_classes,
            enable_tracking=enable_tracking,
            sync_method=sync_method,
            timestamp_files=timestamp_files,
            target_fps=target_fps,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check if running with command line arguments (non-interactive mode)
    if len(sys.argv) > 1:
        print(
            "For command-line usage, import this module and call stitch_videos_2cam() or stitch_videos_3cam() directly."
        )
        print("Running interactive mode instead...")

    # Run interactive menu
    interactive_menu()
