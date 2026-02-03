import cv2
import numpy as np
import time
import os
import logging
import pynvml

logger = logging.getLogger(__name__)

# GPU Stats helpers
GPU_HANDLE = None
GPU_AVAILABLE = False

try:
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not initialize NVML for GPU stats: {e}")


class Stitcher:
    def __init__(self, use_cuda=True):
        self.cachedH = None
        self.blend_start = None
        self.blend_end = None
        self.output_width = None
        self.output_height = None
        self.crop_top = None
        self.crop_bottom = None
        self.focal_length = None  # For cylindrical warping

        # Caching for cylindrical maps
        self.cached_maps = {}  # CPU maps
        self.gpu_maps = {}  # GPU maps (cv2.cuda_GpuMat)

        # Check CUDA availability
        self.use_cuda = use_cuda and self._check_cuda()

        if self.use_cuda:
            self._init_cuda()
        else:
            print("Running on CPU")

        # Always use CPU SIFT for feature detection (more reliable)
        self.detector = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher()

    def _check_cuda(self):
        """Check if CUDA is available and working."""
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                test = cv2.cuda_GpuMat()
                test.upload(np.zeros((10, 10), dtype=np.uint8))
                return True
        except Exception as e:
            print(f"CUDA check failed: {e}")
        return False

    def _init_cuda(self):
        """Initialize CUDA resources."""
        print(f"CUDA enabled! Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.printCudaDeviceInfo(0)

        self.gpu_frame = cv2.cuda_GpuMat()
        self.gpu_warped = cv2.cuda_GpuMat()
        self.gpu_gray = cv2.cuda_GpuMat()
        self.stream = cv2.cuda_Stream()

        print("CUDA initialized - GPU acceleration enabled")

    def cylindricalWarp(self, img, f=None):
        """
        Apply cylindrical warping to an image.
        """
        h, w = img.shape[:2]

        # Estimate focal length if not provided
        if f is None:
            f = w * 0.8  # Good default for typical cameras

        # Store focal length for consistent warping
        if self.focal_length is None:
            self.focal_length = f
        else:
            f = self.focal_length

        # Check cache
        if (h, w) not in self.cached_maps:
            # Center of image
            cx, cy = w / 2, h / 2

            # Create meshgrid for destination coordinates
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

            # Convert to cylindrical coordinates
            theta = (x_coords - cx) / f
            h_cyl = (y_coords - cy) / f

            # Project back to planar coordinates
            x_planar = f * np.tan(theta) + cx
            y_planar = h_cyl * f / np.cos(theta) + cy

            # Create the mapping
            map_x = x_planar.astype(np.float32)
            map_y = y_planar.astype(np.float32)

            self.cached_maps[(h, w)] = (map_x, map_y)

        # Warp the image
        if self.use_cuda:
            try:
                # Prepare GPU resources for warping if not cached
                if (h, w) not in self.gpu_maps:
                    map_x, map_y = self.cached_maps[(h, w)]
                    gpu_map_x = cv2.cuda_GpuMat()
                    gpu_map_y = cv2.cuda_GpuMat()
                    gpu_map_x.upload(map_x)
                    gpu_map_y.upload(map_y)
                    self.gpu_maps[(h, w)] = (gpu_map_x, gpu_map_y)

                gpu_map_x, gpu_map_y = self.gpu_maps[(h, w)]

                # Upload image to GPU
                self.gpu_frame.upload(img)

                # Perform remap on GPU
                # Note: cv2.cuda.remap requires destination size as tuple
                gpu_warped = cv2.cuda.remap(
                    self.gpu_frame,
                    gpu_map_x,
                    gpu_map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                # Download result for downstream processing (feature detection still CPU based)
                warped = gpu_warped.download()

                # For mask: Create ones on GPU?
                # Faster to just cache the CPU mask since it's constant!
                # Or create it once on CPU and cache it.
                # Let's check if we have a cached mask
                mask_key = (h, w, "mask")
                if mask_key not in self.cached_maps:
                    # Create mask on CPU once
                    mask = cv2.remap(
                        np.ones((h, w), dtype=np.uint8) * 255,
                        self.cached_maps[(h, w)][0],
                        self.cached_maps[(h, w)][1],
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    self.cached_maps[mask_key] = mask

                mask = self.cached_maps[mask_key]
                return warped, mask

            except Exception as e:
                # Log the actual error to understand why fallback happened
                logger.error(
                    f"CUDA remap failed: {e}. Falling back to CPU for this frame."
                )

        # CPU Fallback
        map_x, map_y = self.cached_maps[(h, w)]
        warped = cv2.remap(
            img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # Create/Get mask
        mask_key = (h, w, "mask")
        if mask_key not in self.cached_maps:
            mask = cv2.remap(
                np.ones((h, w), dtype=np.uint8) * 255,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            self.cached_maps[mask_key] = mask

        return warped, self.cached_maps[mask_key]

    def inverseCylindricalWarp(self, img, f=None):
        """
        Apply inverse cylindrical warping (for matching features).
        """
        h, w = img.shape[:2]

        if f is None:
            f = self.focal_length if self.focal_length else w * 0.8

        cx, cy = w / 2, h / 2

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        # Inverse mapping: from planar to cylindrical
        x_centered = x_coords - cx
        y_centered = y_coords - cy

        theta = np.arctan2(x_centered, f)
        h_cyl = y_centered * np.cos(theta) / f

        x_cyl = f * theta + cx
        y_cyl = f * h_cyl + cy

        map_x = x_cyl.astype(np.float32)
        map_y = y_cyl.astype(np.float32)

        warped = cv2.remap(
            img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        return warped

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        (imageB, imageA) = images  # left = B, right = A

        if self.cachedH is not None:
            # Apply cylindrical warp to both images
            cylA, maskA = self.cylindricalWarp(imageA)
            cylB, maskB = self.cylindricalWarp(imageB)
            return self.applyWarp(cylA, cylB, self.cachedH, imageA, imageB)

        # Apply cylindrical warp to both images before feature detection
        cylA, maskA = self.cylindricalWarp(imageA)
        cylB, maskB = self.cylindricalWarp(imageB)

        (kpsA, featuresA) = self.detectAndDescribe(cylA)
        (kpsB, featuresB) = self.detectAndDescribe(cylB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            print("Not enough matches to compute homography")
            return None

        (matches, H, status) = M
        if H is None:
            print("Homography is None")
            return None

        H = self._constrainHomography(H, cylA.shape, cylB.shape)
        self.cachedH = H.astype("float32")

        return self.applyWarp(cylA, cylB, self.cachedH, imageA, imageB)

    def _constrainHomography(self, H, shapeA, shapeB):
        H = H / H[2, 2]

        perspective_threshold = 0.002
        if abs(H[2, 0]) > perspective_threshold or abs(H[2, 1]) > perspective_threshold:
            H[2, 0] *= 0.5
            H[2, 1] *= 0.5
            H = H / H[2, 2]

        scale_x = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        scale_y = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)

        if scale_x > 1.3 or scale_x < 0.77:
            H[0, 0] /= scale_x
            H[1, 0] /= scale_x
        if scale_y > 1.3 or scale_y < 0.77:
            H[0, 1] /= scale_y
            H[1, 1] /= scale_y

        H = H / H[2, 2]
        return H

    def applyWarp(self, imageA, imageB, H, origA=None, origB=None):
        """
        CUDA-accelerated warp and blend with black region filling.
        imageA, imageB: cylindrically warped images
        origA, origB: original images for filling black regions
        """
        if origA is None:
            origA = imageA
        if origB is None:
            origB = imageB

        h, w = imageB.shape[:2]

        # Calculate where the warped image corners will be
        corners = np.float32(
            [
                [0, 0],
                [imageA.shape[1], 0],
                [imageA.shape[1], imageA.shape[0]],
                [0, imageA.shape[0]],
            ]
        )
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)

        # Find the rightmost valid point (where content actually ends)
        max_x = int(np.max(warped_corners[:, 0, 0]))

        # Limit canvas width to actual content
        canvas_width = min(max_x + 50, imageA.shape[1] + imageB.shape[1])

        if self.use_cuda:
            try:
                self.gpu_frame.upload(imageA)
                gpu_warped = cv2.cuda.warpPerspective(
                    self.gpu_frame,
                    H,
                    (canvas_width, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                warped = gpu_warped.download()
            except Exception as e:
                print(f"CUDA warp failed: {e}, using CPU")
                warped = cv2.warpPerspective(
                    imageA,
                    H,
                    (canvas_width, h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
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

        # Find overlap region (only on first frame)
        if self.blend_start is None:
            # Find where the warped right image starts (has valid pixels)
            overlap_start = w
            threshold_pixels = int(h * 0.15)

            for x in range(w):
                if np.count_nonzero(warped_gray[:, x] > 15) >= threshold_pixels:
                    overlap_start = x
                    break

            if overlap_start >= w:
                threshold_pixels = int(h * 0.05)
                for x in range(w):
                    if np.count_nonzero(warped_gray[:, x] > 15) >= threshold_pixels:
                        overlap_start = x
                        break

            if overlap_start > w - 30:
                overlap_start = max(0, w - 100)

            # ============ SEAM AT MIDDLE OF OVERLAP ============
            overlap_width = w - overlap_start
            blend_width = 200  # Wide blend zone for smooth transition

            # Seam at center of overlap region
            blend_center = overlap_start + overlap_width // 2
            self.blend_start = max(0, blend_center - blend_width // 2)
            self.blend_end = min(w, blend_center + blend_width // 2)

            # Calculate output width (trim unnecessary black on right)
            valid_cols = np.where(np.any(warped_gray > 15, axis=0))[0]
            if len(valid_cols) > 0:
                self.output_width = min(valid_cols[-1] + 10, canvas_width)
            else:
                self.output_width = canvas_width
            self.output_height = h

            print(f"Seam at MIDDLE of overlap")
            print(f"Blend region: {self.blend_start} to {self.blend_end}")
            print(f"Output size: {self.output_width}x{self.output_height}")

        blend_start = self.blend_start
        blend_end = self.blend_end
        actual_blend_width = blend_end - blend_start

        result = warped.copy()

        # ============ ADVANCED COLOR MATCHING ============
        # Match the right image colors to the left image in the overlap region
        # Then apply a gradual color transition across the right image

        warped = self.match_color_gradient(imageB, warped, blend_start, blend_end, h)

        # ============ SEAMLESS BLENDING (Multi-band Laplacian) ============
        result = warped.copy()
        result[:h, :blend_start] = imageB[:h, :blend_start]

        if actual_blend_width > 0:
            # Use multi-band blending for seamless results
            result = self.multiband_blend(imageB, warped, blend_start, blend_end, h)

        # Fill holes from left image
        left_part = imageB[:h, :w]
        result_part = result[:h, :w]

        left_gray_full = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
        result_gray_full = cv2.cvtColor(result_part, cv2.COLOR_BGR2GRAY)

        holes = (result_gray_full < 10) & (left_gray_full > 10)
        if np.any(holes):
            holes_3 = np.dstack([holes] * 3)
            result_part[holes_3] = left_part[holes_3]

        result[:h, :w] = result_part

        # ============ FILL BLACK REGIONS FROM SOURCE IMAGES ============
        # Use cylindrically warped versions of original images for filling
        cylOrigA, _ = self.cylindricalWarp(origA)
        cylOrigB, _ = self.cylindricalWarp(origB)
        result = self.fillFromSourceImages(result, cylOrigA, cylOrigB, H)

        # Trim to consistent output size
        result = result[: self.output_height, : self.output_width]

        return result

    def fillFromSourceImages(self, result, imageA, imageB, H):
        """
        Fill black regions using actual pixels from source images.
        - Left side black regions: fill from imageB (left image)
        - Right side black regions: fill from warped imageA (right image)
        """
        h, w_left = imageB.shape[:2]
        h_right, w_right = imageA.shape[:2]
        result_h, result_w = result.shape[:2]

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        black_mask = gray < 10

        # If no black regions, return as-is
        if not np.any(black_mask):
            return result

        # ============ FILL LEFT SIDE FROM imageB ============
        # For black pixels in the left portion (where imageB covers)
        left_region_mask = black_mask[:h, :w_left]
        if np.any(left_region_mask):
            left_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # Only fill where imageB has valid content
            valid_fill = left_region_mask & (left_gray > 10)
            if np.any(valid_fill):
                valid_fill_3 = np.dstack([valid_fill] * 3)
                result[:h, :w_left][valid_fill_3] = imageB[valid_fill_3]

        # ============ FILL RIGHT SIDE FROM imageA (inverse warp) ============
        # For remaining black pixels, try to get them from imageA
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        black_mask = gray < 10

        if np.any(black_mask):
            # Compute inverse homography to map result coords back to imageA
            try:
                H_inv = np.linalg.inv(H)

                # Find black pixel coordinates
                black_coords = np.where(black_mask)
                if len(black_coords[0]) > 0:
                    # Create coordinate arrays
                    y_coords = black_coords[0]
                    x_coords = black_coords[1]

                    # Transform to imageA coordinates
                    pts = np.float32(np.column_stack([x_coords, y_coords])).reshape(
                        -1, 1, 2
                    )
                    pts_transformed = cv2.perspectiveTransform(pts, H_inv)
                    pts_transformed = pts_transformed.reshape(-1, 2)

                    # Get pixel values from imageA where valid
                    for i in range(len(y_coords)):
                        src_x = int(round(pts_transformed[i, 0]))
                        src_y = int(round(pts_transformed[i, 1]))

                        # Check if source coordinates are within imageA bounds
                        if 0 <= src_x < w_right and 0 <= src_y < h_right:
                            # Check if source pixel is valid (not black)
                            if np.mean(imageA[src_y, src_x]) > 10:
                                result[y_coords[i], x_coords[i]] = imageA[src_y, src_x]
            except np.linalg.LinAlgError:
                print("Could not invert homography for fill")

        return result

    def match_color_gradient(self, imageB, warped, blend_start, blend_end, h):
        """
        Match colors between left and right images with a gradient falloff.
        This ensures smooth color transition across the entire panorama.
        """
        w_left = imageB.shape[1]
        result_w = warped.shape[1]

        # Sample the overlap region for color statistics
        sample_margin = 50
        sample_start = max(0, blend_start - sample_margin)
        sample_end = min(w_left, blend_end + sample_margin)

        # Get samples from both images in overlap
        sample_left = imageB[:, sample_start:sample_end].copy()
        sample_right = warped[:h, sample_start:sample_end].copy()

        # Create mask for valid pixels (not black)
        right_gray = cv2.cvtColor(sample_right, cv2.COLOR_BGR2GRAY)
        valid_mask = right_gray > 15

        if np.sum(valid_mask) < 500:
            return warped

        # Convert to LAB for better color matching
        left_lab = cv2.cvtColor(sample_left, cv2.COLOR_BGR2LAB).astype(np.float32)
        right_lab = cv2.cvtColor(sample_right, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate color transfer parameters for each channel
        transfer_params = []
        for c in range(3):
            left_vals = left_lab[:, :, c][valid_mask]
            right_vals = right_lab[:, :, c][valid_mask]

            if len(left_vals) > 100:
                left_mean = np.mean(left_vals)
                right_mean = np.mean(right_vals)
                left_std = np.std(left_vals) + 1e-6
                right_std = np.std(right_vals) + 1e-6

                # Scale and offset to match left image
                scale = left_std / right_std
                offset = left_mean - right_mean * scale
                transfer_params.append((scale, offset, right_mean, left_mean))
            else:
                transfer_params.append((1.0, 0.0, 0.0, 0.0))

        # Apply color correction to the entire warped image with gradient falloff
        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Create gradient mask: full correction at blend zone, fading to original at edges
        gradient_mask = np.ones((h, result_w), dtype=np.float32)

        # Fade out the correction towards the right edge
        fade_start = blend_end
        fade_end = result_w
        fade_width = fade_end - fade_start

        if fade_width > 0:
            fade = np.linspace(
                1.0, 0.3, fade_width, dtype=np.float32
            )  # Keep some correction
            gradient_mask[:, fade_start:fade_end] = np.tile(fade, (h, 1))

        # Apply correction with gradient
        for c in range(3):
            scale, offset, right_mean, left_mean = transfer_params[c]

            # Full transfer
            corrected = (warped_lab[:, :, c] - right_mean) * scale + left_mean

            # Blend between original and corrected based on gradient
            original = warped_lab[:, :, c]
            warped_lab[:, :, c] = (
                original * (1 - gradient_mask) + corrected * gradient_mask
            )
            warped_lab[:, :, c] = np.clip(warped_lab[:, :, c], 0, 255)

        warped = cv2.cvtColor(warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return warped

    def multiband_blend(self, imageB, warped, blend_start, blend_end, h):
        """
        Multi-band Laplacian pyramid blending for seamless stitching.
        Seam is at the left edge of right camera - simple gradient blend.
        """
        w_left = imageB.shape[1]
        result_w = warped.shape[1]

        # Create full-size images for blending
        left_full = np.zeros_like(warped)
        left_full[:h, :w_left] = imageB[:h, :w_left]
        right_full = warped.copy()

        # ============ CREATE BLEND MASK ============
        # Simple gradient mask - no motion detection needed
        # because seam is at the edge where there's minimal overlap
        mask = np.zeros((h, result_w), dtype=np.float32)
        mask[:, :blend_start] = 0  # Left image only
        mask[:, blend_end:] = 1  # Right image only

        # Smooth gradient in narrow blend zone
        blend_width = blend_end - blend_start
        if blend_width > 0:
            gradient = np.linspace(0, 1, blend_width, dtype=np.float32)
            gradient = gradient * gradient * (3 - 2 * gradient)  # Smooth curve
            mask[:, blend_start:blend_end] = np.tile(gradient, (h, 1))

        # ============ MULTI-BAND BLENDING ============
        num_levels = 5

        # Build Gaussian pyramids for the mask
        mask_pyr = [mask]
        current_mask = mask
        for i in range(num_levels - 1):
            current_mask = cv2.pyrDown(current_mask)
            mask_pyr.append(current_mask)

        # Build Laplacian pyramids for both images
        left_lap_pyr = self.build_laplacian_pyramid(left_full, num_levels)
        right_lap_pyr = self.build_laplacian_pyramid(right_full, num_levels)

        # Blend each level
        blended_pyr = []
        for i in range(num_levels):
            mask_3 = np.dstack([mask_pyr[i]] * 3)
            blended_level = left_lap_pyr[i] * (1 - mask_3) + right_lap_pyr[i] * mask_3
            blended_pyr.append(blended_level)

        # Reconstruct from blended pyramid
        result = self.reconstruct_from_laplacian(blended_pyr)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def build_laplacian_pyramid(self, img, levels):
        """Build a Laplacian pyramid."""
        img_float = img.astype(np.float32)
        gaussian_pyr = [img_float]

        current = img_float
        for i in range(levels - 1):
            current = cv2.pyrDown(current)
            gaussian_pyr.append(current)

        laplacian_pyr = []
        for i in range(levels - 1):
            size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
            laplacian = gaussian_pyr[i] - expanded
            laplacian_pyr.append(laplacian)

        # Top level is just the smallest Gaussian
        laplacian_pyr.append(gaussian_pyr[-1])

        return laplacian_pyr

    def reconstruct_from_laplacian(self, pyramid):
        """Reconstruct image from Laplacian pyramid."""
        current = pyramid[-1]

        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            current = cv2.pyrUp(current, dstsize=size)
            current = current + pyramid[i]

        return current

    def detectAndDescribe(self, image):
        """Feature detection using CPU SIFT."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, features) = self.detector.detectAndCompute(gray, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """Feature matching."""
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
                    ptsA,
                    ptsB,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=reprojThresh,
                    confidence=0.99,
                    maxIters=2000,
                )
                if M_affine is not None and inliers is not None:
                    H = np.vstack([M_affine, [0, 0, 1]])
                    status = inliers.ravel().astype(np.uint8)
                    return (matches, H, status)
            except:
                pass

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None

        return None


def stitch_video_cylindrical(left_video_path, right_video_path, output_name, **kwargs):
    """
    Stitch two videos using cylindrical warping.
    """
    logger.info(
        f"Starting Cylindrical Stitching: {left_video_path} + {right_video_path}"
    )

    # 1. Open Video Captures
    capL = cv2.VideoCapture(left_video_path)
    capR = cv2.VideoCapture(right_video_path)

    if not capL.isOpened() or not capR.isOpened():
        raise IOError("Could not open one or both video files.")

    # 2. Get Video Properties
    fps = capL.get(cv2.CAP_PROP_FPS)
    width = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(
        min(capL.get(cv2.CAP_PROP_FRAME_COUNT), capR.get(cv2.CAP_PROP_FRAME_COUNT))
    )

    logger.info(f"Video Properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Prepare output filenames
    base_name = output_name[:-4] if output_name.endswith(".mp4") else output_name
    temp_output = f"{base_name}_temp.avi"
    final_output = f"{base_name}.mp4"

    # 3. Initialize Stitcher
    stitcher = Stitcher(use_cuda=True)

    # 4. Initialize Video Writer
    writer = None

    try:
        processed_frames = 0

        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()

            if not retL or not retR:
                break

            # Stitch frame
            result = stitcher.stitch([frameL, frameR])

            if result is None:
                logger.warning(
                    f"Stitching failed at frame {processed_frames}, using fallback"
                )
                # Fallback: Side-by-side
                hL, wL = frameL.shape[:2]
                hR, wR = frameR.shape[:2]

                # Resize to matching height if needed
                if hL != hR:
                    frameR = cv2.resize(frameR, (int(wR * hL / hR), hL))

                result = np.hstack([frameL, frameR])

            # Initialize writer on first successful frame
            if writer is None:
                out_h, out_w = result.shape[:2]
                logger.info(f"Output Video Resolution: {out_w}x{out_h}")

                # Use MJPG for reliable intermediate writing
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))
                if not writer.isOpened():
                    raise IOError(f"Could not open VideoWriter for {temp_output}")

            # Ensure result size matches writer size
            if result.shape[0] != out_h or result.shape[1] != out_w:
                result = cv2.resize(result, (out_w, out_h))

            # Write frame
            writer.write(result)

            processed_frames += 1

            # Progress update
            if processed_frames % 10 == 0:
                progress = processed_frames / total_frames
                if _progress_callback:
                    # Get GPU utilization
                    gpu_util = 0
                    if GPU_AVAILABLE:
                        try:
                            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(
                                GPU_HANDLE
                            ).gpu
                        except:
                            pass

                    gpu_status = "ON" if stitcher.use_cuda else "OFF"
                    msg = f"Stitching frame {processed_frames}/{total_frames} (GPU:{gpu_status} {gpu_util}%)"
                    _progress_callback(progress, msg)

    except Exception as e:
        logger.error(f"Error during stitching: {e}")
        raise e
    finally:
        capL.release()
        capR.release()
        if writer:
            writer.release()

    # Post-process with FFmpeg to ensure web compatibility
    # Phase 2: Re-encode
    output_file = temp_output
    if os.path.exists(temp_output):
        logger.info("=" * 60)
        logger.info("ENCODING (Phase 2)")

        import shutil
        import subprocess

        # Check if NVENC is available
        nvenc_available = False
        try:
            check_result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
            )
            nvenc_available = "h264_nvenc" in check_result.stdout
        except:
            pass

        # Use NVENC GPU encoding if available
        if nvenc_available:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel", "cuda",
                "-i", temp_output,
                "-c:v", "h264_nvenc",
                "-preset", "p4",
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

        try:
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
                        if os.path.exists(temp_output) and not os.path.exists(output_name):
                            shutil.move(temp_output, output_name)
                            output_file = output_name
                else:
                    if os.path.exists(temp_output) and not os.path.exists(output_name):
                        shutil.move(temp_output, output_name)
                        output_file = output_name
        except Exception as e:
            logger.error(f"FFmpeg execution failed: {e}")
            if os.path.exists(temp_output) and not os.path.exists(output_name):
                shutil.move(temp_output, output_name)
                output_file = output_name

    # Ensure we return the path that actually exists
    return output_file


# Global callback
_progress_callback = None


def set_progress_callback(callback):
    global _progress_callback
    _progress_callback = callback
