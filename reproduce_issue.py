import cv2
import numpy as np
import os
import sys

# Add current directory to path so we can import pano
sys.path.append(os.getcwd())

try:
    import pano
except ImportError:
    print("Could not import pano. Make sure you are running this from the project root.")
    sys.exit(1)

def create_dummy_video(filename, width=640, height=480, fps=30, duration=2):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    frames = int(duration * fps)
    for _ in range(frames):
        # Create random noise frame
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    print(f"Created dummy video: {filename}")

def main():
    # Create two dummy videos
    video1 = "dummy_left.mp4"
    video2 = "dummy_right.mp4"
    
    create_dummy_video(video1)
    create_dummy_video(video2)
    
    output_name = "reproduction_output"
    
    print("\nStarting stitching reproduction test...")
    print("These videos are random noise, so feature matching WILL fail.")
    print("We expect the current implementation to skip all frames and produce no valid output video.")
    
    try:
        # Run stitching
        # We expect this to run but produce an empty or non-existent file if all frames fail
        result_file = pano.stitch_videos_2cam(
            video1, video2, 
            output_name=output_name,
            show_preview=False, # Headless
            moving_camera=True # Force feature matching every frame
        )
        
        print(f"\nStitching function returned: {result_file}")
        
        if os.path.exists(result_file):
            size = os.path.getsize(result_file)
            print(f"Output file exists. Size: {size} bytes")
            if size < 1000:
                print("FAILURE CONFIRMED: Output file is too small (likely empty header only).")
            else:
                # If we actually got a huge file from random noise, that's unexpected but interesting.
                print("Output file seems to have content (Unexpected for random noise if logic skips frames).")
        else:
            print("FAILURE CONFIRMED: Output file was not created.")
            
    except Exception as e:
        print(f"\nStitching crashed with error: {e}")

    # Cleanup
    if os.path.exists(video1): os.remove(video1)
    if os.path.exists(video2): os.remove(video2)
    # Don't remove output yet, as we might want to inspect it

if __name__ == "__main__":
    main()
