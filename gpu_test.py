import cv2
import pynvml
import time

print("=" * 40)
print("GPU DIAGNOSTIC TOOL")
print("=" * 40)

# 1. Check OpenCV CUDA
print("\n[1] Checking OpenCV CUDA...")
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA Devices found: {count}")
    if count > 0:
        cv2.cuda.printCudaDeviceInfo(0)
        print("Running dummy CUDA op...")
        dummy = cv2.cuda_GpuMat()
        dummy.upload(cv2.imread(__file__))  # Just upload something or fail
        print("CUDA Upload Success")
    else:
        print("NO CUDA DEVICES FOUND IN OPENCV")
except Exception as e:
    print(f"OpenCV CUDA Error: {e}")

# 2. Check NVML (pynvml)
print("\n[2] Checking Pynvml (Nvidia Management Library)...")
try:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    print(f"NVML Device Count: {deviceCount}")

    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"Device {i}: {name}")

        # Monitor for a few seconds
        print("Monitoring utilizaton for 3 seconds...")
        for _ in range(3):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(
                f"  - GPU: {util.gpu}%, Mem: {util.memory}%, Used: {mem.used / 1024**2:.1f}MB"
            )
            time.sleep(1)

except Exception as e:
    print(f"NVML Error: {e}")

print("\nDone.")
