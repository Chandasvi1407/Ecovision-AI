"""
EcoVision AI - Webcam Diagnostic Test Script
Author: College AI Mini Project Team

Purpose:
Tests whether your HP laptop built-in webcam is accessible using OpenCV's
DirectShow backend (cv2.CAP_DSHOW) on Windows 11.

Usage:
  python webcam_test.py
"""

import sys
import time

try:
    import cv2
except ImportError:
    print("[Error] OpenCV is not installed yet. Please run:")
    print("  pip install opencv-python")
    sys.exit(1)


def test_webcam():
    print("=" * 60)
    print("  EcoVision AI - Laptop Webcam Diagnostic (Windows 11)")
    print("=" * 60)
    print("Attempting to connect to camera via DirectShow (cv2.CAP_DSHOW)...")

    camera_index = 0
    # Use cv2.CAP_DSHOW for fast, reliable initialization on Windows 11
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"[Warning] Camera index {camera_index} failed to open.")
        print("Attempting camera index 1 (external/virtual camera if present)...")
        cap.release()
        camera_index = 1
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("\n[FAILED] Could not access built-in webcam.")
        print("\nTroubleshooting tips for Windows 11:")
        print("1. Check Windows Privacy Settings:")
        print("   Start -> Settings -> Privacy & security -> Camera -> Ensure 'Let apps access your camera' is ON.")
        print("2. Check if another application (Teams, Zoom, Discord, Chrome) is currently using the camera.")
        print("3. Check HP privacy shutter / function key (some HP laptops have an F2/F12 camera killswitch).")
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n[SUCCESS] Webcam connected successfully on index {camera_index}!")
    print(f"  Negotiated Resolution: {width} x {height}")
    print(f"  Reported Hardware FPS: {fps if fps > 0 else 'Variable / Not reported'}")
    print("\nStarting video test loop...")
    print("A diagnostic window will open. Press 'q' on your keyboard to close it.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[Warning] Failed to receive frame from webcam stream.")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # Draw diagnostic overlay
            cv2.putText(
                frame,
                f"EcoVision AI Webcam Test - Index: {camera_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f} | Res: {frame.shape[1]}x{frame.shape[0]}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press 'q' to exit test",
                (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2
            )

            cv2.imshow("EcoVision AI - Webcam Diagnostic", frame)

            # Check if user pressed 'q' to exit or after 30 seconds
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser requested exit ('q' pressed).")
                break
            if elapsed > 30:
                print("\nDiagnostic completed successfully (30s test duration reached).")
                break

    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Diagnostic Complete] Camera hardware released cleanly.\n")

    return True


if __name__ == "__main__":
    success = test_webcam()
    sys.exit(0 if success else 1)
