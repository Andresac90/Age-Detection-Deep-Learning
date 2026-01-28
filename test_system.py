import cv2
import sys

def test_webcam():
    print("Testing webcam access...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Could not access webcam")
        print("\nTroubleshooting:")
        print("  1. Make sure no other application is using the webcam")
        print("  2. Check camera permissions in system settings")
        print("  3. Try a different camera index: 1, 2, etc.")
        return False
    
    print("✓ Webcam accessible!")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret:
        print("✗ Error: Could not read from webcam")
        cap.release()
        return False
    
    print(f"✓ Frame captured: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    cap.release()
    return True


def test_opencv():
    print("\nTesting OpenCV installation...")
    
    try:
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Check if DNN module is available
        if hasattr(cv2, 'dnn'):
            print("✓ OpenCV DNN module available")
        else:
            print("✗ OpenCV DNN module not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*50)
    print("Age Detection - System Test")
    print("="*50)
    
    # Test OpenCV
    opencv_ok = test_opencv()
    
    if not opencv_ok:
        print("\n✗ OpenCV test failed")
        print("Please install OpenCV: pip install opencv-python")
        sys.exit(1)
    
    # Test webcam
    webcam_ok = test_webcam()
    
    if not webcam_ok:
        print("\n✗ Webcam test failed")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
    print("\nYou can now run:")
    print("  1. Download models: python download_models.py")
    print("  2. Run detection: python age_detection.py --mode webcam")


if __name__ == '__main__':
    main()