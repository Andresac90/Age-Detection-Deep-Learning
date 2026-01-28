"""
Complete Setup Script for Age Detection
- Checks and fixes OpenCV installation
- Downloads all required model files
- Verifies everything is ready
"""

import os
import sys
import zipfile
import subprocess

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Model URLs
MODELS_ZIP_URL = 'https://media.geeksforgeeks.org/wp-content/uploads/20250324160232380462/Age-prediction.zip'
AGE_MODEL_URLS = [
    'https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1',
    'https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel',
]


def check_opencv():
    """Check if OpenCV is installed with GUI support."""
    print("\n" + "="*60)
    print("Checking OpenCV Installation")
    print("="*60)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Check for GUI support
        has_gui = hasattr(cv2, 'imshow')
        
        if has_gui:
            print("✓ GUI support available")
            return True
        else:
            print("✗ GUI support NOT available")
            print("  You have opencv-python-headless installed")
            return False
            
    except ImportError:
        print("✗ OpenCV not installed")
        return False


def fix_opencv():
    """Fix OpenCV installation by installing the correct package."""
    print("\n" + "="*60)
    print("Fixing OpenCV Installation")
    print("="*60)
    
    response = input("\nReinstall OpenCV with GUI support? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Skipping OpenCV fix.")
        print("\nWARNING: Webcam mode will not work without GUI support!")
        print("You can still use image and video modes.")
        return False
    
    print("\nUninstalling current OpenCV packages...")
    packages_to_remove = ['opencv-python', 'opencv-contrib-python', 'opencv-python-headless']
    
    for package in packages_to_remove:
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except:
            pass
    
    print("Installing opencv-python with GUI support...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'opencv-python>=4.8.0'
        ])
        print("✓ OpenCV installed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error installing OpenCV: {e}")
        return False


def download_with_requests(url, filename):
    """Download using requests library."""
    if os.path.exists(filename):
        print(f"✓ {filename} already exists, skipping...")
        return True
    
    print(f"\nDownloading {filename}...")
    
    try:
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.get(url, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        sys.stdout.write(f"\r  Progress: {percent}%")
                        sys.stdout.flush()
        
        print(f"\n✓ Downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False


def extract_zip(zip_path, extract_to='models'):
    """Extract zip file and organize files."""
    print(f"\nExtracting to {extract_to}/...")
    
    try:
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            for file in file_list:
                zip_ref.extract(file, extract_to)
                base_name = os.path.basename(file)
                if base_name and not base_name.startswith('.'):
                    print(f"  Extracted: {base_name}")
            
            # Move files if they're in a subfolder
            extracted_items = os.listdir(extract_to)
            
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_to, extracted_items[0])):
                subfolder = os.path.join(extract_to, extracted_items[0])
                
                for item in os.listdir(subfolder):
                    src = os.path.join(subfolder, item)
                    dst = os.path.join(extract_to, item)
                    
                    if os.path.isfile(src):
                        os.rename(src, dst)
                
                try:
                    os.rmdir(subfolder)
                except:
                    pass
        
        print("✓ Extraction complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def download_age_model(models_dir='models'):
    """Download the age_net.caffemodel file."""
    filepath = os.path.join(models_dir, 'age_net.caffemodel')
    
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\n✓ age_net.caffemodel already exists ({file_size:.2f} MB)")
        return True
    
    print("\n" + "="*60)
    print("Downloading age_net.caffemodel (~44 MB)")
    print("="*60)
    
    for i, url in enumerate(AGE_MODEL_URLS, 1):
        print(f"\nAttempt {i}/{len(AGE_MODEL_URLS)}:")
        print(f"  Source: {url.split('/')[2]}")
        
        if download_with_requests(url, filepath):
            # Verify size
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  File size: {file_size:.2f} MB")
            
            if file_size > 40:
                return True
            else:
                print(f"  ✗ File too small, trying next source...")
                os.remove(filepath)
        
        print(f"  ✗ Failed, trying next source...")
    
    return False


def verify_models(models_dir='models'):
    """Verify all model files."""
    required_files = {
        'opencv_face_detector.pbtxt': 0.04,
        'opencv_face_detector_uint8.pb': 2.6,
        'age_deploy.prototxt': 0.01,
        'age_net.caffemodel': 44.0
    }
    
    print("\n" + "="*60)
    print("Verifying Model Files")
    print("="*60)
    
    all_present = True
    
    for filename, expected_size in required_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename}")
            print(f"    Size: {actual_size:.2f} MB (expected ~{expected_size:.2f} MB)")
        else:
            print(f"✗ {filename} - MISSING")
            all_present = False
    
    print("="*60)
    
    return all_present


def main():
    """Main setup function."""
    print("="*60)
    print("Age Detection - Complete Setup")
    print("="*60)
    print("\nThis script will:")
    print("  1. Check/fix OpenCV installation")
    print("  2. Download model files (~50 MB)")
    print("  3. Verify everything is ready")
    print("-"*60)
    
    if not REQUESTS_AVAILABLE:
        print("\n✗ Error: 'requests' library not installed")
        print("\nPlease run: pip install requests")
        return
    
    # Step 1: Check OpenCV
    opencv_ok = check_opencv()
    
    if not opencv_ok:
        if not fix_opencv():
            print("\n⚠ WARNING: OpenCV GUI support not available")
            print("Webcam mode will not work, but you can still use image/video modes.")
    
    # Step 2: Download models
    print("\n" + "="*60)
    print("Downloading Model Files")
    print("="*60)
    
    response = input("\nProceed with model download? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Setup cancelled.")
        return
    
    # Download main package
    zip_filename = 'Age-prediction.zip'
    
    if download_with_requests(MODELS_ZIP_URL, zip_filename):
        if not extract_zip(zip_filename):
            print("\n✗ Extraction failed.")
            return
        
        # Clean up zip
        try:
            os.remove(zip_filename)
            print(f"\n✓ Cleaned up {zip_filename}")
        except:
            pass
    else:
        print("\n✗ Failed to download model package.")
        return
    
    # Download age model
    if not download_age_model():
        print("\n✗ Failed to download age_net.caffemodel")
        print("\nManual download:")
        print("  Visit: https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1")
        print("  Save to: models/age_net.caffemodel")
        return
    
    # Step 3: Verify
    if verify_models():
        print("\n" + "="*60)
        print("✓ Setup Complete!")
        print("="*60)
        print("\nYou can now run:")
        print("  python test_system.py         # Test your setup")
        print("  python age_detection.py --mode webcam  # Run webcam detection")
        print("  python age_detection.py --mode image --input photo.jpg  # Process image")
    else:
        print("\n✗ Setup incomplete. Please check missing files.")


if __name__ == '__main__':
    main()