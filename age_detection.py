import cv2
import numpy as np
import argparse
import os
import sys

# Age categories
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Mean values for model normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class AgeDetector:
    def __init__(self, face_proto, face_model, age_proto, age_model):
        self.face_net = None
        self.age_net = None
        
        # Load models
        self.load_models(face_proto, face_model, age_proto, age_model)
    
    def load_models(self, face_proto, face_model, age_proto, age_model):
        try:
            print("Loading face detection model...")
            self.face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
            print("✓ Face detection model loaded successfully!")
            
            print("Loading age prediction model...")
            self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
            print("✓ Age prediction model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("\nMake sure you have downloaded the required model files.")
            print("See README.md for download instructions.")
            raise
    
    def detect_faces(self, frame, conf_threshold=0.7):
        frame_height, frame_width = frame.shape[:2]
        
        # Prepare blob for face detection
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            [104, 117, 123], False, False
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_boxes = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                
                face_boxes.append([x1, y1, x2, y2])
                
                # Draw rectangle around face
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0),
                    int(round(frame_height / 150)),
                    8
                )
        
        return frame, face_boxes
    
    def predict_age(self, face):
        # Prepare blob for age prediction
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        # Get the age with highest probability
        age = AGE_LIST[age_preds[0].argmax()]
        
        return age
    
    def process_frame(self, frame):
        frame_copy = frame.copy()
        frame_copy, face_boxes = self.detect_faces(frame_copy)
        
        for (x1, y1, x2, y2) in face_boxes:
            # Add padding to face crop
            face = frame[
                max(0, y1 - 20):min(y2 + 20, frame.shape[0] - 1),
                max(0, x1 - 20):min(x2 + 20, frame.shape[1] - 1)
            ]
            
            # Skip if face crop is too small
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            
            # Predict age
            age = self.predict_age(face)
            
            # Add text annotation
            label = f"Age: {age}"
            cv2.putText(
                frame_copy, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2,
                cv2.LINE_AA
            )
        
        return frame_copy
    
    def process_webcam(self, camera_index=0):
        print(f"\nStarting webcam (Camera {camera_index})...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("-" * 50)
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            print("Try a different camera index (0, 1, 2, etc.)")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add instructions overlay
            cv2.putText(
                processed_frame, "Press 'q' to quit | 's' to save",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
            
            # Display the frame
            cv2.imshow('Age Detection - Real-time', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"age_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Frame saved as {filename}")
                frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")
    
    def process_image(self, image_path, output_path=None):
        print(f"\nProcessing image: {image_path}")
        
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Process the image
        processed_frame = self.process_frame(frame)
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"Output saved to: {output_path}")
        else:
            cv2.imshow('Age Detection - Image', processed_frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None):
        print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write to output video
            if out:
                out.write(processed_frame)
            
            # Display progress
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"Processing: {frame_num}/{total_frames} frames", end='\r')
            
            # Show preview
            cv2.imshow('Age Detection - Video', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        print(f"\nProcessed {frame_num} frames")
        
        cap.release()
        if out:
            out.release()
            print(f"Output saved to: {output_path}")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Age Detection using Deep Learning')
    
    parser.add_argument(
        '--mode', type=str, default='webcam',
        choices=['webcam', 'image', 'video'],
        help='Processing mode: webcam, image, or video'
    )
    
    parser.add_argument(
        '--input', type=str,
        help='Path to input image or video file'
    )
    
    parser.add_argument(
        '--output', type=str,
        help='Path to output file (for image/video mode)'
    )
    
    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera index for webcam mode (default: 0)'
    )
    
    parser.add_argument(
        '--face-proto', type=str, default='models/opencv_face_detector.pbtxt',
        help='Path to face detection prototxt file'
    )
    
    parser.add_argument(
        '--face-model', type=str, default='models/opencv_face_detector_uint8.pb',
        help='Path to face detection model file'
    )
    
    parser.add_argument(
        '--age-proto', type=str, default='models/age_deploy.prototxt',
        help='Path to age detection prototxt file'
    )
    
    parser.add_argument(
        '--age-model', type=str, default='models/age_net.caffemodel',
        help='Path to age detection caffemodel file'
    )
    
    args = parser.parse_args()
    
    # Check if model files exist
    model_files = [
        args.face_proto,
        args.face_model,
        args.age_proto,
        args.age_model
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease download the models first. See README.md for instructions.")
        sys.exit(1)
    
    # Initialize detector
    try:
        detector = AgeDetector(
            args.face_proto,
            args.face_model,
            args.age_proto,
            args.age_model
        )
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        sys.exit(1)
    
    # Process based on mode
    print("\n" + "="*50)
    print("Age Detection System - Real-time Processing")
    print("="*50)
    
    if args.mode == 'webcam':
        detector.process_webcam(args.camera)
    
    elif args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            sys.exit(1)
        detector.process_image(args.input, args.output)
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            sys.exit(1)
        detector.process_video(args.input, args.output)


if __name__ == '__main__':
    main()