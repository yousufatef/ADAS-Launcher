import cv2
import numpy as np
from time import time, sleep
import serial
import os

# Initialize serial connection to Arduino
try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Adjust port as needed for Jetson
    sleep(2)  # Wait for connection to establish
except serial.SerialException as e:
    print(f"Error connecting to Arduino: {e}")
    ser = None

# Load face detection model - using more accurate model for Jetson
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For eye detection (to detect if driver is sleeping)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Camera setup for Jetson - may need to adjust for CSI camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)  # Try alternative camera index
    
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Parameters
WARNING_DURATION = 3  # seconds before stopping vehicle for eye closure
CENTER_THRESHOLD = 50  # pixels from center to consider "forward"
WARNING_COUNTER_THRESHOLD = 5  # consecutive frames to trigger head position warning
NO_FACE_THRESHOLD = 30  # frames with no face detected before warning
EYE_CLOSED_THRESHOLD = 10  # frames with eyes not detected before warning
EYE_DETECTION_CONFIDENCE = 0.7  # Confidence threshold for eye detection

# State variables
warning_active = False
warning_start_time = 0
frame_counter = 0
deviation_counter = 0
no_face_counter = 0
eyes_closed_counter = 0
head_position_warning = False
eye_warning = False

def send_arduino_command(command):
    if ser is not None:
        try:
            ser.write(f"{command}\n".encode())
            print(f"Sent command: {command}")  # Debug print
        except serial.SerialException as e:
            print(f"Error sending command to Arduino: {e}")

def draw_center_guidelines(frame):
    height, width = frame.shape[:2]
    cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 1)
    cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 1)
    return frame

def detect_eyes(face_roi_gray):
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 4)
    return len(eyes) > 0  # Returns True if eyes are detected

def adjust_eye_detection_roi(face_rect, head_orientation):
    """Adjust eye detection ROI based on head orientation"""
    x, y, w, h = face_rect
    
    # Standard ROI - upper half of face
    if head_orientation == "Forward":
        return (x, y, w, h//2)
    
    # Adjusted ROIs for different head orientations
    elif head_orientation == "Left":
        # Shift ROI slightly to the right
        return (x + w//4, y, w - w//4, h//2)
    
    elif head_orientation == "Right":
        # Shift ROI slightly to the left
        return (x, y, w - w//4, h//2)
    
    else:
        return (x, y, w, h//2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        frame = draw_center_guidelines(frame)
        frame_width = frame.shape[1]
        frame_center = frame_width // 2
        
        head_position = "No face detected"
        eye_state = "Eyes not detected"
        warning_reason = ""
        
        if len(faces) > 0:
            no_face_counter = 0  # Reset no face counter
            
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Determine head position
            face_center = x + w//2
            deviation = face_center - frame_center
            
            if abs(deviation) < CENTER_THRESHOLD:
                head_position = "Forward"
                deviation_counter = max(0, deviation_counter - 1)
                head_position_warning = False
            elif deviation < 0:
                head_position = "Left"
                deviation_counter += 1
                if deviation_counter >= WARNING_COUNTER_THRESHOLD:
                    head_position_warning = True
                    warning_reason = "Driver not looking forward"
                    send_arduino_command("BUZZER_ON")  # Activate buzzer
            else:
                head_position = "Right"
                deviation_counter += 1
                if deviation_counter >= WARNING_COUNTER_THRESHOLD:
                    head_position_warning = True
                    warning_reason = "Driver not looking forward"
                    send_arduino_command("BUZZER_ON")  # Activate buzzer
            
            # Adjust eye detection ROI based on head position
            eye_roi = adjust_eye_detection_roi((x, y, w, h), head_position)
            x_eye, y_eye, w_eye, h_eye = eye_roi
            face_roi_gray = gray[y_eye:y_eye+h_eye, x_eye:x_eye+w_eye]
            
            # Visualize eye detection ROI
            cv2.rectangle(frame, (x_eye, y_eye), (x_eye+w_eye, y_eye+h_eye), (0, 255, 255), 1)
            
            # Detect eyes in the adjusted ROI
            eyes_detected = detect_eyes(face_roi_gray)
            
            if eyes_detected:
                eyes_closed_counter = 0
                eye_state = "Eyes open"
                eye_warning = False
                if head_position_warning:
                    send_arduino_command("BUZZER_OFF")  # Turn off buzzer if eyes are open
            else:
                eyes_closed_counter += 1
                eye_state = "Eyes closed"
                if eyes_closed_counter >= EYE_CLOSED_THRESHOLD:
                    eye_warning = True
                    warning_reason = "Driver may be sleeping"
                    if not warning_active:
                        warning_active = True
                        warning_start_time = time()
                        send_arduino_command("WARNING")
                        print(f"Warning started - {warning_reason}")
            
            # Display position info
            cv2.putText(frame, f"Head: {head_position}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Deviation: {deviation}px", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Eyes: {eye_state}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        else:
            no_face_counter += 1
            deviation_counter = 0
            eyes_closed_counter = 0
            head_position_warning = False
            
            if no_face_counter >= NO_FACE_THRESHOLD:
                warning_reason = "Driver not detected"
                if not warning_active:
                    warning_active = True
                    warning_start_time = time()
                    send_arduino_command("WARNING")
                    print("Warning started - driver not detected")
            
            if warning_active and no_face_counter < NO_FACE_THRESHOLD:
                warning_active = False
                send_arduino_command("NORMAL")
                
        # Check warning duration for eye closure (only stop for eye closure)
        if eye_warning and (time() - warning_start_time) > WARNING_DURATION:
            print("Stopping vehicle - prolonged eye closure")
            send_arduino_command("STOP")  # Command to stop the motor
            warning_active = False
            eye_warning = False
            
        # For head position warnings, just keep buzzing until corrected
        if head_position_warning and deviation_counter < WARNING_COUNTER_THRESHOLD:
            head_position_warning = False
            send_arduino_command("BUZZER_OFF")
            
        # Display warning status
        status = "NORMAL" 
        color = (0, 255, 0)  # Green
        
        if warning_active or head_position_warning:
            status = f"WARNING: {warning_reason}"
            if warning_active:  # Only show timer for eye warnings
                status += f" ({int(time() - warning_start_time)}s)"
            color = (0, 0, 255)  # Red
            
        cv2.putText(frame, f"Status: {status}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Driver Monitoring', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("Exiting...")
    
finally:
    send_arduino_command("NORMAL")  # Reset Arduino state
    send_arduino_command("BUZZER_OFF")  # Ensure buzzer is off
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()