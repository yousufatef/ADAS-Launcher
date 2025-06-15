import cv2
import numpy as np
import serial
import time
from threading import Thread

# Camera setup
LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection parameters
MIN_CONTOUR_AREA = 1000  # Minimum contour area to consider as an object
DETECTION_THRESHOLD = 30  # Difference threshold for motion detection
ALERT_ZONE_RATIO = 0.3  # Percentage of image width considered alert zone

# Serial communication with Arduino
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600

class BlindSpotMonitor:
    def __init__(self):
        # Initialize cameras
        self.left_cam = cv2.VideoCapture(LEFT_CAMERA_ID)
        self.right_cam = cv2.VideoCapture(RIGHT_CAMERA_ID)
        
        # Set camera resolution
        self.left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Initialize background subtractors
        self.left_bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.right_bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Initialize serial connection to Arduino
        self.serial_conn = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # Wait for connection to establish
        
        # State variables
        self.left_alert = False
        self.right_alert = False
        self.running = True
        
    def process_frame(self, frame, bg_sub, side):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = bg_sub.apply(gray)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(fg_mask, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Define alert zone (right side for left camera, left side for right camera)
        alert_zone_width = int(CAMERA_WIDTH * ALERT_ZONE_RATIO)
        if side == 'left':
            alert_zone = (CAMERA_WIDTH - alert_zone_width, 0, CAMERA_WIDTH, CAMERA_HEIGHT)
        else:
            alert_zone = (0, 0, alert_zone_width, CAMERA_HEIGHT)
        
        # Check for objects in alert zone
        alert = False
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                # Check if object is in alert zone
                if (side == 'left' and x + w > alert_zone[0]) or (side == 'right' and x < alert_zone[2]):
                    alert = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Draw alert zone
        cv2.rectangle(frame, (alert_zone[0], alert_zone[1]), 
                     (alert_zone[2], alert_zone[3]), (0, 255, 255), 2)
        
        return frame, alert
    
    def send_alert_to_arduino(self):
        while self.running:
            # Create command string (L for left, R for right, 1 for alert, 0 for clear)
            command = f"L{1 if self.left_alert else 0}R{1 if self.right_alert else 0}\n"
            self.serial_conn.write(command.encode())
            time.sleep(0.1)  # Send at 10Hz
    
    def run(self):
        # Start thread for sending alerts to Arduino
        arduino_thread = Thread(target=self.send_alert_to_arduino)
        arduino_thread.start()
        
        try:
            while self.running:
                # Read frames from both cameras
                left_ret, left_frame = self.left_cam.read()
                right_ret, right_frame = self.right_cam.read()
                
                if not left_ret or not right_ret:
                    print("Error reading camera frames")
                    break
                
                # Process frames
                left_frame, self.left_alert = self.process_frame(left_frame, self.left_bg_sub, 'left')
                right_frame, self.right_alert = self.process_frame(right_frame, self.right_bg_sub, 'right')
                
                # Display frames
                cv2.imshow('Left Camera', left_frame)
                cv2.imshow('Right Camera', right_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
        finally:
            # Cleanup
            self.running = False
            arduino_thread.join()
            self.left_cam.release()
            self.right_cam.release()
            cv2.destroyAllWindows()
            self.serial_conn.close()

if __name__ == "__main__":
    monitor = BlindSpotMonitor()
    monitor.run()