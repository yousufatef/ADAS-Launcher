import numpy as np
import cv2
import serial
import time

# Serial communication setup
SERIAL_PORT = '/dev/ttyACM0'  # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BAUD_RATE = 9600      # Make sure this matches your Arduino's baud rate
arduino = None
last_send_time = 0
SEND_INTERVAL = 0.1   # Send data every 100ms (10 FPS) to avoid overwhelming Arduino

# Initialize serial connection
def init_serial():
    global arduino
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.001, write_timeout=0.001)  # Very short timeouts
        time.sleep(2)  # Wait for Arduino to initialize
        print(f"Connected to Arduino on {SERIAL_PORT}")
        return True
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        print("Continuing without serial connection...")
        return False

# Send data to Arduino (non-blocking)
def send_to_arduino(angle):
    global arduino, last_send_time
    current_time = time.time()
    
    # Throttle sending to avoid overwhelming the serial buffer
    if current_time - last_send_time < SEND_INTERVAL:
        return
    
    if arduino and arduino.is_open:
        try:
            # Clear input buffer to prevent buildup
            if arduino.in_waiting > 0:
                arduino.reset_input_buffer()
            
            # Convert steering angle to servo range (0-180, where 90 is center)
            # Assuming input angle is in range -90 to +90 degrees
            servo_angle = int(angle + 90)  # Convert -90~+90 to 0~180
            servo_angle = max(0, min(180, servo_angle))  # Constrain to 0-180
            
            # Send angle in Arduino expected format: "S" + angle + newline
            message = f"S{angle}\n"
            arduino.write(message.encode())
            arduino.flush()  # Force write
            last_send_time = current_time
            
            # Print what was sent to Arduino
            print(f"Sent to Arduino: S{servo_angle} (Original angle: {int(angle)}°)")
            
        except (serial.SerialException, serial.SerialTimeoutException) as e:
            print(f"Serial error (non-fatal): {e}")
            # Don't close connection, just skip this send

# Smoothing parameters
SMOOTHING_FACTOR = 0.2  # Adjust this to control sensitivity (0.1 = very smooth, 0.9 = very sensitive)
prev_steering_angle = 0  # Store the previous steering angle for smoothing

# Parameters for Hough Transform and Canny
CANNY_LOW_THRESHOLD = 100  # Increased from 50
CANNY_HIGH_THRESHOLD = 200  # Increased from 150
HOUGH_THRESHOLD = 50  # Increased from 20
HOUGH_MIN_LINE_LENGTH = 50  # Increased from 20
HOUGH_MAX_LINE_GAP = 100  # Decreased from 500

# Frame processing function
def frame_processor(image):
    """
    Processes a single frame to detect lane lines and calculate steering angle.
    """
    global prev_steering_angle  # Use the global variable for smoothing
    
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    
    # Detect edges using Canny with adjusted thresholds
    edges = cv2.Canny(blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    # Select region of interest
    region = region_selection(edges)
    
    # Perform Hough Transform with adjusted parameters
    hough = hough_transform(region)
    
    # Draw lane lines and calculate steering angle
    if hough is not None:
        left_line, right_line = lane_lines(image, hough)
        
        # If only one lane is detected, assume the other lane
        lane_width_pixels = 200  # Adjust based on your camera and lane width
        if left_line is None and right_line is not None:
            left_line = assume_other_lane(image, right_line, -lane_width_pixels)
        elif right_line is None and left_line is not None:
            right_line = assume_other_lane(image, left_line, lane_width_pixels)
        
        # Draw lanes and fill the area between them
        result = draw_lane_lines(image, (left_line, right_line), detected_color=(255, 0, 0), predicted_color=(0, 0, 255), area_color=(0, 255, 0))
        
        # Calculate steering angle
        steering_angle = calculate_steering_angle(image, left_line, right_line)
        
        # Smooth the steering angle
        smoothed_angle = smooth_steering_angle(steering_angle)
        
        # Send smoothed angle to Arduino
        send_to_arduino(smoothed_angle)
        
        # Display steering angle on the frame
        cv2.putText(result, f"Steering Angle: {int(smoothed_angle)} degrees", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        result = image  # If no lines are detected, return the original frame
        # Send 0 to Arduino if no lanes detected
        send_to_arduino(0)
    
    return result

# Smoothing function for steering angle
def smooth_steering_angle(new_angle):
    """
    Applies a low-pass filter to smooth the steering angle.
    """
    global prev_steering_angle
    smoothed_angle = prev_steering_angle + SMOOTHING_FACTOR * (new_angle - prev_steering_angle)
    prev_steering_angle = smoothed_angle
    return smoothed_angle

# Region selection function
def region_selection(image):
    """
    Masks out a specific region of interest (likely the road).
    """
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Hough Transform function
def hough_transform(image):
    """
    Detects lines using the Hough Transform algorithm.
    """
    rho = 1
    theta = np.pi / 180
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=HOUGH_THRESHOLD,
                           minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

# Lane line averaging function
def average_slope_intercept(lines):
    """
    Averages the detected lines to find the left and right lane lines.
    """
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

# Pixel points conversion function
def pixel_points(y1, y2, line):
    """
    Converts slope-intercept form to pixel coordinates.
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # Check for invalid slope (e.g., zero or infinity)
    if abs(slope) < 1e-5:  # Threshold for near-zero slope
        return None
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

# Lane lines computation function
def lane_lines(image, lines):
    """
    Computes the lane lines from the averaged slopes and intercepts.
    """
    if lines is None:
        return None, None  # Return None if no lines are detected
    
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

# Steering angle calculation function
def calculate_steering_angle(image, left_line, right_line):
    """
    Calculates the steering angle required to center the vehicle in the lane.
    """
    if left_line is None or right_line is None:
        return 0  # No lines detected, return 0 angle
    
    # Get the bottom points of the left and right lines
    (x1_left, y1_left), (x2_left, y2_left) = left_line
    (x1_right, y1_right), (x2_right, y2_right) = right_line
    
    # Calculate the midpoint of the lane at the bottom of the image
    lane_center = (x1_left + x1_right) / 2
    
    # Calculate the center of the image
    image_center = image.shape[1] / 2
    
    # Calculate the offset from the center
    offset = lane_center - image_center
    
    # Calculate the steering angle (in degrees)
    focal_length = 1000  # Approximate focal length (adjust based on your camera)
    steering_angle = np.arctan(offset / focal_length) * (180 / np.pi)
    
    return steering_angle

# Function to assume the other lane
def assume_other_lane(image, detected_line, lane_width_pixels):
    """
    Assumes the position of the other lane based on the detected lane and lane width.
    """
    height, width = image.shape[:2]
    
    if detected_line is not None:
        (x1, y1), (x2, y2) = detected_line
        
        # Calculate the slope and intercept of the detected line
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        else:
            slope = float('inf')  # Vertical line
            intercept = x1
        
        # Assume the other lane is `lane_width_pixels` away
        if slope != float('inf'):
            # For non-vertical lines
            x1_assumed = int(x1 - lane_width_pixels)
            x2_assumed = int(x2 - lane_width_pixels)
            y1_assumed = int(slope * x1_assumed + intercept)
            y2_assumed = int(slope * x2_assumed + intercept)
        else:
            # For vertical lines
            x1_assumed = int(x1 - lane_width_pixels)
            x2_assumed = int(x2 - lane_width_pixels)
            y1_assumed = y1
            y2_assumed = y2
        
        assumed_line = ((x1_assumed, y1_assumed), (x2_assumed, y2_assumed))
        return assumed_line
    else:
        return None

# Function to draw lane lines and fill the area between them
def draw_lane_lines(image, lines, detected_color=(255, 0, 0), predicted_color=(0, 0, 255), area_color=(0, 255, 0)):
    """
    Draws the detected and predicted lane lines and fills the area between them.
    """
    line_image = np.zeros_like(image)
    
    # Draw detected lane (blue)
    if lines[0] is not None:
        cv2.line(line_image, *lines[0], detected_color, 12)
    
    # Draw predicted lane (red)
    if lines[1] is not None:
        cv2.line(line_image, *lines[1], predicted_color, 12)
    
    # Fill the area between the lanes (green)
    if lines[0] is not None and lines[1] is not None:
        (x1_left, y1_left), (x2_left, y2_left) = lines[0]
        (x1_right, y1_right), (x2_right, y2_right) = lines[1]
        
        # Create a polygon for the area between the lanes
        pts = np.array([[x1_left, y1_left], [x2_left, y2_left], [x2_right, y2_right], [x1_right, y1_right]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(line_image, [pts], area_color)
    
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

# Main function for webcam processing
def process_webcam():
    """
    Captures video from the webcam and processes each frame to detect lane lines.
    """
    # Initialize serial connection
    serial_connected = init_serial()
    
    # Open the external webcam (use the correct index, e.g., 1 or 2)
    cap = cv2.VideoCapture(0)  # Change this index if necessary           /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
    
    frame_count = 0
    
    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Process the frame
            processed_frame = frame_processor(frame)
            
            # Display the processed frame
            cv2.imshow("Lane Detection", processed_frame)
            
            # Exit on 'q' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Press 'r' to reconnect serial
                if arduino:
                    arduino.close()
                init_serial()
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        # Close serial connection
        if arduino and arduino.is_open:
            arduino.close()
            print("Serial connection closed")

# Run the webcam processing
if __name__ == "__main__":
    process_webcam()