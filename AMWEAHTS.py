#Made by Matt Naganidhi 21st of November 2023
#import libraries
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import dlib
from screeninfo import get_monitors
from pynput import mouse
import threading

#level of sensitivity
sensitivity = 1
last_mouse_position = None
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Function to extract eye coordinates from dlib facial landmarks
def get_eye_coordinates(landmarks, eye_indices):
    return [(landmarks.part(point).x, landmarks.part(point).y) for point in eye_indices]

def estimate_gaze(landmarks, frame_width, frame_height):
    # Define indices for facial landmarks for eyes and face orientation
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263]
    FACE_ORIENTATION_POINTS = [234, 454, 117, 152, 10]  # Example indices for orientation

    # Calculate eye centers
    left_eye_center = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in LEFT_EYE_INDICES], axis=0)
    right_eye_center = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in RIGHT_EYE_INDICES], axis=0)

    # Extract face orientation points
    face_orientation_pts = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in FACE_ORIENTATION_POINTS])

    # Estimate head orientation (Placeholder for actual orientation calculation)
    head_orientation = np.mean(face_orientation_pts, axis=0)

    # Combine head orientation and eye centers for gaze estimation (Placeholder for actual gaze logic)
    # This can be replaced with a more sophisticated algorithm
    gaze_x = (left_eye_center[0] + right_eye_center[0] + head_orientation[0]) / 3
    gaze_y = (left_eye_center[1] + right_eye_center[1] + head_orientation[1]) / 3

    # Normalize and map to screen coordinates
    screen_width, screen_height = pyautogui.size()
    gaze_x = min(max(gaze_x, 0), 1) *  screen_width * sensitivity
    gaze_y = min(max(gaze_y, 0), 1) *  screen_height * sensitivity

    return gaze_x, gaze_y


# Function to display a slider and get the sensitivity value
def show_sensitivity_slider(window_name, sensitivity):
    # Create a window and slider for sensitivity adjustment
    cv2.namedWindow(window_name)
    cv2.createTrackbar('sensitivity', window_name, 0, 100, lambda x: None)
    cv2.setTrackbarPos('sensitivity', window_name, int(sensitivity * 10))
    
    # Update the sensitivity value based on the slider position
    while True:
        sensitivity = cv2.getTrackbarPos('sensitivity', window_name) / 10.0
    return sensitivity


last_position = (0, 0)
change_position = (0, 0)
total_mouse_movements = (0,0)
def on_move(x, y):
    global last_position, total_mouse_movements, dx, dy, change_position
    dx, dy = x - last_position[0], y - last_position[1]
    
    # Storing the movement data
    total_mouse_movements = (total_mouse_movements[0] + dx, total_mouse_movements[1] + dy)
    # Optionally print the movement
    #print(f"Mouse moved: dx={dx}, dy={dy}")
    last_position = (x, y)
    change_position = (dx, dy)
    return total_mouse_movements
def start_mouse_listener():
    with mouse.Listener(on_move=on_move) as listener:
        listener.join() 

# Start the mouse listener in a separate thread
mouse_listener_thread = threading.Thread(target=start_mouse_listener)
mouse_listener_thread.start()
last_gaze_position = (0,0)
def control_mouse(gaze_x, gaze_y, total_mouse_movements):
    global last_mouse_position, last_gaze_position
    try:
        screen_width, screen_height = pyautogui.size()

        # Current mouse position and gaze position
        current_mouse_position = pyautogui.position()
        gaze_position = (screen_width - np.clip(gaze_x, 0, screen_width),
                         np.clip(gaze_y, 0, screen_height))
        # Determine the final cursor position
        if current_mouse_position != last_mouse_position:
            # User is manually moving the mouse
            final_x, final_y = current_mouse_position
        #stabilize the cursor movement from noisy gaze data
        elif (abs(gaze_position[0] - last_gaze_position[0]) < 4 or abs(gaze_position[1] - last_gaze_position[1]) < 4): 
           final_x, final_y = current_mouse_position

        else:
            sensitivity2 = 3
            change_gaze_x = gaze_position[0] - last_gaze_position[0] 
            change_gaze_y = gaze_position[1] - last_gaze_position[1]
            final_x = (change_gaze_x * sensitivity2 + current_mouse_position[0])
            final_y = (change_gaze_y * sensitivity2 + current_mouse_position[1])
        print("x movement" + str(final_x) + "y movement" + str(final_y))
        last_gaze_position = gaze_position
        # Move the cursor to the final position
        pyautogui.moveTo(final_x, final_y)

        # Update last positions
        last_mouse_position = current_mouse_position
    except pyautogui.FailSafeException:
        print("Fail-safe triggered")

def show_sensitivity_slider(window_name):
    global sensitivity
    cv2.namedWindow(window_name)
    cv2.createTrackbar('sensitivity', window_name, 0, 100, lambda x: None)
    cv2.setTrackbarPos('sensitivity', window_name, int(sensitivity * 10))

def update_sensitivity(window_name):
    global sensitivity
    sensitivity = cv2.getTrackbarPos('sensitivity', window_name) / 10.0



def main():
    global sensitivity
    # Initialize your camera, face mesh, etc.
    cam = cv2.VideoCapture(0)

    # Setup the sensitivity slider
    show_sensitivity_slider("Settings")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        flipped_frame = cv2.flip(frame, 1)

        # Convert to RGB and process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                gaze_x, gaze_y = estimate_gaze(face_landmarks, frame.shape[1], frame.shape[0])
                control_mouse(gaze_x, gaze_y, total_mouse_movements)  # Pass sensitivity as an argument
                update_sensitivity("Settings")
        cv2.imshow('AMWEAHTS (Assisting Mouse With Eyes and Head Tracking Software)', flipped_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            exit(0)


        # Break the loop if needed
        #cam.release()
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
