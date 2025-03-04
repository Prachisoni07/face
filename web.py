import cv2

# Open the webcam (default camera, index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Loop to capture and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame from webcam")
        break

    # Display the frame in a window
    cv2.imshow("Webcam Test", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()