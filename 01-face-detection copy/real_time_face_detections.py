import cv2

# Load the Haar Cascade classifier for face detection
# The XML file contains the pre-trained data for detecting faces
face_detector = cv2.CascadeClassifier('../classifiers/haarcascade_frontalface_default.xml')

# Initialize video capture using the default webcam (0 represents the default camera)
video_capture = cv2.VideoCapture(0)

# Start an infinite loop to process video frames in real time
while True:
    # Capture a single frame from the webcam
    # `ret` is a boolean that indicates whether the frame was successfully captured
    # `frame` contains the captured image (if `ret` is True)
    ret, frame = video_capture.read()

    # Convert the captured frame to grayscale for better performance in face detection
    # Haar Cascade classifiers work more efficiently with grayscale images
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection on the grayscale image
    # `minSize=(100, 100)` ensures that only faces larger than 100x100 pixels are detected
    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100))

    # Iterate over all detected faces and draw rectangles around them
    for (x, y, w, h) in detections:
        # Draw a green rectangle around each detected face
        # (x, y) represents the top-left corner of the rectangle
        # (x+w, y+h) represents the bottom-right corner
        # 2 is the thickness of the rectangle's border
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the video frame with the detected faces in a window named 'Video'
    cv2.imshow('Video', frame)

    # Check if the 'q' key is pressed to exit the loop
    # `cv2.waitKey(1)` waits for a key press for 1ms
    # `0xFF` ensures compatibility with different platforms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resource to avoid locking it
video_capture.release()

# Close all OpenCV windows that were opened
cv2.destroyAllWindows()
