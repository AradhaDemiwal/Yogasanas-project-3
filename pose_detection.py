# import cv2
# import mediapipe as mp

# # Initialize mediapipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_draw = mp.solutions.drawing_utils

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process
#     result = pose.process(rgb)

#     if result.pose_landmarks:
#         # Draw joints
#         mp_draw.draw_landmarks(
#             frame,
#             result.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS
#         )

#     cv2.imshow("Joint Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

clicked_joint = None
joint_positions = []

# Mouse click function
def mouse_click(event, x, y, flags, param):
    global clicked_joint

    if event == cv2.EVENT_LBUTTONDOWN:
        for (jx, jy, name) in joint_positions:
            distance = math.hypot(x - jx, y - jy)

            if distance < 20:  # click radius
                clicked_joint = name
                break

cv2.namedWindow("Pose")
cv2.setMouseCallback("Pose", mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    joint_positions = []  # reset every frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, c = frame.shape

        for id, lm in enumerate(result.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            name = mp_pose.PoseLandmark(id).name

            # store positions
            joint_positions.append((cx, cy, name))

            # draw small circle
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Show clicked joint name
    if clicked_joint:
        cv2.putText(frame, f"Selected: {clicked_joint}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Pose", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()