import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os

def convert_video_to_25fps(input_video_path, output_video_path):
    try:
        subprocess.run(['ffmpeg', '-i', input_video_path, '-r', '25', output_video_path], check=True)
        print(f"Video converted to 25 fps and saved as {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("An error occurred during video conversion:", e)

def get_lips_bbox(landmarks, lip_indices):
    lip_points = np.array([landmarks[i] for i in lip_indices])
    x, y, w, h = cv2.boundingRect(lip_points)
    return x, y, w, h

# Define a function to align the face using eye landmarks.
def align_face(image, landmarks, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
    if desired_face_height is None:
        desired_face_height = desired_face_width

    # The indices for the left and right eye corners.
    left_eye_idx = 130
    right_eye_idx = 359

    # Extract the left and right eye (x, y) coordinates.
    left_eye_center = landmarks[left_eye_idx]
    right_eye_center = landmarks[right_eye_idx]

    # Compute the angle between the eye centroids.
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Calculate the desired right eye x-coordinate based on the desired x-coordinate of the left eye.
    desired_right_eye_x = 1.0 - desired_left_eye[0]

    # Determine the scale of the new resulting image by taking the ratio of the distance between eyes in the current image to the ratio of distance in the desired image.
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_face_width
    scale = desired_dist / dist

    # Compute center (x, y)-coordinates between the two eyes in the input image.
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

    # Grab the rotation matrix for rotating and scaling the face.
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Update the translation component of the matrix.
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    # Apply the affine transformation.
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # Return the aligned face and the transformation matrix.
    return output, M

# Function to transform landmarks using the same transformation as the face alignment
def transform_landmarks(landmarks, M):
    transformed_landmarks = []
    for landmark in landmarks:
        # Apply the transformation matrix to each landmark point
        x, y = landmark
        transformed_point = np.dot(M, np.array([x, y, 1]))
        transformed_landmarks.append((int(transformed_point[0]), int(transformed_point[1])))
    return transformed_landmarks


def get_video_crops(video_path):
    print('1')
    lip_indices = [187, 411, 136, 365]
    # Initialize MediaPipe Face Detection and Face Mesh
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) 
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.2, refine_landmarks=True)
    convert_video_to_25fps(video_path,'temp.mp4')
    cap = cv2.VideoCapture('temp.mp4')
    lips_crops_bw_list=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use face detection to find faces
        detection_results = face_detection.process(rgb_frame)
        if detection_results.detections:
            for detection in detection_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Crop and display the face
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    face_crop = frame[y:y+h, x:x+w]
                    face_crop = cv2.resize(face_crop, (400, 400))
                    # Align the face crop
                    face_mesh_results = face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    if face_mesh_results.multi_face_landmarks:
                        for face_landmarks in face_mesh_results.multi_face_landmarks:
                            points = [(int(p.x * face_crop.shape[1]), int(p.y * face_crop.shape[0])) for p in face_landmarks.landmark]
                            aligned_face, _ = align_face(face_crop, points)

                            # Crop lips from the aligned face
                            transformed_landmarks = transform_landmarks(points, _)
                            x, y, w, h = get_lips_bbox(transformed_landmarks, lip_indices)
                            lips_crop = aligned_face[y:y+h, x:x+w]
                            lips_crop = cv2.resize(lips_crop, (88, 88))  # Resize for better visibility
                            lips_crop_bw = cv2.cvtColor(lips_crop, cv2.COLOR_BGR2GRAY)
                            # cv2.imshow('Lips Crop', lips_crop_bw)
                            lips_crops_bw_list.append(lips_crop_bw)

        # Press 'q' to quit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # np.save('lips_crops_bw.npy', np.array(lips_crops_bw_list))
        # Release resources
    face_mesh.close()
    face_detection.close()
    cap.release()
    cv2.destroyAllWindows()
    os.remove('temp.mp4')
    print(np.array(lips_crops_bw_list).shape)
    return np.array(lips_crops_bw_list)

