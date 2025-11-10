import os
import cv2
import dlib
import numpy as np
from sklearn import svm
import pickle

def train_dlib_model(train_folder="TrainingImage", model_folder="DlibModel"):
    """
    Train a Dlib + SVM face recognition model on images from train_folder,
    and save the model & employee IDs to model_folder.
    """
    os.makedirs(model_folder, exist_ok=True)

    # Load Dlib face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    # predictor_path = "shape_predictor_68_face_landmarks.dat"        # Ensure file exists
    # sp = dlib.shape_predictor(predictor_path)
    # facerec_path = "dlib_face_recognition_resnet_model_v1.dat"      # Ensure file exists
    # facerec = dlib.face_recognition_model_v1(facerec_path)
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (parent directory of dlib_logic)
    project_root = os.path.dirname(script_dir)

    # Construct absolute paths to dlib model files
    predictor_path = os.path.join(project_root, "shape_predictor_68_face_landmarks.dat")
    facerec_path = os.path.join(project_root, "dlib_face_recognition_resnet_model_v1.dat")
    # Check if files exist before loading
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Shape predictor file not found: {predictor_path}\nPlease ensure the file is in the project root directory.")
    if not os.path.exists(facerec_path):
        raise FileNotFoundError(f"Face recognition model file not found: {facerec_path}\nPlease ensure the file is in the project root directory.")
    
    # Load dlib models with error handling
    try:
        sp = dlib.shape_predictor(predictor_path)
        facerec = dlib.face_recognition_model_v1(facerec_path)
    except Exception as e:
        raise RuntimeError(f"Error loading dlib models: {str(e)}\nPlease ensure the model files are valid and not corrupted.")
    # Read images and labels
    image_files = [f for f in os.listdir(train_folder) if f.endswith('.jpg')]
    embeddings = []
    labels = []

    for file in image_files:
        path = os.path.join(train_folder, file)
        img = cv2.imread(path)
        
        # Fix 6: Validate image loading
        if img is None:
            print(f"Warning: Could not read image {file}. Skipping...")
            continue
        
        # Fix 13: Use BGR image for face detection (consistent with prediction)
        # Dlib detector works on both grayscale and color, but using color is more consistent
        faces = detector(img)  # Use BGR image for consistency with prediction
        if len(faces) == 0:
            continue  # skip if no face detected in this image

        # Take first detected face
        face = faces[0]
        shape = sp(img, face)
        face_embedding = facerec.compute_face_descriptor(img, shape)
        
        embeddings.append(np.array(face_embedding))
        
        # Fix 7: Validate filename format before parsing
        try:
            parts = file.split(".")
            if len(parts) < 3:
                print(f"Warning: Invalid filename format '{file}'. Expected format: Name.ID.Num.jpg. Skipping...")
                continue
            emp_id = int(parts[1])  # Second part should be the ID
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse employee ID from filename '{file}'. Skipping...")
            continue
        
        labels.append(emp_id)

    if len(labels) == 0:
        print("No faces found in training images.")
        return 0, 0  # Return tuple instead of None

    # Train SVM classifier
    clf = svm.SVC(C=1.0, kernel='linear', probability=True)
    clf.fit(embeddings, labels)

    # Save model
    with open(os.path.join(model_folder, "dlib_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # Save embeddings IDs (optional)
    np.save(os.path.join(model_folder, "employee_ids.npy"), np.array(labels))

    print(f"Training complete on {len(labels)} faces for {len(set(labels))} employees.")
    return len(labels), len(set(labels))