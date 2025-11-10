import cv2
import dlib
import pickle
import pandas as pd
import numpy as np
import datetime
import os

def predict_dlib_faces(model_folder="DlibModel", employee_csv="EmployeeDetails/EmployeeDetails.csv", att_dir="Attendance"):
    os.makedirs(att_dir, exist_ok=True)

    # Fix 4: Check if trained model exists before loading
    model_path = os.path.join(model_folder, "dlib_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found: {model_path}\nPlease train the model first using 'Train Images (Dlib)' button.")

    # Load trained SVM model with error handling
    try:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading trained model: {str(e)}\nThe model file may be corrupted. Please retrain the model.")

    # Load employee mapping with error handling (Fix 11)
    try:
        if not os.path.exists(employee_csv) or os.path.getsize(employee_csv) == 0:
            raise FileNotFoundError(f"Employee CSV file is empty or doesn't exist: {employee_csv}")
        df = pd.read_csv(employee_csv)
        if 'Id' not in df.columns or 'Name' not in df.columns:
            raise ValueError(f"Employee CSV must have 'Id' and 'Name' columns. Found: {df.columns.tolist()}")
        if len(df) == 0:
            raise ValueError("Employee CSV is empty. Please add employees first.")
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except Exception as e:
        raise RuntimeError(f"Error reading employee CSV: {str(e)}")

    # Dlib setup
    detector = dlib.get_frontal_face_detector()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (parent directory of dlib_logic)
    project_root = os.path.dirname(script_dir)
    
    # Construct absolute paths to dlib model files
    # sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")              # <- old hardcoded path
    # facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # <- old hardcoded path
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

    # Attendance DataFrame
    attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])
    recognized_ids = set()

    # Fix 5: Camera validation
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open camera!\nPlease check if camera is connected and not being used by another application.")
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Press 'q' to quit. Unknown faces will be marked as 'Unknown'.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        faces = detector(frame)
        for face in faces:
            shape = sp(frame, face)
            embedding = np.array(facerec.compute_face_descriptor(frame, shape)).reshape(1, -1)

            # Predict using SVM (Fix 12: Add error handling)
            try:
                probs = clf.predict_proba(embedding)
                conf = probs.max()
                pred_id = clf.classes_[probs.argmax()]
                
                if conf > 0.6:  # recognition threshold
                    emp_row = df[df['Id'] == pred_id]
                    name = emp_row['Name'].values[0] if not emp_row.empty else "Unknown"
                else:
                    name = "Unknown"
            except Exception as e:
                # If prediction fails, mark as unknown
                print(f"Warning: Prediction error for face: {str(e)}")
                name = "Unknown"
                conf = 0.0
                pred_id = None

            label = f"{name} ({int(conf*100)}%)" if name != "Unknown" else "Unknown"

            # Mark attendance
            if name != "Unknown" and pred_id is not None and pred_id not in recognized_ids:
                ts = datetime.datetime.now()
                attendance.loc[len(attendance)] = [pred_id, name, ts.strftime('%Y-%m-%d'), ts.strftime('%H:%M:%S')]
                recognized_ids.add(pred_id)

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), font, 0.7, (255, 255, 255), 2)

        cv2.imshow("Dlib Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(attendance) > 0:
        filename = os.path.join(att_dir, f"Attendance_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        attendance.to_csv(filename, index=False)
        return len(attendance), filename
    else:
        return 0, None


# Example standalone run
if __name__ == "__main__":
    predict_dlib_faces()
