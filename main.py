import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import pandas as pd
# Fix 9: Removed unused imports (datetime and time)

# Import Dlib-based logic
from dlib_logic.train_dlib import train_dlib_model
from dlib_logic.predict_dlib import predict_dlib_faces

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System - Dlib")
        self.root.geometry('1024x720')
        self.root.configure(bg="#f0f0f0")

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Face Recognition Attendance System", font=('Arial', 25, 'bold'),
                 bg="#282c34", fg="white", pady=20).pack(fill=tk.X)

        form_frame = tk.Frame(self.root, bg="#f0f0f0")
        form_frame.pack(pady=20)

        tk.Label(form_frame, text="Employee ID", font=('Arial', 14)).grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
        self.emp_id_entry = tk.Entry(form_frame, font=('Arial', 14))
        self.emp_id_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(form_frame, text="Employee Name", font=('Arial', 14)).grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
        self.emp_name_entry = tk.Entry(form_frame, font=('Arial', 14))
        self.emp_name_entry.grid(row=1, column=1, padx=10, pady=10)

        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Capture Images", font=('Arial', 14), command=self.take_images).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="Train Images (Dlib)", font=('Arial', 14), command=self.train_images).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="Track Attendance (Dlib)", font=('Arial', 14), command=self.track_images).pack(side=tk.LEFT, padx=20)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def take_images(self):
        """Use your existing capture logic from train.py"""
        Id = self.emp_id_entry.get().strip()
        name = self.emp_name_entry.get().strip()

        if not (self.is_number(Id) and name.isalpha()):
            messagebox.showerror("Error", "Invalid ID or Name\nID must be a number\nName must contain only letters")
            return

        # Check if employee already exists (Fix 11: Add error handling for CSV)
        os.makedirs("EmployeeDetails", exist_ok=True)
        csv_path = 'EmployeeDetails/EmployeeDetails.csv'
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                df = pd.read_csv(csv_path)
                if int(Id) in df['Id'].values:
                    messagebox.showerror("Error", f"Employee with ID {Id} already exists!")
                    return
            except pd.errors.EmptyDataError:
                # CSV exists but is empty, will create new one
                pass
            except Exception as e:
                messagebox.showerror("Error", f"Error reading employee database: {str(e)}")
                return

        # Fix 5: Camera validation
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Could not open camera!\nPlease check if camera is connected and not being used by another application.")
            return
        
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        sampleNum = 0
        os.makedirs("TrainingImage", exist_ok=True)

        messagebox.showinfo("Instructions", "Press 'q' to stop capturing\nMake sure your face is clearly visible.\nPlease turn your head slightly left/right/up/down during capture to add variation.")

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sampleNum += 1
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                face_equalized = cv2.equalizeHist(face_resized)
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", face_equalized)

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, f'Captured: {sampleNum}/100', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Capturing Image', img)
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Add employee to CSV (Fix 11: Add error handling)
        try:
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                with open(csv_path, 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(['Id', 'Name'])
                    writer.writerow([Id, name])
            else:
                with open(csv_path, 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([Id, name])
        except Exception as e:
            messagebox.showerror("Error", f"Error saving employee details: {str(e)}")
            return

        messagebox.showinfo("Success", f"Images Saved for ID: {Id}, Name: {name}\nPlease train the Dlib model before tracking attendance.")

    def train_images(self):
        """Call Dlib training logic"""
        if not os.path.exists("TrainingImage") or len([f for f in os.listdir("TrainingImage") if f.endswith('.jpg')]) == 0:
            messagebox.showerror("Error", "No training images found!\nPlease capture images first.")
            return

        os.makedirs("DlibModel", exist_ok=True)
        
        # Fix 14: Add error handling in train_images method
        try:
            num_faces, num_employees = train_dlib_model(train_folder="TrainingImage", model_folder="DlibModel")
            messagebox.showinfo("Training Complete", f"Training complete on {num_faces} images for {num_employees} employees.")
        except Exception as e:
            messagebox.showerror("Training Error", f"Error during training: {str(e)}")


    def track_images(self):
        """Call Dlib prediction/attendance logic"""
        # Fix 4: Check if trained model file exists (not just folder)
        model_path = os.path.join("DlibModel", "dlib_model.pkl")
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "No trained Dlib model found!\nPlease train the model first.")
            return

        # Fix 15: Add error handling in track_images method
        try:
            num_present, filename = predict_dlib_faces(model_folder="DlibModel", employee_csv="EmployeeDetails/EmployeeDetails.csv")
            if num_present > 0:
                messagebox.showinfo("Attendance Recorded", f"Attendance recorded for {num_present} employees.\nSaved to {filename}")
            else:
                messagebox.showinfo("Attendance Recorded", "No recognized employees in this session.")
        except Exception as e:
            messagebox.showerror("Attendance Error", f"Error during attendance tracking: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()