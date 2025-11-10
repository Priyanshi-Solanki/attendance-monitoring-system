import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recogniser Attendance System")
        self.root.geometry('1024x720')
        self.root.configure(bg="#f0f0f0")

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Face Recognition Attendance System", font=('Arial', 25, 'bold'), bg="#282c34", fg="white", pady=20).pack(fill=tk.X)

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
        tk.Button(btn_frame, text="Train Images", font=('Arial', 14), command=self.train_images).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="Track Attendance", font=('Arial', 14), command=self.track_images).pack(side=tk.LEFT, padx=20)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def take_images(self):
        Id = self.emp_id_entry.get().strip()
        name = self.emp_name_entry.get().strip()

        if not (self.is_number(Id) and name.isalpha()):
            messagebox.showerror("Error", "Invalid ID or Name\nID must be a number\nName must contain only letters")
            return

        # Check if employee already exists
        os.makedirs("EmployeeDetails", exist_ok=True)
        csv_path = 'EmployeeDetails/EmployeeDetails.csv'
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                df = pd.read_csv(csv_path)
                if int(Id) in df['Id'].values:
                    messagebox.showerror("Error", f"Employee with ID {Id} already exists!")
                    return
            except Exception as e:
                # If CSV is empty or corrupted, create new one
                pass

        cam = cv2.VideoCapture(0)
        # detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        sampleNum = 0

        os.makedirs("TrainingImage", exist_ok=True)

        # messagebox.showinfo("Instructions", "Press 'q' to stop capturing\nMake sure your face is clearly visible in the camera")
        messagebox.showinfo("Instructions", "Press 'q' to stop capturing\nMake sure your face is clearly visible.\nPlease turn your head slightly left/right/up/down during capture to add variation.")

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # normalize lighting
            gray = cv2.equalizeHist(gray)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sampleNum += 1
                face_roi = gray[y:y+h, x:x+w]
                # resize to consistent size for training
                # face_resized = cv2.resize(face_roi, (200, 200))
                face_resized = cv2.resize(face_roi, (200, 200))
                face_equalized = cv2.equalizeHist(face_resized)
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", face_equalized)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, f'Captured: {sampleNum}/100', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
                # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # cv2.putText(img, f'Captured: {sampleNum}/30', (x, y-10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Capturing Image', img)
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Add employee to CSV (check if file exists and has header)
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            with open(csv_path, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['Id', 'Name'])
                writer.writerow([Id, name])
        else:
            with open(csv_path, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([Id, name])

        messagebox.showinfo("Success", f"Images Saved for ID: {Id}, Name: {name}\nPlease train the model before tracking attendance.")

    def train_images(self):
        # Check if training images exist
        if not os.path.exists("TrainingImage") or len([f for f in os.listdir("TrainingImage") if f.endswith('.jpg')]) == 0:
            messagebox.showerror("Error", "No training images found!\nPlease capture images first.")
            return

        # Check if cv2.face module is available
        if not hasattr(cv2, 'face'):
            messagebox.showerror("Error", "OpenCV face module not found!\n\nPlease install opencv-contrib-python:\npip install opencv-contrib-python\n\n(Not opencv-python)")
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            messagebox.showerror("Error", "LBPH Face Recognizer not available!\n\nPlease install opencv-contrib-python:\npip install opencv-contrib-python")
            return

        # detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

        faces, Ids = self.get_images_and_labels("TrainingImage")
        
        if len(faces) == 0:
            messagebox.showerror("Error", "No valid training images found!")
            return
        
        recognizer.train(faces, np.array(Ids))

        os.makedirs("TrainingImageLabel", exist_ok=True)
        recognizer.save("TrainingImageLabel/Trainer.yml")

        messagebox.showinfo("Training Complete", f"Model trained successfully!\nTrained on {len(faces)} images from {len(set(Ids))} employee(s).")

    def get_images_and_labels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces, Ids = [], []
        # sort imagePaths for deterministic ordering
        imagePaths = sorted(imagePaths)
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            # ensure uniform size (in case some files differ)
            imageNp = cv2.resize(imageNp, (200, 200))
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        # for imagePath in imagePaths:
        #     pilImage = Image.open(imagePath).convert('L')
        #     imageNp = np.array(pilImage, 'uint8')
        #     Id = int(os.path.split(imagePath)[-1].split(".")[1])
        #     faces.append(imageNp)
        #     Ids.append(Id)
        return faces, Ids

    def track_images(self):
        # Check if trained model exists
        if not os.path.exists("TrainingImageLabel/Trainer.yml"):
            messagebox.showerror("Error", "No trained model found!\nPlease train the model first.")
            return

        # Check if employee details exist
        if not os.path.exists("EmployeeDetails/EmployeeDetails.csv") or os.path.getsize("EmployeeDetails/EmployeeDetails.csv") == 0:
            messagebox.showerror("Error", "No employee details found!")
            return

        # Check if cv2.face module is available
        if not hasattr(cv2, 'face'):
            messagebox.showerror("Error", "OpenCV face module not found!\n\nPlease install opencv-contrib-python:\npip install opencv-contrib-python\n\n(Not opencv-python)")
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("TrainingImageLabel/Trainer.yml")
        except AttributeError:
            messagebox.showerror("Error", "LBPH Face Recognizer not available!\n\nPlease install opencv-contrib-python:\npip install opencv-contrib-python")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Error loading trained model!\n\n{str(e)}\n\nPlease train the model first.")
            return

        # faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

        
        # Read employee details (handle header)
        try:
            df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv")
            # Ensure column names are correct
            if 'Id' not in df.columns or 'Name' not in df.columns:
                df = pd.read_csv("EmployeeDetails/EmployeeDetails.csv", header=None, names=["Id", "Name"])
        except:
            messagebox.showerror("Error", "Error reading employee details file!")
            return

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])
        recognized_ids = set()  # Track IDs that have been recognized in this session

        messagebox.showinfo("Instructions", "Press 'q' to stop tracking attendance\nUnknown faces will be marked as 'Unknown'")

        while True:
            ret, im = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            # Apply histogram equalization for consistent lighting (same as training)
            gray = cv2.equalizeHist(gray)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                # Ensure test face matches training dimensions
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))

                Id, conf = recognizer.predict(face_resized)

                # Confidence handling: LBPH ~ lower=better
                if conf < 50:  # relaxed threshold for lighting variance
                    name_data = df[df['Id'] == Id]
                    if not name_data.empty:
                        name = name_data['Name'].values[0]
                        label = f"{name} ({int(conf)}%)"

                        if Id not in recognized_ids:
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                            recognized_ids.add(Id)

                        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for recognized
                    else:
                        label = f"Unknown ID: {Id}"
                        cv2.rectangle(im, (x, y), (0, 165, 255), 2)  # Orange if weird mismatch
                else:
                    label = f"Unknown ({int(conf)}%)"
                    # cv2.rectangle(im, (x, y), (0, 0, 255), 2)  # Red for unknown
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for unknown


                cv2.putText(im, label, (x, y-10), font, 0.7, (255, 255, 255), 2)
            # ---------------------------------------------------------------------
            # for (x, y, w, h) in faces:
            #     Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                
            #     # In LBPH: Lower confidence = better match (typically < 50 is good, < 30 is excellent)
            #     # Only accept recognition if confidence is low AND ID exists in database
            #     if conf < 50:
            #         # Check if this ID exists in our employee database
            #         name_data = df[df['Id'] == Id]
            #         if not name_data.empty:
            #             name = name_data['Name'].values[0]
            #             label = f"{Id} - {name} (Present)"
            #             # Only add to attendance once per session per person
            #             if Id not in recognized_ids:
            #                 ts = time.time()
            #                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            #                 timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            #                 attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
            #                 recognized_ids.add(Id)
            #             cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for recognized
            #         else:
            #             # ID recognized but not in database (shouldn't happen if training is correct)
            #             label = f"Unknown ID: {Id}"
            #             cv2.rectangle(im, (x, y), (x+w, y+h), (0, 165, 255), 2)  # Orange
            #     else:
            #         # Confidence too high = unknown face
            #         label = "Unknown"
            #         cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for unknown

            #     cv2.putText(im, label, (x, y-10), font, 0.75, (255, 255, 255), 2)
            # --------------------------------------------------------------------------------------------------

            cv2.imshow('Tracking Attendance', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save attendance if any records exist
        if len(attendance) > 0:
            attendance.drop_duplicates(subset=['Id'], inplace=True)
            os.makedirs("Attendance", exist_ok=True)
            filename = f"Attendance/Attendance_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            attendance.to_csv(filename, index=False)
            messagebox.showinfo("Attendance Saved", f"Attendance saved to {filename}\n{len(attendance)} employee(s) marked present.")
        else:
            messagebox.showinfo("No Attendance", "No recognized employees in this session.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
