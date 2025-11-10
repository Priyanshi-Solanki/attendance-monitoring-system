# Face Recognition Attendance System - Dlib Edition

A Tkinter-based face recognition attendance system using **Dlib + SVM** for high-accuracy face recognition. This system can capture face images, train a recognition model, and track attendance in real-time with automatic unknown face detection.

## 🎯 Features

- ✅ **High Accuracy**: Uses Dlib's deep learning face recognition model (ResNet-based)
- ✅ **Capture face images** for multiple users
- ✅ **Train recognition model** with Dlib + SVM classifier
- ✅ **Real-time attendance tracking** with live camera feed
- ✅ **Automatic unknown face detection**
- ✅ **CSV-based attendance records** with timestamps
- ✅ **Prevents duplicate attendance** entries per session
- ✅ **User-friendly GUI interface**
- ✅ **Robust error handling** and validation

## 📋 Requirements

### System Requirements
- **Python 3.7+** (Python 3.8, 3.9, 3.10, or 3.11 recommended)
- **Webcam/Camera** (for face capture and attendance tracking)
- **Windows/Linux/macOS** (tested on Windows)

### Python Dependencies
- `opencv-python` or `opencv-contrib-python` (4.5.0+)
- `dlib` (19.24.1+)
- `scikit-learn` (for SVM classifier)
- `pandas` (for CSV handling)
- `numpy` (for array operations)
- `tkinter` (usually comes with Python)

### Required Model Files
The following files must be in the project root directory:
- `shape_predictor_68_face_landmarks.dat` (face landmark detector)
- `dlib_face_recognition_resnet_model_v1.dat` (face recognition model)

**Note:** These files are included in the repository. If missing, download from:
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (extract the .bz2 file)
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) (extract the .bz2 file)

## 🚀 Installation

### Step 1: Clone or Download the Repository

```bash
git clone <your-repository-url>
cd "final cv project 200"
```

Or download and extract the ZIP file.

### Step 2: Install Python Dependencies

#### Option A: Using pip (Recommended)

```bash
# Install basic dependencies
pip install opencv-python pandas numpy scikit-learn

# Install dlib (this can be tricky, see below)
pip install dlib
```

#### Option B: Install dlib from Wheel File (Windows - Python 3.11)

If you're on Windows with Python 3.11, you can use the provided wheel file:

```bash
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

#### Option C: Install dlib from Source (if pip fails)

**For Windows:**
1. Install Visual Studio Build Tools
2. Install CMake: `pip install cmake`
3. Install dlib: `pip install dlib`

**For Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
```

**For macOS:**
```bash
brew install cmake
pip install dlib
```

### Step 3: Verify Installation

```bash
python -c "import cv2, dlib, sklearn, pandas, numpy; print('All dependencies installed!')"
```

If no errors, you're ready to go!

## 🎮 Quick Start

### Running the Application

```bash
python main.py
```

The GUI window will open with three buttons:
1. **Capture Images** - Capture face images for training
2. **Train Images (Dlib)** - Train the face recognition model
3. **Track Attendance (Dlib)** - Start real-time attendance tracking

## 📖 How to Use

### Step 1: Add Employees

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Enter Employee Details:**
   - **Employee ID**: Must be a number (e.g., `1001`, `1002`)
   - **Employee Name**: Must contain only letters (e.g., `John`, `Jane`)

3. **Capture Images:**
   - Click **"Capture Images"** button
   - Position your face in front of the camera
   - The system will automatically capture up to 100 images
   - **Tips for better recognition:**
     - Ensure good lighting
     - Keep face clearly visible
     - Turn head slightly left/right/up/down during capture
     - Remove glasses/hats if possible
   - Press **'q'** to stop capturing early

4. **Repeat for all employees:**
   - Add each employee following the same process
   - Each employee needs multiple images (recommended: 50-100)

### Step 2: Train the Model

1. **After capturing images for all employees:**
   - Click **"Train Images (Dlib)"** button
   - Wait for training to complete (may take 30 seconds to 2 minutes depending on number of images)
   - You'll see a success message: "Training complete on X images for Y employees."

**⚠️ IMPORTANT:** 
- You **MUST** train the model after adding new employees
- If you add more employees later, you need to retrain the model
- Training uses all images in the `TrainingImage/` folder

### Step 3: Track Attendance

1. **Click "Track Attendance (Dlib)" button**
2. **Position yourself in front of the camera:**
   - 🟢 **Green box** = Recognized employee (known face)
   - 🔴 **Red box** = Unknown face (not in system)
3. **Press 'q'** to stop tracking
4. **Attendance is automatically saved** to `Attendance/` folder as CSV file with timestamp

## 📁 Project Structure

```
final cv project 200/
├── main.py                              # Main application (Dlib-based) - RUN THIS
├── train.py                             # Old LBPH version (not used)
├── dlib_logic/                          # Dlib face recognition module
│   ├── __init__.py                      # Package initialization
│   ├── train_dlib.py                    # Training logic
│   └── predict_dlib.py                  # Prediction/attendance logic
├── shape_predictor_68_face_landmarks.dat # Dlib face landmark detector (REQUIRED)
├── dlib_face_recognition_resnet_model_v1.dat # Dlib face recognition model (REQUIRED)
├── haarcascade_frontalface_alt2.xml     # OpenCV face detector (for image capture)
├── TrainingImage/                       # Stores captured face images
│   └── {Name}.{ID}.{number}.jpg        # Format: Name.ID.Number.jpg
├── DlibModel/                           # Stores trained model
│   ├── dlib_model.pkl                   # Trained SVM classifier
│   └── employee_ids.npy                 # Employee ID mappings
├── EmployeeDetails/                     # Employee database
│   └── EmployeeDetails.csv              # CSV with Id,Name columns
├── Attendance/                          # Attendance records
│   └── Attendance_YYYY-MM-DD_HH-MM-SS.csv # Timestamped attendance files
└── README.md                            # This file
```

## 📝 File Formats

### Training Images
- **Format:** `{Name}.{ID}.{number}.jpg`
- **Example:** `John.1001.1.jpg`, `John.1001.2.jpg`, `John.1001.3.jpg`
- **Location:** `TrainingImage/` folder
- **Count:** Up to 100 images per user (captured automatically)

### Employee Details CSV
- **Format:** `Id,Name`
- **Example:**
  ```csv
  Id,Name
  1001,John
  1002,Jane
  1003,Alice
  ```
- **Location:** `EmployeeDetails/EmployeeDetails.csv`
- **Auto-generated** when you capture images

### Attendance CSV
- **Format:** `Id,Name,Date,Time`
- **Example:**
  ```csv
  Id,Name,Date,Time
  1001,John,2024-01-15,10:30:45
  1002,Jane,2024-01-15,10:31:12
  ```
- **Location:** `Attendance/Attendance_YYYY-MM-DD_HH-MM-SS.csv`
- **Auto-generated** with timestamp when attendance is tracked

## 🔧 Technical Details

### Face Recognition Algorithm
- **Algorithm:** Dlib Face Recognition (ResNet-based deep learning) + SVM Classifier
- **Library:** `dlib` (face recognition model v1)
- **Classifier:** Support Vector Machine (SVM) with linear kernel
- **Confidence Threshold:** 0.6 (60% probability) - higher = better match
- **Training Images:** 50-100 per user recommended
- **Image Format:** BGR (color) JPG
- **Face Embedding:** 128-dimensional vector per face

### Face Detection
- **Algorithm:** Dlib HOG-based face detector
- **Library:** `dlib.get_frontal_face_detector()`
- **For Image Capture:** OpenCV Haar Cascade (`haarcascade_frontalface_alt2.xml`)

### Key Features
- **Duplicate Prevention:** Cannot add same employee ID twice
- **Session-based Attendance:** Each person marked present only once per session
- **Automatic Timestamping:** Date and time automatically recorded
- **CSV Export:** Attendance records saved as CSV files
- **Error Handling:** Comprehensive error handling with user-friendly messages
- **Path Independence:** Works from any directory (absolute paths used)

## 🐛 Troubleshooting

### Problem: "No module named 'dlib'"
**Solution:** 
- Install dlib: `pip install dlib`
- On Windows, you may need to install Visual Studio Build Tools first
- Or use the provided wheel file: `pip install dlib-19.24.1-cp311-cp311-win_amd64.whl`

### Problem: "Shape predictor file not found"
**Solution:** 
- Ensure `shape_predictor_68_face_landmarks.dat` is in the project root directory
- Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Extract the .bz2 file to get the .dat file

### Problem: "Face recognition model file not found"
**Solution:** 
- Ensure `dlib_face_recognition_resnet_model_v1.dat` is in the project root directory
- Download from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
- Extract the .bz2 file to get the .dat file

### Problem: Camera doesn't open
**Solution:** 
- Check if webcam is connected and working
- Close other applications using the camera (Zoom, Teams, etc.)
- Try running as administrator (Windows)
- Check camera permissions (Linux/macOS)

### Problem: "No training images found!"
**Solution:** 
- Capture images first using "Capture Images" button
- Ensure images are in `TrainingImage/` folder
- Check that images follow the format: `Name.ID.Number.jpg`

### Problem: "No trained Dlib model found!"
**Solution:** 
- Click "Train Images (Dlib)" after capturing user images
- Ensure `DlibModel/dlib_model.pkl` exists after training

### Problem: Known users marked as "Unknown"
**Solution:**
- Retrain the model: Click "Train Images (Dlib)" again
- Ensure good lighting during capture and recognition
- Make sure face is clearly visible and not obstructed
- Capture more training images (50-100 recommended)
- Check that employee exists in `EmployeeDetails/EmployeeDetails.csv`

### Problem: "Employee with ID X already exists!"
**Solution:** 
- The system prevents duplicate IDs
- Use a different ID or delete the existing employee first
- Check `EmployeeDetails/EmployeeDetails.csv` for existing IDs

### Problem: Application window doesn't open
**Solution:**
- Check if Python is installed: `python --version`
- Check if tkinter is available: `python -c "import tkinter"`
- On Linux, install tkinter: `sudo apt-get install python3-tk`
- On macOS, tkinter should come with Python

### Problem: "Error loading dlib models"
**Solution:**
- Ensure model files are not corrupted
- Re-download the .dat files if needed
- Check file permissions

## ⚠️ Important Notes

1. **Always train the model after adding new users** - Otherwise they won't be recognized
2. **Capture 50-100 images per user** - More images = better recognition accuracy
3. **Good lighting is essential** - Poor lighting significantly affects recognition
4. **Face should be clearly visible** - No obstructions, good angle, face the camera
5. **Each user needs a unique ID** - Duplicate IDs are not allowed
6. **The system prevents duplicate attendance** - Each person marked present only once per session
7. **Unknown faces are NOT added to attendance** - Only recognized employees are recorded
8. **Model files are required** - Ensure both .dat files are in the project root
9. **Training takes time** - Be patient, especially with many images/employees

## ⌨️ Keyboard Shortcuts

- **`q` key:** Stop camera/exit capture or tracking mode

## 🎯 Button Functions

- **Capture Images:** Captures up to 100 face images for training
- **Train Images (Dlib):** Trains the Dlib + SVM face recognition model
- **Track Attendance (Dlib):** Starts real-time attendance tracking with camera

## 🔄 Differences from LBPH Version

This Dlib-based version (`main.py`) offers:
- ✅ **Higher accuracy** than LBPH
- ✅ **Better performance** with deep learning models
- ✅ **More robust** to lighting variations
- ✅ **Better handling** of face angles and expressions

The old LBPH version (`train.py`) is still included but not recommended for new installations.

## 📦 Dependencies Summary

```bash
# Core dependencies
opencv-python>=4.5.0
dlib>=19.24.1
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0

# Usually included with Python
tkinter
```

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `main.py` and `dlib_logic/` files
3. Open an issue on GitHub

---

**Happy Tracking! 🎉**

*Built with Dlib, OpenCV, and Python*
