"""
Face Recognition GUI with persistent profiles

Features:
- Create profiles: give a name, capture images from webcam (saves to profiles/<id>/)
- Train recognizer: trains OpenCV LBPHFaceRecognizer on saved images and saves model to recognizer.yml
- Live recognition: opens webcam and shows recognized profile name (or Unknown)
- Manage profiles: list and delete profiles
- Profiles metadata stored in a local SQLite database (profiles.db)

Dependencies:
- Python 3.8+
- opencv-contrib-python
- numpy
- Pillow

Install:
    pip install opencv-contrib-python numpy pillow

Run:
    python face_recognition_gui.py

Notes:
- Requires a working webcam. If your webcam index isn't 0, change CAMERA_INDEX in the code.
- LBPH works reasonably well for small-scale systems. For production or better accuracy, consider deep-learning based embeddings (face_recognition library).
"""

import os
import cv2
import sqlite3
import threading
import shutil
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, 'profiles')
DB_PATH = os.path.join(BASE_DIR, 'profiles.db')
MODEL_PATH = os.path.join(BASE_DIR, 'recognizer.yml')
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CAMERA_INDEX = 0  # change if your webcam is on a different index

os.makedirs(PROFILES_DIR, exist_ok=True)

# -------------------- Database helpers --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


def add_profile_to_db(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO profiles (name) VALUES (?)', (name,))
    conn.commit()
    pid = c.lastrowid
    conn.close()
    return pid


def get_profiles():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name FROM profiles ORDER BY id')
    rows = c.fetchall()
    conn.close()
    return rows


def delete_profile_from_db(pid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM profiles WHERE id = ?', (pid,))
    conn.commit()
    conn.close()

# -------------------- Face data helpers --------------------
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def capture_images_for_profile(pid, samples=30):
    profile_path = os.path.join(PROFILES_DIR, str(pid))
    os.makedirs(profile_path, exist_ok=True)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        raise RuntimeError('Could not open webcam')

    count = 0
    while count < samples:
        ret, frame = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(profile_path, f'{count:03d}.jpg')
            cv2.imwrite(file_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Captured {count}/{samples}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            break

        cv2.imshow('Capturing faces (press q to abort)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def prepare_training_data():
    labels = []
    faces = []
    profiles = get_profiles()
    for pid, _ in profiles:
        folder = os.path.join(PROFILES_DIR, str(pid))
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith('.jpg'):
                continue
            path = os.path.join(folder, fname)
            img = Image.open(path).convert('L')
            arr = np.array(img, dtype=np.uint8)
            faces.append(arr)
            labels.append(pid)
    return faces, labels


def train_recognizer():
    faces, labels = prepare_training_data()
    if len(faces) == 0:
        raise RuntimeError('No training data found. Create profiles first.')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)


def load_recognizer():
    if not os.path.exists(MODEL_PATH):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer

# -------------------- GUI --------------------
class FaceApp:
    def __init__(self, root):
        self.root = root
        root.title('Face Recognition - Profiles')
        root.geometry('700x420')

        self.recognizer = load_recognizer()

        # Left frame: controls
        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        btn_new = tk.Button(left, text='Create profile', width=20, command=self.create_profile)
        btn_new.pack(pady=4)

        btn_train = tk.Button(left, text='Train recognizer', width=20, command=self.train_model)
        btn_train.pack(pady=4)

        btn_live = tk.Button(left, text='Start live recognition', width=20, command=self.start_recognition)
        btn_live.pack(pady=4)

        btn_list = tk.Button(left, text='List profiles', width=20, command=self.list_profiles)
        btn_list.pack(pady=4)

        btn_delete = tk.Button(left, text='Delete profile', width=20, command=self.delete_profile)
        btn_delete.pack(pady=4)

        btn_export = tk.Button(left, text='Export profile images', width=20, command=self.export_profile)
        btn_export.pack(pady=4)

        # Right frame: preview
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.preview_label = tk.Label(right, text='No camera active', compound=tk.TOP)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        self.preview_image = None
        self.running = False

    def create_profile(self):
        name = simpledialog.askstring('Profile name', 'Enter a name for this profile:')
        if not name:
            return
        pid = add_profile_to_db(name)
        messagebox.showinfo('Capture', f'Profile created with id {pid}. Now capturing images from webcam.')
        try:
            capture_images_for_profile(pid, samples=30)
            messagebox.showinfo('Done', 'Captured images. You should train the recognizer now.')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def train_model(self):
        try:
            train_recognizer()
            self.recognizer = load_recognizer()
            messagebox.showinfo('Trained', 'Recognizer trained and saved.')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def start_recognition(self):
        if self.running:
            self.running = False
            return
        if self.recognizer is None:
            messagebox.showwarning('No model', 'No trained model found. Train first.')
            return
        self.running = True
        t = threading.Thread(target=self._recognition_loop, daemon=True)
        t.start()

    def _recognition_loop(self):
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened():
            messagebox.showerror('Camera error', 'Cannot open webcam')
            self.running = False
            return

        profiles = dict(get_profiles())

        while self.running:
            ret, frame = cam.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            display_name = 'No face'
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                label, conf = self.recognizer.predict(face_img)
                # LBPH gives lower confidence for better matches but scale varies
                name = profiles.get(label, 'Unknown')
                display_name = f'{name} ({label}) conf={conf:.0f}'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                break

            # Convert to PhotoImage for Tkinter display
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((480, 360))
            imgtk = ImageTk.PhotoImage(img)
            self.preview_image = imgtk
            self.preview_label.configure(image=imgtk, text=display_name, compound=tk.TOP)
            self.preview_label.update()

        cam.release()
        self.preview_label.configure(image='', text='No camera active')

    def list_profiles(self):
        rows = get_profiles()
        if not rows:
            messagebox.showinfo('Profiles', 'No profiles found.')
            return
        txt = '\n'.join([f'{r[0]}: {r[1]}' for r in rows])
        messagebox.showinfo('Profiles', txt)

    def delete_profile(self):
        rows = get_profiles()
        if not rows:
            messagebox.showinfo('Delete', 'No profiles to delete.')
            return
        choices = '\n'.join([f'{r[0]}: {r[1]}' for r in rows])
        pid = simpledialog.askinteger('Delete profile', 'Enter profile id to delete:\n' + choices)
        if not pid:
            return
        delete_profile_from_db(pid)
        folder = os.path.join(PROFILES_DIR, str(pid))
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        # Re-train automatically if model exists
        if os.path.exists(MODEL_PATH):
            try:
                train_recognizer()
                self.recognizer = load_recognizer()
            except Exception:
                pass
        messagebox.showinfo('Deleted', f'Profile {pid} deleted.')

    def export_profile(self):
        rows = get_profiles()
        if not rows:
            messagebox.showinfo('Export', 'No profiles to export.')
            return
        choices = '\n'.join([f'{r[0]}: {r[1]}' for r in rows])
        pid = simpledialog.askinteger('Export profile', 'Enter profile id to export:\n' + choices)
        if not pid:
            return
        folder = os.path.join(PROFILES_DIR, str(pid))
        if not os.path.isdir(folder):
            messagebox.showerror('Error', 'No images for that profile')
            return
        dest = filedialog.askdirectory(title='Select export directory')
        if not dest:
            return
        target = os.path.join(dest, f'profile_{pid}')
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(folder, target)
        messagebox.showinfo('Exported', f'Exported to {target}')

# -------------------- Main --------------------

def main():
    init_db()
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
