# vision/face_auth.py
"""
LBPH-based FaceAuthenticator using OpenCV only.

Features:
- enroll_image(name, image_path): add face image file for label `name`
- enroll_from_camera(name, camera_index=0, show_preview=True): capture one image via webcam
- train(save_path="known_faces/lbph_model.yml"): trains the LBPH model from known_faces/<label>/*.jpg
- authenticate(camera_index=0, timeout=8.0, required_matches=2, threshold=60): attempts live recognition
- stores labels mapping in known_faces/labels.json and model in known_faces/lbph_model.yml
- purely OpenCV & numpy (pip install opencv-python numpy)
"""

import os
import time
import json
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

KNOWN_DIR_DEFAULT = "known_faces"
LABELS_FILE = "labels.json"
MODEL_FILE = "lbph_model.yml"

# helper: ensure folder exists
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class FaceAuthenticator:
    def __init__(self, known_faces_dir: str = KNOWN_DIR_DEFAULT):
        self.known_faces_dir = known_faces_dir
        _ensure_dir(self.known_faces_dir)
        self.labels_path = os.path.join(self.known_faces_dir, LABELS_FILE)
        self.model_path = os.path.join(self.known_faces_dir, MODEL_FILE)
        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels: Dict[str, int] = {}
        self.rev_labels: Dict[int, str] = {}
        # if model exists, try loading it
        self._load_labels()
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
            except Exception:
                # model invalid -> ignore
                pass

    # ---------- Enrollment helpers ----------
    def enroll_image(self, name: str, image_path: str) -> bool:
        """
        Copy a file into known_faces/<name>/ and then call train().
        Returns True on copy success (train not performed here).
        """
        if not os.path.exists(image_path):
            return False
        person_dir = os.path.join(self.known_faces_dir, name)
        _ensure_dir(person_dir)
        # choose a filename (timestamp)
        dst = os.path.join(person_dir, f"{int(time.time()*1000)}.jpg")
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                # fallback to simple copy if imdecode fails
                import shutil
                shutil.copy(image_path, dst)
            else:
                cv2.imwrite(dst, img)
            return True
        except Exception:
            try:
                import shutil
                shutil.copy(image_path, dst)
                return True
            except Exception:
                return False

    def enroll_from_camera(self, name: str, camera_index: int = 0, show_preview: bool = True, save_crop: bool = True) -> bool:
        """
        Opens the camera and lets user press SPACE to capture an image for `name`.
        Returns True if a file was saved.
        """
        person_dir = os.path.join(self.known_faces_dir, name)
        _ensure_dir(person_dir)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False

        saved = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                display = frame.copy()
                if show_preview:
                    cv2.putText(display, "Press SPACE to capture, ESC to cancel", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Enroll - press SPACE", display)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    break
                if key == 32:  # SPACE
                    # save full frame or try to detect face and crop
                    fname = os.path.join(person_dir, f"{int(time.time()*1000)}.jpg")
                    if save_crop:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self._detect_faces(gray)
                        if faces:
                            # take first face bounding box
                            x, y, w, h = faces[0]
                            crop = frame[y:y+h, x:x+w]
                            cv2.imwrite(fname, crop)
                        else:
                            cv2.imwrite(fname, frame)
                    else:
                        cv2.imwrite(fname, frame)
                    saved = True
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return saved

    # ---------- Training ----------
    def train(self) -> Tuple[bool, str]:
        """
        Trains LBPH from images in known_faces/<label>/*.jpg
        Saves model and labels file. Returns (success, message).
        """
        labels = {}
        faces = []
        ids = []
        next_id = 0

        # iterate people directories
        for name in sorted(os.listdir(self.known_faces_dir)):
            person_dir = os.path.join(self.known_faces_dir, name)
            if not os.path.isdir(person_dir):
                continue
            # assign id
            if name in labels:
                pid = labels[name]
            else:
                pid = next_id
                labels[name] = pid
                next_id += 1
            # list images
            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                path = os.path.join(person_dir, fname)
                try:
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    # detect face and crop if possible
                    faces_bb = self._detect_faces(img)
                    if faces_bb:
                        x, y, w, h = faces_bb[0]
                        face_img = img[y:y+h, x:x+w]
                    else:
                        # resize to reasonable size
                        face_img = cv2.resize(img, (200, 200))
                    # LBPH needs single-channel (grayscale)
                    face_resized = cv2.resize(face_img, (200, 200))
                    faces.append(face_resized)
                    ids.append(pid)
                except Exception:
                    continue

        if not faces or not ids:
            return False, "No training images found. Put images in known_faces/<label>/*.jpg and try again."

        try:
            self.recognizer.train(faces, np.array(ids))
            # save model
            self.recognizer.write(self.model_path)
            # save labels
            with open(self.labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2)
            # reload mapping
            self._load_labels()
            return True, "Trained successfully."
        except Exception as e:
            return False, f"Training failed: {e}"

    # ---------- Authenticate ----------
    def authenticate(self, camera_index: int = 0, timeout: float = 8.0, required_matches: int = 2, threshold: int = 70) -> Optional[Tuple[str, float]]:
        """
        Live recognition loop:
        - threshold: lower = stricter (distance/score). LBPH gives 'confidence' where lower is better.
        - threshold default ~70 (tweak for your environment)
        - required_matches: number of frames with match to accept.
        Returns (label, confidence) on success or None.
        """
        if not self.labels:
            return None
        # open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None

        match_counts: Dict[str, int] = {name: 0 for name in self.labels.keys()}
        start = time.time()
        try:
            while time.time() - start < timeout:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detect_faces(gray)
                if not faces:
                    time.sleep(0.02)
                    continue
                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_img, (200, 200))
                    try:
                        label_id, confidence = self.recognizer.predict(face_resized)
                    except Exception:
                        continue
                    name = self.rev_labels.get(int(label_id))
                    if name is None:
                        continue
                    # LBPH: lower confidence value means better match (0 = perfect). We'll accept if confidence <= threshold
                    if confidence <= threshold:
                        match_counts[name] += 1
                        if match_counts[name] >= required_matches:
                            return name, float(confidence)
                time.sleep(0.02)
        finally:
            cap.release()
        return None

    # ---------- Utilities ----------
    def _load_labels(self):
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                    # ensure keys are str and values int
                    self.labels = {str(k): int(v) for k, v in self.labels.items()}
                    self.rev_labels = {int(v): k for k, v in self.labels.items()}
            except Exception:
                self.labels = {}
                self.rev_labels = {}
        else:
            self.labels = {}
            self.rev_labels = {}

    def _detect_faces(self, gray_image) -> List[Tuple[int, int, int, int]]:
        """
        Simple cascade face detector. Returns list of x,y,w,h bounding boxes.
        """
        # load the default haarcascade classifier (bundled with OpenCV)
        try:
            # try a local copy first (if shipped)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            # convert to list-of-tuples
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        except Exception:
            return []

    def add_label_and_train(self, name: str, image_path: Optional[str] = None, from_camera: bool = False, camera_index: int = 0) -> Tuple[bool, str]:
        """
        Convenience: enroll (image or camera) then train.
        """
        if from_camera:
            ok = self.enroll_from_camera(name, camera_index=camera_index, show_preview=True)
            if not ok:
                return False, "Could not capture from camera."
        elif image_path:
            ok = self.enroll_image(name, image_path)
            if not ok:
                return False, "Could not copy image."
        else:
            return False, "No image provided."

        # after enroll, train
        return self.train()

    def list_known(self) -> List[str]:
        """
        Returns list of known label names (folder names under known_faces)
        """
        names = [d for d in sorted(os.listdir(self.known_faces_dir)) if os.path.isdir(os.path.join(self.known_faces_dir, d))]
        return names
