# enroll_from_camera.py
# Usage: python enroll_from_camera.py <label>
# Example: python enroll_from_camera.py ansh
# Press SPACE to capture, ESC to cancel. After capture, model will be trained.

import os
import sys
from vision.face_auth import FaceAuthenticator

def main():
    if len(sys.argv) < 2:
        print("Usage: python enroll_from_camera.py <label>")
        sys.exit(1)

    label = sys.argv[1].strip()
    project_root = os.getcwd()
    fa = FaceAuthenticator(known_faces_dir=os.path.join(project_root, "known_faces"))

    print("Camera will open. Press SPACE to capture an image for label:", label)
    ok = fa.enroll_from_camera(label, camera_index=0, show_preview=True, save_crop=True)
    if not ok:
        print("Capture cancelled or camera error.")
        sys.exit(2)
    print("Captured image for label:", label)

    trained, msg = fa.train()
    print("Train result:", trained, msg)
    if trained:
        print("Model saved:", fa.model_path)
        print("Labels saved:", fa.labels_path)
    else:
        print("Training did not run or failed:", msg)

if __name__ == "__main__":
    main()
