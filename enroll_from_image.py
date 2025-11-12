# enroll_from_image.py
# Usage: python enroll_from_image.py <label> <path/to/image.jpg>
# Example: python enroll_from_image.py ansh ansh.jpg

import os
import sys
from vision.face_auth import FaceAuthenticator

def main():
    if len(sys.argv) < 3:
        print("Usage: python enroll_from_image.py <label> <image_path>")
        sys.exit(1)

    label = sys.argv[1].strip()
    image_path = sys.argv[2].strip()
    project_root = os.getcwd()
    print("Project root:", project_root)
    print("Label:", label)
    print("Image path:", image_path)

    fa = FaceAuthenticator(known_faces_dir=os.path.join(project_root, "known_faces"))

    ok = fa.enroll_image(label, image_path)
    if not ok:
        print("Failed to enroll image. Check path and permissions.")
        sys.exit(2)
    print("Image enrolled successfully into known_faces/%s/" % label)

    trained, msg = fa.train()
    print("Train result:", trained, msg)
    if trained:
        print("Model saved:", fa.model_path)
        print("Labels saved:", fa.labels_path)
    else:
        print("Training did not run or failed:", msg)

if __name__ == "__main__":
    main()
