import shutil
import os
import difflib


src = "../experiments/audio-visual/"
dst = "../results/"
d = difflib.Differ()

i = 0
for root, dirs, files in os.walk(src):
    if "checkpoints" in root or "models" in root or "testing" in root:
        continue
    i += 1
    for f in files:
        new_loc = os.path.join(dst, root.split(src, 1)[-1])
        os.makedirs(new_loc, exist_ok=True)
        shutil.copy(os.path.join(root, f), os.path.join(new_loc, f))
