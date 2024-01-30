import random
import threading
import time
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import sys
import cv2
import numpy as np
import os

###############################################################################
#
# Utils
#
###############################################################################


def get_input_filenames_without_extension(path):
    """
    Returns a list of the filenames for all the inputs, without the extension.
    In other words, if the directory has the following files:
    ```
        agri_0_0.jpeg
        agri_0_0.txt
        agri_0_1.jpeg
        agri_0_0.txt
    ```
    The output will be:
    ```
        ['agri_0_0', 'agri_0_1']
    ```
    """

    f = []
    for (_, _, filenames) in os.walk(path):
        f.extend([f.split(".")[0] for f in filenames])
        break
    return list(set(f))


def draw_rect_and_label(img, xywh, class_id, transparency, thickness, confidence : float | None = None):
    x, y, w, h = xywh * 512
    x -= w/2
    y -= h/2

    color = (0, 255, 0)
    if class_id == 1:
        color = (255, 0, 0)
    p1 = (int(x), int(y))
    p2 = (int(x + w), int(y + h))

    overlay = img.copy()

    # Rectangle + outline
    image_new = cv2.rectangle(overlay, p1, p2, (0, 0, 0), thickness + 4)
    cv2.rectangle(image_new, p1, p2, color, thickness)
    image_new = cv2.addWeighted(image_new, transparency, img, 1 - transparency, 0)
    
    return image_new


TEST_PATH = "datasets/mydataset/test/"
fnames = get_input_filenames_without_extension(TEST_PATH)
model = None


def load_model():
    print("Loading AI model...")

    global model
    from ultralytics import YOLO
    model = YOLO('runs/detect/train12/weights/best.pt')

    print("Finished...")


def get_ground_truth(fname_truth):
    truth = []

    with open(TEST_PATH + '/' + fname_truth) as f:
        lines = f.readlines()
        for l in lines:
            items = l.split()
            truth.append({
                'class': int(items[0]),
                'xywh': np.array([float(x) for x in items[1:]])
            })
    return truth


def get_eval_result(fname_img):
    if model is None:
        load_model()

    got = []
    res = model(TEST_PATH +  fname_img)
    for r in res:
        for b in r.boxes:
            got.append({
                'class': int(b.cls[0]),
                'xywh': np.array([float(f) for f in (b.xywh / 512)[0]]),
                'confidence': float(b.conf[0])
            })
    return got


###############################################################################
 
 
class Window(QWidget):
    def __init__(self):
        super().__init__()
 
        self.acceptDrops()
        self.setWindowTitle("iSeeWeeds GUI")
        W = 512 * 2 + 32
        H = 512 + 32
        self.setGeometry(int((1920 - W) / 2), int((1080 - H) / 2), W, H)

        self.label1 = QLabel()
        self.label2 = QLabel()

        l = QVBoxLayout()
        h = QHBoxLayout()


        self.btn_random_image = QPushButton("Random image")
        self.btn_random_image.clicked.connect(self.on_click_btn_random_image)

        l.addWidget(self.btn_random_image)
        h.addWidget(self.label1)
        h.addWidget(self.label2)
        l.addItem(h)

        self.setLayout(l)

        self.show()


    def eval_img(self, fname: str):
        fname_img = f"{fname}.jpeg"
        fname_truth = f"{fname}.txt"

        self.label1.setPixmap(QPixmap(TEST_PATH + fname_img))

        def evaulate_and_draw_rects():
            got = get_eval_result(fname_img)
            truth = get_ground_truth(fname_truth)

            img = cv2.imread(TEST_PATH +  fname_img)
            for t in truth:
                img = draw_rect_and_label(img, t['xywh'], t['class'], 0.2, -1)
            for g in got:
                img = draw_rect_and_label(img, g['xywh'], g['class'], 0.75, 2)

            # Display second image
                     
            img2 = PyQt5.QtGui.QImage(img.data, img.shape[1], img.shape[0], PyQt5.QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(PyQt5.QtGui.QPixmap.fromImage(img2))

        t = threading.Thread(target=evaulate_and_draw_rects)
        t.start()


    def on_click_btn_random_image(self):
        random.seed(time.time())
        fname = random.choice(fnames)
        self.eval_img(fname)
 

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())