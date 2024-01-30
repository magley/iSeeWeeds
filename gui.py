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
        color = (0, 0, 255)
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


def get_ground_truth(fname_truth):
    try:
        truth = []
        with open(fname_truth) as f:
            lines = f.readlines()
            for l in lines:
                items = l.split()
                truth.append({
                    'class': int(items[0]),
                    'xywh': np.array([float(x) for x in items[1:]])
                })
        return truth
    except FileNotFoundError:
        print(f"No truth file found {fname_truth}")
        return []


def get_eval_result(fname_img):
    got = []
    res = model(fname_img)
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
        self.setWindowTitle("iSeeWeeds")
        W = 512 * 2 + 32
        H = 512 + 32
        self.setGeometry(int((1920 - W) / 2), int((1080 - H) / 2), W, H)

        self.label1 = QLabel()
        self.label2 = QLabel()

        l = QVBoxLayout()
        h = QHBoxLayout()
        h2 = QHBoxLayout()
        h3 = QHBoxLayout()

        self.btn_random_image = QPushButton("Random image")
        self.btn_random_image.clicked.connect(self.on_click_btn_random_image)

        self.btn_load_image = QPushButton("Open image...")
        self.btn_load_image.clicked.connect(self.on_click_btn_load_image)

        self.cb_show_got = QCheckBox("Show result")
        self.cb_show_got.setChecked(True)
        self.cb_show_got.toggled.connect(self.on_toggle_cb_show_got)

        self.cb_show_truth = QCheckBox("Show truth")
        self.cb_show_truth.setChecked(True)
        self.cb_show_truth.toggled.connect(self.on_toggle_cb_show_truth)

        self.btn_history_back = QPushButton()
        self.btn_history_back.setToolTip("Evaluate previous image")
        self.btn_history_back.clicked.connect(self.history_back)
        self.btn_history_back.clicked.connect(self.set_history_buttons_enabled)
        self.btn_history_back.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))

        self.btn_history_forward = QPushButton()
        self.btn_history_forward.setToolTip("Evaluate next image")
        self.btn_history_forward.clicked.connect(self.history_forward)
        self.btn_history_forward.clicked.connect(self.set_history_buttons_enabled)
        self.btn_history_forward.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))

        self.lbl_log = QLabel()
   
        h2.addWidget(self.btn_history_back, stretch=1)
        h2.addWidget(self.btn_history_forward, stretch=1)
        h2.addWidget(self.btn_random_image, stretch=50)
        h2.addWidget(self.btn_load_image, stretch=20)

        h3.addWidget(self.cb_show_got, stretch=1)
        h3.addWidget(self.cb_show_truth, stretch=3)
        h3.addWidget(self.lbl_log, stretch=20)
        h3.addStretch()

        h.addWidget(self.label1)
        h.addWidget(self.label2)

        l.addItem(h2)
        l.addItem(h3)
        l.addItem(h)
        l.addStretch()

        self.setLayout(l)

        self.show()

        self.current_img = {}
        self.history = [] # Fname history
        self.history_i = 0
        self.set_history_buttons_enabled()

        self.setWindowIcon(PyQt5.QtGui.QIcon('ico_iseeweeds.png'))

    def set_log(self, log = ""):
        self.lbl_log.setText(log)

    def set_history_buttons_enabled(self):
        self.btn_history_back.setEnabled(self.history_i > 0)
        self.btn_history_forward.setEnabled(self.history_i < len(self.history) - 1)

    def history_back(self):
        self.history_i -= 1
        fname = self.history[self.history_i]
        self.eval_img(fname, False, True)

    def history_forward(self):
        self.history_i += 1
        fname = self.history[self.history_i]
        self.eval_img(fname, False, True)

    def on_toggle_cb_show_got(self):     
        self.draw_bboxes()

    def on_toggle_cb_show_truth(self):      
        self.draw_bboxes()       

    def load_ai_model(self):
        self.set_log("Loading AI model...")
        global model
        from ultralytics import YOLO
        model = YOLO('iSeeWeeds.pt')
        self.set_log("")

    def draw_bboxes(self):
        draw_got = self.cb_show_got.isChecked()
        draw_truth = self.cb_show_truth.isChecked()

        if 'fname' not in self.current_img or 'truth' not in self.current_img or 'got' not in self.current_img:
            return
        
        img = cv2.imread(self.current_img['fname'], cv2.COLOR_RGB2BGR)

        if draw_truth:
            for t in self.current_img['truth']:
                img = draw_rect_and_label(img, t['xywh'], t['class'], 0.2, -1)
        if draw_got:
            for g in self.current_img['got']:
                img = draw_rect_and_label(img, g['xywh'], g['class'], 0.75, 2)
        
        img2 = PyQt5.QtGui.QImage(img.data, img.shape[1], img.shape[0], PyQt5.QtGui.QImage.Format_BGR888)
        self.label2.setPixmap(PyQt5.QtGui.QPixmap.fromImage(img2))

    def eval_img(self, fname: str, new_one = True, full_image_path = False):
        fname_img = ""
        fpath_img = ""
        fname_truth = ""
        fpath_truth = ""
        if not full_image_path:
            fname_img = f"{fname}.jpeg"
            fname_truth = f"{fname}.txt"
            fpath_img = TEST_PATH + fname_img
            fpath_truth = TEST_PATH + '/' + fname_truth
        else:
            fname_img = fname
            fpath_img = fname_img
            fname_truth = fname.split("/")[-1].split(".")[0]
            fpath_truth = "/".join(fname.split("/")[:-1]) + "/" + fname_truth + ".txt"

        if new_one:
            self.history.append(fpath_img)
            self.history_i = len(self.history) - 1

        self.label1.setPixmap(QPixmap(fpath_img))

        self.current_img = {
            'fname': fpath_img
        }

        def evaulate_and_draw_rects():
            if model is None:
                self.load_ai_model()

            self.set_log(f"Evaluating {fname_img}...")
            got = get_eval_result(fpath_img)
            truth = get_ground_truth(fpath_truth)

            self.current_img['got'] = got
            self.current_img['truth'] = truth

            self.set_log("")

            self.draw_bboxes()
            self.set_history_buttons_enabled()

        t = threading.Thread(target=evaulate_and_draw_rects)
        t.start()

    def on_click_btn_random_image(self):
        random.seed(time.time())
        fname = random.choice(fnames)
        self.eval_img(TEST_PATH + fname + ".jpeg", full_image_path = True)

    def on_click_btn_load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Open an image", os.getcwd(), "All Files (*);;JPEG (*.jpeg)", options=options)
        if fileName:
            self.eval_img(fileName, full_image_path = True)
 

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())