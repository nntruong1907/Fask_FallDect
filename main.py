from flask import Flask, redirect, url_for, render_template, request, flash, Response
from flask_sqlalchemy import SQLAlchemy

import cv2
from datetime import datetime
import time
import tensorflow as tf
# import threading
import numpy as np

import os
# from werkzeug.utils import secure_filename
from data import BodyPart
from movenet import Movenet
import utils_pose as utils
from def_lib import detect, get_keypoint_landmarks, landmarks_to_embedding, draw_prediction_on_image, predict_pose, draw_class_on_image

db = SQLAlchemy()

app = Flask(__name__)

movenet = Movenet('movenet_thunder')
model = tf.keras.models.load_model("./models/model_fall.h5")
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'videos')
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
app.config['SECRET_KEY'] = "random string"

db.init_app(app)


class Videos(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(1000))
    time = db.Column(db.String(1000))

    def __init__(self, location, time):
        self.location = location
        self.time = time


def write_video(file_path, frames, fps):
    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def gen_video(path):
    # model = tf.keras.models.load_model("./models/model_fall.h5")
    cap = cv2.VideoCapture(path)
    time_step = 5
    label = "waiting"
    i = 0
    lm = []
    list = []

    while cap.isOpened():
        ret, frame = cap.read()
        # Reshape Image
        if ret == True:
            img = frame.copy()
            img = cv2.resize(img, (854, 480))
#             img = cv2.resize(img, (640, 360))
            img = tf.convert_to_tensor(img, dtype=tf.uint8)
            i = i + 1
        #     print(img)

            print(f"Start detect: frame {i}")
            person = detect(img)
            landmarks = get_keypoint_landmarks(person)
            lm_pose = landmarks_to_embedding(tf.reshape(
                tf.convert_to_tensor(landmarks), (1, 51)))
    #         print(lm_pose)
            lm.append(lm_pose)
            img = np.array(img)
            img = draw_prediction_on_image(img, person, crop_region=None,
                                           close_figure=False, keep_input_size=True)
            if (len(lm) == time_step):
                lm = tf.reshape(lm, (1, 34, 5))

                label = predict_pose(model, lm, label)
                lm = []

            img = np.array(img)
            img = draw_class_on_image(label, img)
            list.append(img)

        else:
            break

    cap.release()
    write_video(path, np.array(list), 24)


def stream_video(path):
    cam = cv2.VideoCapture(path)
    while True:
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            time.sleep(0.01)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def stream_camera1():
    model = tf.keras.models.load_model("./models/model_fall.h5")
    # cap = cv2.VideoCapture('https://192.168.239.166:8080/video')
    cap = cv2.VideoCapture(0)
    lm = []
    time_step = 5
    label = "waiting"
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img = frame.copy()
            img = cv2.resize(img, (854, 480))
            img = tf.convert_to_tensor(img, dtype=tf.uint8)
            i = i + 1

            print(f"Start detect: frame {i}")
            person = detect(img)
            landmarks = get_keypoint_landmarks(person)
            lm_pose = landmarks_to_embedding(tf.reshape(
                tf.convert_to_tensor(landmarks), (1, 51)))
    #         print(lm_pose)
            lm.append(lm_pose)
            img = np.array(img)
            img = draw_prediction_on_image(img, person, crop_region=None,
                                           close_figure=False, keep_input_size=True)
            if (len(lm) == time_step):
                lm = tf.reshape(lm, (1, 34, 5))

                label = predict_pose(model, lm, label)

            img = np.array(img)
            img = draw_class_on_image(label, img)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def stream_camera():
    cam = cv2.VideoCapture(0)
    while True:
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload():
    global Tmp_path
    if request.method == 'POST':
        if not request.files['file']:
            flash('Please choose your video', 'error')
            return redirect(url_for('upload'))
        else:
            # flash('Uploading... please wait a moment')
            file = request.files['file']
            # filename = secure_filename(file.filename)
            filename = file.filename

            f_name, _ = filename.split(".")
            t = datetime.now()
            t = t.strftime("%H%M%S")
            file_name = t + f_name + ".mp4"
            print(file_name)
            path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            # path = os.path.join('./static/videos/', file_name)
            # Tmp_path = os.path.join('./videos/', file_name)

            print(path)

            file.save(path)
            # flash('Video successfully uploaded')
            Tmp_path = path
            gen_video(path)
            now = datetime.now()
            time = now.strftime("%d/%m/%Y %H:%M:%S")
            video = Videos(file_name, time)

            db.session.add(video)
            db.session.commit()

            return redirect(url_for('play'))

    else:
        return render_template('index.html')


@app.route('/play', methods=["GET", 'POST'])
def play():
    global Tmp_path
    if request.method == 'POST':
        video_id = request.form.get('video_id')
        video = Videos.query.filter_by(id=video_id).first()
        Tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], video.location)

        return render_template('show_result.html')

    return render_template('show_result.html')


@app.route('/stream')
def stream():
    global Tmp_path
    print(Tmp_path)
    return Response(stream_video(Tmp_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera')
def camera():
    return render_template('camera.html', listVideo=video)


@app.route('/stream_cam')
def stream_cam():
    return Response(stream_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/videos')
def video():
    global Tmp_path
    video = Videos.query.all()
    return render_template('video.html', listVideo=video)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
