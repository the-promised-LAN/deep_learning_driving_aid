import os
import sys
import cv2
import time
import imutils
import numpy as np
import mrcnn.model as modellib
from mrcnn import utils, visualize
from imutils.video import WebcamVideoStream
import random
from flask import Flask, render_template, Response
import pdb
import lane_detect

app = Flask(__name__)

OPTIMIZE_CAM = False
SHOW_FPS = False
SHOW_FPS_WO_COUNTER = True # faster
PROCESS_IMG = True

# Root directory of the project
from samples.coco.coco import CocoConfig

ROOT_DIR = os.path.abspath("./")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Camera():
    def __init__(self):

        config = InferenceConfig()
        config.display()

        # For frame skipping
        self.count = 1
        self.trafficCnt = 0

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        self.colors = visualize.random_colors(len(self.class_names))

        self.gentle_grey = (45, 65, 79)
        self.white = (255, 255, 255)

        if OPTIMIZE_CAM:
            #vs = WebcamVideoStream(src=0).start()
            self.vs = cv2.VideoCapture("drive15_trim.mp4") #"test4_t1_480p_15fps.mp4"
        else:
            #vs = cv2.VideoCapture(0)
            self.vs = cv2.VideoCapture("drive15_trim.mp4")

        if SHOW_FPS:
            self.fps_caption = "FPS: 0"
            self.fps_counter = 0
            self.start_time = time.time()

        SCREEN_NAME = 'Mask RCNN LIVE'
        #cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty(SCREEN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def get_frame(self): 
        stop_sign = 0
        person = 0
        traffic_light = 0
        parking = 0
        lane_drift = 0

        # Capture frame-by-frame
        if OPTIMIZE_CAM:
            frame = self.vs.read()
        else:
             grabbed, frame = self.vs.read()
             if not grabbed:
                print("Failed to get video source")
                return
        
        #cv2.imshow(SCREEN_NAME, frame)
        if SHOW_FPS_WO_COUNTER:
            self.start_time = time.time() # start time of the loop

        
        # FRAME SKIPPING
        if self.count % 3 == 0:
            self.count = 1

            if PROCESS_IMG:
                #interest_area = frame[0:360,200:500,:].copy()       # cropping to only detect in RoI (y:y+h, x:x+w) 
                # converting to grayscale for performance   
                #interest_area_gray = cv2.cvtColor(cv2.cvtColor(interest_area, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

                gFrame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                results = self.model.detect([gFrame], verbose=1)
                r = results[0]

                # Run detection
                masked_image, stop_sign, traffic_light, parking, person = visualize.display_instances_10fps(gFrame, r['rois'], r['masks'], 
                    r['class_ids'], self.class_names, r['scores'], colors=self.colors, real_time=True)
                
                #frame[0:360,200:500,:] = masked_image.copy()    # overlaying RoI detections onto original frame
                
            #if PROCESS_IMG:
            #    s = masked_image
            #else:
            s = masked_image
            # print("Image shape: {1}x{0}".format(s.shape[0], s.shape[1]))


            # START OF LANE DETECT
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gaus = cv2.GaussianBlur(gray_frame, (9,9), 10.0)
            unsharp = cv2.addWeighted(gray_frame, 1.5, gaus, -0.5, 0, gaus)

            canny_image = lane_detect.canny_gray(unsharp)
            cropped_image = lane_detect.region_of_interest(canny_image)

            try:
                lines = cv2.HoughLinesP(cropped_image, 5, np.pi/180, 100, np.array([]), minLineLength=60, maxLineGap=50)

                average_lines = lane_detect.average_slope_intercept(frame, lines)

                line_image = lane_detect.display_lines(frame, average_lines)

                s = cv2.addWeighted(s, 0.8, line_image, 1, 1)

                # 480p input
                s = cv2.circle(s, (320, 473), 10, color=(255,255,255), thickness=-1)

                x1, y1, x2, y2 = average_lines[0]
                if 320 - x1 < 100:
                    s = cv2.line(s, (320, 473), (x1, 473), color=(0,0,255), thickness=2)
                else:
                    s = cv2.line(s, (320, 473), (x1, 473), color=(0,255,255), thickness=2)

                x1, y1, x2, y2 = average_lines[1]
                if x1 - 320 < 100:
                    s = cv2.line(s, (320, 473), (x1, 473), color=(0,0,255), thickness=2)
                else:
                    s = cv2.line(s, (320, 473), (x1, 473), color=(0,255,255), thickness=2)
            except:
                print("ERROR IN LANE DETECTION")      
                
            y_icon_offset = 350

             # GUI prototype -- icon display
            x_offset = int((s.shape[1] / 10) * 9)
            if (stop_sign == 1):
                # draw stop sign icon in frame
                stop_sign_icon = cv2.imread('icons/stop_sign.png')
                y_offset_1 = int((s.shape[0] / 20) * 1)
                s[y_offset_1:y_offset_1 + stop_sign_icon.shape[0], x_offset:x_offset + stop_sign_icon.shape[1]] = stop_sign_icon

            if (traffic_light == 1):
                self.trafficCnt = 45
            
            if (self.trafficCnt > 0):
                # draw traffic light icon in frame
                traffic_light_icon = cv2.imread('icons/traffic_light.png')
                y_offset_2 = int((s.shape[0] / 20) * 3)
                s[y_offset_2:y_offset_2 + traffic_light_icon.shape[0], x_offset:x_offset + traffic_light_icon.shape[1]] = traffic_light_icon
                self.trafficCnt -= 1

            if (person == 1):
                # draw person icon in frame
                person_icon = cv2.imread('icons/pedastrian_sign.png')
                y_offset_3 = int((s.shape[0] / 20) * 5)
                s[y_offset_3:y_offset_3 + person_icon.shape[0], x_offset:x_offset + person_icon.shape[1]] = person_icon

            if (lane_drift == 1):
                # draw lane drift icon in frame
                lane_drift_icon = cv2.imread('icons/triangle_yellow_warning.png')
                y_offset_4 = int((s.shape[0] / 20) * 7)
                s[y_offset_4:y_offset_4 + lane_drift_icon.shape[0], x_offset:x_offset + lane_drift_icon.shape[1]] = lane_drift_icon

            if (parking == 1):
                # draw parking icon in frame
                parking_icon = cv2.imread('icons/parking_icon.png')
                y_offset_5 = int((s.shape[0] / 20) * 9)
                s[y_offset_5:y_offset_5 + parking.shape[0], x_offset:x_offset + parking.shape[1]] = parking_icon

            s = cv2.resize(s,(1920,1080))
        
            ret, jpeg = cv2.imencode('.jpg', s)
            
            return jpeg.tobytes()
            #cv2.imshow(SCREEN_NAME, s)
            #cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        else:
            self.count += 1       

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        #try:
        frame = camera.get_frame()
        if frame is None:
            print("empty frame")
            continue
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        #except:
         #   sys.exit(0)

@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', debug=True)
# When everything done, release the capture
if OPTIMIZE_CAM:
    vs.stop()
else:
    vs.release()
    cv2.destroyAllWindows()