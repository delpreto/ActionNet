import cv2
import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device
import pupil_labs
import datetime
import time

# Look for devices. Returns as soon as it has found the first device.
print("Looking for the next best device...")
device = discover_one_device(max_search_duration_seconds=10)
if device is None:
    print("No device found.")
    raise SystemExit(-1)

print(f"Connecting to {device}...")

cap = cv2.VideoCapture("rtsp://pi.local:8086/?camera=world")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

# cap = cv2.VideoCapture("rtp://pi.local:8086/?camera=world")
now = datetime.datetime.now()
videoFileName = 'D:/Github/ActionNet/data/tests/eyetracking/' + str(now).replace(':', '').replace('.','') +'EyetrackingVideo.avi'

print(str(now).replace(':', '').replace('.',''))
codec = "DIVX"
fourcc = cv2.VideoWriter_fourcc(*codec)
out = cv2.VideoWriter(videoFileName, fourcc, 30.0, (int(width), int(height)))

while True:
    # print('trying')
    frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
    # print(gaze[2])
    if gaze[2] == True:
        cv2.circle(
            frame.bgr_pixels,
            (int(gaze.x), int(gaze.y)),
            radius=10,
            color=(0, 0, 255),
            thickness=3,
        )

    unixTime = str(time.time())
    currentTime = str(datetime.datetime.now())
    startTime = str(now)
    cv2.putText(frame.bgr_pixels, startTime, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(frame.bgr_pixels, currentTime, (400, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(frame.bgr_pixels, unixTime, (800, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    out.write(frame.bgr_pixels)
    cv2.imshow("Scene camera with gaze overlay", frame.bgr_pixels)

    cv2.waitKey(1)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('e'):
        cv2.destroyWindow('rtsp scene stream')

    else:
        pass
        # print("No Stream opened")

out.release()