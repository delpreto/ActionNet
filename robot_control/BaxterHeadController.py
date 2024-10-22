#!/usr/bin/env python


import rospy
import cv2
import cv_bridge
from std_msgs.msg import Bool, Int32, UInt16, Float32
from sensor_msgs.msg import Image
from baxter_core_msgs.msg import HeadPanCommand, HeadState
import sys, os, time
from random import randrange as rand
from threading import Thread
import numpy as np

class BaxterHeadController():
  def __init__(self, logStatus=True):
    if logStatus:
      self.log("Initializing node for Baxter's head")
    try:
      rospy.init_node("baxterHead")
    except rospy.ROSException, e:
      if 'rospy.init_node() has already been called with different arguments' in str(e):
        pass
      else:
        raise
    
    # general state
    self._screen_width = 1024
    self._screen_height = 600
    
    # set up publishers
    self._screenPub = rospy.Publisher('/robot/xdisplay', Image, queue_size=1, latch=True)
    self._headPub = rospy.Publisher('/robot/head/command_head_pan', HeadPanCommand, queue_size=5, latch=True)
    self._headAngle = 1000
    self._nodPub = rospy.Publisher('/robot/head/command_head_nod', Bool, queue_size=2, latch=True)
    self._nodding = False
    self._sonarPub = rospy.Publisher('/robot/sonar/head_sonar/set_sonars_enabled', UInt16, queue_size=10)
    self._haloPub_red   = rospy.Publisher('/robot/sonar/head_sonar/lights/set_red_level',   Float32, queue_size=10)
    self._haloPub_green = rospy.Publisher('/robot/sonar/head_sonar/lights/set_green_level', Float32, queue_size=10)
    self._sonarLEDPub = rospy.Publisher('/robot/sonar/head_sonar/lights/set_lights', UInt16, queue_size=10)
    # set up subscribers
    headSub = rospy.Subscriber('/robot/head/head_state', HeadState, self._on_head_state)  # used to set _headAngle and _nodding in background

    # set up LED control (halo and sonar)
    self._sonarLEDCmd = None
    self._haloRedCmd = None
    self._haloGreenCmd = None
    self._LEDPubThread = None
    self._LEDPubThreadInit(rate=100) # if less than 100 Hz will time out and revert to default LED control
    self._LEDPubThreadStart()
    # see http://sdk.rethinkrobotics.com/wiki/API_Reference#Sonar_LED_Indicators
    self._sonarLEDModes = {
      'DEFAULT_BEHAVIOR' : 0x0000,
      'OVERRIDE_ENABLE'  : 0x8000,
    }
    self._sonarLEDOnCodes = [
      0x0001,
      0x0002,
      0x0004,
      0x0008,
      0x0010,
      0x0020,
      0x0040,
      0x0080,
      0x0100,
      0x0200,
      0x0400,
      0x0800,
    ]
    self._sonarLEDAllOnCode = 0x8FFF
    self._sonarLEDAllOffCode = 0x8000

    # set up slideshow control
    self._slideshowImgDir = None
    self._slideshowImages = None
    self._slideshowImagesShown = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

  """
  Callback to store current head state
  """
  def _on_head_state(self, msg):
    self._headAngle = msg.pan
    self._nodding = msg.isNodding

  """
  Show an image on the screen
  """
  def showImage(self, imageFile=None, imageMsg=None, logStatus=True):
    if imageFile is not None:
      if logStatus:
        self.log("Showing image file %s" % imageFile)
      img = cv2.imread(imageFile)
      if img is None:
        if logStatus:
          self.log("INVALID IMAGE file %s" % imageFile)
      else:
        imageMsg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
      self._screenPub.publish(imageMsg)
      rospy.sleep(0.1)
    elif imageMsg is not None:
      if logStatus:
        self.log("Showing image message")
      self._screenPub.publish(imageMsg)
      rospy.sleep(0.1)
  
  def showColor(self, color_bgr=(0,0,0)):
    if isinstance(color_bgr, str):
      if color_bgr.lower() in ['w', 'white']:
        color_bgr = (255, 255, 255)
      elif color_bgr.lower() in ['k', 'black']:
        color_bgr = (0, 0, 0)
      elif color_bgr.lower() in ['r', 'red']:
        color_bgr = (0, 0, 255)
      elif color_bgr.lower() in ['b', 'blue']:
        color_bgr = (255, 0, 0)
      elif color_bgr.lower() in ['g', 'green']:
        color_bgr = (0, 255, 0)
    img = np.zeros((self._screen_height, self._screen_width, 3), dtype=np.uint8)
    img[:,:,:] = color_bgr
    imageMsg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    self._screenPub.publish(imageMsg)
    rospy.sleep(0.1)
    
  """
  Show a video on the screen [blocking]
  """
  def showVideo(self, videoFile, fps_vid=None, fps_display=10, playback_speed_factor=1, logStatus=True):
    fpsRate = rospy.Rate(fps_display*playback_speed_factor)
    videoCapture = cv2.VideoCapture(videoFile)
    if videoCapture is None:
      return
    if fps_vid is None:
      fps_vid = videoCapture.get(cv2.CAP_PROP_FPS)
    if logStatus:
      self.log("Playing video file %s" % videoFile)
    have_frame = True
    while have_frame:
      frame = None
      for i in range(int(round(fps_vid / fps_display))):
        have_frame, frame = videoCapture.read()
      if have_frame:
        frame = np.array(frame)
        scale_factor_width = float(self._screen_width)/float(frame.shape[1])
        scale_factor_height = float(self._screen_height)/float(frame.shape[0])
        scale_factor = min([scale_factor_width, scale_factor_height])
        frame = cv2.resize(frame, (0,0), None, scale_factor, scale_factor)
        pad_width = int(round((self._screen_width - frame.shape[1])/2))
        pad_height = int(round((self._screen_height - frame.shape[0])/2))
        if pad_width > 1:
          frame = cv2.copyMakeBorder(frame, pad_height, pad_height,
                                     pad_width, pad_width,
                                     cv2.BORDER_CONSTANT, value=(0,0,0))
        imageMsg = cv_bridge.CvBridge().cv2_to_imgmsg(frame, encoding="bgr8")
        self._screenPub.publish(imageMsg)
      fpsRate.sleep()
    videoCapture.release()
    cv2.destroyAllWindows()
    if logStatus:
      self.log("Finished playing video file %s" % videoFile)

  """
  Set the folder of images that will be used for slideshows
  Folder should contain only images
  """
  def slideshowSetImageDir(self, imgDir):
    self._slideshowImgDir = imgDir
    self._slideshowImages = [f for f in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, f))]
    self.slideshowReset()

  """
  Reset which images have already been shown in calls to slideshow
  """
  def slideshowReset(self):
    self._imagesShown = []

  """
  Show a slideshow of images from the specified folder
  Will remember which images were previously shown and choose new ones when possible
  """
  def slideshow(self, numImages, imageDuration=8, totalDuration=None, printStatus=False, logStatus=False):
    if self._slideshowImages is None or len(self._slideshowImages) == 0:
      print 'No images specified to include in slideshow.  Call slideshowSetImageDir first.'
      return

    if imageDuration is None:
      imageDuration = totalDuration / numImages
    if totalDuration is None:
      totalDuration = imageDuration * numImages

    if printStatus:
      print 'Showing a %d second slideshow' % totalDuration

    if len(self._slideshowImagesShown) + numImages >= len(self._slideshowImages):
      self.slideshowReset()
    shownThisShow = 0
    for i in range(numImages):
      image = self._slideshowImages[rand(0, len(self._slideshowImages))]
      while image in self._slideshowImagesShown and len(self._slideshowImagesShown) < len(self._slideshowImages):
        image = self._slideshowImages[rand(0, len(self._slideshowImages))]
      shownThisShow += 1
      if printStatus:
        print("\tShowing image %s of %s" % (shownThisShow, numImages))
      self.showImage(os.path.join(self._slideshowImgDir, image), logStatus=logStatus)
      time.sleep(imageDuration)
      self._slideshowImagesShown.append(image)

  """
  Pan the head to a specified angle
  Note that maximum reachable angle is about 80 degrees
  """
  def pan(self, angle, speed=25, degrees=True, tolerance=3 * 3.14159 / 180.0, logStatus=True):
    if logStatus:
      self.log("Moving head pan to %s %s" % (angle, "degrees" if degrees else "radians"))
    if degrees:
      angle = angle * 3.14159 / 180.0
    headCmnd = HeadPanCommand(angle, speed/100.0, HeadPanCommand.REQUEST_PAN_VOID)  # REQUEST_PAN_VOID means don't change pan request setting
    count = 0
    self._headPub.publish(headCmnd)
    while (abs(self._headAngle - angle) > tolerance):
      time.sleep(0.01)
      count = count + 1
      if count > 2 / 0.01:
        if logStatus:
          self.log("ERROR MOVING HEAD did not reach desired tolerance in 2 seconds")
        break

  """
  Nod the head a specified number of times
  """
  def nod(self, times=3, internod_delay_s=0):
    for i in range(times):
      self._nodPub.publish(True)
      # wait for nod to start
      wait_startTime = time.time()
      while not self._nodding and time.time() - wait_startTime < 1:
        time.sleep(0.05)
      # wait for nod to complete
      wait_startTime = time.time()
      while self._nodding and time.time() - wait_startTime < 3:
        time.sleep(0.05)
      # wait before next nod
      if i < times and internod_delay_s is not None and internod_delay_s > 0:
        time.sleep(internod_delay_s)

  """
  Set the halo LED ring to a desired mix of red and green
  """
  def setHaloLED(self, red_percent, green_percent):
    if (red_percent >= 0 and red_percent <= 100 and green_percent >= 0 and green_percent <= 100)  \
        or (red_percent is None and green_percent is None):
      # the commands will be published by the LEDPubThread thread
      # note: probably only need single publish instead of constant publishing, but already have the thread anyway
      self._haloRedCmd = int(red_percent)
      self._haloGreenCmd = int(green_percent)

  """
  Set the yellow sonar LEDs
  led_states can be a 12-element list of 0/1, or a list of indices to turn on
  led_states can also be 'auto'/'default', 'on', or 'off'
  """
  def setSonarLEDs(self, led_states):
    if led_states == 'auto' or led_states == 'default' or led_states is None:
      cmd = self._sonarLEDModes['DEFAULT_BEHAVIOR']
    elif led_states == 'on':
      cmd = self._sonarLEDModes['OVERRIDE_ENABLE']
      cmd |= self._sonarLEDAllOnCode
    elif led_states == 'off':
      cmd = self._sonarLEDModes['OVERRIDE_ENABLE']
      cmd |= self._sonarLEDAllOffCode
    elif isinstance(led_states, (list, tuple)):
      cmd = self._sonarLEDModes['OVERRIDE_ENABLE']
      if len(led_states) == 12 and max(led_states) <= 1:
        for i in range(len(led_states)):
          if led_states[i]:
            cmd |= self._sonarLEDOnCodes[i]
      elif max(led_states) <= 11 and min(led_states) >= 0:
        for i in led_states:
          cmd |= self._sonarLEDOnCodes[i]
    self._sonarLEDCmd = cmd # will be published by the LEDPubThread thread

  """
  Enable or disable the sonar
  """
  def setSonar(self, state):
    self._sonarPub.publish(1 if state else 0)

  """
  Log a message to the ros log
  """
  def log(self, msg):
    rospy.loginfo(rospy.get_caller_id() + ": " + msg)

  """
  Clean up and quit
  """
  def quit(self):
    self._haloRedCmd = 0
    self._haloGreenCmd = 100
    self._LEDPubThreadDestroy()

  """
  Initialize a thread for running the LED publish loop
  @param rate [int] the target control loop rate (Hz)
  """
  def _LEDPubThreadInit(self, rate=100):
    # terminate/destroy any existing thread
    self._LEDPubThreadDestroy()
    # create the new thread
    self._LEDPubThreadRunning = False
    self._LEDPubThreadDestroyed = False
    self._LEDPubThread = Thread(target=self._LEDPubLoop, args=(rate,))
    # start the thread (note that it won't do anything since pidThreadRunning is False)
    self._LEDPubThread.start()

  """
  Start or stop the LED publish loop control thread (can be called multiple times)
  """
  def _LEDPubThreadState(self, state):
    if state:
      self._LEDPubThreadStart()
    else:
      self._LEDPubThreadStop()

  def _LEDPubThreadStart(self):
    if self._LEDPubThread is None:
      raise AssertionError('LEDPubThreadInit should be called before LEDPubThreadStart')
    self._LEDPubThreadRunning = True

  def _LEDPubThreadStop(self):
    if self._LEDPubThread is None:
      raise AssertionError('LEDPubThreadInit should be called before LEDPubThreadStart')
    self._LEDPubThreadRunning = False

  """
  Destroy the LED publish loop control thread (must call LEDPubThreadInit afterwards if want to use thread again)
  """
  def _LEDPubThreadDestroy(self):
    if self._LEDPubThread is not None:
      # signal the thread to stop
      self._LEDPubThreadRunning = False
      self._LEDPubThreadDestroyed = True
      # wait for it to terminate
      self._LEDPubThread.join()
      # reset the thread state
      self._LEDPubThread = None
      self._LEDPubThreadRunning = False
      self._LEDPubThreadDestroyed = False

  """
  Publish the LED state at the desired rate (meant for running in its own thread)
  """
  def _LEDPubLoop(self, rate):
    rate = rospy.Rate(rate)
    while not self._LEDPubThreadDestroyed and not rospy.is_shutdown():
      if self._LEDPubThreadRunning:
        if self._sonarLEDCmd is not None:
          self._sonarLEDPub.publish(self._sonarLEDCmd)
        if self._sonarLEDCmd == self._sonarLEDModes['DEFAULT_BEHAVIOR']:
          self._sonarLEDCmd = None # publish the switch to default once, then cease publishing
        if self._haloRedCmd is not None:
          self._haloPub_red.publish(self._haloRedCmd)
        if self._haloGreenCmd is not None:
          self._haloPub_green.publish(self._haloGreenCmd)
      rate.sleep()

"""
Test code
"""
if __name__ == '__main__':
  try:
    head = BaxterHeadController()
    count = 0
    for arg in sys.argv:
      try:
        (keyword, value) = arg.split('=')
        if len(value) == 0:
          continue
        if 'image' in keyword:
          image = value.strip()
          head.showImage(image)
          count = count + 1
        elif 'video' in keyword:
          video = value.strip()
          head.showVideo(video)
          count = count + 1
        elif 'degrees' in keyword:
          angle = float(value.strip())
          head.pan(angle)
          count = count + 1
        elif 'radians' in keyword:
          angle = float(value.strip())
          head.pan(angle, degrees=False)
          count = count + 1
        elif 'nod' in keyword:
          times = int(value.strip())
          head.nod(times)
          count = count + 1
        elif 'sonarLEDs' in keyword:
          try:
            states = eval(value.strip())
          except:
            states = value.strip()
          head.setSonarLEDs(states)
          count = count + 1
          time.sleep(2)
        elif 'sonar' in keyword:
          state = int(value.strip())
          head.setSonar(state)
          count = count + 1
        elif 'halo' in keyword:
          percents = eval(value.strip())
          head.setHaloLED(percents[0], percents[1])
          count = count + 1
          time.sleep(0.5)
      except ValueError:
        continue
    if count == 0:
      print("Usage: BaxterScreen.py image=path/to/image video=path/to/video degrees=angle radians=angle nod=3 sonarLEDs=[2,5] sonar=0 halo=[0,100]")
      print("  Any subset of arguments may be provided, and all will be treated separately in order provided")
      print("  image and video will show the image or video on the head")
      print("  degrees and radians will pan the head to that angle")
      print("  sonarLEDs is a 12-element list of 0/1, a list of indices to turn on, 'auto', 'on', or 'off' (no spaces in lists)")
      print("  sonar is 0 or 1")
      print("  halo is [red_percent,green_percent] (no spaces)")
  except:
    head.quit()
    raise
  head.quit()

















