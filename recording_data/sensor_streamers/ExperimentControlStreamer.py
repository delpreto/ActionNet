
from sensor_streamers.SensorStreamer import SensorStreamer

import tkinter
from tkinter import ttk

import sys
import time
import textwrap
from collections import OrderedDict
import traceback
import ctypes

from utils.print_utils import *

################################################
################################################
# A class to create a GUI that can record experiment events.
# Includes calibration periods, activities, and arbitrary user input.
################################################
################################################
class ExperimentControlStreamer(SensorStreamer):
  
  ########################
  ###### INITIALIZE ######
  ########################
  
  def __init__(self, streams_info=None,
               log_player_options=None, visualization_options=None,
               print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                            log_player_options=log_player_options,
                            visualization_options=visualization_options,
                            print_status=print_status, print_debug=print_debug,
                            log_history_filepath=log_history_filepath)
    self._wait_after_stopping = False
    self._always_run_in_main_process = True
    self._log_source_tag = 'control'
    
    self._tkinter_root = None
    
    # Define state for the current status.
    self._experiment_is_running = False
    self._calibrating = False
    self._performing_activity = False
    
    # Define lists of environment targets, poses, activities, etc.
    self._target_positions_cm = OrderedDict([
      ('Right heel at origin, facing +x', (0, 0)),
      ('Right heel at target, facing -x', (100, 150)),
      ('Gaze Calibration Stance', ((154.6 + 183.2)/2, -((28.5-5) + (57.0-5))/2)),
      ('Gaze Calibration Screen', (243, -107)),
      ])
    self._target_names = list(self._target_positions_cm.keys())
    self._arm_poses = [
      'Upper arm down, forearm straight out, palm facing in',
      'Straight down, palms inward (N pose)',
      'Straight out, palms downward (T pose)',
      'Straight up, palms inward',
      ]
    self._hand_poses = [
      'Fist',
      'Spread fingers',
      'Wave in',
      'Wave out',
      ]
    self._scale_scenarios = [
      'Flat hand on scale',
      'Flat hand on bumps on scale',
      'Single fingers on scale',
      'Mug (by handle)',
      'Pan (by handle)',
      'Plate (from side)',
      'Knife (by handle)',
      'Jar (full side grasp)',
      ]
    self._thirdparty_calibrations = [
      'Xsens: N-Pose with Walking',
      'Xsens: T-Pose with Walking',
      'Xsens: N-Pose without Walking',
      'Xsens: T-Pose without Walking',
      'PupilLabs: Target Grid',
      'PupilLabs: Single Target',
      'Manus: IMUs',
      'Manus: Poses Left',
      'Manus: Poses Right',
      ]
    self._activities = [
      'Get/replace items from refrigerator/cabinets/drawers',
      'Clear cutting board',
      'Peel a cucumber',
      'Slice a cucumber',
      'Peel a potato',
      'Slice a potato',
      'Slice bread',
      'Spread almond butter on a bread slice',
      'Spread jelly on a bread slice',
      'Open/close a jar of almond butter',
      'Pour water from a pitcher into a glass',
      'Clean a plate with a sponge',
      'Clean a plate with a towel',
      'Clean a pan with a sponge',
      'Clean a pan with a towel',
      'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
      'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
      'Stack on table: 3 each large/small plates, bowls',
      'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
      'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
      ]
    
    # Define the calibration data streams and their corresponding GUI inputs.
    # The input names will be updated later once they are added to the GUI.
    max_notes_length = 75
    self._calibration_device_name = 'experiment-calibration'
    self._calibration_streams = OrderedDict([
      ('body', {
               'description': 'Periods when the person assumed a known '
                              'body pose, location, and orientation.  '
                              'Useful for calibrating IMU-based orientations such as the Xsens.',
               'data_type': 'S%d' % ((int(max([len(x) for x in self._target_names] + [max_notes_length])/10)+1)*10),
               'tab_label': 'Body',
               'inputs': [
                  {'label': 'Location', 'type': 'combo', 'values': self._target_names,        'name': None},
                  {'label': 'Facing',   'type': 'combo', 'values': self._target_names,        'name': None},
                  {'label': 'Pose',     'type': 'combo', 'values': ['T-Pose', 'N-Pose'], 'name': None},
                ]
               }),
      ('arms', {
               'description': 'Periods when the person assumed a known '
                              'arm pose and pointing direction orientation.  '
                              'Useful for calibrating IMU-based orientations such as the Myos.',
               'data_type': 'S%d' % ((int(max([len(x) for x in self._target_names+self._arm_poses] + [max_notes_length])/10)+1)*10),
               'tab_label': 'Arms',
               'inputs': [
                  {'label': 'Body location',             'type': 'combo', 'values': self._target_names,   'name': None},
                  {'label': 'Left arm pose',             'type': 'combo', 'values': self._arm_poses, 'name': None},
                  {'label': 'Left forearm pointing at',  'type': 'combo', 'values': self._target_names,   'name': None},
                  {'label': 'Right arm pose',            'type': 'combo', 'values': self._arm_poses, 'name': None},
                  {'label': 'Right forearm pointing at', 'type': 'combo', 'values': self._target_names,   'name': None},
                ]
               }),
      ('hand_poses', {
               'description': 'Periods when the person assumed a known '
                              'hand poses.  Useful for calibrating IMU-based orientations'
                              'such as the Xsens gloves, how the Myo is worn, and forearm EMG levels.',
               'data_type': 'S%d' % ((int(max([len(x) for x in self._hand_poses] + [max_notes_length])/10)+1)*10),
               'tab_label': 'Hands',
               'inputs': [
                  {'label': 'Left hand pose',  'type': 'combo', 'values': self._hand_poses, 'name': None},
                  {'label': 'Right hand pose', 'type': 'combo', 'values': self._hand_poses, 'name': None},
                ]
               }),
      ('gaze', {
               'description': 'Periods when the person assumed a known '
                              'body position and gazed at a known target.  '
                              'Useful for calibrating the eye tracking.',
               'data_type': 'S%d' % ((int(max([len(x) for x in self._target_names] + [max_notes_length])/10)+1)*10),
               'tab_label': 'Gaze',
               'inputs': [
                  {'label': 'Body location', 'type': 'combo', 'values': self._target_names, 'name': None},
                  {'label': 'Gazing at',     'type': 'combo', 'values': self._target_names, 'name': None},
                ]
               }),
      ('tactile_gloves', {
               'description': 'Periods when the person exerted a known '
                              'force against a known part of the glove.',
        'data_type': 'S%d' % ((int(max([len('%10.6f' % 0)] + [max_notes_length])/10)+1)*10),
               'tab_label': 'Tactile',
               'inputs': [
                  {'label': 'Left hand pose/object' , 'type': 'combo', 'values': self._scale_scenarios, 'name': None},
                  {'label': 'Right hand pose/object', 'type': 'combo', 'values': self._scale_scenarios, 'name': None},
                ]
               }),
      ('third_party', {
               'description': 'Periods when calibration routines from '
                              'third-party software was running, such as '
                              'Xsens or PupilLabs.',
               'data_type': 'S%d' % ((int(max([len(x) for x in self._thirdparty_calibrations+self._target_names] + [max_notes_length])/10)+1)*10),
               'tab_label': 'Third-Party',
               'inputs': [
                  {'label': 'Calibration Type',       'type': 'combo', 'values': self._thirdparty_calibrations, 'name': None},
                  {'label': 'Body starting location', 'type': 'combo', 'values': self._target_names,   'name': None},
                  {'label': 'Body ending location',   'type': 'combo', 'values': self._target_names,   'name': None},
                  {'label': 'Screen location with gaze targets', 'type': 'combo', 'values': self._target_names, 'name': None},
                ]
               }),
    ])
    self._last_calibration_times_s = OrderedDict([(calibration_type, None) for calibration_type in self._calibration_streams])
    
    # Create the streams unless an existing log is being replayed
    #  (in which case SensorStreamer will create the streams automatically).
    if not self._replaying_data_logs:
      for (stream_name, info) in self._calibration_streams.items():
        self.add_stream(device_name=self._calibration_device_name, stream_name=stream_name,
                        data_type=info['data_type'], sample_size=[3+len(info['inputs'])], sampling_rate_hz=None,
                        timesteps_before_solidified=2, # will update the start and stop entries with valid status and notes
                        data_notes=OrderedDict([
                          ('Description', info['description']),
                          (SensorStreamer.metadata_data_headings_key,
                            ['Start/Stop', 'Valid', 'Notes']
                            + [input['label'] for input in info['inputs']])
                          ]),
                        )

    # Define the activity stream.
    # Data in each entry will be: activity, start/stop, good/maybe/bad, notes
    self._activities_device_name = 'experiment-activities'
    self._activities_stream_name = 'activities'
    if not self._replaying_data_logs:
      self.add_stream(device_name=self._activities_device_name, stream_name=self._activities_stream_name,
                      data_type='S500', sample_size=[4], sampling_rate_hz=None,
                      timesteps_before_solidified=2, # will update the start and stop entries with valid status and notes
                      data_notes=OrderedDict([
                        ('Description', 'The activity being performed.'),
                        (SensorStreamer.metadata_data_headings_key,
                         ['Activity', 'Start/Stop', 'Valid', 'Notes']),
                      ]),
                      )
    self._activities_counts = OrderedDict([
      (activity, {'Good':0, 'Maybe':0, 'Bad':0}) for activity in self._activities
    ])
    
    # Define the notes stream.
    self._notes_device_name = 'experiment-notes'
    self._notes_stream_name = 'notes'
    if not self._replaying_data_logs:
      self.add_stream(device_name=self._notes_device_name, stream_name=self._notes_stream_name,
                      data_type='S500', sample_size=[1], sampling_rate_hz=None,
                      data_notes=OrderedDict([
                        ('Description', 'Notes that the experimenter entered '
                                        'during the trial, timestamped to '
                                        'align with collected data'),
                      ]),
                      )

    # Add to the metadata.
    self._metadata.setdefault(self._calibration_device_name, {})
    self._metadata.setdefault(self._activities_device_name, {})
    self._metadata[self._calibration_device_name]['Target Locations [cm]'] = self._target_positions_cm
    self._metadata[self._calibration_device_name]['Arm Poses'] = self._arm_poses
    self._metadata[self._calibration_device_name]['Hand Poses'] = self._hand_poses
    self._metadata[self._calibration_device_name]['Scale Scenarios'] = self._scale_scenarios
    self._metadata[self._calibration_device_name]['Third-Party Calibrations'] = self._thirdparty_calibrations
    self._metadata[self._activities_device_name]['Activities'] = self._activities
    self._metadata[self._activities_device_name]['Target Locations [cm]'] = self._target_positions_cm

  def _connect(self, timeout_s=10):
    # Without this, the window can be really tiny if matplotlib is used before the GUI generation.
    # Make sure this is called before any matplotlib.
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    return True
  
  def _on_quit(self):
    self._experiment_is_running = False
    self._tkinter_root.destroy()
  
  # Get whether the user has ended the experiment.
  def experiment_is_running(self):
    return self._experiment_is_running

  # Helpers to validate text inputs.
  def _validate_text_input_float(self, new_text):
    return len(new_text) == 0 or new_text.replace('.','',1).isnumeric()
  def _validate_text_input_int(self, new_text):
    return len(new_text) == 0 or new_text.isnumeric()
  
  ###############################
  ###### CALIBRATION LOGIC ######
  ###############################
  
  # Helpers to get calibration elements.
  def _get_active_calibration_type(self):
    calibration_frame = self._tkinter_root.nametowidget('calibration_frame')
    notebook_calibration = calibration_frame.nametowidget('notebook_calibration')
    tab_label = notebook_calibration.tab(notebook_calibration.select(), 'text')
    for (calibration_type, info) in self._calibration_streams.items():
      if tab_label == info['tab_label']:
        return calibration_type
    return None
  def _get_calibration_tab(self, calibration_type=None):
    calibration_frame = self._tkinter_root.nametowidget('calibration_frame')
    notebook_calibration = calibration_frame.nametowidget('notebook_calibration')
    if calibration_type is None:
      calibration_type = self._get_active_calibration_type()
    tab_calibration = notebook_calibration.nametowidget('tab_calibration_%s' % calibration_type)
    return tab_calibration
  
  # Callback function when calibrations are started/stopped.
  def _button_callback_startStop_calibration(self):
    # Get the calibration configuration.
    calibration_type = self._get_active_calibration_type()
    tab_calibration = self._get_calibration_tab(calibration_type=calibration_type)
    calibration_time_s = time.time()
    
    # Toggle the calibration status.
    self._calibrating = not self._calibrating
    self._last_calibration_type = calibration_type
    
    # Add data to the appropriate stream.
    data = []
    is_string_type = self._calibration_streams[calibration_type]['data_type'].find('S') == 0
    if self._calibrating:
      data.append('Start' if is_string_type else 1)
    else:
      data.append('Stop' if is_string_type else 0)
    data.append('') # placeholder for good/bad indicator (will be added later)
    data.append('') # placeholder for notes (will be added later)
    for input in self._calibration_streams[calibration_type]['inputs']:
      input_element = tab_calibration.nametowidget(input['name'])
      if input['type'] == 'combo':
        data.append(input_element.get())
      elif input['type'] == 'text':
        input_text = input_element.get()
        if 'float' in self._calibration_streams[calibration_type]['data_type'].lower() and len(input_text) > 0:
          data.append(float(input_text))
        elif 'int' in self._calibration_streams[calibration_type]['data_type'].lower() and len(input_text) > 0:
          data.append(int(input_text))
        else:
          data.append(input_text)
    self.append_data(self._calibration_device_name, calibration_type, calibration_time_s, data)
  
    # Disable editing the calibration configuration if calibration started.
    comboboxes_to_toggle = []
    entries_to_toggle = []
    for input in self._calibration_streams[calibration_type]['inputs']:
      if input['type'] == 'combo':
        comboboxes_to_toggle.append(tab_calibration.nametowidget(input['name']))
      else:
        entries_to_toggle.append(tab_calibration.nametowidget(input['name']))
    if self._calibrating:
      for combobox in comboboxes_to_toggle:
        combobox['state'] = 'disabled'
      for entry in entries_to_toggle:
        entry['state'] = 'disabled'
    else:
      for combobox in comboboxes_to_toggle:
        combobox['state'] = 'readonly'
      for entry in entries_to_toggle:
        entry['state'] = 'normal'
        
    # Update the start/stop button text/format.
    comboboxes_to_toggle = []
    entries_to_toggle = []
    for input in self._calibration_streams[calibration_type]['inputs']:
      if input['type'] == 'combo':
        comboboxes_to_toggle.append(tab_calibration.nametowidget(input['name']))
      else:
        entries_to_toggle.append(tab_calibration.nametowidget(input['name']))
    if self._calibrating:
      self._button_calibration_startStop.configure(style="Blue.TButton")
      self._button_calibration_startStop['text'] = 'STOP Calibration'
    else:
      self._button_calibration_startStop.configure(style="TButton")
      self._button_calibration_startStop['text'] = 'Start Calibration'
      self._button_calibration_startStop['state'] = 'disabled'
      
    # Update the mark good/maybe/bad buttons and notes field.
    if self._calibrating:
      self._button_calibration_markGood['state'] = 'disabled'
      self._button_calibration_markMaybe['state'] = 'disabled'
      self._button_calibration_markBad['state'] = 'disabled'
      self._text_calibration_notes['state'] = 'disabled'
    else:
      self._button_calibration_markGood['state'] = 'normal'
      self._button_calibration_markMaybe['state'] = 'normal'
      self._button_calibration_markBad['state'] = 'normal'
      self._text_calibration_notes['state'] = 'normal'
      self._button_calibration_markGood.configure(style='DarkGreen.TButton')
      self._button_calibration_markMaybe.configure(style='DarkYellow.TButton')
      self._button_calibration_markBad.configure(style='DarkRed.TButton')
  
  # Callbacks for when the calibration is marked as good/maybe/bad.
  def _button_callback_mark_calibration_good(self):
    self._button_callback_mark_calibration('Good')
  def _button_callback_mark_calibration_maybe(self):
    self._button_callback_mark_calibration('Maybe')
  def _button_callback_mark_calibration_bad(self):
    self._button_callback_mark_calibration('Bad')
  def _button_callback_mark_calibration(self, mark_status):
    # Get then clear the notes.
    calibration_notes = self._text_calibration_notes.get('1.0', tkinter.END).strip()
    self._text_calibration_notes.delete('1.0', tkinter.END)
    # Update both the start and stop entries in the data log.
    self._data[self._calibration_device_name][self._last_calibration_type]['data'][-1][1] = mark_status
    self._data[self._calibration_device_name][self._last_calibration_type]['data'][-1][2] = calibration_notes
    self._data[self._calibration_device_name][self._last_calibration_type]['data'][-2][1] = mark_status
    self._data[self._calibration_device_name][self._last_calibration_type]['data'][-2][2] = calibration_notes
    # Update the last calibration time if it was marked as good.
    if mark_status == 'Good':
      self._last_calibration_times_s[self._last_calibration_type] = self._data[self._calibration_device_name][self._last_calibration_type]['time_s'][-1]
    # Update field states/formats.
    self._button_calibration_markGood['state'] = 'disabled'
    self._button_calibration_markMaybe['state'] = 'disabled'
    self._button_calibration_markBad['state'] = 'disabled'
    self._text_calibration_notes['state'] = 'disabled'
    self._button_calibration_startStop['state'] = 'normal'
    self._button_calibration_markGood.configure(style='TButton')
    self._button_calibration_markMaybe.configure(style='TButton')
    self._button_calibration_markBad.configure(style='TButton')

  ############################
  ###### ACTIVITY LOGIC ######
  ############################
  
  # Callback for when an activity is started/stopped.
  def _button_callback_startStop_activity(self):
    # Get the GUI elements.
    activities_frame = self._tkinter_root.nametowidget('activities_frame')
    button_activity_startStop = activities_frame.nametowidget('button_activity_startStop')
    combo_activities = activities_frame.nametowidget('combo_activities')
    activity = combo_activities.get()

    # Abort if no activity was selected.
    if len(activity) == 0:
      return
    # Toggle the activity status.
    self._performing_activity = not self._performing_activity
    self._last_activity = activity

    # Add data to the appropriate stream.
    data = []
    data.append(activity)
    data.append('Start' if self._performing_activity else 'Stop')
    data.append('') # placeholder for good/bad indicator (will be added later)
    data.append('') # placeholder for notes (will be added later)
    self.append_data(self._activities_device_name, self._activities_stream_name,
                     time.time(), data)

    # Disable editing the activity configuration if calibration started.
    if self._performing_activity:
      combo_activities['state'] = 'disabled'
    else:
      combo_activities['state'] = 'readonly'
    
    # Update the start/stop button text/format.
    if self._performing_activity:
      button_activity_startStop.configure(style="Blue.TButton")
      button_activity_startStop['text'] = 'STOP Activity'
    else:
      button_activity_startStop.configure(style="TButton")
      button_activity_startStop['text'] = 'Start Activity'

    # Update the mark good/maybe/bad buttons and notes field.
    if self._performing_activity:
      self._button_activity_markGood['state'] = 'disabled'
      self._button_activity_markMaybe['state'] = 'disabled'
      self._button_activity_markBad['state'] = 'disabled'
      self._text_activity_notes['state'] = 'disabled'
    else:
      self._button_activity_markGood['state'] = 'normal'
      self._button_activity_markMaybe['state'] = 'normal'
      self._button_activity_markBad['state'] = 'normal'
      self._text_activity_notes['state'] = 'normal'
      self._button_activity_markGood.configure(style='DarkGreen.TButton')
      self._button_activity_markMaybe.configure(style='DarkYellow.TButton')
      self._button_activity_markBad.configure(style='DarkRed.TButton')
    
  # Callbacks for marking an activity trial as good or bad.
  def _button_callback_mark_activity_good(self):
    self._button_callback_mark_activity('Good')
  def _button_callback_mark_activity_maybe(self):
    self._button_callback_mark_activity('Maybe')
  def _button_callback_mark_activity_bad(self):
    self._button_callback_mark_activity('Bad')
  def _button_callback_mark_activity(self, mark_status):
    # Get then clear the notes.
    activity_notes = self._text_activity_notes.get('1.0', tkinter.END).strip()
    self._text_activity_notes.delete('1.0', tkinter.END)
    # Update both the start and stop entries in the data log.
    self._data[self._activities_device_name][self._activities_stream_name]['data'][-1][2] = mark_status
    self._data[self._activities_device_name][self._activities_stream_name]['data'][-1][3] = activity_notes
    self._data[self._activities_device_name][self._activities_stream_name]['data'][-2][2] = mark_status
    self._data[self._activities_device_name][self._activities_stream_name]['data'][-2][3] = activity_notes
    # Update activity counts.
    self._activities_counts[self._last_activity][mark_status] = 1 + self._activities_counts[self._last_activity][mark_status]
    self._update_activities_counts_text()
    # Update field states/formats.
    self._button_activity_markGood['state'] = 'disabled'
    self._button_activity_markMaybe['state'] = 'disabled'
    self._button_activity_markBad['state'] = 'disabled'
    self._text_activity_notes['state'] = 'disabled'
    self._button_activity_startStop['state'] = 'normal'
    self._button_activity_markGood.configure(style='TButton')
    self._button_activity_markMaybe.configure(style='TButton')
    self._button_activity_markBad.configure(style='TButton')
  
  def _define_ttk_styles(self):
    ttk_style = ttk.Style()
    ttk_style.configure('CalibrationFrame.TFrame', background='lightyellow')
    ttk_style.configure('ActivitiesFrame.TFrame', background='lightgreen')
    ttk_style.configure('NotesFrame.TFrame', background='lightblue')
    ttk_style.configure('LogFrame.TFrame', background='darkgray')
    ttk_style.configure("Blue.TButton", foreground="white", background="blue")
    ttk_style.configure("DarkRed.TButton", foreground="white", background="darkred")
    ttk_style.configure("DarkYellow.TButton", foreground="black", background="darkorange")
    ttk_style.configure("DarkGreen.TButton", foreground="black", background="darkgreen")
    return ttk_style
  
  # Callback for when general notes are submitted.
  def _button_callback_notes_submit(self):
    input_element = self._tkinter_root.nametowidget('notes_frame').nametowidget('general_notes')
    notes = input_element.get('1.0', tkinter.END) # line 1 character 0 through the end
    notes = notes.strip()
    if len(notes) > 0:
      self.append_data(self._notes_device_name, self._notes_stream_name,
                       time.time(), notes)
    input_element.delete('1.0', tkinter.END)

  #################################
  ###### GUI LAYOUT/CREATION ######
  #################################
  
  # Create the GUI!
  def _create_gui(self):
    # Create the root.
    self._tkinter_root = tkinter.Tk()
    self._tkinter_root.title("Let's make an ActionNet!")
    # Create some room around all the internal frames
    self._tkinter_root['padx'] = 5
    self._tkinter_root['pady'] = 5

    grid_positions = {
      'calibration_frame': {'row':1, 'column':1, 'rowspan':1, 'columnspan':1},
      'activities_frame':  {'row':2, 'column':1, 'rowspan':1, 'columnspan':1},
      'notes_frame':       {'row':3, 'column':1, 'rowspan':1, 'columnspan':1},
      'logViewer_frame':   {'row':1, 'column':2, 'rowspan':3, 'columnspan':1},
      'quit_button':       {'row':4, 'column':3, 'rowspan':1, 'columnspan':1},
  
      'calibration_label':            {'row':1, 'column':1, 'rowspan':1, 'columnspan':4},
      'calibration_notebook':         {'row':2, 'column':1, 'rowspan':1, 'columnspan':4},
      'calibration_button_startStop': {'row':3, 'column':1, 'rowspan':1, 'columnspan':1},
      'calibration_button_markGood':  {'row':3, 'column':2, 'rowspan':1, 'columnspan':1},
      'calibration_button_markMaybe': {'row':3, 'column':3, 'rowspan':1, 'columnspan':1},
      'calibration_button_markBad':   {'row':3, 'column':4, 'rowspan':1, 'columnspan':1},
      'calibration_notes':            {'row':4, 'column':1, 'rowspan':1, 'columnspan':4},
      'calibration_times':            {'row':1, 'column':5, 'rowspan':4, 'columnspan':1},
  
      'activities_label':            {'row':1, 'column':1, 'rowspan':1, 'columnspan':4},
      'activities_combo':            {'row':2, 'column':1, 'rowspan':1, 'columnspan':4},
      'activities_button_startStop': {'row':3, 'column':1, 'rowspan':1, 'columnspan':1},
      'activities_button_markGood':  {'row':3, 'column':2, 'rowspan':1, 'columnspan':1},
      'activities_button_markMaybe': {'row':3, 'column':3, 'rowspan':1, 'columnspan':1},
      'activities_button_markBad':   {'row':3, 'column':4, 'rowspan':1, 'columnspan':1},
      'activities_notes':            {'row':4, 'column':1, 'rowspan':1, 'columnspan':4},
      'activities_counts':           {'row':1, 'column':5, 'rowspan':4, 'columnspan':1},
  
      'notes_label':         {'row':1, 'column':1, 'rowspan':1, 'columnspan':1},
      'notes_button_submit': {'row':2, 'column':1, 'rowspan':1, 'columnspan':1},
      'notes_notes':         {'row':1, 'column':2, 'rowspan':1, 'columnspan':3},
  
      'logViewer_label': {'row':1, 'column':1, 'rowspan':1, 'columnspan':1},
      'logViewer_log':   {'row':2, 'column':1, 'rowspan':1, 'columnspan':1},
      'logViewer_scrollbar_vertical':   {'row':2, 'column':2, 'rowspan':1, 'columnspan':1},
      'logViewer_scrollbar_horizontal': {'row':3, 'column':1, 'rowspan':1, 'columnspan':1},
    }
    ttk_style = self._define_ttk_styles()
    
    # Calibration!
    calibration_frame = ttk.Frame(self._tkinter_root, name='calibration_frame', style='CalibrationFrame.TFrame')
    calibration_frame.grid(sticky=tkinter.W, padx=5, pady=15,
                           **grid_positions['calibration_frame'])

    # Create a notebook that will have tabs for various calibrations.
    ttk.Label(calibration_frame, text='Calibration')\
      .grid(sticky=tkinter.W, **grid_positions['calibration_label'])
    notebook_calibration = ttk.Notebook(calibration_frame, name='notebook_calibration')
    notebook_calibration.grid(**grid_positions['calibration_notebook'],
                              sticky=tkinter.E + tkinter.W + tkinter.N + tkinter.S,
                              padx=5, pady=5)
    # Create all calibration tabs based on the defined streams.
    # Will update the 'name' values in self._calibration_streams[calibration_type]['inputs']
    #  to reflect the names of the elements that are added to the GUI.
    for (calibration_type, stream_info) in self._calibration_streams.items():
      # Create a new tab in the notebook.
      tab = tkinter.Frame(notebook_calibration, name='tab_calibration_%s' % calibration_type)
      notebook_calibration.add(tab, text=stream_info['tab_label'], compound=tkinter.TOP)
      # Add a label and input element for each input.
      row = 1
      for (input_index, input_info) in enumerate(stream_info['inputs']):
        label_text = input_info['label']
        keyword = label_text.replace(' ', '-').replace('[','').replace(']','').lower()
        ttk.Label(tab, text=label_text).grid(row=row, column=1, sticky=tkinter.W)
        if input_info['type'] == 'combo':
          input_name = 'combo_%s_%s' % (calibration_type, keyword)
          input_element = ttk.Combobox(tab, state='readonly', # no custom values
                                       name=input_name,
                                       values=['']+input_info['values'],
                                       width=max(20, max([len(x) for x in input_info['values']])),
                                       )
        elif input_info['type'] == 'text':
          input_name = 'text_%s_%s' % (calibration_type, keyword)
          if input_info['validation'] == 'float':
            validate_command = (tab.register(self._validate_text_input_float), '%P')
          elif input_info['validation'] == 'int':
            validate_command = (tab.register(self._validate_text_input_int), '%P')
          elif input_info['validation'] is None:
            validate_command = None
          else:
            raise AssertionError('Unknown validation type: %s' % input_info['validation'])
          input_element = ttk.Entry(tab, width=6, name=input_name,
                                    validate='key', validatecommand=validate_command)
        else:
          raise AssertionError('Unknown input type: %s' % input_info['type'])
        input_element.grid(row=row, column=2, sticky=tkinter.W)
        row = row+1
        self._calibration_streams[calibration_type]['inputs'][input_index]['name'] = input_name
    # Add a button to start/stop this calibration.
    # calibration_done_frame = ttk.Frame(calibration_frame, name='calibration_done_frame')
    # calibration_done_frame.grid(row=3, column=1, columnspan=2,
    #                             sticky=tkinter.W, padx=5, pady=25)
    self._button_calibration_startStop = ttk.Button(calibration_frame,
                                           text='Start Calibration',
                                           name='button_calibration_startStop',
                                           command=self._button_callback_startStop_calibration,
                                           )
    self._button_calibration_startStop.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['calibration_button_startStop'])
    # Add a field to enter notes about the trial.
    self._text_calibration_notes = tkinter.Text(calibration_frame,
                                                width=25, height=1, name='calibration_notes')
    self._text_calibration_notes.grid(sticky=tkinter.W, padx=5, pady=5, **grid_positions['calibration_notes'])
    self._text_calibration_notes['state'] = 'disabled'
    # Add buttons to mark the calibration as good/maybe/bad.
    self._button_calibration_markGood = ttk.Button(calibration_frame,
                                           text='Mark Good',
                                           name='button_calibration_good',
                                           command=self._button_callback_mark_calibration_good,
                                           )
    self._button_calibration_markGood['state'] = 'disabled'
    self._button_calibration_markGood.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['calibration_button_markGood'])
    self._button_calibration_markMaybe = ttk.Button(calibration_frame,
                                           text='Mark Maybe',
                                           name='button_calibration_mabybe',
                                           command=self._button_callback_mark_calibration_maybe,
                                           )
    self._button_calibration_markMaybe['state'] = 'disabled'
    self._button_calibration_markMaybe.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['calibration_button_markMaybe'])
    self._button_calibration_markBad = ttk.Button(calibration_frame,
                                           text='Mark Bad',
                                           name='button_calibration_bad',
                                           command=self._button_callback_mark_calibration_bad,
                                           )
    self._button_calibration_markBad['state'] = 'disabled'
    self._button_calibration_markBad.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['calibration_button_markBad'])
    # Add time since each calibration type.
    calibration_times_element = ttk.Label(calibration_frame, text='')
    calibration_times_element.grid(padx=5, pady=5, sticky=tkinter.W + tkinter.N,
                                   **grid_positions['calibration_times'])
    def update_calibration_time_text():
      txt = 'Time since last calibrations:'
      for (calibration_type, last_time_s) in self._last_calibration_times_s.items():
        txt += '\n'
        if last_time_s is None or time.time() - last_time_s > 10*60:
          txt += ' *** '
        else:
          txt += '       '
        if last_time_s is None:
          txt += '   --------'
        else:
          txt += ' %04.1f min' % ((time.time() - last_time_s)/60)
        txt += ':   %s' % calibration_type
      calibration_times_element['text'] = txt
      calibration_times_element.after(2000, update_calibration_time_text)
    update_calibration_time_text()
    
    # Do activities!
    activities_frame = ttk.Frame(self._tkinter_root, name='activities_frame', style='ActivitiesFrame.TFrame')
    activities_frame.grid(sticky=tkinter.W, padx=5, pady=15,
                          **grid_positions['activities_frame'])
    ttk.Label(activities_frame, text='Activities') \
          .grid(sticky=tkinter.W + tkinter.N, **grid_positions['activities_label'])
    # Add a dropdown to select the activity.
    ttk.Combobox(activities_frame, state='readonly', # no custom values
                 name='combo_activities',
                 values=['']+self._activities,
                 width=max(20, max([len(x) for x in self._activities])),
                 )\
          .grid(sticky=tkinter.W + tkinter.N, padx=5, pady=5, **grid_positions['activities_combo'])
    # Add a button to start/stop the activity.
    self._button_activity_startStop = ttk.Button(activities_frame,
                                         text='Start Activity',
                                         name='button_activity_startStop',
                                         command=self._button_callback_startStop_activity,
                                         )
    self._button_activity_startStop.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['activities_button_startStop'])
    # Add a field to enter notes about the trial.
    self._text_activity_notes = tkinter.Text(activities_frame, width=25, height=1, name='activity_notes')
    self._text_activity_notes.grid(sticky=tkinter.W, padx=5, pady=5, **grid_positions['activities_notes'])
    self._text_activity_notes['state'] = 'disabled'
    # Add buttons to mark trial as good/maybe/bad.
    self._button_activity_markGood = ttk.Button(activities_frame,
                                       text='Mark Good',
                                       name='button_activity_good',
                                       command=self._button_callback_mark_activity_good,
                                       )
    self._button_activity_markGood['state'] = 'disabled'
    self._button_activity_markGood.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['activities_button_markGood'])
    self._button_activity_markMaybe = ttk.Button(activities_frame,
                                       text='Mark Maybe',
                                       name='button_activity_maybe',
                                       command=self._button_callback_mark_activity_maybe,
                                       )
    self._button_activity_markMaybe['state'] = 'disabled'
    self._button_activity_markMaybe.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['activities_button_markMaybe'])
    self._button_activity_markBad = ttk.Button(activities_frame,
                                       text='Mark Bad',
                                       name='button_activity_bad',
                                       command=self._button_callback_mark_activity_bad,
                                       )
    self._button_activity_markBad['state'] = 'disabled'
    self._button_activity_markBad.grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['activities_button_markBad'])
    # Add number of trials of each activity.
    activities_counts_element = ttk.Label(activities_frame, text='')
    activities_counts_element.grid(padx=5, pady=5, sticky=tkinter.W + tkinter.N, **grid_positions['activities_counts'])
    def update_activities_counts_text():
      txt = 'Number of each activity performed [good/maybe/bad]:'
      for (activity, counts) in self._activities_counts.items():
        txt += '\n'
        txt += ' %3d / %3d / %3d:  %s' % (counts['Good'], counts['Maybe'], counts['Bad'], activity)
      activities_counts_element['text'] = txt
    self._update_activities_counts_text = update_activities_counts_text
    self._update_activities_counts_text()
    
    # Add an input for general notes.
    notes_frame = ttk.Frame(self._tkinter_root, name='notes_frame', style='NotesFrame.TFrame')
    notes_frame.grid(sticky=tkinter.W, padx=5, pady=15, **grid_positions['notes_frame'])
    ttk.Label(notes_frame, text='General notes')\
          .grid(sticky=tkinter.W + tkinter.N, **grid_positions['notes_label'])
    tkinter.Text(notes_frame, width=40, height=2, name='general_notes')\
          .grid(sticky=tkinter.W, padx=5, pady=5, **grid_positions['notes_notes'])
    ttk.Button(notes_frame,
                text='Submit Notes',
                name='button_notes_submit',
                command=self._button_callback_notes_submit,
                ).grid(padx=5, pady=5, sticky=tkinter.W, **grid_positions['notes_button_submit'])

    # Add a log view.
    logViewer_frame = ttk.Frame(self._tkinter_root, name='logViewer_frame', style='LogFrame.TFrame')
    logViewer_frame.grid(sticky=tkinter.N + tkinter.W, padx=5, pady=15, **grid_positions['logViewer_frame'])
    ttk.Label(logViewer_frame, text='Status Log') \
      .grid(sticky=tkinter.W + tkinter.N, **grid_positions['logViewer_label'])
    # logViewer_log = ttk.Label(logViewer_frame, text='')
    logViewer_log = tkinter.Text(logViewer_frame, width=60, height=40,
                                 wrap=tkinter.NONE,
                                 font=('Courier', 8), name='logViewer_text')
    logViewer_log.grid(padx=5, pady=5, sticky=tkinter.W + tkinter.N, **grid_positions['logViewer_log'])
    logViewer_scroll_vertical = tkinter.Scrollbar(logViewer_frame, command=logViewer_log.yview)
    logViewer_scroll_vertical.grid(padx=5, pady=5, sticky='nsew', **grid_positions['logViewer_scrollbar_vertical'])
    logViewer_scroll_horizontal = tkinter.Scrollbar(logViewer_frame, command=logViewer_log.xview, orient='horizontal')
    logViewer_scroll_horizontal.grid(padx=5, pady=5, sticky='nsew', **grid_positions['logViewer_scrollbar_horizontal'])
    logViewer_log['yscrollcommand'] = logViewer_scroll_vertical.set
    logViewer_log['xscrollcommand'] = logViewer_scroll_horizontal.set
    logViewer_log.xview_moveto(0.09)
    logViewer_log.yview_moveto(1)
    def update_logViewer_text():
      # Abort if no log is being saved.
      if self._log_history_filepath is None:
        logViewer_log['state'] = 'normal'
        logViewer_log.delete(1.0, tkinter.END)
        logViewer_log.insert(tkinter.END, '\n\n'
                                        + '*'*25
                                        + '\nNo log file was specified\n'
                                        + '*'*25
                                        + '\n\n')
        logViewer_log['state'] = 'disabled'
        return
      # Determine whether the user has moved the scrollbar or it's still in autoscroll mode.
      scroll_yview_original = logViewer_log.yview()[1]
      scroll_xview_original = logViewer_log.xview()[0]
      if round(scroll_yview_original, 2) == 1:
        # Get the log entries to display.
        with open(self._log_history_filepath, 'r') as fin:
          log = [x.strip() for x in fin.readlines()]
        log_entries = log #log[-1000:]
        if not self._print_debug:
          log_entries = [entry for entry in log_entries if '[debug]' not in entry]
        if not self._print_status:
          log_entries = [entry for entry in log_entries if '[normal]' not in entry]
        log_entries = '\n'.join(log_entries).split('\n') # some single entries may have newlines
        log_entries = [entry[11:] for entry in log_entries] # remove the date to save space
        log_entries = [entry[0:12] + entry[15:] for entry in log_entries] # remove the post-millisecond decimals to save space
        # log_entries = log_entries[-1000:]
        txt = '\n'.join(log_entries)
        # Update the viewer.
        logViewer_log['state'] = 'normal'
        logViewer_log.delete(1.0, tkinter.END)
        logViewer_log.insert(tkinter.END, txt)
        logViewer_log['state'] = 'disabled'
        # Scroll to the bottom left.
        logViewer_log.yview_moveto(1)
        logViewer_log.xview_moveto(scroll_xview_original)
        # logViewer_log.see(tkinter.END) # scrolls both vertically and horizontally
      # Update again soon
      logViewer_log.after(1000, update_logViewer_text)
    update_logViewer_text()
    
    # Handle quitting, via a button or closing the window.
    self._tkinter_root.protocol("WM_DELETE_WINDOW", self._on_quit)
    quit_button = ttk.Button(self._tkinter_root, text='End Experiment', command=self._on_quit)
    quit_button.grid(padx=5, pady=50, sticky=tkinter.E, **grid_positions['quit_button'])

  #####################
  ###### RUNNING ######
  #####################
  
  # Loop until self._running is False
  def _run(self):
    self._experiment_is_running = True
    self._create_gui()
    try:
      while self._running and self.experiment_is_running():
        self._tkinter_root.mainloop()
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except:
      self._log_error('\n\n***ERROR RUNNING ExperimentControlStreamer:\n%s\n' % traceback.format_exc())
    finally:
      self._log_status('Done!')
  
  # Clean up and quit
  def quit(self):
    self._on_quit()
    if self._print_debug:
      self._log_debug('ExperimentControlStreamer quitting')
      self._log_debug('ExperimentControlStreamer data:')
      self._log_debug(self._data)
    SensorStreamer.quit(self)
    

#####################
###### TESTING ######
#####################
if __name__ == '__main__':
  
  duration_s = 30
  print_status = True
  print_debug = True
  
  # Create a streamer.
  streamer = ExperimentControlStreamer(print_status=print_status, print_debug=print_debug)
  
  # Run
  streamer.run()
  time_start_s = time.time()
  while streamer.experiment_is_running() and time.time() - time_start_s < 1200:
    time.sleep(0.2)
  streamer.stop()
  time_stop_s = time.time()
  duration_s = time_stop_s - time_start_s
  # Print some sample rates.
  print('Stream duration [s]: ', duration_s)
  print_var(streamer._data, 'streamer data')
  













