
from sensor_streamers.SensorStreamer import SensorStreamer

import sys
import time
import textwrap
from collections import OrderedDict
import traceback


################################################
################################################
# A class to record arbitrary user input during an experiment.
# Each note will be timestamped so it can be matched with sensor data and events.
################################################
################################################
class NotesStreamer(SensorStreamer):

  ########################
  ###### INITIALIZE ######
  ########################

  # Note that if this streamer will run in a child process,
  #  then input will fail unless the main stdin is provided.
  #  This can probably be done via the custom_stdin argument as follows (found online but not tested yet):
  #   custom_stdin = os.fdopen(os.dup(sys.stdin.fileno()))
  def __init__(self, streams_info=None, custom_stdin=None,
                log_player_options=None, visualization_options=None,
                print_status=True, print_debug=False, log_history_filepath=None):
    SensorStreamer.__init__(self, streams_info,
                              log_player_options=log_player_options,
                              visualization_options=visualization_options,
                              print_status=print_status, print_debug=print_debug,
                              log_history_filepath=log_history_filepath)
    self._wait_after_stopping = False
    self._always_run_in_main_process = True
    self._log_source_tag = 'notes'

    # Create the stream unless an existing log is being replayed
    #  (in which case SensorStreamer will create the stream automatically).
    if not self._replaying_data_logs:
      self.add_stream(device_name='experiment-notes', stream_name='notes',
                      data_type='S500', sample_size=[1], sampling_rate_hz=None,
                      data_notes=OrderedDict([
                        ('Description', 'Notes that the experimenter entered '
                                        'during the trial, timestamped to '
                                        'align with collected data'),
                      ]))


    # Set a custom stdin if desired.
    if custom_stdin is not None:
      sys.stdin = custom_stdin

  def _connect(self, timeout_s=10):
    return True

  # Get the most recent notes string that the user has entered.
  def get_last_notes(self):
    device_name = self.get_device_names()[0]
    stream_name = self.get_stream_names()[0]
    data = self.get_data(device_name, stream_name, starting_index=-1, ending_index=None)
    if data is not None:
      return data['data'][0]
    return data

  #####################
  ###### RUNNING ######
  #####################

  # Loop until self._running is False
  def _run(self):
    msg = textwrap.dedent('''
    ----------
    You may log timestamped experiment notes
     at any time by typing them and pressing enter
     (even if other printouts have occurred in the meantime)
    ----------
    ''')
    print(msg)

    try:
      while self._running:
        try:
          notes = input('Enter experiment notes: ').strip()
        except UnicodeDecodeError:
          self._log_warn('\nWarning: NotesStreamer could not decode the user input.\n')
          notes = ''
        notes_times_s = time.time()
        if len(notes) > 0:
          self.append_data(self.get_device_names()[0], 'notes',
                           notes_times_s, notes)
          self._log_debug(' Logged an experiment note at %f' % notes_times_s)
    except KeyboardInterrupt: # The program was likely terminated
      pass
    except EOFError: # The program may have been terminated
      self._log_warn('\nNotesStreamer encounted EOFError')
    except:
      self._log_error('\n\n***ERROR RUNNING NotesStreamer:\n%s\n' % traceback.format_exc())
    finally:
      pass

  # Clean up and quit
  def quit(self):
    if self._print_debug:
      self._log_debug('NotesStreamer quitting')
      self._log_debug('NotesStreamer data:')
      self._log_debug(self._data)
    SensorStreamer.quit(self)














