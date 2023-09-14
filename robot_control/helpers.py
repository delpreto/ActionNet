
##############################################
# Miscellaneous helper functions
##############################################

from __future__ import print_function

from collections import OrderedDict
import math
# for waitForInputOrTimeout
import os, time
if os.name != 'nt': # using Linux
  import sys, tty, termios, select
else: # using Windows
  import msvcrt

"""
prepend/append a string to other strings
will do so for all keys if a dict, all items if a list, or simply the given variable otherwise
example, if given addString(['again', 'goodbye'], 'hello ', False) will output ['hello again', 'hello goodbye']
@param invar the variable with strings to manipulate
@param toAdd [str] the string to append or prepend
@param append [bool] whether to append or prepend (default is append)
"""
def addString(invar, toAdd, append=True):
  if isinstance(invar, OrderedDict):
    return OrderedDict([(addString(key, toAdd, append), invar[key]) for key in list(invar.keys())])
  if isinstance(invar, dict):
    return dict([(addString(key, toAdd, append), invar[key]) for key in list(invar.keys())])
  if isinstance(invar, list):
    return [addString(key, toAdd, append) for key in invar]
  if isinstance(invar, tuple):
    return tuple([addString(key, toAdd, append) for key in invar])
  if append:
    return '%s%s' % (invar, toAdd)
  else:
    return '%s%s' % (toAdd, invar)


"""
print a given variable in a nice format
@param toPrint [dict, OrderedDict, list, tuple, str, int, float, None] the variable to print
@param name [str] the name of the variable if desired
@return [str] a string with the variable nicely formatted for printing
"""
def printVariable(toPrint, name=None, level=0):
  print(getVariableStr(invar=toPrint, name=name, level=level))

"""
get a string of a given variable in a nice format for printing
@param toPrint [dict, OrderedDict, list, tuple, str, int, float, None] the variable to print
@param name [str] the name of the variable if desired
@return [str] a string with the variable nicely formatted for printing
"""
def getVariableStr(invar, name=None, level=0):
  variableStr = ''
  if name is not None and len(name) > 0:
    if ':' != name.strip()[-1]:
      name = name + ':'
    variableStr += '%s ' % name
    level += 1
  if isinstance(invar, dict):
    variableStr += '(OrderedDict)' if isinstance(invar, OrderedDict) else '(dict)'
    variableStr += '\n'
    for (key, value) in invar.items():
      variableStr += '\t' * level
      variableStr += '%s: ' % key
      variableStr += getVariableStr(value, level=level + 1)
      variableStr += '\n'
  elif isinstance(invar, (list, tuple)):
    variableStr += '(list)' if isinstance(invar, list) else '(tuple)'
    variableStr += '\n'
    for i, item in enumerate(invar):
      variableStr += '\t' * level
      variableStr += '%02d: ' % i
      variableStr += getVariableStr(item, level=level + 1)
      variableStr += '\n'
  else:
    variableStr += '\t' * level
    variableStr += str(invar)
    variableStr += '\n'
  return variableStr.strip()


"""
convert degrees to radians
will do so for all values if a dict/OrderedDict, all items if a list, or simply the given variable otherwise
"""
def deg2rad(invar):
  if isinstance(invar, OrderedDict):
    return OrderedDict([(key, deg2rad(invar[key])) for key in list(invar.keys())])
  if isinstance(invar, dict):
    return dict([(key, deg2rad(invar[key])) for key in list(invar.keys())])
  if isinstance(invar, list):
    return [deg2rad(key) for key in invar]
  if isinstance(invar, tuple):
    return tuple([deg2rad(key) for key in invar])
  if invar is None:
    return None
  return math.radians(invar)


"""
convert radians to degrees
will do so for all values if a dict/OrderedDict, all items if a list, or simply the given variable otherwise
"""
def rad2deg(invar):
  if isinstance(invar, OrderedDict):
    return OrderedDict([(key, rad2deg(invar[key])) for key in list(invar.keys())])
  if isinstance(invar, dict):
    return dict([(key, rad2deg(invar[key])) for key in list(invar.keys())])
  if isinstance(invar, list):
    return [rad2deg(key) for key in invar]
  if isinstance(invar, tuple):
    return tuple([rad2deg(key) for key in invar])
  if invar is None:
    return None
  return math.degrees(invar)

"""
scale numbers by a given factor
will do so for all values if a dict/OrderedDict, all items if a list, or simply the given variable otherwise
"""
def scale(invar, factor):
  if isinstance(invar, OrderedDict):
    return OrderedDict([(key, scale(invar[key], factor)) for key in list(invar.keys())])
  if isinstance(invar, dict):
    return dict([(key, scale(invar[key], factor)) for key in list(invar.keys())])
  if isinstance(invar, list):
    return [scale(key, factor) for key in invar]
  if isinstance(invar, tuple):
    return tuple([scale(key, factor) for key in invar])
  if invar is None:
    return None
  return invar * factor

"""
Clamp a value according to the provided limits
will do so for all values if a dict/OrderedDict, all items if a list, or simply the given variable otherwise
@param value the value(s) to clamp
@param limits [list, tuple] a two-element array
    the first element is the minimum limit and the second element is the maximum limit
    either or both elements may be None
@return the clamped value(s)
"""
def clamp(invar, limits):
  if isinstance(invar, OrderedDict):
    return OrderedDict([(key, clamp(invar[key], limits)) for key in list(invar.keys())])
  if isinstance(invar, dict):
    return dict([(key, clamp(invar[key], limits)) for key in list(invar.keys())])
  if isinstance(invar, list):
    return [clamp(key, limits) for key in invar]
  if isinstance(invar, tuple):
    return tuple([clamp(key, limits) for key in invar])
  if invar is None:
    return None
  res = invar
  if limits[0] is not None: res = max(res, limits[0])
  if limits[1] is not None: res = min(res, limits[1])
  return res

if __name__ == '__main__':
  test = OrderedDict([
                      ('one', 1),
                      ('two', 2),
                      ('three', [3,4,5,None]),
                      ('four', {'fourA':6, 'fourB':7, 'fourC':8}),
                      ('five', (3.14, 90, 180, 1.57)),
                      ])
  printVariable(test, 'my test variable')
  print('----------------------')
  printVariable(addString(test, 'hi_', False), 'my hi test variable')
  print('----------------------')
  printVariable(rad2deg(test), 'my test in degrees')
  print('----------------------')
  printVariable(deg2rad(test), 'my test in radians')
  print('----------------------')
  printVariable(scale(test, 2), 'my test twice over')
  print('----------------------')
  printVariable(clamp(test, [3,6]), 'my test clamped')
  
  
"""
Wait for user to enter a termination sequence or for a timeout.
Keypresses will not show up in the terminal but will be silently captured.
@param exit_codes [str or list] the sequence(s) that the user should enter to exit the waiting
@param timeout [float] how long to wait for the input sequence before exiting
@param exit_condition_fn an optional function handle that will also cause the waiting to break if it returns True
@return (got_exit_code, timeout_reached, exit_condition_fn_met, user_input)
"""
def waitForInputCodeOrTimeout(exit_codes, timeout, exit_condition_fn=None):
  if not isinstance(exit_codes, (list, tuple)):
    exit_codes = [exit_codes]
  if exit_condition_fn is None:
    exit_condition_fn = lambda:False
  if os.name != 'nt': # using Linux
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    got_input_code = False
    try:
      start_wait_time = time.time()
      user_input = ''
      #tty.setraw(sys.stdin.fileno())
      tty.setcbreak(sys.stdin.fileno())
      timedout = False
      exit_condition_fn_met = False
      while not got_input_code and not timedout and not exit_condition_fn_met:
        user_input = ''
        while len(select.select([sys.stdin], [], [], 0.001)[0]) > 0:
          user_input += sys.stdin.read(1)
        got_input_code = True in [exit_code in user_input for exit_code in exit_codes]
        exit_condition_fn_met = exit_condition_fn()
        timedout = time.time() - start_wait_time >= timeout
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  else: # using Windows
    got_input_code = False
    start_wait_time = time.time()
    user_input = ''
    timedout = False
    exit_condition_fn_met = False
    while not got_input_code and not timedout and not exit_condition_fn_met:
      user_input = ''
      while msvcrt.kbhit():
        user_input += msvcrt.getch()
      got_input_code = True in [exit_code in user_input for exit_code in exit_codes]
      exit_condition_fn_met = exit_condition_fn()
      timedout = time.time() - start_wait_time >= timeout
  return (got_input_code, timedout, exit_condition_fn_met, user_input)
  
  
  
  
  
