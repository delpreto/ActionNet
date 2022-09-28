////////////////////////
//
// Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// See https://action-net.csail.mit.edu for more usage information.
// Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
//
////////////////////////

#define SEND_DEBUG_VALUES 0 // Whether to send known values instead of sensor readings.
                            // Can set this in SerialStreamer.py too to validate the data.

const int sensor_pins[] = {A0, A1}; // The number of pins should match the sample size expected by SerialStreamer
const int num_sensors = sizeof(sensor_pins)/sizeof(int);
const int poll_period_ms = 10; // Should match the expected sampling rate used by SerialStreamer
const int baud_rate = 1000000; // Should match the baud rate used by SerialStreamer

//====================================================
// SETUP
//====================================================
void setup()
{
  // Begin Serial communication.
  Serial.begin(baud_rate);
  while(!Serial);
  Serial.println();

  // Configure input pins if needed.
  for(int sensor_index = 0; sensor_index < num_sensors; sensor_index++)
    pinMode(sensor_pins[sensor_index], INPUT);
}

//====================================================
// MAIN LOOP
//====================================================
void loop()
{
  // Read data from each sensor.
  for(int sensor_index = 0; sensor_index < num_sensors; sensor_index++)
  {
    // Send sensor data or debug values as desired.
    #if SEND_DEBUG_VALUES
    Serial.print(sensor_index);
    #else
    Serial.print(analogRead(sensor_pins[sensor_index]));
    #endif
    // Send the delimiter if there will be another value.
    // This should match the delimiter used by SerialStreamer.
    if(sensor_index+1 < num_sensors)
      Serial.print(" ");
  }
  // Finish the line of data from all sensors.
  Serial.println();
  // Wait to create the desired sampling rate.
  delay(poll_period_ms);
}
