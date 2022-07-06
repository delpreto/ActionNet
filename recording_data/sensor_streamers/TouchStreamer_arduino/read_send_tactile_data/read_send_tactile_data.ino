////////////
/*
Copyright (c) 2022 MIT CSAIL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See https://action-net.csail.mit.edu for more usage information.

*/
////////////

// Specify which microcontroller is being used
#define USING_ARDUINO 1
#define USING_ESP     0

// Specify whether to send data wirelessly via ESP-NOW or wired via Serial.
#define SEND_DATA_ESPNOW 0
#define SEND_DATA_SERIAL 1
#define SERIAL_DEBUG_PRINTS 0

// Specify whether to wait for explicit data requests (old paradigm)
//  or to constantly send data terminated by a newline.
#define SERIAL_WAIT_FOR_REQUEST 1

// Include libraries.
#if USING_ESP
#include <driver/adc.h>
#endif
#if SEND_DATA_ESPNOW
#include <esp_now.h>
#include <WiFi.h>
#endif

// Specify the tactile matrix shape.
#define NUM_TACTILE_ROWS 32
#define NUM_TACTILE_COLS 32

#ifdef SEND_DATA_ESPNOW
// Configure settings for sending data via ESP-NOW.
// NOTE: the address below should match the MAC address of the receiver.
const uint8_t espnow_receiver_address[] = {0x7C, 0x87, 0xCE, 0xF6, 0xCE, 0xA8}; // {0x94, 0xB9, 0x7E, 0x6B, 0x51, 0x7C};
#define ESPNOW_CHANNEL 1
// Define a structure to hold the tactile data.
typedef struct espnow_tactile_data_struct {
  uint8_t matrix_index; // The index of the current matrix.
  uint8_t row_index;    // The index of the row being sent within the current matrix.
  uint16_t data[NUM_TACTILE_COLS];    // The row data.
} espnow_tactile_data_struct;
espnow_tactile_data_struct tactile_data;
#endif

// Configure settings for sending data or COM messages via Serial.
#define SERIAL_RATE 1000000
// Will raise each byte so the data will never contain a newline.
uint8_t newline_offset = '\n'+1;

// Specify the number of samples to average for each reading.
#define SAMPLING_AVERAGE_COUNT 1

// Send incrementing values instead of actual sensor readings
//  (but still take the sensor readings, so the timing should be the same).
#define SEND_DEBUG_VALUES 0

// Define pins for sampling and outputs (mux channel selections, LEDs, etc).
#if USING_ESP
const int pins_selectRows[] = {A0, A1, A5, 12, 13};
const int pins_selectCols[] = {27, 33, 15, 32, 14};
#endif
#if USING_ARDUINO
const int pins_selectRows[] = {2, 3, 4, 5, 6};
const int pins_selectCols[] = {8, 9, 10, 11, 12};
const int pin_analog_input = A7; // A7 for newer boards, A0 for old big board
#endif
const int num_row_pins = sizeof(pins_selectRows)/sizeof(int);
const int num_col_pins = sizeof(pins_selectCols)/sizeof(int);
const int led_pin = 13;

// Initialize state.
uint16_t analog_reading_averaged = 0;
uint8_t matrix_index = 0;
int matrix_counter = 0; // for debugging purposes
// Will send serial data as a single buffer, which is faster than individual bytes.
#if USING_ESP
#if SEND_DATA_SERIAL
#if !SERIAL_WAIT_FOR_REQUEST
const int buffer_to_write_length = 2*NUM_TACTILE_COLS + 2 + 1; // two bytes per sample, the matrix and row indexes, and the newline
#else
const int buffer_to_write_length = 2*NUM_TACTILE_COLS*NUM_TACTILE_ROWS + 1 + 1; // two bytes per sample, the matrix index, and the newline
#endif // !SERIAL_WAIT_FOR_REQUEST
uint8_t buffer_to_write[buffer_to_write_length];
int buffer_to_write_index = 0;
#endif // SEND_DATA_SERIAL
#endif // USING_ESP

// =============================
// SETUP
// =============================
void setup()
{
  // Set up serial.
  Serial.begin(SERIAL_RATE, SERIAL_8N1);

  // Set up pin modes.
  for(int i = 0; i < num_row_pins; i++)
  {
    pinMode(pins_selectRows[i], OUTPUT);
    digitalWrite(pins_selectRows[i], LOW);
  }
  for(int i = 0; i < num_col_pins; i++)
  {
    pinMode(pins_selectCols[i], OUTPUT);
    digitalWrite(pins_selectCols[i], LOW);
  }
  pinMode(led_pin, OUTPUT);
  // Set up the ADC.
  #if USING_ESP
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_11);
  #endif

  // Set up ESP-NOW.
  #if SEND_DATA_ESPNOW
  WiFi.mode(WIFI_STA);
  // This is the MAC address of the Master in Station Mode.
  #ifdef SERIAL_DEBUG_PRINTS
  Serial.print("MAC address: "); Serial.println(WiFi.macAddress());
  #endif
  if (esp_now_init() != ESP_OK)
  {
    #ifdef SERIAL_DEBUG_PRINTS
    Serial.println("Error initializing ESP-NOW");
    #endif
    return;
  }
  // Register the peer (the receiver).
  esp_now_peer_info_t peer_info;
  memset(&peer_info, 0, sizeof(peer_info));
  memcpy(peer_info.peer_addr, espnow_receiver_address, 6);
  peer_info.channel = ESPNOW_CHANNEL;
  peer_info.encrypt = 0;
  // Add the peer (the receiver).
  esp_err_t add_status = esp_now_add_peer(&peer_info);
  #if SERIAL_DEBUG_PRINTS
  if(add_status == ESP_OK)
  {
    // Pair success
    Serial.println("ESP-NOW paired successfully");
  }
  else if(add_status == ESP_ERR_ESPNOW_NOT_INIT)
    Serial.println("ERROR ADDING ESP-NOW PEER: ESP-NOW is not initialized");
  else if(add_status == ESP_ERR_ESPNOW_ARG)
    Serial.println("ERROR ADDING ESP-NOW PEER: Invalid argument");
  else if(add_status == ESP_ERR_ESPNOW_FULL)
    Serial.println("ERROR ADDING ESP-NOW PEER: Peer list full");
  else if(add_status == ESP_ERR_ESPNOW_NO_MEM)
    Serial.println("ERROR ADDING ESP-NOW PEER: Out of memory");
  else if(add_status == ESP_ERR_ESPNOW_EXIST)
    Serial.println("ERROR ADDING ESP-NOW PEER: Peer already exists");
  else
    Serial.println("ERROR ADDING ESP-NOW PEER: Not sure what happened");
  #endif

  // Initialize the tactile data state.
  tactile_data.matrix_index = 0;

  #endif // SEND_DATA_ESPNOW
}

// =============================
// MAIN LOOP
// =============================
void loop()
{
  // Request paradigm - wait for a serial command then send entire matrix of data.
  #if SERIAL_WAIT_FOR_REQUEST && SEND_DATA_SERIAL
  switch(read_serial_char())
  {
    case 'a':
      digitalWrite(led_pin, HIGH);
      scan_send_matrix();
      digitalWrite(led_pin, LOW);
      break;
  }
  #else
  // Streaming paradigm - constantly send data with each row of the matrix as a new line.
  digitalWrite(led_pin, HIGH);
  scan_send_matrix();
  digitalWrite(led_pin, LOW);
  #endif
}

// =============================
// HELPERS
// =============================
char read_serial_char()
{
  // Wait for serial input
  while(Serial.available() <= 0);
  // Read the command
  return (short) Serial.read();
}

// Read all sensors in the matrix and send the results via Serial or ESP-NOW.
// If sending via Serial using the request paradigm, will send the whole matrix as a line:
//   The line format is [matrix_index][matrix_data][\n]
// If sending via Serial using the streaming paradigm, will send each row as a line.
//   The line format is [matrix_index][row_index][row_data][\n]
// If sending via ESP-NOW, will send each row as a message.
//   Each message will be a espnow_tactile_data_struct containing row data.
void scan_send_matrix()
{
  // Send the matrix index if using the Serial request paradigm.
  #if SEND_DATA_SERIAL && SERIAL_WAIT_FOR_REQUEST
    #if !SERIAL_DEBUG_PRINTS
    #if USING_ESP
    buffer_to_write_index = 0; // Set the index to the beginning of the buffer.
    buffer_to_write[buffer_to_write_index++] = (uint8_t)matrix_index + newline_offset;
    #endif
    #if USING_ARDUINO
    Serial.write((uint8_t)matrix_index + newline_offset);
    #endif
    #else
    Serial.println(matrix_index); Serial.println("------");
    #endif
  #endif

  for(int row_index = 0; row_index < NUM_TACTILE_ROWS; row_index++)
  {
    select_mux_pin(row_index);
    // Send the row index if using the Serial streaming paradigm.
    #if SEND_DATA_SERIAL && !SERIAL_WAIT_FOR_REQUEST
    #if USING_ESP
    buffer_to_write_index = 0; // Set the index to the beginning of the buffer.
    buffer_to_write[buffer_to_write_index++] = (uint8_t)matrix_index + newline_offset;
    buffer_to_write[buffer_to_write_index++] = (uint8_t)row_index + newline_offset;
    #endif
    #if USING_ARDUINO
    Serial.write((uint8_t)matrix_index + newline_offset);
    Serial.write((uint8_t)row_index + newline_offset);
    #endif
    #endif

    for(int column_index = 0; column_index < NUM_TACTILE_COLS; column_index++)
    {
      select_read_switch(column_index);
      // delayMicroseconds(20);
      analog_reading_averaged = 0;
      for(int avg_i = 0; avg_i < SAMPLING_AVERAGE_COUNT; avg_i++)
      {
          #if USING_ESP
          analog_reading_averaged += adc1_get_raw(ADC1_CHANNEL_6);
          #endif
          #if USING_ARDUINO
          analog_reading_averaged += analogRead(pin_analog_input);
          #endif
      }
      analog_reading_averaged /= SAMPLING_AVERAGE_COUNT;

      // Send incrementing debug values if desired.
      // Note that the mux control and analog read is still performed,
      //  so sampling rate estimates are still accurate.
      #if SEND_DEBUG_VALUES
      analog_reading_averaged = 32*row_index + column_index;
      #endif

      // Store or send data as needed based on the communication paradigm.

      #if SEND_DATA_ESPNOW
      tactile_data.data[column_index] = analog_reading_averaged;
      #endif

      #if SEND_DATA_SERIAL
        // Send the 12-bit reading as two bytes.
        #if !SERIAL_DEBUG_PRINTS
        #if USING_ESP
        buffer_to_write[buffer_to_write_index++] = (uint8_t)(analog_reading_averaged >> 6) + newline_offset;
        buffer_to_write[buffer_to_write_index++] = (uint8_t)(analog_reading_averaged & 0b00111111) + newline_offset;
        #endif
        #if USING_ARDUINO
        Serial.write((uint8_t)(analog_reading_averaged >> 5) + newline_offset);
        Serial.write((uint8_t)(analog_reading_averaged & 0b00011111) + newline_offset);
        #endif
        #else
        Serial.print(" ");
        Serial.print(analog_reading_averaged);
        #endif
      #endif
    }
    // Have now finished processing an entire row.
    // Send the data and/or a newline if needed.
    #if SEND_DATA_ESPNOW
    tactile_data.matrix_index = matrix_index;
    tactile_data.row_index = row_index;
    esp_err_t result = esp_now_send(espnow_receiver_address, (uint8_t *) &tactile_data, sizeof(tactile_data));
    delayMicroseconds(500);
    #if SERIAL_DEBUG_PRINTS
    if (result == ESP_OK)
      Serial.println("Sent row data successfully");
    else
      Serial.println("Error sending row data");
    #endif
    #endif
    #if SEND_DATA_SERIAL && !SERIAL_WAIT_FOR_REQUEST
    #if USING_ESP
    buffer_to_write[buffer_to_write_index++] = (uint8_t)'\n';
    // Send the data!
    Serial.write(buffer_to_write, buffer_to_write_length);
    #endif
    #if USING_ARDUINO
    Serial.write((uint8_t)'\n');
    #endif
    #endif
  }
  // Have now finished processing an entire matrix.
  // Update state, and send a newline if needed.
  matrix_index = matrix_index == (254-newline_offset) ? 0 : (matrix_index+1);
  #if SEND_DATA_SERIAL && SERIAL_WAIT_FOR_REQUEST
  #if USING_ESP
  buffer_to_write[buffer_to_write_index++] = (uint8_t)'\n';
  // Send the data!
  Serial.write(buffer_to_write, buffer_to_write_length);
  #endif
  #if USING_ARDUINO
  Serial.write((uint8_t)'\n');
  #endif
  #endif

//  Serial.println(matrix_counter);
  matrix_counter++;
}

void select_mux_pin(byte pin)
{
  #if USING_ESP
  for (int i = 0; i < num_row_pins; i++)
  {
    if(pin & (1<<i))
      digitalWrite(pins_selectRows[i], HIGH);
    else
      digitalWrite(pins_selectRows[i], LOW);
  }
  #endif
  #if USING_ARDUINO
  PORTD = pin << 2;
  #endif
}

void select_read_switch(byte pin)
{
  #if USING_ESP
  for(int i = 0; i < num_col_pins; i++)
  {
    if(pin & (1<<i))
      digitalWrite(pins_selectCols[i], HIGH);
    else
      digitalWrite(pins_selectCols[i], LOW);
  }
  #endif
  #if USING_ARDUINO
  PORTB = pin;
  #endif
}
