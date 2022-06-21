
// Include libraries.
#include <esp_now.h>
#include <WiFi.h>

#define SERIAL_DEBUG_PRINTS 0

// Specify the tactile matrix shape.
#define NUM_TACTILE_ROWS 32
#define NUM_TACTILE_COLS 32

// Configure settings for sending data and/or COM messages via Serial.
#define SERIAL_RATE 1000000
// Will raise each byte so the data will never contain a newline.
uint8_t newline_offset = '\n'+1;

// Configure settings for receiving data via ESP-NOW.
#define ESPNOW_CHANNEL 1
// Define a structure to hold the tactile data.
typedef struct espnow_tactile_data_struct {
  uint8_t matrix_index; // The index of the current matrix.
  uint8_t row_index;    // The index of the row being sent within the current matrix.
  uint16_t data[NUM_TACTILE_COLS];    // The row data.
} espnow_tactile_data_struct;
espnow_tactile_data_struct tactile_data;

// Define pins for outputs.
const int led_pin = 13;

// Initialize state.
// Will send serial data as a single buffer, which is faster than individual bytes.
const int buffer_to_write_length = 2*NUM_TACTILE_COLS + 2 + 1; // two bytes per sample, the matrix and row indexes, and the newline
uint8_t buffer_to_write[buffer_to_write_length];
int buffer_to_write_index = 0;

// =============================
// SETUP
// =============================
void setup()
{
  // Set up serial.
  Serial.begin(SERIAL_RATE, SERIAL_8N1);

  // Set up pin modes.
  pinMode(led_pin, OUTPUT);
  digitalWrite(led_pin, HIGH);

  // Initialize ESP-NOW.
  WiFi.mode(WIFI_STA); // Set this device as a WiFi Station.
  #if SERIAL_DEBUG_PRINTS
  Serial.println(WiFi.macAddress());
  #endif
  if (esp_now_init() == ESP_OK)
  {
    #if SERIAL_DEBUG_PRINTS
    Serial.println("ESP-NOW initialized successfully.");
    #endif
  }
  else
  {
    #if SERIAL_DEBUG_PRINTS
    Serial.println("ESP-NOW initialization failed.  Restarting.");
    #endif
    ESP.restart();
  }

  // Initialize state.
  digitalWrite(led_pin, LOW);

  // Register the callback for received data now that ESP-NOW is successfully initialized.
  esp_now_register_recv_cb(process_received_data);
}

// =============================
// MAIN LOOP
// =============================
void loop()
{
  // See the callback function for the real work.
}

// Define a callback function that will be executed when data is received.
// Will send each row of data as a new line of the format [matrix_index][row_index][row_data][\n].
void process_received_data(const uint8_t* mac_address, const uint8_t* in_data, int data_length)
{
  // Update the tactile data structure with the new data.
  memcpy(&tactile_data, in_data, sizeof(tactile_data));

  // Indicate that data is being processed.
  digitalWrite(led_pin, HIGH);
//  long start_micros = micros();

  // Set the index to the beginning of the buffer.
  buffer_to_write_index = 0;

  // Send metadata via Serial.
  #if !SERIAL_DEBUG_PRINTS
  buffer_to_write[buffer_to_write_index++] = (uint8_t)tactile_data.matrix_index + newline_offset;
  buffer_to_write[buffer_to_write_index++] = (uint8_t)tactile_data.row_index + newline_offset;
  #else
  Serial.print(tactile_data.matrix_index); Serial.print(" ");
  Serial.print(tactile_data.row_index); Serial.println();
  Serial.println("------");
  #endif
  // Send the actual data via Serial.
  for(int col = 0; col < NUM_TACTILE_COLS; col++)
  {
    // Send the 12-bit reading as two bytes.
    #if !SERIAL_DEBUG_PRINTS
    buffer_to_write[buffer_to_write_index++] = (uint8_t)(tactile_data.data[col] >> 6) + newline_offset;
    buffer_to_write[buffer_to_write_index++] = (uint8_t)(tactile_data.data[col] & 0b00111111) + newline_offset;
    #else
    Serial.print(tactile_data.data[col]); Serial.print(" ");
    #endif
  }
  // Terminate the line of data.
  buffer_to_write[buffer_to_write_index++] = (uint8_t)'\n';

  // Send the data!
  // Doing it all at once is much faster than writing individual bytes.
  Serial.write(buffer_to_write, buffer_to_write_length);

//  long end_micros = micros();
//  Serial.println("||");
//  Serial.println(buffer_to_write_length);
//  Serial.println((float)(end_micros - start_micros));
//  Serial.println("||");

  // Indicate that data is done being processed.
  digitalWrite(led_pin, LOW);
}













