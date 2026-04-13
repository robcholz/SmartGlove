/* Includes ---------------------------------------------------------------- */
#include <ESP32_Gesture_Detector_inferencing.h> // <-- CHANGE THIS to your actual library name
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// This is where we store the data before sending it to the "Brain"
float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1);
    }
    Serial.println("MPU6050 Found!");
}

void loop() {
    Serial.println("Sampling...");

    // 1. Fill the buffer with 2 seconds of data (matching your 2000ms window)
    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);
        
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        // Fill buffer with X, Y, Z
        buffer[ix + 0] = a.acceleration.x;
        buffer[ix + 1] = a.acceleration.y;
        buffer[ix + 2] = a.acceleration.z;

        // Maintain the 100Hz timing
        while (micros() < next_tick) { /* wait */ }
    }

    // 2. Run the Classifier
    signal_t signal;
    int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    
    ei_impulse_result_t result = { 0 };
    err = run_classifier(&signal, &result, false);

    // 3. Print the results
    Serial.println("Predictions:");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        Serial.print("    ");
        Serial.print(result.classification[ix].label);
        Serial.print(": ");
        Serial.println(result.classification[ix].value);
    }

    delay(1000); // Wait a second before the next detection
}