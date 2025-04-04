#include <TensorFlowLite.h>
#include "soc_lstm.tflite"  // Your converted model

// Sensor setup (INA219 for voltage/current)
#include <Wire.h>
#include <Adafruit_INA219.h>
Adafruit_INA219 ina219;

// TFLite interpreter
tflite::MicroErrorReporter error_reporter;
const tflite::Model* model = tflite::GetModel(soc_lstm_tflite);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, 2048);

void setup() {
  Serial.begin(115200);
  ina219.begin();
  interpreter.AllocateTensors();
}

void loop() {
  float voltage = ina219.getBusVoltage_V();
  float current = ina219.getCurrent_mA();
  float temp = 25.0;  // Replace with DS18B20 reading

  // Normalize inputs (same as Python scaler)
  float inputs[3] = {voltage / 5.0, current / 10.0, temp / 50.0};  # Adjust scaling factors

  // Run inference
  TfLiteTensor* input = interpreter.input(0);
  for (int i = 0; i < 3; i++) input->data.f[i] = inputs[i];
  interpreter.Invoke();
  float soc = interpreter.output(0)->data.f[0] * 100.0;  # Scale back to 0-100%

  Serial.print("SOC: "); Serial.println(soc);
  delay(1000);
}