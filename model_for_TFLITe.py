# Convert to TFLite (quantized for ESP32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()

# Save for ESP32
with open('soc_lstm.tflite', 'wb') as f:
    f.write(tflite_model)