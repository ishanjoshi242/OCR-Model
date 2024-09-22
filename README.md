# OCR Model Conversion: GPU to CPU Optimization

## Project Overview

This project demonstrates the conversion of an OCR (Optical Character Recognition) model, originally designed to run on a GPU, to a CPU-optimized version while maintaining or improving both accuracy and inference speed (FPS). OCR models are traditionally resource-intensive, leveraging GPUs for parallel computation. However, many applications require deployment on systems without GPUs, necessitating efficient CPU-based solutions. This project focuses on achieving this conversion, comparing the performance of both the CPU and GPU versions, and providing insights into the challenges and optimizations involved.

---

## Objectives

1. **Model Conversion**: Convert the GPU-based OCR model to run efficiently on a CPU, ensuring minimal accuracy drop while maintaining or improving FPS.
2. **Performance Evaluation**: Conduct a comparative analysis of the GPU and CPU models based on accuracy, speed, and resource usage.
3. **Video Processing**: Use both models to process a provided video file and demonstrate their performance in terms of real-time text recognition.
4. **Optimization Techniques**: Explore methods for optimizing the OCR model for CPU, including reducing computational complexity and fine-tuning architecture.

---

## Key Features

- **Efficient Model Conversion**: The project provides a method to load a model trained on GPU and optimize it to run on CPU using techniques such as reducing CNN and LSTM layer complexities.
  
- **Performance Benchmarking**: The project compares the **frames per second (FPS)** and **accuracy** of the OCR model on both CPU and GPU, offering detailed metrics on how well the conversion was achieved.
  
- **Real-Time Video Processing**: A demonstration of the OCR model processing frames from a video in real-time. The output video showcases a side-by-side comparison of the performance between the GPU and CPU implementations.
  
- **Challenges & Solutions**: The README and demo video explain the hurdles faced during the conversion process and provide innovative solutions to handle potential accuracy loss or speed drops when switching from GPU to CPU.

---

## Steps in the Process

### 1. **OCR Model Architecture**
The OCR model is based on a CRNN (Convolutional Recurrent Neural Network) architecture:

- **CNN Layers**: Extract visual features from input images.
- **RNN Layers**: Bi-directional LSTM layers learn the sequence dependencies in the extracted features.
- **Dense Layer**: Outputs the final text predictions based on softmax activation.

### 2. **Optimizing for CPU**
Running deep learning models on CPUs introduces challenges due to reduced parallelism compared to GPUs. To address this:
- **Layer Pruning**: Reduced the number of filters in CNN layers and units in LSTM layers to make the model lighter for CPU inference.
- **Efficient Data Preprocessing**: Optimized the image preprocessing pipeline for CPU without compromising input quality.
  
```python
with tf.device('/cpu:0'):
    model = build_crnn_model(input_shape, num_classes)
```

### 3. **Performance Comparison**
The CPU and GPU models' **FPS** and **accuracy** are measured using a sample dataset. We calculated the time taken for both models to process frames and compared their speed. Despite a lower computational power, the CPU model aims to match the performance of the GPU-based model.

```python
def measure_fps(model, images, num_samples=100):
    start_time = time.time()
    for i in range(num_samples):
        _ = model.predict(np.expand_dims(images[i], axis=0))
    end_time = time.time()
    fps = num_samples / (end_time - start_time)
    return fps
```

### 4. **Video Inference and Output**
A provided video file is processed by both the GPU and CPU models, and the real-time OCR output is overlaid on the frames. The processed video compares inference speeds and accuracy in text recognition.

```python
def process_video_with_ocr(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            img = preprocess_image(frame)
            text = ocr_inference(model, img)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
```

---

## Results and Insights

- **Accuracy**: The CPU version of the OCR model maintained a high level of accuracy compared to the GPU version, with minimal loss.
- **FPS**: The FPS for the CPU model was optimized to match real-time performance, although slightly lower than the GPU model due to hardware limitations.
- **Resource Utilization**: CPU resource utilization was significantly higher, but adjustments to the model architecture ensured efficient usage.

---

## How to Run

1. Clone the repository
2. Install the required packages
3. Run the CPU-optimized model
4. Run the performance benchmark
---

## Future Work

- Further reduce model complexity to improve CPU efficiency.
- Implement hybrid CPU-GPU models to balance performance and resource usage.
- Explore deployment on edge devices with limited computational power.

---
