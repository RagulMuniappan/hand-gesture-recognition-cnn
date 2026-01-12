## Hand Gesture Number Recognition

A deep learning–based computer vision system that recognizes the **number shown by a human hand gesture** and predicts the corresponding digit in real time or from images.  
This project uses a **Convolutional Neural Network (CNN)** trained on grayscale hand gesture images to classify digits **0–9**, along with an **unknown** class.

---

### Project Overview

Hand gesture recognition plays a key role in human–computer interaction.  
This project focuses on identifying **numeric hand gestures** using deep learning and image processing techniques.

The system:
- Takes a hand gesture image as input
- Preprocesses it into grayscale format
- Uses a trained CNN model to predict the digit being shown

---

### Features

- Recognizes digits **0–9**
- Handles **unknown / invalid gestures**
- CNN-based deep learning model
- Grayscale image processing
- Batch image prediction support
- Trained model saved for reuse

---

### Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- CNN (Convolutional Neural Network)

---

### Model Architecture

- Input: **256 × 256 × 1 (Grayscale Image)**
- Convolution + MaxPooling layers
- Fully connected dense layers
- Softmax output layer with **11 classes**
  - Digits: `0–9`
  - `unknown`

---

### Project Structure

Hand-Gesture-Number-Recognition/
│
├── dataset/
│ ├── train/
│ └── test/
│
├── train.py # Model training script
├── main.py # Prediction script
├── best_model.keras # Best saved model
├── final_hand_model.keras# Final trained model
├── train_history.pkl # Training history
├── README.md


---

### How to Run the Project

#### 1️⃣ Clone the Repository

    ```bash
    git clone https://github.com/your-username/hand-gesture-number-recognition.git
    cd hand-gesture-number-recognition

#### 2️⃣ Install Dependencies

    ```bash
    pip install tensorflow keras opencv-python numpy

#### 3️⃣ Train the Model

    ```bash
    python train.py

#### 4️⃣ Run Prediction

  Place test images inside the check/ folder
  Run:

    ```bash
    python main.py

The program will display:

Predicted digit

Confidence score

---

### Output Example

Image: sample1.jpg
Prediction: 5
Confidence: 0.97

---

### Future Enhancements

- Real-time webcam prediction 
- MediaPipe-based hand tracking
- Mobile / web deployment
- Expand to alphabets or gestures
- Improve accuracy with deeper CNNs

---

### License

This project is intended for educational purposes.
