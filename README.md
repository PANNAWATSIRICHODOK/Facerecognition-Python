# Face Recognition System

This project is a face recognition system using a pre-trained model from the `face_recognition` library. It detects faces from a webcam feed, logs recognized individuals with timestamps, and provides a GUI for starting the recognition process and displaying the recognized names.

## Features

- Real-time face recognition using a webcam.
- Logs recognized names and timestamps.
- GUI for starting face recognition and displaying recognized names.
- Saves log data to a CSV file.

## Installation

Ensure you have a folder named images in the project directory with the images of the people you want to recognize. Each image file should be named after the person it represents (e.g., john_doe.jpg). The system uses these images to learn the faces it needs to recognize.

To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required libraries using the following commands:

```bash
pip install opencv-python
pip install numpy
pip install pandas
pip install dlib
pip install face_recognition
