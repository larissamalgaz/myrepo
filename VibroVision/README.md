# VibroVision

VibroVision is an intelligent user interface application designed to provide haptic feedback based on object detection in the classic game Super Mario Bros. The application uses a custom-trained YOLOv8 model to detect objects in the game and triggers corresponding haptic feedback using a bHaptics TactSuit.

## Course Information

- **Course**: COMP 537 Intelligent User Interfaces
- **Term**: Spring 2024
- **Institution**: Koç University

## Project Structure

```
VibroVision/
├── assets/
│   ├── games/
│   ├── icons/
│   ├── models/
│   │   └── yolov8m_vibrovision.pt
│   └── tacts/
├── main.py
└── requirements.txt
```

## Requirements

Ensure you have the following dependencies installed. You can install them using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy==1.26.4
opencv-python==4.9.0.80
pyautogui==0.9.54
websocket-client==1.8.0
ultralytics==8.2.27
PyQt5==5.15.10
```

## Usage

### Running the Application

1. **Navigate to the project directory**:
    ```sh
    cd /path/to/VibroVision
    ```

2. **Run the application**:
    ```sh
    python main.py
    ```

### Application Overview

The application provides a graphical user interface (GUI) with several tabs:

1. **Library Tab**: Select a game (currently supports Super Mario Bros).
2. **Interactions Tab**: Configure haptic feedback interactions based on in-game events.
3. **Capture Tab**: Select a window to capture and process screenshots.
4. **Settings Tab**: Adjust application settings such as haptic feedback intensity.

### Core Functionalities

- **Object Detection**: Uses a custom-trained YOLOv8 model to detect objects in the game.
- **Haptic Feedback**: Sends haptic feedback signals to the bHaptics TactSuit based on detected interactions.
- **Screenshot Capture**: Captures and processes screenshots from the selected game window.

## Code Overview

### BhapticsManager Class

Manages the connection and communication with the bHaptics TactSuit, including registering and submitting haptic feedback.

### YOLOModel Class

Handles object detection using a custom-trained YOLOv8 model. It predicts objects in the provided screenshots and returns their bounding boxes, confidence scores, and class labels.

### ScreenCapture Class

Handles capturing screenshots from a specified window.

### VibroVisionApp Class

The main application class that initializes the GUI, handles user interactions, and integrates all functionalities (object detection, haptic feedback, and screenshot capture).

## Code Example

Here is an example of the main components of the code:

```python
# Standard library imports
import os
import sys
import time
import json
import socket
import threading

# Third-party library imports
import numpy as np
import cv2
import pyautogui
import pygetwindow as gw

# Enum for defining enumerations
from enum import Enum

# WebSocket for real-time communication
from websocket import create_connection, WebSocket

# YOLO from the ultralytics package for object detection
from ultralytics import YOLO

# PyQt5 imports for GUI components
from PyQt5.QtCore import QTimer, pyqtSlot, Qt, QSettings, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QSplitter, QTabWidget, QSlider, QGridLayout, QToolButton
)

# BhapticsManager class to manage haptic feedback
class BhapticsManager:
    ...
    
# YOLOModel class to handle object detection
class YOLOModel:
    ...

# ScreenCapture class to handle window capturing
class ScreenCapture:
    ...

# VibroVisionApp class to handle the application UI and logic
class VibroVisionApp(QMainWindow):
    ...

if __name__ == "__main__":
    # Initialize the application
    app = QApplication(sys.argv)

    # Initialize the YOLO model
    yolo_model = YOLOModel("path/to/yolov8m_vibrovision.pt")

    # Initialize and register the BhapticsManager with the haptic feedback assets
    bhaptics_manager = BhapticsManager()
    bhaptics_manager.register("1", "assets/tacts/coin.tact")
    ...
    
    # Initialize and show the main application window
    main_window = VibroVisionApp(yolo_model, bhaptics_manager)
    main_window.show()

    # Start the application event loop
    sys.exit(app.exec_())
```