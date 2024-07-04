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
import torch
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
    
    # Enum class to define haptic positions
    class BhapticsPosition(Enum):
        Vest = "Vest"
        VestFront = "VestFront"
        VestBack = "VestBack"
        ForearmL = "ForearmL"
        ForearmR = "ForearmR"
        Head = "Head"
        HandL = "HandL"
        HandR = "HandR"
        FootL = "FootL"
        FootR = "FootR"
        GloveL = "GloveL"
        GloveR = "GloveR"
        active_keys = set([])
        connected_positions = set([])

    # Custom WebSocket class to handle incoming frames
    class WebSocketReceiver(WebSocket):
        def recv_frame(self):
            """Receive a frame from the WebSocket server."""
            frame = super().recv_frame()
            try:
                frame_obj = json.loads(frame.data)
                self.active_keys = set(frame_obj['ActiveKeys'])
                self.connected_positions = set(frame_obj['ConnectedPositions'])
            except json.JSONDecodeError:
                print('Failed to decode JSON frame')
            return frame

    def __init__(self):
        """Initialize the BhapticsManager class."""
        self.active_keys = set([])
        self.connected_positions = set([])
        self.ws = None
        self.connected = False
        self.initialize()

    def initialize(self):
        """Initialize the WebSocket connection."""
        try:
            self.ws = create_connection(
                "ws://localhost:15881/v2/feedbacks",
                sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),),
                class_=self.WebSocketReceiver
            )
            threading.Thread(target=self.thread_function).start()
            self.connected = True
        except Exception as e:
            print(f"Couldn't connect: {e}")
            return

    def thread_function(self):
        """Thread function to keep the WebSocket connection alive."""
        while True:
            if self.ws is not None:
                self.ws.recv_frame()

    def destroy(self):
        """Close the WebSocket connection."""
        if self.ws is not None:
            self.ws.close()

    def is_playing(self):
        """Check if any feedback is currently playing."""
        return len(self.active_keys) > 0

    def is_playing_key(self, key):
        """Check if the key is currently playing feedback."""
        return key in self.active_keys

    def is_device_connected(self, position):
        """Check if the device at the given position is connected."""
        return position in self.connected_positions

    def register(self, key, file_directory):
        """Register a key with the given file directory."""
        with open(file_directory, 'r') as f:
            data = json.load(f)
        
        project = data["project"]
        layout = project["layout"]
        tracks = project["tracks"]
        request = {
            "Register": [{
                "Key": key,
                "Project": {
                    "Tracks": tracks,
                    "Layout": layout
                }
            }]
        }
        self.__submit(json.dumps(request))

    def submit_registered(self, key):
        """Submit a registered key."""
        request = {
            "Submit": [{
                "Type": "key",
                "Key": key,
            }]
        }
        self.__submit(json.dumps(request))

    def submit_registered_with_option(self, key, alt_key, scale_option, rotation_option):
        """Submit a registered key with options."""
        request = {
            "Submit": [{
                "Type": "key",
                "Key": key,
                "Parameters": {
                    "altKey": alt_key,
                    "rotationOption": rotation_option,
                    "scaleOption": scale_option,
                }
            }]
        }
        self.__submit(json.dumps(request))

    def submit(self, key, frame):
        """Submit a key with a frame."""
        request = {
            "Submit": [{
                "Type": "frame",
                "Key": key,
                "Frame": frame
            }]
        }
        self.__submit(json.dumps(request))

    def __submit(self, json_str):
        """Send the JSON string to the WebSocket server."""
        if self.ws is not None:
            self.ws.send(json_str)

#buraya kadar









# YOLOModel class to handle object detection
class YOLOModel:
    def __init__(self, model_path):
        """
        Initialize the YOLO model.

        Parameters:
        model_path (str): Path to the pre-trained YOLO model.
        """
        self.model = YOLO(model_path)  # Load the YOLO model
        self.previous_detections = []  # List to store previous detections
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the device (GPU or CPU)
        self.model.to(self.device)  # Move the model to the appropriate device

    def predict(self, image):
        """
        Make predictions on the input image.

        Parameters:
        image (ndarray): Input image for object detection.

        Returns:
        tuple: Arrays of predictions (bounding boxes), confidences, and classes.
        """
        results = self.model.predict(image, conf=0.4, verbose=False)  # Run the prediction with a confidence threshold

        if results and results[0].boxes:
            # Extract bounding boxes, confidences, and class labels
            predictions = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
        else:
            predictions, confidences, classes = [], [], []  # Return empty arrays if no detections

        return np.array(predictions), np.array(confidences), np.array(classes)

# ScreenCapture class to handle window capturing
class ScreenCapture:
    def list_windows(self):
        """
        List all the windows available on the system.

        Returns:
        list: Titles of all available windows except the tool window.
        """
        tool_window_title = "Window Screenshot Tool"
        return [win.title for win in gw.getAllWindows() if win.title and win.title != tool_window_title]

    def take_screenshot(self, window_title):
        """
        Take a screenshot of the specified window.

        Parameters:
        window_title (str): Title of the window to capture.

        Returns:
        Image: Screenshot of the specified window, or None if the window is not found.
        """
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            window = windows[0]
            was_minimized = window.isMinimized
            if was_minimized:
                window.maximize()
                time.sleep(0.2)  # Wait for the window to maximize

            # Get the window boundaries
            left, top = window.left, window.top
            right, bottom = window.left + window.width, window.top + window.height

            # Capture the screenshot of the specified region
            screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))

            return screenshot
        return None

# VibroVisionApp class to handle the application UI and logic
class VibroVisionApp(QMainWindow):
    def __init__(self, model, bhaptics_manager):
        """
        Initialize the VibroVisionApp class.

        Parameters:
        model (YOLOModel): The YOLO model for object detection.
        bhaptics_manager (BhapticsManager): The manager for haptic feedback.
        """
        super().__init__()
        self.model = model
        self.bhaptics_manager = bhaptics_manager
        self.screen_capture = ScreenCapture()
        self.settings = QSettings("MyCompany", "VibroVision")
        self.feedback_options = ["1", "2", "3", "4", "5", "6", "7", "8"]
        self.interactions = [
            "Mario collected a coin",
            "Mario collected a fire-flower",
            "Mario collected a starman",
            "Mario collected a super-mushroom",
            "Mario interacted with an enemy",
            "Mario finished the level",
            "Mario encountered the Mushroom Retainer",
            "Mario encountered the Princess"
        ]
        self.interaction_feedback = {}
        self.intensity = self.settings.value("intensity", 1.0, type=float)
        self.game_selected = None  # Attribute to store the selected game
        self.initUI()
        self.load_interactions()  # Load interactions after UI is initialized
        self.load_settings()  # Load settings after UI is initialized
        self.previous_objects = []
        self.current_objects = []
        self.mario_box = None
        self.previous_time = None
        self.speeds = {}
        self.flag_timer = 0
        self.retainer_timer = 0
        self.princess_timer = 0
        self.coin_timer = 0
        self.powerup_timer = 0
        self.enemy_timer = 0
        self.timer = 10
        self.feedback = []

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("VibroVision")
        self.setWindowIcon(QIcon('assets/icons/app_icon.png'))
        self.setGeometry(100, 100, 800, 500)  # Adjust the initial size as needed
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QLabel {
                color: #000000;
            }
            QComboBox {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0; /* Change to desired hover color */
                border: 1px solid #a0a0a0; /* Optional: change border color on hover */
            }
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
            }
        """)

        self.tabs = QTabWidget()
        self.library_tab = QWidget()
        self.interactions_tab = QWidget()
        self.capture_tab = QWidget()
        self.settings_tab = QWidget()

        self.tabs.addTab(self.library_tab, "Library")
        self.tabs.addTab(self.interactions_tab, "Interactions")
        self.tabs.addTab(self.capture_tab, "Capture")
        self.tabs.addTab(self.settings_tab, "Settings")

        self.init_library_tab()
        self.init_interactions_tab()
        self.init_capture_tab()
        self.init_settings_tab()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.process_screenshot)

    def update_status(self, message, success=True):
        """Update the status label with a message."""
        color = "green" if success else "red"
        self.status_label.setStyleSheet(f"color: {color}")
        self.status_label.setText(message)

    def init_library_tab(self):
        """Initialize the library tab."""
        layout = QGridLayout()
        games = [
            ("Mario", "assets/games/mario_poster.png"),
            # Add more games here
        ]

        for i, (game_name, game_image_path) in enumerate(games):
            button = QToolButton(self)
            button.setIcon(QIcon(game_image_path))
            button.setIconSize(QSize(250, 361))  # Adjust to match the poster aspect ratio
            button.setFixedSize(250, 361)  # Set a fixed size to ensure consistent button size
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)  # Remove text
            button.setStyleSheet("""
                QToolButton {
                    margin: 5px; 
                    background-color: transparent; 
                    border: none; 
                }
                QToolButton::hover QLabel {
                    background-color: rgba(0, 0, 0, 128);
                    color: white;
                    font-size: 24px;
                    qproperty-alignment: AlignCenter;
                }
            """)  # Add margin for spacing and transparent background
            button.clicked.connect(lambda checked, game=game_name: self.on_game_selected(game))

            hover_label = QLabel(game_name, button)
            hover_label.setFixedSize(250, 361)
            hover_label.setAlignment(Qt.AlignCenter)
            hover_label.setStyleSheet("QLabel { background-color: transparent; color: transparent; }")

            layout.addWidget(button, i // 4, i % 4)  # 4 buttons per row

        self.library_tab.setLayout(layout)

    def on_game_selected(self, game):
        """Handle game selection."""
        self.game_selected = game
        self.tabs.setCurrentIndex(1)  # Switch to the Interactions tab

    def init_interactions_tab(self):
        """Initialize the interactions tab."""
        layout = QVBoxLayout()

        if self.bhaptics_manager.connected:
            self.connection_status_label = QLabel("Haptic vest is connected", self)
            self.connection_status_label.setStyleSheet("color: green;")
        else:
            self.connection_status_label = QLabel("Haptic vest is not connected", self)
            self.connection_status_label.setStyleSheet("color: red;")
        self.connection_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.connection_status_label)

        connect_button = QPushButton("Connect Haptic Vest", self)
        connect_button.setIcon(QIcon('assets/icons/connect_icon.png'))
        connect_button.clicked.connect(self.connect_haptic_vest)
        layout.addWidget(connect_button)

        for index, message in enumerate(self.interactions):
            label = QLabel(message, self)
            combo_box = QComboBox(self)
            combo_box.addItems(self.feedback_options)
            combo_box.setCurrentText(str(index + 1))  # Default feedback option

            play_button = QPushButton(self)
            play_button.setIcon(QIcon('assets/icons/start_icon.png'))
            play_button.setFixedSize(24, 24)  # Set a fixed size for the play button
            play_button.clicked.connect(lambda checked, combo_box=combo_box: self.play_interaction(combo_box.currentText()))

            interaction_layout = QHBoxLayout()
            interaction_layout.addWidget(label)
            interaction_layout.addWidget(combo_box)
            interaction_layout.addWidget(play_button)

            layout.addLayout(interaction_layout)
            self.interaction_feedback[index] = combo_box

        reset_button = QPushButton("Reset Interactions", self)
        reset_button.setIcon(QIcon('assets/icons/refresh_icon.png'))
        reset_button.clicked.connect(self.reset_interactions)
        layout.addWidget(reset_button)

        apply_button = QPushButton("Apply Interactions", self)
        apply_button.setIcon(QIcon('assets/icons/apply_icon.png'))
        apply_button.clicked.connect(self.apply_interactions)
        layout.addWidget(apply_button)

        layout.addStretch(1)

        self.interactions_tab.setLayout(layout)

    def load_interactions(self):
        """Load interactions from settings."""
        for index, combo_box in self.interaction_feedback.items():
            saved_value = self.settings.value(f"interaction_{index}", str(index + 1))
            combo_box.setCurrentText(saved_value)

    def reset_interactions(self):
        """Reset interactions to default values."""
        for index, combo_box in self.interaction_feedback.items():
            combo_box.setCurrentText(str(index + 1))  # Reset to default value
        self.show_message("Interactions reset to default")

    def apply_interactions(self):
        """Apply interactions and save to settings."""
        for index, combo_box in self.interaction_feedback.items():
            self.settings.setValue(f"interaction_{index}", combo_box.currentText())
        self.show_message("Interactions applied")

    def connect_haptic_vest(self):
        """Connect to the haptic vest."""
        self.bhaptics_manager.initialize()
        if self.bhaptics_manager.connected:
            self.connection_status_label.setText("Haptic vest is connected")
            self.connection_status_label.setStyleSheet("color: green;")  # Green color for connected
        else:
            self.connection_status_label.setText("Haptic vest is not connected")
            self.connection_status_label.setStyleSheet("color: red;")  # Red color for not connected

    def play_interaction(self, interaction):
        """Play the selected interaction."""
        if not self.bhaptics_manager.connected:
            self.show_message("Vest is not connected", error=True)
            return
        self.bhaptics_manager.submit_registered_with_option(
            interaction, "alt2",
            scale_option={"intensity": self.intensity, "duration": 1},
            rotation_option={"offsetAngleX": 0, "offsetY": 0}
        )

    def show_message(self, message, error=False):
        """Show a message in the status label."""
        color = "red" if error else "green"
        self.connection_status_label.setStyleSheet(f"color: {color}")
        self.connection_status_label.setText(message)

    def init_capture_tab(self):
        """Initialize the capture tab."""
        self.label = QLabel("Select a window", self)

        self.combo_box = QComboBox(self)
        self.combo_box.addItems(self.screen_capture.list_windows())

        self.refresh_button = QPushButton("Refresh List", self)
        self.refresh_button.setIcon(QIcon('assets/icons/refresh_icon.png'))
        self.refresh_button.clicked.connect(self.refresh_window_list)

        self.start_button = QPushButton("Start Capture", self)
        self.start_button.setIcon(QIcon('assets/icons/start_icon.png'))
        self.start_button.clicked.connect(self.start_capture)

        self.stop_button = QPushButton("Stop Capture", self)
        self.stop_button.setIcon(QIcon('assets/icons/stop_icon.png'))
        self.stop_button.clicked.connect(self.stop_capture)

        self.clear_button = QPushButton("Clear Log", self)
        self.clear_button.setIcon(QIcon('assets/icons/clear_icon.png'))
        self.clear_button.clicked.connect(self.clear_log)

        self.status_label = QLabel("", self)

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)

        layout = QVBoxLayout()
        label_layout = QHBoxLayout()
        label_layout.addWidget(self.label)
        label_layout.addWidget(self.combo_box)
        layout.addLayout(label_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)

        # Create splitter for dynamic resizing
        splitter = QSplitter()
        upper_widget = QWidget()
        upper_widget.setLayout(layout)
        splitter.addWidget(upper_widget)
        splitter.addWidget(self.log_text)
        splitter.setOrientation(Qt.Vertical)

        capture_layout = QVBoxLayout()
        capture_layout.addWidget(splitter)
        self.capture_tab.setLayout(capture_layout)

    def init_settings_tab(self):
        """Initialize the settings tab."""
        layout = QVBoxLayout()

        intensity_layout = QHBoxLayout()
        intensity_label = QLabel(f"Intensity: {self.intensity:.2f}", self)
        self.intensity_slider = QSlider(Qt.Horizontal, self)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(400)
        self.intensity_slider.setValue(int(self.intensity * 100))
        self.intensity_slider.setTickInterval(5)
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.valueChanged.connect(lambda: self.on_intensity_changed(intensity_label))

        intensity_layout.addWidget(intensity_label)
        intensity_layout.addWidget(self.intensity_slider)

        layout.addLayout(intensity_layout)

        reset_button = QPushButton("Reset Intensity", self)
        reset_button.setIcon(QIcon('assets/icons/refresh_icon.png'))
        reset_button.clicked.connect(self.reset_to_default)
        layout.addWidget(reset_button)

        layout.addStretch(1)  # Add stretch to push items to the top

        self.settings_tab.setLayout(layout)

    def on_intensity_changed(self, intensity_label):
        """Handle intensity slider change."""
        self.intensity = self.intensity_slider.value() / 100.0
        self.settings.setValue("intensity", self.intensity)
        intensity_label.setText(f"Intensity: {self.intensity:.2f}")

    def on_library_changed(self):
        """Handle library change event."""
        game = self.game_selected
        # Enable or disable the interactions and capture tab based on game selection
        self.tabs.setTabEnabled(1, game != "")
        self.tabs.setTabEnabled(2, game != "")

    def reset_to_default(self):
        """Reset intensity to default value."""
        self.intensity_slider.setValue(100)
        self.settings.setValue("intensity", 1.0)
        self.intensity = 1.0

    def load_settings(self):
        """Load settings from QSettings."""
        self.intensity = self.settings.value("intensity", 1.0, type=float)
        self.intensity_slider.setValue(int(self.intensity * 100))

    @pyqtSlot()
    def refresh_window_list(self):
        """Refresh the list of available windows."""
        self.combo_box.clear()
        self.combo_box.addItems(self.screen_capture.list_windows())
        self.update_status("Window list refreshed", success=True)

    def log_to_gui(self, message):
        """Log a message to the GUI."""
        self.log_text.append(message)

    @pyqtSlot()
    def clear_log(self):
        """Clear the log text."""
        self.log_text.clear()

    def process_screenshot(self):
        """Process a screenshot of the selected window."""
        screenshot = self.screen_capture.take_screenshot(self.combo_box.currentText())
        if screenshot:
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            screenshot_resized = cv2.resize(screenshot_cv, (640, 640), interpolation=cv2.INTER_LINEAR)
            predictions, confidences, classes = self.model.predict(screenshot_resized)
            self.current_objects = [(*pred, cls) for pred, cls in zip(predictions, classes)]

            self.mario_box = None
            for obj in self.current_objects:
                if obj[-1] == 14:
                    self.mario_box = obj

            current_time = time.time()
            if self.previous_time is not None:
                self.calculate_speeds(current_time - self.previous_time)

            self.previous_time = current_time
            self.detect_interactions()
            self.previous_objects = self.current_objects.copy()

    def calculate_speeds(self, time_diff):
        """Calculate speeds of detected objects."""
        if time_diff <= 0:
            return

        speeds = {}
        for current_obj in self.current_objects:
            cx1, cy1, cx2, cy2, cls = current_obj
            speed = (0, 0)

            for previous_obj in self.previous_objects:
                px1, py1, px2, py2, pcls = previous_obj
                if cls == pcls:
                    dx = (cx1 + cx2) / 2 - (px1 + px2) / 2
                    dy = (cy1 + cy2) / 2 - (py1 + py2) / 2
                    speed = (dx / time_diff, dy / time_diff)
                    break

            speeds[int(cls)] = speed

        self.speeds = speeds

    def detect_interactions(self):
        """Detect interactions between Mario and the objects in the scene."""
        if not self.mario_box:
            return

        mario_x1, mario_y1, mario_x2, mario_y2, _ = self.mario_box
        mario_width = mario_x2 - mario_x1
        mario_height = mario_y2 - mario_y1
        proximity_threshold = 1.5 * max(mario_width, mario_height)

        enemies = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 16, 17, 21, 22, 25]

        for obj in self.current_objects:
            x1, y1, x2, y2, cls = obj
            if cls == 8:  # FLAG
                if self.flag_timer <= 0 and self.will_collide(mario_x1, mario_y1, mario_x2, mario_y2, x1, y1, x2, y2, cls):
                    self.log_to_gui(self.interactions[5])
                    self.bhaptics_manager.submit_registered_with_option(
                        self.feedback[5][0], "alt2",
                        scale_option={"intensity": self.intensity, "duration": 1},
                        rotation_option={"offsetAngleX": 0, "offsetY": 0}
                    )
                    self.flag_timer = self.timer * 100
            if cls == 15:  # MUSHROOM RETAINER
                if self.retainer_timer <= 0 and self.check_appearance_disappearance(15):
                    self.log_to_gui(self.interactions[6])
                    self.bhaptics_manager.submit_registered_with_option(
                        self.feedback[6][0], "alt2",
                        scale_option={"intensity": self.intensity, "duration": 1},
                        rotation_option={"offsetAngleX": 0, "offsetY": 0}
                    )
                    self.retainer_timer = self.timer * 100
            if cls == 18:  # PRINCESS
                if self.princess_timer <= 0 and self.check_appearance_disappearance(18):
                    self.log_to_gui(self.interactions[7])
                    self.bhaptics_manager.submit_registered_with_option(
                        self.feedback[7][0], "alt2",
                        scale_option={"intensity": self.intensity, "duration": 1},
                        rotation_option={"offsetAngleX": 0, "offsetY": 0}
                    )
                    self.princess_timer = self.timer * 100

        if self.coin_timer <= 0 and self.check_appearance_disappearance(5) and (self.was_near_mario(proximity_threshold, 5) or self.is_above_mario(proximity_threshold, 5)):
            self.log_to_gui(self.interactions[0])
            self.bhaptics_manager.submit_registered_with_option(
                self.feedback[0][0], "alt2",
                scale_option={"intensity": self.intensity, "duration": 1},
                rotation_option={"offsetAngleX": 0, "offsetY": 0}
            )
            self.coin_timer = self.timer
        if self.powerup_timer <= 0 and self.check_appearance_disappearance(7) and (self.was_near_mario(proximity_threshold, 7) or self.is_near_mario(proximity_threshold, 7)):
            self.bhaptics_manager.submit_registered_with_option(
                self.feedback[1][0], "alt2",
                scale_option={"intensity": self.intensity, "duration": 1},
                rotation_option={"offsetAngleX": 0, "offsetY": 0}
            )
            self.log_to_gui(self.interactions[1])
            self.powerup_timer = self.timer
        if self.powerup_timer <= 0 and self.check_appearance_disappearance(23) and (self.was_near_mario(proximity_threshold, 23) or self.is_near_mario(proximity_threshold, 23)):
            self.bhaptics_manager.submit_registered_with_option(
                self.feedback[2][0], "alt2",
                scale_option={"intensity": self.intensity, "duration": 1},
                rotation_option={"offsetAngleX": 0, "offsetY": 0}
            )
            self.log_to_gui(self.interactions[2])
            self.powerup_timer = self.timer
        if self.powerup_timer <= 0 and self.check_appearance_disappearance(24) and (self.was_near_mario(proximity_threshold, 24) or self.is_near_mario(proximity_threshold, 24)):
            self.bhaptics_manager.submit_registered_with_option(
                self.feedback[3][0], "alt2",
                scale_option={"intensity": self.intensity, "duration": 1},
                rotation_option={"offsetAngleX": 0, "offsetY": 0}
            )
            self.log_to_gui(self.interactions[3])
            self.powerup_timer = self.timer
        if self.enemy_timer <= 0 and self.check_appearance_disappearance(enemies) and (self.was_near_mario(proximity_threshold, enemies) or self.is_near_mario(proximity_threshold, enemies)):
            self.log_to_gui(self.interactions[4])
            self.bhaptics_manager.submit_registered_with_option(
                self.feedback[4][0], "alt2",
                scale_option={"intensity": self.intensity, "duration": 1},
                rotation_option={"offsetAngleX": 0, "offsetY": 0}
            )
            self.enemy_timer = self.timer

        # Decrease timers
        self.decrease_timers()

    def decrease_timers(self):
        """Decrease interaction timers."""
        if self.flag_timer > 0:
            self.flag_timer -= 1
        if self.retainer_timer > 0:
            self.retainer_timer -= 1
        if self.princess_timer > 0:
            self.princess_timer -= 1
        if self.coin_timer > 0:
            self.coin_timer -= 1
        if self.powerup_timer > 0:
            self.powerup_timer -= 1
        if self.enemy_timer > 0:
            self.enemy_timer -= 1

    def will_collide(self, mx1, my1, mx2, my2, ox1, oy1, ox2, oy2, cls):
        """Predict if Mario will collide with an object in the next frame."""
        mario_speed = self.speeds.get(14, (0, 0))
        object_speed = self.speeds.get(cls, (0, 0))

        future_mx1 = mx1 + mario_speed[0]
        future_my1 = my1 + mario_speed[1]
        future_mx2 = mx2 + mario_speed[0]
        future_my2 = my2 + mario_speed[1]

        future_ox1 = ox1 + object_speed[0]
        future_oy1 = oy1 + object_speed[1]
        future_ox2 = ox2 + object_speed[0]
        future_oy2 = oy2 + object_speed[1]

        return future_mx1 < future_ox2 and future_mx2 > future_ox1 and future_my1 < future_oy2 and future_my2 > future_oy1

    def check_appearance_disappearance(self, classes):
        """Check if any objects of specified classes appeared or disappeared."""
        if isinstance(classes, int):
            classes = [classes]
        for current_cls in classes:
            current_count = sum(1 for obj in self.current_objects if obj[-1] == current_cls)
            previous_count = sum(1 for obj in self.previous_objects if obj[-1] == current_cls)
            if current_count != previous_count:
                return True
        return False

    def is_above_mario(self, proximity_threshold, classes):
        """Check if any objects of specified classes are above Mario."""
        if isinstance(classes, int):
            classes = [classes]
        if not self.mario_box:
            return False
        mario_x1, mario_y1, mario_x2, mario_y2, _ = self.mario_box
        for obj in self.current_objects:
            x1, y1, x2, y2, cls = obj
            if cls in classes and y2 - mario_y1 < proximity_threshold * 3 and abs(mario_x1 - x1) < proximity_threshold and abs(mario_x2 - x2) < proximity_threshold:
                return True
        return False

    def is_near_mario(self, proximity_threshold, classes):
        """Check if any objects of specified classes are near Mario."""
        if isinstance(classes, int):
            classes = [classes]
        if not self.mario_box:
            return False
        mario_x1, mario_y1, mario_x2, mario_y2, _ = self.mario_box
        for obj in self.current_objects:
            x1, y1, x2, y2, cls = obj
            if cls in classes:
                if abs(mario_x1 - x1) < proximity_threshold and abs(mario_y1 - y1) < proximity_threshold and abs(mario_x2 - x2) < proximity_threshold and abs(mario_y2 - y2) < proximity_threshold:
                    return True
        return False

    def was_near_mario(self, proximity_threshold, classes):
        """Check if any objects of specified classes were near Mario."""
        if isinstance(classes, int):
            classes = [classes]
        if not self.mario_box:
            return False
        mario_x1, mario_y1, mario_x2, mario_y2, _ = self.mario_box
        for obj in self.previous_objects:
            x1, y1, x2, y2, cls = obj
            if cls in classes:
                if abs(mario_x1 - x1) < proximity_threshold and abs(mario_y1 - y1) < proximity_threshold and abs(mario_x2 - x2) < proximity_threshold and abs(mario_y2 - y2) < proximity_threshold:
                    return True
        return False

    @pyqtSlot()
    def start_capture(self):
        """Start the screenshot capture process."""
        self.feedback = [(self.interaction_feedback[index].currentText(), self.interactions[index]) for index in range(len(self.interactions))]
        self.capture_timer.start()
        self.update_status("Capturing is started", success=True)

    @pyqtSlot()
    def stop_capture(self):
        """Stop the screenshot capture process."""
        self.capture_timer.stop()
        self.update_status("Capturing is stopped", success=True)

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full paths to the haptic feedback assets
    coin_tact_path = os.path.join(script_dir, "assets", "tacts", "coin.tact")
    fire_flower_tact_path = os.path.join(script_dir, "assets", "tacts", "fire_flower.tact")
    starman_tact_path = os.path.join(script_dir, "assets", "tacts", "starman.tact")
    super_mushroom_tact_path = os.path.join(script_dir, "assets", "tacts", "super_mushroom.tact")
    enemy_tact_path = os.path.join(script_dir, "assets", "tacts", "enemy.tact")
    flag_tact_path = os.path.join(script_dir, "assets", "tacts", "flag.tact")
    level_end_tact_path = os.path.join(script_dir, "assets", "tacts", "level_end.tact")
    princess_tact_path = os.path.join(script_dir, "assets", "tacts", "princess.tact")

    # Initialize and register the BhapticsManager with the haptic feedback assets
    bhaptics_manager = BhapticsManager()
    bhaptics_manager.register("coin", "assets/tacts/coin.tact")
    bhaptics_manager.register("2", fire_flower_tact_path)
    bhaptics_manager.register("3", starman_tact_path)
    bhaptics_manager.register("4", super_mushroom_tact_path)
    bhaptics_manager.register("5", enemy_tact_path)
    bhaptics_manager.register("6", flag_tact_path)
    bhaptics_manager.register("7", level_end_tact_path)
    bhaptics_manager.register("8", princess_tact_path)

    # Path to the YOLO model
    yolo_model_path = os.path.join(script_dir, "assets", "models", "yolov8m_vibrovision.pt")

    # Initialize the application
    app = QApplication(sys.argv)

    # Initialize the YOLO model
    yolo_model = YOLOModel(yolo_model_path)

    # Initialize and show the main application window
    main_window = VibroVisionApp(yolo_model, bhaptics_manager)
    main_window.show()

    # Start the application event loop
    sys.exit(app.exec_())
