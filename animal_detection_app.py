import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
from ultralytics import YOLO
import os

class AnimalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.video_capture = None
        self.is_video_playing = False
        self.carnivorous_animals = ['lion', 'tiger', 'leopard', 'wolf', 'bear', 'fox', 'crocodile', 'shark']
        self.input_size = (224, 224)  # Based on your model's input size
        self.confidence_threshold = 0.5  # Threshold for considering a detection
        
        # Class names from your model
        self.class_names = ['antelope', 'bear', 'beaver', 'bee', 'bison', 'blackbird', 'buffalo', 'butterfly', 
                          'camel', 'cat', 'cheetah', 'chimpanzee', 'chinchilla', 'cow', 'crab', 'crocodile', 
                          'deer', 'dog', 'dolphin', 'donkey', 'duck', 'eagle', 'elephant', 'falcon', 'ferret', 
                          'flamingo', 'fox', 'frog', 'giraffe', 'goat', 'goose', 'gorilla', 'grasshopper', 
                          'hawk', 'hedgehog', 'hippopotamus', 'hyena', 'iguana', 'jaguar', 'kangaroo', 'koala', 
                          'lemur', 'leopard', 'lizard', 'lynx', 'mole', 'mongoose', 'ostrich', 'otter', 'owl', 
                          'panda', 'peacock', 'penguin', 'porcupine', 'raccoon', 'seal', 'sheep', 'snail', 
                          'snake', 'spider', 'squid', 'walrus', 'whale', 'wolf']
        
        self.setup_gui()
        self.load_models()
        
    def setup_gui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create buttons
        ttk.Button(self.button_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        
        # Create confidence threshold slider
        self.conf_scale = ttk.Scale(self.button_frame, from_=0.0, to=1.0, 
                                  orient='horizontal', length=200,
                                  command=self.update_confidence)
        self.conf_scale.set(self.confidence_threshold)
        self.conf_scale.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.button_frame, text="Confidence Threshold").pack(side=tk.LEFT)
        
        # Create display area
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.display_label = ttk.Label(self.display_frame)
        self.display_label.pack()
        
        # Create status label
        self.status_label = ttk.Label(self.main_frame, text="Ready")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
        
    def load_models(self):
        try:
            # Load your custom classification model
            model_path = "my_model.keras"
            self.custom_model = tf.keras.models.load_model(model_path)
            
            # Load YOLOv8 model
            self.yolo_model = YOLO('yolov8x.pt')
            
            self.status_label.config(text="Models loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def update_confidence(self, value):
        self.confidence_threshold = float(value)
            
    def preprocess_for_custom_model(self, img):
        # Convert OpenCV image (BGR) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, self.input_size)
        
        # Convert to float32 and add batch dimension
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for ResNet
        img_array = tf.keras.applications.resnet.preprocess_input(img_array)
        return img_array
            
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.process_image(file_path)
            
    def process_image(self, image_path):
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Failed to load image")
            
            # Get YOLO detections
            yolo_results = self.yolo_model(image)[0]
            
            # Track carnivorous animals
            carnivorous_count = 0
            
            # Process each detection
            for box in yolo_results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Extract the region of interest
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Preprocess ROI for custom model
                processed_roi = self.preprocess_for_custom_model(roi)
                
                # Get custom model prediction
                custom_predictions = self.custom_model.predict(processed_roi)
                class_idx = np.argmax(custom_predictions[0])
                confidence = custom_predictions[0][class_idx]
                
                if confidence >= self.confidence_threshold:
                    class_name = self.class_names[class_idx]
                    
                    # Check if animal is carnivorous
                    is_carnivorous = class_name.lower() in self.carnivorous_animals
                    if is_carnivorous:
                        carnivorous_count += 1
                    
                    # Draw detection
                    self.draw_detection(image, class_name, confidence, (x1, y1, x2, y2), is_carnivorous)
            
            # Show warning if carnivorous animals detected
            if carnivorous_count > 0:
                messagebox.showwarning("Warning", f"Detected {carnivorous_count} carnivorous animal(s)!")
            
            # Display the processed image
            self.display_image(image)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            
    def draw_detection(self, image, class_name, confidence, bbox, is_carnivorous):
        x1, y1, x2, y2 = bbox
        
        # Set colors and thickness
        box_color = (0, 0, 255) if is_carnivorous else (0, 255, 0)  # Red for carnivorous, Green for others
        text_color = (255, 255, 255)  # White text
        box_thickness = 3 if is_carnivorous else 2
        text_thickness = 2
        
        # Draw bounding box with semi-transparency
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # Create label
        label = f"{class_name} ({confidence:.2f})"
        
        # Get text size
        font_scale = 0.8
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        
        # Draw label background
        label_y = max(y1 - 10, text_height + 10)
        cv2.rectangle(image, 
                     (x1, label_y - text_height - 10),
                     (x1 + text_width + 10, label_y + 5),
                     box_color, -1)
        
        # Draw label text
        cv2.putText(image, label,
                    (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, text_thickness)
            
    def open_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.is_video_playing = True
            self.update_video_frame()
            
    def update_video_frame(self):
        if self.is_video_playing and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                try:
                    # Get YOLO detections
                    yolo_results = self.yolo_model(frame)[0]
                    
                    # Track carnivorous animals
                    carnivorous_count = 0
                    
                    # Process each detection
                    for box in yolo_results.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Extract the region of interest
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        
                        # Preprocess ROI for custom model
                        processed_roi = self.preprocess_for_custom_model(roi)
                        
                        # Get custom model prediction
                        custom_predictions = self.custom_model.predict(processed_roi)
                        class_idx = np.argmax(custom_predictions[0])
                        confidence = custom_predictions[0][class_idx]
                        
                        if confidence >= self.confidence_threshold:
                            class_name = self.class_names[class_idx]
                            
                            # Check if animal is carnivorous
                            is_carnivorous = class_name.lower() in self.carnivorous_animals
                            if is_carnivorous:
                                carnivorous_count += 1
                            
                            # Draw detection
                            self.draw_detection(frame, class_name, confidence, (x1, y1, x2, y2), is_carnivorous)
                    
                    # Display the processed frame
                    self.display_image(frame)
                    
                    # Schedule next frame update
                    self.root.after(30, self.update_video_frame)
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    self.root.after(30, self.update_video_frame)
            else:
                self.stop_video()
                
    def stop_video(self):
        self.is_video_playing = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            
    def display_image(self, image):
        # Resize image to fit display
        height, width = image.shape[:2]
        max_size = 800
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            
        # Convert to PhotoImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update display
        self.display_label.config(image=photo)
        self.display_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalDetectionApp(root)
    root.mainloop() 