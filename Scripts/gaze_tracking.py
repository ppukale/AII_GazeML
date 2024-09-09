import torch
import torchvision.models as models
from torch import nn
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, OrderedDict
from sklearn.linear_model import LinearRegression
import pyautogui
import time

# Define the custom gaze model based on VGG16
class GazeModel(nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        # Load a pre-trained VGG16 model
        self.vgg16 = models.vgg16()  # Avoid deprecated 'pretrained' parameter
        # Modify the classifier for gaze estimation
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, 2)  # Output layer has 2 neurons for (x, y)

    def forward(self, x):
        return self.vgg16(x)

# Load the pretrained gaze model
model = GazeModel()
model_path = r'C:/Users/prash/MyProjects/gitrepos/eye-model/venv/models/gaze_model_pytorch_vgg16_prl_mpii_allsubjects4.model'  # Corrected path format

# Load the state dictionary with adjustments if needed
try:
    # Adjust the state_dict keys if necessary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()

    # This assumes the pretrained model has specific key naming conventions
    for k, v in state_dict.items():
        new_key = k.replace("left_features", "vgg16.features")  # Example of key adjustment; modify as needed
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to allow partial loading
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()  # Set the model to evaluation mode

# Define transformation for input frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize frame to 224x224 as required by VGG16
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Function to save gaze data to CSV
def save_to_csv(gaze_data, filename="gaze_data.csv"):
    if gaze_data:
        df = pd.DataFrame(gaze_data, columns=['x', 'y'])
        df.to_csv(filename, index=False)
        print(f"Gaze data saved to {filename}")
    else:
        print("No gaze data to save.")

# Function to generate heatmap
def generate_heatmap(gaze_data, filename="heatmap.png"):
    if gaze_data:
        df = pd.DataFrame(gaze_data, columns=['x', 'y'])
        plt.figure(figsize=(10, 6))
        
        # Corrected seaborn kdeplot call with appropriate arguments
        sns.kdeplot(data=df, x='x', y='y', cmap="Reds", fill=True, bw_adjust=0.5)  # Correct use of `x`, `y`, and `fill=True`
        
        plt.title('Gaze Heatmap')
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.savefig(filename)
        plt.close()  # Close the plot to prevent hanging
        print(f"Heatmap saved to {filename}")
    else:
        print("No gaze data available to generate a heatmap.")

# Capture video from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gaze_data = deque()  # To store gaze coordinates

# Get screen size
screen_width, screen_height = pyautogui.size()

# Calibration points on the screen (example coordinates)
calibration_points = [(100, 100), (screen_width - 100, 100), 
                      (100, screen_height - 100), (screen_width - 100, screen_height - 100), 
                      (screen_width // 2, screen_height // 2)]
calibration_gaze_data = []

# Calibration process
print("Starting calibration. Please look at the red coordinates as they appear on the screen.")
for point in calibration_points:
    x, y = point
    print(f"Look at point: {point}")

    # Blink the red text on the screen
    for _ in range(10):  # Blink 10 times
        # Create a blank image with the same screen size
        screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(screen, f'{x}, {y}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('Calibration', screen)
        cv2.waitKey(50)  # Display for 500ms
        screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)  # Clear screen
        cv2.imshow('Calibration', screen)
        cv2.waitKey(50)  # Blank for 500ms

    time.sleep(5)  # Wait for 5 seconds after displaying the coordinates

    # Capture gaze data for current screen point
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        continue

    # Convert frame to PIL image and apply transformations
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)

    # Predict gaze using the model
    with torch.no_grad():
        predicted_gaze = model(input_tensor)

    # Extract gaze coordinates (x, y)
    gaze_coords = predicted_gaze.numpy()[0]
    calibration_gaze_data.append((gaze_coords, point))

cv2.destroyAllWindows()  # Close the calibration window

# Prepare data for regression
model_gaze = np.array([g[0] for g in calibration_gaze_data])  # Gaze coordinates from model
screen_points = np.array([g[1] for g in calibration_gaze_data])  # Actual screen coordinates

# Fit linear regression models for x and y coordinates
reg_x = LinearRegression().fit(model_gaze[:, 0].reshape(-1, 1), screen_points[:, 0])
reg_y = LinearRegression().fit(model_gaze[:, 1].reshape(-1, 1), screen_points[:, 1])

# Function to predict screen location
def predict_screen_location(gaze_coords):
    screen_x = reg_x.predict([[gaze_coords[0]]])[0]
    screen_y = reg_y.predict([[gaze_coords[1]]])[0]
    return int(screen_x), int(screen_y)

# Main loop to process webcam feed and predict gaze location on screen
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Convert frame to PIL image and apply transformations
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)

    # Predict gaze using the model
    with torch.no_grad():
        predicted_gaze = model(input_tensor)

    # Extract gaze coordinates (x, y)
    gaze_coords = predicted_gaze.numpy()[0]

    # Predict screen location
    screen_x, screen_y = predict_screen_location(gaze_coords)

    # Print predicted screen location
    print(f"User is looking at: ({screen_x}, {screen_y})")

    # Append gaze coordinates to the deque
    gaze_data.append((screen_x, screen_y))

    # Optional: Draw gaze point on the frame (visual feedback)
    cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)  # Draw green circle at the predicted screen point

    # Display the frame
    cv2.imshow('Gaze Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save data and clean up
cap.release()
cv2.destroyAllWindows()
save_to_csv(gaze_data)
generate_heatmap(gaze_data)
