import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained PyTorch model
model = torch.load('good_model(3).pt', map_location=device)
model.eval()
model.to(device)

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust according to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image_path = 'picture.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

num_males = 0
num_females = 0

for (x, y, w, h) in faces:
    # Extract the face region
    face = image[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Convert the face to PIL Image
    face_pil = Image.fromarray(face_rgb)

    # Preprocess the face
    face_tensor = preprocess(face_pil)
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    face_tensor = face_tensor.to(device)  # Move tensor to the correct device

    # Perform inference
    with torch.no_grad():
        outputs = model(face_tensor)
        prediction = outputs.argmax(dim=1).item()  # Get the predicted class (0 for male, 1 for female)

    # Update counters and set colors
    if prediction == 0:
        label = "Female"
        num_females += 1
        box_color = (255, 0, 255)  # Pink for female
        text_color = (255, 0, 255)  # Pink text for female
    else:
        label = "Male"
        num_males += 1
        box_color = (0, 0, 255)  # Red for male
        text_color = (0, 0, 255)  # Red text for male

    # Draw the bounding box and label on the image
    cv2.rectangle(image, (x, y), (x+w, y+h), box_color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

# Display the counter on the top left corner
cv2.putText(image, f'Males: {num_males}  Females: {num_females}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save the output image
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image)

# Resize the window to fit the image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
