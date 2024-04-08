import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('my_model_6_classes.h5')

# Function to preprocess a single image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Path to the image you want to test
test_image_path = 'the-beautiful-cala-goloritze-in-sardinia-royalty-free-image-1149841315-1559070275.jpg'

# Preprocess the image
test_image = preprocess_image(test_image_path)

# Make predictions
predictions = model.predict(test_image)

# Map predicted indices to class names
class_names = {
    0: 'beach',
    1: 'cities',
    2: 'desert',
    3: 'island',
    4: 'mountains',
    5: 'sea'
}

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

# Print the predicted class name
print("Predicted class:", predicted_class_name)
