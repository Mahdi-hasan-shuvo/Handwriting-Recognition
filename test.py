import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
 
from keras.models import load_model
model = load_model('/content/my_modelcnn.h5')
 
# Use cv2_imshow instead of cv2.imshow
from google.colab.patches import cv2_imshow 
 
word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
} 
image = cv2.imread('/content/k.PNG')
image_copy = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (400,440))
 
image_copy = cv2.GaussianBlur(image_copy, (7,7), 0)
gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
 
final_image = cv2.resize(img_thresh, (28,28))
final_image =np.reshape(final_image, (1,28,28,1))
 
prediction = word_dict[np.argmax(model.predict(final_image))]
 
cv2.putText(image, "Prediction: " + prediction, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (0,255,0))

# Display the image using cv2_imshow
cv2_imshow(image) 
 
# cv2.waitKey(0)  # Wait for a key press (not needed in Colab)
cv2.destroyAllWindows() # Close the window


# for multiple text prediction

import cv2
import tensorflow as tf
import numpy as np

from keras.models import load_model
from google.colab.patches import cv2_imshow

# Load your model (make sure the path is correct)
model = load_model('/content/my_modelcnn.h5')

# Define your character dictionary
word_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

def predict_characters(image_path):
    image = cv2.imread(image_path)
    if image is None:  # Check if image loaded successfully
        print(f"Error: Could not open or read image: {image_path}")
        return ""

    image_copy = image.copy()
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.boundingRect)

    predictions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter contours (adjust these thresholds as needed)
        if w > 10 and h > 20:  # Example: Filter out small noise
            char_image = img_thresh[y:y+h, x:x+w]

            # Check if the character image is not empty
            if char_image.size > 0:
                final_image = cv2.resize(char_image, (28, 28))
                final_image = np.reshape(final_image, (1, 28, 28, 1))
                final_image = final_image.astype('float32') / 255.0  # Normalize pixel values

                prediction_probabilities = model.predict(final_image)  # Get probabilities
                predicted_class = np.argmax(prediction_probabilities)
                prediction = word_dict.get(predicted_class, "?")  # Handle unknown characters

                predictions.append(prediction)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Warning: Empty character image encountered. Skipping.")


    cv2_imshow(image)
    cv2.destroyAllWindows()
    return "".join(predictions)

# Example usage:
image_path = '/content/how-can-i-improve-my-handwriting-v0-29nhfjcqbije1.webp'  # Replace with your image path
predicted_word = predict_characters(image_path)
print("Predicted word:", predicted_word)
