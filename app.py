import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# 1. Load the model (Cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('digit_recognizer.keras')
    except:
        return tf.keras.models.load_model('save_weights.hdf5')

model = load_model()

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) inside the box below, and the AI will predict it.")

# 2. Create the Drawing Canvas
# We use a white background with black ink, as that is intuitive for users.
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Here:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=15,      # Thicker stroke matches MNIST training data better
        stroke_color="#000000", # Black ink
        background_color="#FFFFFF", # White background
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# 3. Process the Input
if canvas_result.image_data is not None:
    # Get the numpy array (RGBA) from the canvas
    input_numpy = np.array(canvas_result.image_data)
    
    # Check if the user has actually drawn something (not just empty canvas)
    # We check if the sum of alpha channel is greater than 0
    if np.sum(input_numpy[:, :, 3]) > 0:
        
        # Convert to PIL Image
        input_image = Image.fromarray(input_numpy.astype('uint8'), 'RGBA')
        
        # Convert to Grayscale (L) - this makes the background white and ink black
        # Because the canvas background is effectively white
        input_image = input_image.convert('L')
        
        # INVERSION: We need White Digit on Black Background for MNIST
        input_image = ImageOps.invert(input_image)
        
        # Resize to 28x28
        input_image = input_image.resize((28, 28))
        
        # Show what the model sees (Debugging / Educational)
        with col2:
            st.subheader("What the Model Sees:")
            st.image(input_image, width=150, caption="Processed Image (28x28)")

        # Normalize
        img_array = np.array(input_image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # 4. Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown(f"### Prediction: **{predicted_class}**")
        st.caption(f"Confidence: {confidence:.2%}")
        
        # Optional: Show bar chart of probabilities
        st.bar_chart(prediction[0])

    else:
        with col2:
            st.info("Please draw a digit to see the prediction.")