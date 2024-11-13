import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Setting up Streamlit app
st.title("Morphological Image Processing Tool")

# Step 1: Uploading an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
if uploaded_image is not None:
    # Converting image to grayscale for morphological processing
    image = Image.open(uploaded_image)
    img_array = np.array(image)

    # Check if the image is not grayscale and convert it if necessary
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        st.write("Converted image to grayscale.")

    # Display the grayscale image
    st.image(img_array, caption="Grayscale Image", use_column_width=True)

    # Step 2: Select morphological operation
    operation = st.selectbox("Choose a morphological operation",
                             ["Dilation", "Erosion", "Opening", "Closing",
                              "Morphological Gradient", "Top-Hat", "Black-Hat",
                              "Hit-or-Miss", "Boundary Extraction",
                              "Region Filling", "Connected Component Labeling"])

    # Slider for structuring element size
    kernel_size = st.slider("Kernel size (odd number)", 1, 21, 5, step=2)
    kernel_shape = st.selectbox("Kernel shape", ["Rectangle", "Ellipse", "Cross"])

    # Creating structuring element based on selection
    if kernel_shape == "Rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "Ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == "Cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))


    # Function for region filling
    def region_fill(img, seed_point):
        filled_img = np.zeros_like(img)
        filled_img[seed_point] = 255  # Start filling at the seed point
        while True:
            dilated = cv2.dilate(filled_img, kernel, iterations=1)
            filled_img = cv2.bitwise_and(dilated, cv2.bitwise_not(img))
            if np.array_equal(filled_img, dilated):
                break
        return cv2.bitwise_or(filled_img, img)


    # Optioning for binary conversion for certain operations
    if operation in ["Hit-or-Miss", "Connected Component Labeling"]:
        threshold = st.slider("Binary Threshold", 0, 255, 128)
        _, img_array = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
        st.write("Converted image to binary format.")

    # Applying the selected operation
    if st.button("Apply"):
        if operation == "Dilation":
            result_img = cv2.dilate(img_array, kernel, iterations=1)
        elif operation == "Erosion":
            result_img = cv2.erode(img_array, kernel, iterations=1)
        elif operation == "Opening":
            result_img = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
        elif operation == "Closing":
            result_img = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        elif operation == "Morphological Gradient":
            result_img = cv2.morphologyEx(img_array, cv2.MORPH_GRADIENT, kernel)
        elif operation == "Top-Hat":
            result_img = cv2.morphologyEx(img_array, cv2.MORPH_TOPHAT, kernel)
        elif operation == "Black-Hat":
            result_img = cv2.morphologyEx(img_array, cv2.MORPH_BLACKHAT, kernel)
        elif operation == "Hit-or-Miss":
            hit_miss_kernel = np.array([[1, 1, 1], [0, 1, -1], [0, 1, -1]], dtype=np.int8)
            result_img = cv2.morphologyEx(img_array, cv2.MORPH_HITMISS, hit_miss_kernel)
        elif operation == "Boundary Extraction":
            eroded = cv2.erode(img_array, kernel, iterations=1)
            result_img = cv2.subtract(img_array, eroded)
        elif operation == "Region Filling":
            seed_x = st.slider("Seed Point X", 0, img_array.shape[1] - 1, 0)
            seed_y = st.slider("Seed Point Y", 0, img_array.shape[0] - 1, 0)
            result_img = region_fill(img_array, (seed_y, seed_x))
        elif operation == "Connected Component Labeling":
            num_labels, labels_im = cv2.connectedComponents(img_array)
            result_img = (labels_im * (255 // num_labels)).astype(np.uint8)

        # Display the processed image
        st.image(result_img, caption=f"Result of {operation}", use_column_width=True)

        # Option to download the processed image
        result_img_pil = Image.fromarray(result_img)
        buf = io.BytesIO()
        result_img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Image", data=byte_im, file_name="processed_image.png", mime="image/png")
