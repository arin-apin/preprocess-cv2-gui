import cv2
import numpy as np
from tkinter import *
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from tkinter import ttk
# from custom_integer_dialog import CustomIntegerDialog

# Filter functions
def apply_invert(img):
    """Inverts image."""
    filtered_img = cv2.bitwise_not(img)
    return filtered_img

def apply_dilate(img, kernel_size, iterations):
    """Applies dilation to an image with a given kernel size and number of iterations."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_img = cv2.dilate(img, kernel, iterations=iterations)
    return filtered_img

def apply_erode(img, kernel_size, iterations):
    """Applies erosion to an image with a given kernel size and number of iterations."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_img = cv2.erode(img, kernel, iterations=iterations)
    return filtered_img

def apply_grayscale(img):
    """Converts an image to grayscale."""
    filtered_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return filtered_img

def apply_threshold(img, threshold):
    """Applies a binary threshold to an image."""
    _, filtered_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return filtered_img

def apply_threshold_band(img, lower_threshold, upper_threshold, invert):
    """Applies a threshold band to an image, with an option to invert the result."""
    _, img_thresh1 = cv2.threshold(img, lower_threshold, 255, cv2.THRESH_BINARY)
    _, img_thresh2 = cv2.threshold(img, upper_threshold, 255, cv2.THRESH_BINARY_INV)
    filtered_img = cv2.bitwise_and(img_thresh1, img_thresh2)
    if invert:
        filtered_img = cv2.bitwise_not(filtered_img)
    return filtered_img

def apply_smooth(img, kernel_size, iterations):
    """Applies Gaussian blur to an image with a given kernel size and number of iterations."""
    filtered_img = img
    for _ in range(iterations):
        filtered_img = cv2.GaussianBlur(filtered_img, (kernel_size, kernel_size), 0)
    return filtered_img

def apply_sharpen(img, kernel):
    """Applies sharpening to an image using a convolution kernel."""
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

def apply_canny(img, threshold1, threshold2):
    """Applies the Canny edge detector to an image."""
    return cv2.Canny(img, threshold1, threshold2)

def apply_normalize(img, alpha, beta):
    """Normalizes the brightness and contrast of an image."""
    return cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)

def show_configuration(filter_name, index):
    """Shows configuration dialog for each filter based on selected filter type."""
    configuration = configurations[index]
    if filter_name == "Threshold":
        configuration['threshold'] = simpledialog.askinteger("Configure Threshold", "Enter threshold value:",
                                                              initialvalue=configuration.get('threshold', 127))
    elif filter_name == "Dilate":
        configuration['kernel_size'] = simpledialog.askinteger("Configure Dilate", "Enter kernel size:",
                                                                initialvalue=configuration.get('kernel_size', 3))
        configuration['iterations'] = simpledialog.askinteger("Configure Dilate", "Enter number of iterations:",
                                                              initialvalue=configuration.get('iterations', 1))
    elif filter_name == "Erode":
        configuration['kernel_size'] = simpledialog.askinteger("Configure Erode", "Enter kernel size:",
                                                               initialvalue=configuration.get('kernel_size', 3))
        configuration['iterations'] = simpledialog.askinteger("Configure Erode", "Enter number of iterations:",
                                                              initialvalue=configuration.get('iterations', 1))
    elif filter_name == "Threshold Band":
        configuration['lower_threshold'] = simpledialog.askinteger("Configure Lower Threshold", "Enter lower threshold:",
                                                                   initialvalue=configuration.get('lower_threshold', 50))
        configuration['upper_threshold'] = simpledialog.askinteger("Configure Upper Threshold", "Enter upper threshold:",
                                                                   initialvalue=configuration.get('upper_threshold', 150))
        configuration['invert'] = simpledialog.askinteger("Invert Result", "Invert (0=no, 1=yes):",
                                                          initialvalue=configuration.get('invert', 0))
    elif filter_name == "Smooth":
        configuration['kernel_size'] = simpledialog.askinteger("Configure Smooth", "Enter kernel size:",
                                                               initialvalue=configuration.get('kernel_size', 5))
        configuration['iterations'] = simpledialog.askinteger("Configure Smooth", "Enter number of iterations:",
                                                              initialvalue=configuration.get('iterations', 1))
    elif filter_name == "Sharpen":
        configuration['version'] = simpledialog.askinteger("Configure Sharpen", "0 = Basic, 1 = Strong, 2 = Excessive, 3 = Edge enhance",
                                                          initialvalue=configuration.get('version', 0))
        if configuration['version'] == 0:
            configuration['kernel'] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        elif configuration['version'] == 1:
            configuration['kernel'] = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        elif configuration['version'] == 2:
            configuration['kernel'] = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]]) 
        elif configuration['version'] == 3:
            configuration['kernel'] = np.array([
                                        [-1, -1, -1, -1, -1],
                                        [-1, 2, 2, 2, -1],
                                        [-1, 2, 8, 2, -1],
                                        [-1, 2, 2, 2, -1],
                                        [-1, -1, -1, -1, -1]
                                    ]) / 8.0 
        else:
            configuration['kernel'] = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            configuration['version'] = 0
    elif filter_name == "Canny":
        configuration['threshold1'] = simpledialog.askinteger("Configure Canny", "Enter threshold1:",
                                                              initialvalue=configuration.get('threshold1', 100))
        configuration['threshold2'] = simpledialog.askinteger("Configure Canny", "Enter threshold2:",
                                                              initialvalue=configuration.get('threshold2', 200))
    elif filter_name == "Normalize":
        configuration['alpha'] = simpledialog.askinteger("Configure Normalize", "Enter alpha:",
                                                         initialvalue=configuration.get('alpha', 0))
        configuration['beta'] = simpledialog.askinteger("Configure Normalize", "Enter beta:",
                                                        initialvalue=configuration.get('beta', 255))
    # Note: Grayscale doesn't require additional configuration
    apply_filters()

def apply_filters():
    """Applies selected filters in order to the original image and displays the result."""
    if img_original is None:
        return
    processed_img = img_original.copy()
    for i, combo in enumerate(combos_filter):
        filter_name = combo.get()
        configuration = configurations[i]
        if filter_name == "Grayscale":
            processed_img = apply_grayscale(processed_img)
        elif filter_name == "Threshold":
            processed_img = apply_threshold(processed_img, configuration.get('threshold', 127))
        elif filter_name == "Dilate":
            kernel_size = configuration.get('kernel_size', 3)
            iterations = configuration.get('iterations', 1)
            processed_img = apply_dilate(processed_img, kernel_size, iterations)
        elif filter_name == "Erode":
            kernel_size = configuration.get('kernel_size', 3)
            iterations = configuration.get('iterations', 1)
            processed_img = apply_erode(processed_img, kernel_size, iterations)
        elif filter_name == "Threshold Band":
            lower_threshold = configuration.get('lower_threshold', 50)
            upper_threshold = configuration.get('upper_threshold', 150)
            invert = configuration.get('invert', 0)
            processed_img = apply_threshold_band(processed_img, lower_threshold, upper_threshold, invert)
        elif filter_name == "Smooth":
            kernel_size = configuration.get('kernel_size', 5)
            iterations = configuration.get('iterations', 1)
            processed_img = apply_smooth(processed_img, kernel_size, iterations)  
        elif filter_name == "Sharpen":
            kernel = configuration.get('kernel', np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
            processed_img = apply_sharpen(processed_img, kernel)
        elif filter_name == "Canny":
            threshold1 = configuration.get('threshold1', 100)
            threshold2 = configuration.get('threshold2', 200)
            processed_img = apply_canny(processed_img, threshold1, threshold2)
        elif filter_name == "Normalize":
            alpha = configuration.get('alpha', 0)
            beta = configuration.get('beta', 255)
            processed_img = apply_normalize(processed_img, alpha, beta)
        elif filter_name == "Invert":
            processed_img = apply_invert(processed_img)

    display_image(processed_img)


# Function to load and display the original image
def load_image():
    """Loads an image from disk and applies selected filters."""
    global img_original
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img_original = cv2.imread(file_path)
    apply_filters()

# Function to display the image in the GUI
def display_image(img):
    """Displays an image in the GUI."""
    height, width = img.shape[:2]
    while width>1080 or height > 768:
        img = cv2.resize(img,(width//2,height//2))
        height, width = img.shape[:2]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)
    lbl_image.config(image=img_tk)
    lbl_image.image = img_tk

def on_combo_change(index, *args):
    """Callback function triggered when a filter selection changes."""
    apply_filters()

# Initial application setup
window = ThemedTk(theme="equilux")  # 'equilux' es un ejemplo de tema oscuro
window.title("Preprocess cv2 GUI")
frame = ttk.Frame(window)
frame.pack(expand=True, fill='both')
img_original = None
configurations = [{} for _ in range(5)]  # Configuration for each filter

# Image widget
lbl_image = ttk.Label(frame)
lbl_image.pack()

# Load image button
btn_load = ttk.Button(frame, text="Load Image", command=load_image)
btn_load.pack()

# Comboboxes and configuration buttons for each filter
filter_options = ["None", 
                'Canny',
                'Dilate',
                'Erode',
                'Grayscale',
                'Invert',
                'Normalize',
                'Sharpen',
                'Smooth',
                'Threshold',
                'Threshold Band']
combos_filter = []
for i in range(5):
    combo_var = StringVar()  # Create a control variable for each Combobox
    combo_var.set(filter_options[0])  # Set the initial option
    combo = ttk.Combobox(frame, textvariable=combo_var, values=filter_options, state="readonly")
    combo.bind('<<ComboboxSelected>>', lambda event, index=i: on_combo_change(event, index))  # Handle selection change
    combo.pack(side=LEFT, padx=5, pady=5)  # Add some padding for aesthetics
    combos_filter.append(combo_var)  # Save the control variable for later use

    # Create and pack the configuration button for each filter
    ttk.Button(frame, text="Configure", command=lambda index=i: show_configuration(combos_filter[index].get(), index)).pack(side=LEFT, padx=5)

window.mainloop()
