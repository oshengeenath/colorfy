import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import io
import time
import os
import subprocess
import tempfile
from basicsr.archs.colorfy_arch import Colorfy
import glob

def select_file_using_subprocess(is_folder=False):
    """Use a standalone Python script to select a file or folder and return the path"""
    # Create a temporary file to store the result
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_file.close()
    
    # Create a temporary file for the file dialog script
    script_file = tempfile.NamedTemporaryFile(delete=False, suffix='.py')
    script_file.write(b"""
import tkinter as tk
from tkinter import filedialog
import sys
import os

# Function to select file/folder and save the path to a temporary file
def select_path(output_file, is_folder=False):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    
    if is_folder:
        path = filedialog.askdirectory(title="Select Folder")
    else:
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("PyTorch files", "*.pth"), ("All files", "*.*"))
        )
    root.destroy()
    
    # Write the path to the output file
    with open(output_file, 'w') as f:
        f.write(path)
    
    return path

if __name__ == "__main__":
    if len(sys.argv) > 2:
        output_file = sys.argv[1]
        is_folder = sys.argv[2].lower() == 'true'
        select_path(output_file, is_folder)
    else:
        print("Error: Arguments not provided correctly")
        sys.exit(1)
""")
    script_file.close()
    
    # Run the script as a separate process
    try:
        subprocess.run(["python", script_file.name, temp_file.name, str(is_folder)], check=True)
        
        # Read the selected path
        with open(temp_file.name, 'r') as f:
            path = f.read().strip()
        
        # Clean up temporary files
        os.unlink(temp_file.name)
        os.unlink(script_file.name)
        
        return path
    except Exception as e:
        st.error(f"Error running file dialog: {str(e)}")
        # Clean up temporary files
        try:
            os.unlink(temp_file.name)
            os.unlink(script_file.name)
        except:
            pass
        return None

class ImageColorizer:
    def __init__(self, model_path, input_size=512):
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        """Load the Colorfy model"""
        config = {
            "encoder_name": "convnext-l",
            "decoder_name": "MultiScaleColorDecoder",
            "input_size": [self.input_size, self.input_size],
            "num_output_channels": 2,
            "last_norm": "Spectral",
            "do_normalize": False,
            "num_queries": 100,
            "num_scales": 3,
            "dec_layers": 9,
        }
        
        model = Colorfy(**config)
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Extract params if needed
        if "params" in state_dict:
            state_dict = state_dict["params"]
            
        # Filter compatible weights
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        
        print(f"Loaded {len(filtered_dict)} of {len(state_dict)} layers")
        model.load_state_dict(filtered_dict, strict=False)
        return model.to(self.device)
    
    @torch.no_grad()
    def colorize(self, img):
        """Colorize a grayscale or color image"""
        if img is None:
            raise ValueError("Empty image received")
            
        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        # Save original dimensions and extract L channel
        height, width = img.shape[:2]
        img_float = img.astype(np.float32) / 255.0
        orig_l = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)[:, :, :1]
        
        # Resize and prepare for model
        img_resized = cv2.resize(img_float, (self.input_size, self.input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Process with model
        tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        output_ab = self.model(tensor).cpu()
        
        # Resize output and combine with original L channel
        output_ab = torch.nn.functional.interpolate(
            output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
        
        # Combine L with predicted AB
        output_lab = np.concatenate((orig_l, output_ab), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        
        # Convert to uint8
        return (output_bgr * 255.0).round().astype(np.uint8)

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format (BGR)"""
    rgb_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) to PIL Image"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def resize_image(image, max_width=350, max_height=350):
    """Resize the image to fit within max_width and max_height while maintaining aspect ratio.
    Further reduced from 250x250 to 180x180 for even smaller previews."""
    original_width, original_height = image.size
    if original_width > max_width or original_height > max_height:
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_ratio = min(width_ratio, height_ratio)
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    return image

def process_single_image(colorizer, image, input_size):
    """Process a single image through the colorizer"""
    cv2_image = pil_to_cv2(image)
    colored_cv2 = colorizer.colorize(cv2_image)
    colored_image = cv2_to_pil(colored_cv2)
    return colored_image

def process_folder(colorizer, folder_path, output_folder, progress_bar, status_text, preview_containers):
    """Process all images in a folder with live previews"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    # Check if any images were found
    if not image_files:
        return 0, []
    
    # Process each image
    results = []
    orig_container, colored_container = preview_containers
    prev_original = None
    prev_colored = None
    
    for i, img_path in enumerate(image_files):
        try:
            # Update progress bar
            progress_value = (i) / max(1, len(image_files) - 1)
            progress_bar.progress(progress_value)
            status_text.text(f"Processing image {i+1} of {len(image_files)}: {os.path.basename(img_path)}")
            
            # Open image
            image = Image.open(img_path)
            
            # Process the full-resolution image
            colored_image = process_single_image(colorizer, image, colorizer.input_size)
            
            # Create preview-sized images (different from the processed images)
            resized_original = resize_image(image)
            resized_colored = resize_image(colored_image)
            
            # Make sure the preview images are different from the previous ones
            if i > 0 and prev_original is not None and prev_colored is not None:
                orig_container.empty()
                colored_container.empty()
            
            # Display the preview images
            orig_container.image(resized_original, caption=f"Original: {os.path.basename(img_path)}", use_container_width=False)
            colored_container.image(resized_colored, caption=f"Colorized: {os.path.basename(img_path)}", use_container_width=False)
            
            # Remember these images for comparison in the next iteration
            prev_original = resized_original
            prev_colored = resized_colored
            
            # Save the full-resolution colorized image
            output_path = os.path.join(output_folder, f"colorized_{os.path.basename(img_path)}")
            colored_image.save(output_path)
            
            # Add to results list
            results.append({
                "original_path": img_path,
                "output_path": output_path,
                "filename": os.path.basename(img_path)
            })
            
        except Exception as e:
            status_text.text(f"Error processing {img_path}: {str(e)}")
            time.sleep(2)  # Show the error briefly
    
    # Set progress to 100% when done
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {len(results)} images")
    
    return len(results), results

def main():
    st.set_page_config(page_title="Colorfy App", layout="wide")
    
    st.title("COLORFY - Image Colorization")
    st.write("Upload a black and white image or select a folder to colorize images using COLORFY.")
    
    # Initialize session state
    if 'model_path' not in st.session_state:
        st.session_state.model_path = ""
    if 'file_dialog_active' not in st.session_state:
        st.session_state.file_dialog_active = False
    if 'output_folder' not in st.session_state:
        st.session_state.output_folder = ""
    if 'folder_dialog_active' not in st.session_state:
        st.session_state.folder_dialog_active = False
    
    # Sidebar for model parameters
    st.sidebar.header("Model Settings")
    
    # File selection button for model
    if st.sidebar.button("Browse for Model File", disabled=st.session_state.file_dialog_active):
        st.session_state.file_dialog_active = True
        with st.spinner("Opening file dialog..."):
            file_path = select_file_using_subprocess(is_folder=False)
            if file_path:
                st.session_state.model_path = file_path
                st.sidebar.success(f"Selected: {os.path.basename(file_path)}")
            else:
                st.sidebar.warning("No file selected")
        st.session_state.file_dialog_active = False
        st.rerun()
    
    # Display selected model path
    if st.session_state.model_path:
        st.sidebar.success(f"Selected: {os.path.basename(st.session_state.model_path)}")
        st.sidebar.text(f"Path: {st.session_state.model_path}")
    else:
        st.sidebar.warning("No model selected. Please browse for your .pth file.")
    
    # Model parameters
    input_size = st.sidebar.slider("Input Size", 256, 1024, 512, 64)
    
    # Show device info
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"Using device: {device}")
    
    # Mode selection: Single Image or Folder
    mode = st.radio("Choose processing mode:", ["Single Image", "Folder Processing"])
    
    if mode == "Single Image":
        # Single image processing flow
        uploaded_file = st.file_uploader("Choose an image to colorize", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Load image and create small preview version
            image = Image.open(uploaded_file)
            small_preview = resize_image(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(small_preview, use_container_width=False)  # Use the small preview version
            
            # Process button
            process_btn = st.button("Colorize Image")
            
            if process_btn:
                if not st.session_state.model_path or not os.path.exists(st.session_state.model_path):
                    st.error("Please select a valid model file first.")
                    return
                    
                with st.spinner("Processing..."):
                    try:
                        # Initialize colorizer with the selected model path
                        colorizer = ImageColorizer(st.session_state.model_path, input_size)
                        
                        # Process the full-resolution image
                        start_time = time.time()
                        colored_image = process_single_image(colorizer, image, input_size)
                        processing_time = time.time() - start_time
                        
                        # Create small preview of the colorized result
                        small_colored_preview = resize_image(colored_image)
                        
                        # Display processed image preview
                        with col2:
                            st.subheader("Colorized Image")
                            st.image(small_colored_preview, use_container_width=False)  # Use the small preview version
                        
                        st.success(f"Image processed in {processing_time:.2f} seconds")
                        
                        # Add download button for the full-resolution result
                        buffered = io.BytesIO()
                        colored_image.save(buffered, format="JPEG")
                        st.download_button(
                            label="Download Colorized Image",
                            data=buffered.getvalue(),
                            file_name="colorized_image.jpg",
                            mime="image/jpeg"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Please upload an image to begin.")
    
    else:  # Folder Processing
        # Input folder selection
        st.subheader("Input Folder")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Select Input Folder", disabled=st.session_state.folder_dialog_active):
                st.session_state.folder_dialog_active = True
                with st.spinner("Opening folder dialog..."):
                    folder_path = select_file_using_subprocess(is_folder=True)
                    if folder_path:
                        st.session_state.input_folder = folder_path
                        st.success(f"Selected: {folder_path}")
                    else:
                        st.warning("No folder selected")
                st.session_state.folder_dialog_active = False
                st.rerun()
        
        # Output folder selection
        st.subheader("Output Folder")
        with col2:
            if st.button("Select Output Folder", disabled=st.session_state.folder_dialog_active):
                st.session_state.folder_dialog_active = True
                with st.spinner("Opening folder dialog..."):
                    folder_path = select_file_using_subprocess(is_folder=True)
                    if folder_path:
                        st.session_state.output_folder = folder_path
                        st.success(f"Selected: {folder_path}")
                    else:
                        st.warning("No folder selected")
                st.session_state.folder_dialog_active = False
                st.rerun()
        
        # Display selected folders
        if 'input_folder' in st.session_state and st.session_state.input_folder:
            st.info(f"Input Folder: {st.session_state.input_folder}")
        if 'output_folder' in st.session_state and st.session_state.output_folder:
            st.info(f"Output Folder: {st.session_state.output_folder}")
        
        # Live preview section with proper styling for smaller images
        st.subheader("Live Processing Preview")
        preview_cols = st.columns(2)
        
        # Create containers with styling for smaller images
        with preview_cols[0]:
            st.write("Original Image")
            original_container = st.empty()
        with preview_cols[1]:
            st.write("Colorized Result")
            colorized_container = st.empty()
            
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process folder button
        process_folder_btn = st.button("Process All Images")
        
        if process_folder_btn:
            if not st.session_state.model_path or not os.path.exists(st.session_state.model_path):
                st.error("Please select a valid model file first.")
                return
                
            if not hasattr(st.session_state, 'input_folder') or not st.session_state.input_folder:
                st.error("Please select an input folder.")
                return
                
            if not hasattr(st.session_state, 'output_folder') or not st.session_state.output_folder:
                st.error("Please select an output folder.")
                return
            
            status_text.text("Initializing...")
            
            try:
                # Initialize colorizer with the selected model path
                colorizer = ImageColorizer(st.session_state.model_path, input_size)
                
                # Process all images in the folder with live preview
                start_time = time.time()
                count, results = process_folder(
                    colorizer, 
                    st.session_state.input_folder, 
                    st.session_state.output_folder,
                    progress_bar,
                    status_text,
                    (original_container, colorized_container)
                )
                processing_time = time.time() - start_time
                
                if count > 0:
                    st.success(f"Processed {count} images in {processing_time:.2f} seconds")
                    
                    # Show a gallery of the processed images - using small previews
                    st.subheader("Processed Results Gallery")
                    
                    # Determine how many images to show in each row (more per row since images are smaller)
                    images_per_row = 4
                    for i in range(0, min(12, len(results)), images_per_row):
                        row_cols = st.columns(images_per_row)
                        for j in range(images_per_row):
                            if i + j < len(results):
                                with row_cols[j]:
                                    # Load and create small preview for gallery
                                    result_img = Image.open(results[i + j]["output_path"])
                                    small_gallery_img = resize_image(result_img)
                                    st.image(small_gallery_img, 
                                             caption=results[i + j]["filename"],
                                             use_container_width=False)
                    
                    # Path to results
                    st.info(f"All colorized images saved to: {st.session_state.output_folder}")
                else:
                    st.warning(f"No images found in the selected folder: {st.session_state.input_folder}")
                
            except Exception as e:
                st.error(f"Error processing folder: {str(e)}")
    
    # Display helpful information
    with st.expander("Help & Information"):
        st.markdown("""
        ### How to use this app:
        
        #### For Single Image Mode:
        1. Click "Browse for Model File" in the sidebar and select your .pth model file
        2. Upload an image you want to colorize
        3. Click the "Colorize Image" button
        4. Download the colorized result
        
        #### For Folder Processing Mode:
        1. Click "Browse for Model File" in the sidebar and select your .pth model file
        2. Select an input folder containing your images
        3. Select an output folder for the colorized results
        4. Click "Process All Images" button
        5. Watch the live processing preview as each image is colorized
        
        ### Troubleshooting:
        
        - If the file dialog doesn't appear, check if it opened behind your browser window
        - For large models, the first processing may take some time to load
        - Try adjusting the Input Size if you experience memory issues
        - If you encounter errors selecting large model files, try moving the file to a location with a shorter path
        """)

    # Display model size information if available
    if st.session_state.model_path and os.path.exists(st.session_state.model_path):
        file_size_mb = os.path.getsize(st.session_state.model_path) / (1024 * 1024)
        st.sidebar.info(f"Model file size: {file_size_mb:.1f} MB")

if __name__ == "__main__":
    main()

#streamlit run inference\app.py