import argparse
import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
from basicsr.archs.colorfy_arch import Colorfy


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
        state_dict = torch.load(model_path, map_location="cpu")
        
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


def main():
    parser = argparse.ArgumentParser(description="Colorize grayscale images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--input", type=str, default="assets", help="Input folder")
    parser.add_argument("--output", type=str, default="results", help="Output folder")
    parser.add_argument("--input_size", type=int, default=512, help="Input size")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize colorizer
    colorizer = ImageColorizer(args.model_path, args.input_size)
    
    # Get list of images
    img_list = os.listdir(args.input)
    if not img_list:
        print("No images found in the input directory")
        return
    
    # Process images
    for name in tqdm(img_list, desc="Colorizing images"):
        img_path = os.path.join(args.input, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Skipping {name}: Unable to load image")
            continue
            
        try:
            colored_img = colorizer.colorize(img)
            cv2.imwrite(os.path.join(args.output, name), colored_img)
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    print(f"Processed images saved in {args.output}")


if __name__ == "__main__":
    main()

#python inference\colorization_pipeline.py --model_path net_g_20000.pth --input assets --output results --input_size 512