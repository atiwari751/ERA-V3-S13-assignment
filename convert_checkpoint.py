import torch
from smollm2_135M import create_model
from checkpoint_utils import save_lightweight_checkpoint
import os

def convert_to_lightweight():
    # Create model
    model = create_model()
    
    # Load the full checkpoint
    checkpoint = torch.load("checkpoints/model_latest_10000.pt")
    
    # Get the model state dict and remove the "_orig_mod." prefix from keys
    state_dict = checkpoint['model_state_dict']
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            fixed_state_dict[new_key] = value
        else:
            fixed_state_dict[key] = value
    
    # Load the fixed state dict
    model.load_state_dict(fixed_state_dict)
    
    # Save lightweight version
    save_lightweight_checkpoint(model)
    
    # Print size comparison
    full_size = os.path.getsize("checkpoints/model_latest_10000.pt") / (1024 * 1024)  # MB
    light_size = os.path.getsize("checkpoints/model_lightweight_10000.pt") / (1024 * 1024)  # MB
    print(f"Full checkpoint size: {full_size:.2f} MB")
    print(f"Lightweight checkpoint size: {light_size:.2f} MB")

if __name__ == "__main__":
    convert_to_lightweight() 