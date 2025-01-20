import os
from pathlib import Path
import torch

def save_checkpoint(model, optimizer, scaler, step, loss, save_dir="checkpoints"):
    """
    Save model checkpoint, overwriting previous checkpoint.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        scaler: The gradient scaler
        step: Current training step
        loss: Current loss value
        save_dir: Directory to save checkpoints
    """
    # Create checkpoint directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Single checkpoint name
    checkpoint_path = os.path.join(save_dir, "model_latest.pt")
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved at step {step}")

def load_checkpoint(model, optimizer, scaler, checkpoint_path="checkpoints/model_latest_10000.pt"):
    """
    Load model checkpoint.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        scaler: The gradient scaler
        checkpoint_path: Path to checkpoint file
    
    Returns:
        step: The training step from checkpoint
        loss: The loss value from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['step'], checkpoint['loss']

def save_lightweight_checkpoint(model, save_dir="checkpoints"):
    """
    Save only the model weights for deployment, excluding optimizer and training states.
    
    Args:
        model: The PyTorch model
        save_dir: Directory to save the lightweight checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Use a different name for the lightweight version
    checkpoint_path = os.path.join(save_dir, "model_lightweight_10000.pt")
    
    # Save only the model state dict
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nLightweight checkpoint saved at: {checkpoint_path}")

def load_lightweight_checkpoint(model, checkpoint_path="checkpoints/model_lightweight_10000.pt"):
    """
    Load only model weights from lightweight checkpoint.
    
    Args:
        model: The PyTorch model
        checkpoint_path: Path to lightweight checkpoint file
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load state dict with appropriate device
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device) 