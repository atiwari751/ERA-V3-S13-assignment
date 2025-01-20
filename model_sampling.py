from transformers import AutoTokenizer
import torch

def decode_tokens(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

def sample_model_output(model, x, tokenizer, max_preview_length=200):
    """
    Generate and display a sample output from the model.
    
    Args:
        model: The language model
        x: Input tensor batch
        tokenizer: HuggingFace tokenizer
        max_preview_length: Number of characters to show in preview
    """
    print("\n=== Sample Output ===")
    
    # Take the first sequence from the batch
    input_tokens = x[0].cpu().tolist()
    input_text = decode_tokens(tokenizer, input_tokens)
    print("\nInput Text:")
    print(input_text[:max_preview_length] + "...")
    
    # Get model prediction
    outputs = model(x)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    pred_tokens = torch.argmax(logits[0], dim=-1).cpu().numpy().tolist()
    pred_text = decode_tokens(tokenizer, pred_tokens)
    print("\nModel Output:")
    print(pred_text[:max_preview_length] + "...")
    print("\n==================\n")

def initialize_tokenizer(model_path, hf_token=None):
    """Initialize and return the tokenizer."""
    return AutoTokenizer.from_pretrained(model_path, token=hf_token)

if __name__ == "__main__":
    # Test code that only runs if this file is executed directly
    print("This module provides sampling functionality for the language model.") 