import streamlit as st
import torch
import torch.nn.functional as F
from smollm2_135M import create_model
from checkpoint_utils import load_lightweight_checkpoint
from transformers import AutoTokenizer

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer (cached by Streamlit)"""
    # Create and load model
    model = create_model()
    model = load_lightweight_checkpoint(model, "checkpoints/model_lightweight_10000.pt")
    model.eval()
    
    # Load tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained("tokenizer", local_files_only=True)
    
    return model, tokenizer

def continue_text(model, tokenizer, prompt, max_new_tokens=50):
    """Continue the prompt text using the trained model"""
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    
    # Generate one token at a time
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions
            logits, _ = model(input_ids)
            
            # Get the next token (most likely)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Add the new token to our sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Decode the entire sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI
st.title("Text Continuation with SmolLM2")

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Input prompt
prompt = st.text_area("Enter your prompt:", height=100)

# Number of new tokens to generate
max_new_tokens = st.slider("Number of new tokens", min_value=10, max_value=200, value=50)

# Generate button
if st.button("Continue Text"):
    if prompt:
        with st.spinner("Generating..."):
            continued_text = continue_text(model, tokenizer, prompt, max_new_tokens)
            st.write("### Generated Continuation:")
            st.write(continued_text)
    else:
        st.warning("Please enter a prompt!") 