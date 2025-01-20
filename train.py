import torch
from torch.amp import autocast, GradScaler
from smollm2_135M import create_model
from cosmopedia_dataloader import create_cosmopedia_loader
from model_sampling import initialize_tokenizer, sample_model_output
from checkpoint_utils import save_checkpoint, load_checkpoint
import time

def train(
    batch_size: int = 16,
    sequence_length: int = 800,
    learning_rate: float = 3e-4,
    num_steps: int = 5000,
    model_path: str = 'HuggingFaceTB/SmolLM2-135M-Instruct',
    hf_token: str = None,
    sample_frequency: int = 500,
    checkpoint_frequency: int = 500,
    resume_from_checkpoint: bool = False
):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    torch.set_float32_matmul_precision('high')
    
    # Create model from scratch
    print("Creating model...")
    model = create_model()
    model.to(device)
    model.gradient_checkpointing_enable()
    
    # Compile model
    print("Compiling model...")
    model = torch.compile(model)
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(model_path, hf_token)
    
    # Create data loader
    train_loader = create_cosmopedia_loader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        tokenizer_path=model_path,
        hf_token=hf_token
    )
    
    # Initialize optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=True,
    )
    scaler = GradScaler()
    
    # Load checkpoint if resuming
    start_step = 0
    if resume_from_checkpoint:
        print("Loading checkpoint...")
        start_step, last_loss = load_checkpoint(model, optimizer, scaler)
        print(f"Resuming from step {start_step} with loss {last_loss:.6f}")
    
    # Training loop
    model.train()
    print("Starting training...")
    
    for step in range(start_step, num_steps):
        t0 = time.time()
        
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        # Forward pass with mixed precision
        with autocast(device_type=device):
            outputs = model(x, labels=y)
            loss = outputs[1] if isinstance(outputs, tuple) else outputs.loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Calculate timing and throughput
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (batch_size * sequence_length) / (t1 - t0)
        
        print(f'step{step} | loss: {loss.item():.6f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}')
        
        # Show sample output
        if step > 0 and step % sample_frequency == 0:
            print(f"\n=== Sample at step {step} ===")
            model.eval()
            with torch.no_grad(), autocast(device_type=device):
                sample_model_output(model, x, tokenizer)
            model.train()
            
        # Save checkpoint
        if step > 0 and step % checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, scaler, step, loss.item())

if __name__ == "__main__":
    HF_TOKEN = "hidden_for_security"
    
    train(
        batch_size=16,
        sequence_length=800,
        hf_token=HF_TOKEN,
        num_steps=5051,
        sample_frequency=500,
        checkpoint_frequency=5000,
        resume_from_checkpoint=True
    )