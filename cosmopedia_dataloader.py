import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple, Iterator
import random

class CosmopediaDataLoader:
    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        tokenizer_path: str = 'HuggingFaceTB/SmolLM2-135M-Instruct',
        dataset_name: str = "HuggingFaceTB/smollm-corpus",
        subset: str = "cosmopedia-v2",
        streaming: bool = True,
        hf_token: str = None
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, subset, streaming=streaming, token=hf_token)["train"]
        self.data_iter = iter(self.dataset)
        
        # Buffer settings
        self.token_buffer = []
        self.prefetch_factor = 5  # Keep 5 batches worth of data
        self.min_buffer_size = batch_size * sequence_length * self.prefetch_factor
        
        # Initial buffer fill
        self._fill_buffer()
        print("Initialized Cosmopedia dataloader")
    
    def _tokenize_text(self, example: dict) -> list:
        """Tokenize text and extract random coherent chunks"""
        text = example['text']
        
        # Split into sentences (rough approximation)
        sentences = text.split('.')
        tokens_list = []
        
        # Tokenize each sentence separately
        for sentence in sentences:
            if not sentence.strip():
                continue
            tokens = self.tokenizer(
                sentence.strip() + ".",  # Add period back
                truncation=False,
                padding=False,
                return_tensors=None
            )['input_ids']
            tokens_list.append(tokens)
        
        # Combine sentences into chunks of approximately sequence_length
        chunks = []
        current_chunk = []
        current_length = 0
        
        for tokens in tokens_list:
            if current_length + len(tokens) > self.sequence_length:
                if current_chunk:  # Save current chunk if it exists
                    chunks.append(current_chunk)
                current_chunk = tokens
                current_length = len(tokens)
            else:
                current_chunk.extend(tokens)
                current_length += len(tokens)
        
        if current_chunk:  # Don't forget the last chunk
            chunks.append(current_chunk)
        
        # Return random chunk if we have any
        if chunks:
            # Randomly select a chunk
            chunk = random.choice(chunks)
            # Pad or truncate to exact sequence length
            if len(chunk) > self.sequence_length:
                return chunk[:self.sequence_length]
            else:
                return chunk + [self.tokenizer.pad_token_id] * (self.sequence_length - len(chunk))
        return []
    
    def _fill_buffer(self) -> None:
        """Fill buffer with tokens"""
        print("Refilling buffer...")
        current_size = len(self.token_buffer)
        
        try:
            while len(self.token_buffer) < self.min_buffer_size:
                example = next(self.data_iter)
                tokens = self._tokenize_text(example)
                if tokens:
                    self.token_buffer.extend(tokens)
                
                if torch.cuda.is_available() and random.random() < 0.1:
                    torch.cuda.empty_cache()
                    
        except StopIteration:
            self.data_iter = iter(self.dataset)
            if len(self.token_buffer) == 0:
                self._fill_buffer()
        
        # Shuffle at sequence boundaries
        if len(self.token_buffer) > current_size:
            sequences = [self.token_buffer[i:i + self.sequence_length] 
                        for i in range(0, len(self.token_buffer), self.sequence_length)]
            random.shuffle(sequences)
            self.token_buffer = [token for seq in sequences for token in seq]
        
        print(f"Buffer refilled: {current_size} → {len(self.token_buffer)} tokens")
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch while maintaining sequence coherence"""
        if len(self.token_buffer) < self.min_buffer_size // 2:
            self._fill_buffer()
        
        # Get complete sequences for this batch
        tokens_needed = self.batch_size * self.sequence_length
        tokens = self.token_buffer[:tokens_needed]
        self.token_buffer = self.token_buffer[tokens_needed:]
        
        # Create input and target tensors
        x = torch.tensor(tokens).view(self.batch_size, self.sequence_length)
        y = torch.roll(x, shifts=-1, dims=-1)
        y[:, -1] = x[:, 0]  # Wrap around for last token
        
        return x, y
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self.next_batch()

def create_cosmopedia_loader(
    batch_size: int,
    sequence_length: int,
    tokenizer_path: str = 'HuggingFaceTB/SmolLM2-135M-Instruct',
    hf_token: str = None
) -> CosmopediaDataLoader:
    return CosmopediaDataLoader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        tokenizer_path=tokenizer_path,
        hf_token=hf_token
    )

if __name__ == "__main__":
    # Test the dataloader
    HF_TOKEN = "hidden_for_security"  # Replace with your Hugging Face token
    loader = create_cosmopedia_loader(batch_size=4, sequence_length=128, hf_token=HF_TOKEN)
    x, y = loader.next_batch()
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    print(f"Sample tokens: {x[0, :10]}")  # Show first 10 tokens of first sequence 