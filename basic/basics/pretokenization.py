import os
from typing import BinaryIO, Dict, List
import re
from collections import defaultdict
import multiprocessing as mp
from functools import partial

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer implementation with parallel pre-tokenization.
    """
    
    def __init__(self, special_tokens: List[str] = None):
        self.vocab = {}  # token -> id mapping
        self.inverse_vocab = {}  # id -> token mapping
        self.merges = []  # list of merge rules (byte_pair -> new_token)
        self.merge_rules = {}  # (byte1, byte2) -> merged_token mapping
        
        # Special tokens handling
        self.special_tokens = special_tokens or ["<|endoftext|>", "<|startoftext|>"]
        self.special_token_pattern = self._build_special_token_pattern()
        
    def _build_special_token_pattern(self):
        """Build regex pattern for splitting on special tokens."""
        if not self.special_tokens:
            return None
        
        # Escape special regex characters in tokens
        escaped_tokens = [re.escape(token) for token in self.special_tokens]
        # Join with | for OR pattern
        pattern = "|".join(escaped_tokens)
        return re.compile(f"({pattern})")
    
    def _chunk_corpus(self, text: str, chunk_size: int = 100000) -> List[str]:
        """
        Chunk corpus ensuring boundaries occur at special tokens.
        
        Args:
            text: Input corpus
            chunk_size: Approximate size of each chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # If not at the end, find the next special token boundary
            if end < len(text) and self.special_token_pattern:
                # Look ahead for special token
                search_start = end
                search_end = min(end + chunk_size // 4, len(text))  # Look ahead up to 25% of chunk size
                
                # Find the next special token after the current end position
                match = self.special_token_pattern.search(text, search_start, search_end)
                if match:
                    end = match.start()
                else:
                    # No special token found, use original end but try to break at whitespace
                    while end > start and end < len(text) and text[end] not in ' \n\t':
                        end -= 1
                    if end == start:  # Couldn't find whitespace, use original end
                        end = min(start + chunk_size, len(text))
            
            chunks.append(text[start:end])
            start = end
        
        return chunks
    
    def _process_chunk(self, chunk: str) -> List[str]:
        """
        Process a single chunk: remove special tokens and pre-tokenize.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            List of pre-tokens
        """
        if not self.special_token_pattern:
            # No special tokens, just pre-tokenize the whole chunk
            return self._pretokenize_text(chunk)
        
        # Split on special tokens
        segments = self.special_token_pattern.split(chunk)
        
        all_tokens = []
        for segment in segments:
            # Skip empty segments and special tokens themselves
            if segment and segment not in self.special_tokens:
                tokens = self._pretokenize_text(segment)
                all_tokens.extend(tokens)
        
        return all_tokens
    
    def _pretokenize_parallel(self, text: str, num_processes: int = None) -> List[str]:
        """
        Pre-tokenize text in parallel with special token handling.
        
        Args:
            text: Input text corpus
            num_processes: Number of processes to use (None for CPU count)
            
        Returns:
            List of pre-tokens
        """
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Chunk the corpus
        chunks = self._chunk_corpus(text)
        
        if len(chunks) == 1 or num_processes == 1:
            # Single chunk or single process, process directly
            return self._process_chunk(chunks[0])
        
        # Process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            chunk_results = pool.map(self._process_chunk, chunks)
        
        # Flatten results
        all_tokens = []
        for chunk_tokens in chunk_results:
            all_tokens.extend(chunk_tokens)
        
        return all_tokens

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str] = None, 
              min_frequency: int = 2, use_parallel: bool = True, num_processes: int = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train BPE tokenizer on corpus with parallel pre-tokenization.
        
        Args:
            input_path: Path to a text file with BPE tokenizer training data
            vocab_size: Target vocabulary size (including base bytes, merges, and special tokens)
            special_tokens: List of strings to add to the vocabulary
            min_frequency: Minimum frequency for merges
            use_parallel: Whether to use parallel pre-tokenization
            num_processes: Number of processes for parallel processing
            
        Returns:
            vocab: dict[int, bytes] - Mapping from token ID to token bytes
            merges: list[tuple[bytes, bytes]] - List of BPE merges in order of creation
        """
        print(f"Training BPE tokenizer with vocab_size={vocab_size}")
        
        # Update special tokens if provided
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.special_token_pattern = self._build_special_token_pattern()
            
        print(f"Special tokens: {self.special_tokens}")
        print(f"Using parallel processing: {use_parallel}")
        
        # Read the training data from file
        with open(input_path, 'r', encoding='utf-8') as f:
            text_corpus = f.read()
        
        # Step 1: Initialize vocabulary with individual bytes and special tokens
        self._initialize_base_vocab()
        self._add_special_tokens()
        
        # Step 2: Pre-tokenize corpus (with parallel processing and special token handling)
        if use_parallel:
            pretokens = self._pretokenize_parallel(text_corpus, num_processes)
        else:
            pretokens = self._process_chunk(text_corpus)
        
        print(f"Pre-tokenization complete. Found {len(pretokens)} tokens.")
        
        # Step 3: Convert pretokens to list of bytes for each word
        word_freqs = defaultdict(int)
        for token in pretokens:
            word_freqs[token] += 1
            
        print(f"Unique words: {len(word_freqs)}")
        
        # Convert words to byte sequences
        word_splits = {}
        for word in word_freqs:
            # Store as bytes objects, not integers
            word_splits[word] = [bytes([b]) for b in word.encode('utf-8')]
        
        # Step 4: Iteratively merge most frequent pairs
        current_vocab_size = len(self.vocab)
        
        while current_vocab_size < vocab_size:
            # Count all pairs
            pairs = self._get_pairs(word_splits, word_freqs)
            
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < min_frequency:
                break
                
            # Merge the best pair
            new_token_id = current_vocab_size
            merged_token = best_pair[0] + best_pair[1]
            
            # Add to vocabularies
            self.vocab[merged_token] = new_token_id
            self.inverse_vocab[new_token_id] = merged_token
            
            # Add merge rule
            self.merges.append(best_pair)
            self.merge_rules[best_pair] = merged_token
            
            # Update word splits
            word_splits = self._merge_vocab(best_pair, word_splits)
            
            current_vocab_size += 1
            
            if current_vocab_size % 1000 == 0:
                print(f"Vocab size: {current_vocab_size}")
        
        print(f"Training complete. Final vocab size: {len(self.vocab)}")
        
        # Return vocab and merges in the required format
        return self.inverse_vocab, self.merges

    def _pretokenize_text(self, text: str) -> List[str]:
        """
        Pre-tokenize text using regex pattern similar to GPT-2 tokenizer.
        This splits on whitespace and punctuation while keeping words intact.
        """
        # GPT-2 style regex pattern for pre-tokenization
        pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+")
        
        tokens = pattern.findall(text)
        # Filter out empty strings
        return [token for token in tokens if token]
        
    def _initialize_base_vocab(self):
        """Initialize vocabulary with all possible bytes."""
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Add all 256 possible byte values
        for i in range(256):
            byte_token = bytes([i])
            self.vocab[byte_token] = i
            self.inverse_vocab[i] = byte_token
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        current_id = len(self.vocab)
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.vocab:
                self.vocab[token_bytes] = current_id
                self.inverse_vocab[current_id] = token_bytes
                current_id += 1
    
    def _get_pairs(self, word_splits: Dict[str, list], word_freqs: Dict[str, int]) -> Dict[tuple, int]:
        """Count frequency of adjacent pairs across all words."""
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word_splits[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
                
        return dict(pairs)
    
    def _merge_vocab(self, pair: tuple, word_splits: Dict[str, list]) -> Dict[str, list]:
        """Apply merge rule to all word splits."""
        new_word_splits = {}
        merged_token = pair[0] + pair[1]  # This concatenates bytes
        
        for word in word_splits:
            new_split = []
            i = 0
            symbols = word_splits[word]
            
            while i < len(symbols):
                if (i < len(symbols) - 1 and 
                    symbols[i] == pair[0] and 
                    symbols[i + 1] == pair[1]):
                    # Merge the pair
                    new_split.append(merged_token)
                    i += 2
                else:
                    new_split.append(symbols[i])
                    i += 1
                    
            new_word_splits[word] = new_split
            
        return new_word_splits
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs using trained BPE with special token handling.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Process text with special token handling
        pretokens = self._process_chunk(text)
        token_ids = []
        
        for pretoken in pretokens:
            # Convert to bytes and apply BPE
            word_bytes = [bytes([b]) for b in pretoken.encode('utf-8')]
            
            # Apply merge rules
            while len(word_bytes) > 1:
                pairs = [(word_bytes[i], word_bytes[i + 1]) for i in range(len(word_bytes) - 1)]
                
                # Find first pair that has a merge rule
                merge_idx = None
                for i, pair in enumerate(pairs):
                    if pair in self.merge_rules:
                        merge_idx = i
                        break
                
                if merge_idx is None:
                    break
                    
                # Apply merge
                merged = self.merge_rules[pairs[merge_idx]]
                new_word = word_bytes[:merge_idx] + [merged] + word_bytes[merge_idx + 2:]
                word_bytes = new_word
            
            # Convert to token IDs
            for token in word_bytes:
                if isinstance(token, bytes) and token in self.vocab:
                    token_ids.append(self.vocab[token])
                elif isinstance(token, int):
                    # Single byte value, convert to bytes
                    byte_token = bytes([token])
                    if byte_token in self.vocab:
                        token_ids.append(self.vocab[byte_token])
                    else:
                        raise ValueError(f"Unknown byte token: {byte_token}")
                else:
                    raise ValueError(f"Unknown token type: {type(token)} - {token}")
        
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if not self.inverse_vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Convert IDs to tokens
        byte_sequence = b""
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                byte_sequence += token
            else:
                raise ValueError(f"Unknown token ID: {token_id}")
        
        # Decode bytes to string
        try:
            return byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            return byte_sequence.decode('utf-8', errors='replace')
    
    def save_vocab(self, vocab_path: str, merges_path: str):
        """Save vocabulary and merges to files."""
        import json
        
        # Save vocabulary (convert bytes to hex strings for JSON)
        vocab_serializable = {}
        for token, id in self.vocab.items():
            if isinstance(token, bytes):
                vocab_serializable[token.hex()] = id
            else:
                vocab_serializable[str(token)] = id
                
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_serializable, f, indent=2)
        
        # Save merges
        merges_serializable = []
        for pair in self.merges:
            merges_serializable.append([pair[0].hex(), pair[1].hex()])
            
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_serializable, f, indent=2)
    
    def load_vocab(self, vocab_path: str, merges_path: str):
        """Load vocabulary and merges from files."""
        import json
        
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = {}
        self.inverse_vocab = {}
        for token_hex, id in vocab_data.items():
            token = bytes.fromhex(token_hex)
            self.vocab[token] = id
            self.inverse_vocab[id] = token
        
        # Load merges
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_data = json.load(f)
        
        self.merges = []
        self.merge_rules = {}
        for pair_hex in merges_data:
            pair = (bytes.fromhex(pair_hex[0]), bytes.fromhex(pair_hex[1]))
            self.merges.append(pair)
            self.merge_rules[pair] = pair[0] + pair[1]


# Updated example usage
if __name__ == "__main__":
    # Test with sample text containing special tokens
    sample_text = """Hello world! This is document 1.<|endoftext|>This is document 2 with more text.<|endoftext|>And here's document 3!"""
    
    # Create a temporary file for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(sample_text)
        temp_file_path = f.name
    
    print("=== Enhanced BPE Tokenizer Example ===")
    print("Original text:", sample_text)
    print()
    
    try:
        # Train BPE tokenizer with special tokens
        special_tokens = ["<|endoftext|>", "<|startoftext|>"]
        tokenizer = BPETokenizer(special_tokens=special_tokens)
        vocab, merges = tokenizer.train(temp_file_path, vocab_size=300, special_tokens=special_tokens, min_frequency=1, use_parallel=True)
        print()
        
        print(f"Returned vocab type: {type(vocab)}")
        print(f"Returned merges type: {type(merges)}")
        print(f"Vocab size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")
        
        # Encode text
        token_ids = tokenizer.encode(sample_text)
        print("Encoded token IDs:", token_ids)
        print(f"Number of tokens: {len(token_ids)}")
        print()
        
        # Decode back
        decoded_text = tokenizer.decode(token_ids)
        print("Decoded text:", decoded_text)
        print("Matches original:", decoded_text == sample_text)
        print()
        
        # Show some vocabulary
        print("Sample vocabulary entries:")
        for i, (token_id, token_bytes) in enumerate(list(vocab.items())[:20]):
            try:
                readable = token_bytes.decode('utf-8')
                print(f"  {token_id}: {token_bytes} -> '{readable}'")
            except:
                print(f"  {token_id}: {token_bytes}")
            if i >= 10:
                break
        
        print(f"\nTotal vocabulary size: {len(vocab)}")
        print(f"Number of merge rules: {len(merges)}")
        
        # Show some merges
        print("\nSample merge rules:")
        for i, (token1, token2) in enumerate(merges[:10]):
            try:
                readable1 = token1.decode('utf-8')
                readable2 = token2.decode('utf-8')
                print(f"  {i}: {token1} + {token2} -> '{readable1}' + '{readable2}'")
            except:
                print(f"  {i}: {token1} + {token2}")
                
    finally:
        # Clean up temporary file
        import os
        os.unlink(temp_file_path)