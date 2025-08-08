import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


import re
from collections import defaultdict
from typing import Dict


def pretokenize_text(text: str) -> list[str]:
    """
    Pre-tokenize text using regex pattern similar to GPT-2 tokenizer.
    This splits on whitespace and punctuation while keeping words intact.
    """
    # GPT-2 style regex pattern for pre-tokenization (compatible with Python re)
    pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+")
    
    tokens = pattern.findall(text)
    # Filter out empty strings
    return [token for token in tokens if token]


def count_pretokens_in_chunk(text_chunk: str) -> Dict[str, int]:
    """
    Count pre-tokens in a text chunk for BPE training.
    """
    pretokens = pretokenize_text(text_chunk)
    counts = defaultdict(int)
    
    for token in pretokens:
        counts[token] += 1
    
    return dict(counts)


def get_byte_pairs_from_pretoken(pretoken: str) -> list[tuple[bytes, bytes]]:
    """
    Get all adjacent byte pairs from a pre-token for BPE training.
    """
    token_bytes = pretoken.encode('utf-8')
    pairs = []
    
    for i in range(len(token_bytes) - 1):
        pair = (bytes([token_bytes[i]]), bytes([token_bytes[i + 1]]))
        pairs.append(pair)
    
    return pairs


def count_byte_pairs_in_text(text: str) -> Dict[tuple[bytes, bytes], int]:
    """
    Count all byte pairs in pre-tokenized text for BPE training.
    """
    pretokens = pretokenize_text(text)
    pair_counts = defaultdict(int)
    
    for pretoken in pretokens:
        pairs = get_byte_pairs_from_pretoken(pretoken)
        for pair in pairs:
            pair_counts[pair] += 1
    
    return dict(pair_counts)


## Usage - Complete BPE preprocessing pipeline
def process_corpus_for_bpe(file_path: str, num_processes: int = 4) -> Dict[tuple[bytes, bytes], int]:
    """
    Process entire corpus for BPE training by chunking and counting byte pairs.
    """
    total_pair_counts = defaultdict(int)
    
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Process each chunk  
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Count byte pairs in this chunk
            chunk_pair_counts = count_byte_pairs_in_text(chunk)
            
            # Add to total counts
            for pair, count in chunk_pair_counts.items():
                total_pair_counts[pair] += count
    
    return dict(total_pair_counts)


# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = "Hello world! This is a test. <|endoftext|> More text here."
    
    print("=== Basic Pre-tokenization Example ===")
    print("Original text:", sample_text)
    print()
    
    # Pre-tokenize
    pretokens = pretokenize_text(sample_text)
    print("Pre-tokens:", pretokens)
    print()
    
    # Count byte pairs
    pair_counts = count_byte_pairs_in_text(sample_text)
    print("First 10 byte pairs:")
    for i, (pair, count) in enumerate(list(pair_counts.items())[:10]):
        print(f"  {pair}: {count}")
    print()
    
    # Example of how to use with actual corpus files
    print("=== Corpus Processing Example ===")
    print("To process a corpus file, use:")
    print("  pair_counts = process_corpus_for_bpe('path/to/corpus.txt', num_processes=4)")
    print("This will:")
    print("  1. Split the file into chunks at <|endoftext|> boundaries") 
    print("  2. Pre-tokenize each chunk")
    print("  3. Count byte pairs across all chunks")
    print("  4. Return aggregated counts for BPE training")
