import argparse
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train(corpus_path, output_path, vocab_size, special_tokens):
    # Validate inputs
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if not special_tokens:
        special_tokens = []
    if vocab_size <= len(special_tokens):
        raise ValueError(f"vocab_size ({vocab_size}) must be larger than number of special tokens ({len(special_tokens)})")

    # Prepare output directory
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # If corpus exists, stream lines (avoid loading very large files into memory).
    if os.path.exists(corpus_path):
        try:
            file_size = os.path.getsize(corpus_path)
        except Exception:
            file_size = None

        # For small files we read into memory to show a line count; for large files we stream.
        if file_size is not None and file_size <= 50 * 1024 * 1024:  # 50 MB
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = [l.rstrip('\n') for l in f if l.strip()]
            line_iter = iter(lines)
            approx_count = len(lines)
            print(f"Training BPE tokenizer on {approx_count} lines; vocab_size={vocab_size}")
        else:
            def line_iter_gen():
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    for l in f:
                        s = l.strip()
                        if s:
                            yield s

            line_iter = line_iter_gen()
            if file_size is None:
                print(f"Training BPE tokenizer from '{corpus_path}' (unknown size); vocab_size={vocab_size}")
            else:
                print(f"Training BPE tokenizer from '{corpus_path}' (size={file_size/(1024**2):.1f} MB); vocab_size={vocab_size}")
    else:
        # Small fallback so script doesn't crash
        fallback = [
            "Hello world.",
            "This is a small fallback corpus. Replace with a real corpus for best results."
        ]
        line_iter = iter(fallback)
        print(f"Corpus not found at '{corpus_path}'. Using fallback corpus; vocab_size={vocab_size}")

    # Build tokenizer and trainer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # Train and save with basic error handling
    try:
        tokenizer.train_from_iterator(line_iter, trainer=trainer)
    except Exception as e:
        raise RuntimeError(f"Tokenizer training failed: {e}")

    try:
        tokenizer.save(output_path)
        print(f"Tokenizer saved to: {output_path}")
        print("Resulting vocab size:", len(tokenizer.get_vocab()))
    except Exception as e:
        raise RuntimeError(f"Failed to save tokenizer to '{output_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--corpus", "-c", default="corpus.txt", help="Path to corpus file (one or many lines).")
    parser.add_argument("--output", "-o", default="tokenizer.json", help="Output tokenizer file.")
    parser.add_argument("--vocab-size", "-v", type=int, default=2000000, help="Vocabulary size (e.g. 5000).")
    parser.add_argument("--special-tokens", "-s", nargs="*", default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "USER:", "AI:", "<END>"],
                        help="Special tokens to include.")
    args = parser.parse_args()

    train(args.corpus, args.output, args.vocab_size, args.special_tokens)