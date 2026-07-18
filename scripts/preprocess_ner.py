import torch
import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Process NER CoNLL data into PyTorch Tensors")
    parser.add_argument("--clean_train", type=str, required=True, help="Path to the clean training data")
    parser.add_argument("--noisy_train", type=str, default=None, help="Path to noisy training data (if applicable)")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the dev/validation data")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the clean test data")
    parser.add_argument("--out_prefix", type=str, required=True, help="Prefix for output files (e.g., 'crowd', 'clean')")
    return parser.parse_args()

def read_conll(filepath):
    """Reads vertical CoNLL files into lists of words and labels."""
    sentences = []
    words, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"): continue
            if not line:
                if words:
                    sentences.append({"tokens": words, "labels": labels})
                    words, labels = [], []
            else:
                splits = line.split()
                words.append(splits[0])
                labels.append(splits[-1])
        if words:
            sentences.append({"tokens": words, "labels": labels})
    return sentences

def main():
    args = parse_args()

    print("1. Loading datasets...")
    clean_train = read_conll(args.clean_train)
    noisy_train = read_conll(args.noisy_train) if args.noisy_train else None
    dev_data = read_conll(args.dev_file)
    test_data = read_conll(args.test_file)

    # 2. Build the Universal Label Dictionary
    unique_tags = set()
    for dataset in [clean_train, dev_data, test_data]:
        for doc in dataset:
            for tag in doc["labels"]:
                unique_tags.add(tag)
    tag2id = {tag: i for i, tag in enumerate(sorted(list(unique_tags)))}
    print(f"2. Universal Label Mapping:\n{tag2id}\n")

    print("3. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./roberta_local", add_prefix_space=True)

    def process_data(clean_dataset, noisy_dataset=None):
        input_ids, attention_masks, labels, noise_labels = [], [], [], []
    
        for i in range(len(clean_dataset)):
            tokens = clean_dataset[i]["tokens"]
            c_labels = clean_dataset[i]["labels"]
            # If no noisy dataset is provided, just use the clean labels (0 noise)
            n_labels = noisy_dataset[i]["labels"] if noisy_dataset else c_labels
    
            # Tokenize with Padding and Truncation to 128 tokens
            encoded = tokenizer(
                tokens, is_split_into_words=True,
                max_length=128, padding='max_length', truncation=True
            )
    
            word_ids = encoded.word_ids()
            aligned_labels = []
            aligned_noise = []
            prev_word_idx = None
    
            for w_idx in word_ids:
                if w_idx is None:
                    aligned_labels.append(-100)
                    aligned_noise.append(-100)
                elif w_idx != prev_word_idx:
                    aligned_labels.append(tag2id[n_labels[w_idx]])
                    # Calculate noise (1 if corrupted, 0 if clean)
                    is_noise = 1 if c_labels[w_idx] != n_labels[w_idx] else 0
                    aligned_noise.append(is_noise)
                else:
                    aligned_labels.append(-100)
                    aligned_noise.append(-100)
                prev_word_idx = w_idx
    
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
            labels.append(aligned_labels)
            noise_labels.append(aligned_noise)
    
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(labels),
            "noise_labels": torch.tensor(noise_labels)
        }

    print(f"4. Tokenizing and Padding for '{args.out_prefix}' dataset...")
    print("   -> Processing Train Data...")
    train_tensors = process_data(clean_train, noisy_train)

    print("   -> Processing Dev Data...")
    dev_tensors = process_data(dev_data)

    print("   -> Processing Test Data...")
    test_tensors = process_data(test_data)

    print("5. Saving to disk...")
    torch.save(train_tensors, f"train_ner_{args.out_prefix}.pt")
    torch.save(dev_tensors, f"dev_ner_{args.out_prefix}.pt")
    torch.save(test_tensors, f"test_ner_{args.out_prefix}.pt")

    print(f"SUCCESS! {args.out_prefix} data is fully prepped for the GPU.")

if __name__ == "__main__":
    main()