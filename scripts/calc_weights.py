import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="train_ner_crowd30.pt",
                    help="Path to the processed .pt training tensor")
args = parser.parse_args()
# 1. Load your training data
data = torch.load(args.input, map_location='cpu')
labels = data['labels']

# 2. Flatten the tensor and remove the -100 padding tokens
valid_labels = labels[labels != -100]

N = valid_labels.numel()
C = 9 # Number of unique classes

# 3. Count occurrences of each class (0 through 8)
counts = torch.bincount(valid_labels, minlength=C)

# 4. Calculate Mathematical Weights: W_i = N / (C * n_i)
weights = N / (C * counts.float() + 1e-8) # 1e-8 prevents division by zero

# 5. Apply Square Root Smoothing to prevent Gradient Explosions
smoothed_weights = torch.sqrt(weights)

# Print the results
tag2id = {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}

print(f"Total Valid Tokens (N): {N}")
print("-" * 65)
weights_array = []

for tag, idx in tag2id.items():
    raw_w = round(weights[idx].item(), 4)
    smooth_w = round(smoothed_weights[idx].item(), 4)
    weights_array.append(smooth_w)
    
    print(f"Class {idx} ({tag:<6}) | Count: {counts[idx]:<6} | Raw: {raw_w:<7} | Smoothed: {smooth_w}")

print("-" * 65)
print("Copy-paste this mathematically smoothed array into your code:")
print(f"weights_array = {weights_array}")
