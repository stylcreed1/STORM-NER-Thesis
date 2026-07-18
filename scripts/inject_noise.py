import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inject synthetic noise into CoNLL NER data")
    parser.add_argument("--input", type=str, required=True, help="Clean train file")
    parser.add_argument("--output", type=str, required=True, help="Output noisy file")
    parser.add_argument("--noise_rate", type=float, default=0.30, help="Percentage of tokens to corrupt (0.0 to 1.0)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Standard CoNLL-2003 Tags
    entity_tags = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER']
    all_tags = entity_tags + ['O']

    total_tokens = 0
    corrupted_tokens = 0

    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            # Preserve document boundaries and blank lines
            if not line or line.startswith("-DOCSTART-"):
                f_out.write(line + "\n")
                continue

            parts = line.split()
            word = parts[0]
            true_label = parts[-1]
            total_tokens += 1

            # Roll the dice! Does this token get corrupted?
            if random.random() < args.noise_rate:
                corrupted_tokens += 1
                
                # If it's an entity, corrupt it by turning it into 'O' (Missing Entity) 
                # or a wrong entity type (Classification Error)
                if true_label != 'O':
                    choices = [t for t in all_tags if t != true_label]
                    new_label = random.choice(choices)
                
                # If it's an 'O', hallucinate a fake entity (Spurious Entity)
                else:
                    new_label = random.choice(entity_tags)
                
                f_out.write(f"{word}\t{new_label}\n")
            else:
                f_out.write(f"{word}\t{true_label}\n")

    actual_rate = (corrupted_tokens / total_tokens) * 100
    print(f"\n☢️  SYNTHETIC NOISE INJECTION COMPLETE ☢️")
    print(f"Target Noise Rate: {args.noise_rate * 100}%")
    print(f"Tokens Corrupted:  {corrupted_tokens} out of {total_tokens} ({actual_rate:.2f}%)")
    print(f"Saved to: {args.output}\n")

if __name__ == '__main__':
    main()
