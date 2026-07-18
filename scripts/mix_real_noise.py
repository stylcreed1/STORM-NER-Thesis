import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Mix Real-World Noisy and Clean Sentences")
    parser.add_argument("--clean_input", type=str, required=True, help="Clean train file")
    parser.add_argument("--noisy_input", type=str, required=True, help="Real-world noisy train file")
    parser.add_argument("--output", type=str, required=True, help="Mixed output file")
    parser.add_argument("--noise_rate", type=float, default=0.30, help="Percentage of sentences to take from the noisy file")
    return parser.parse_args()

def get_sentences(filepath):
    """Reads a CoNLL formatted file and yields sentences as lists of lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        current_sentence = []
        for line in f:
            if line.strip() == "" or line.startswith("-DOCSTART-"):
                if current_sentence:
                    yield current_sentence
                    current_sentence = []
                yield [line] # Yield the empty line or DOCSTART as its own block
            else:
                current_sentence.append(line)
        if current_sentence:
            yield current_sentence

def main():
    args = parse_args()

    clean_sentences = list(get_sentences(args.clean_input))
    noisy_sentences = list(get_sentences(args.noisy_input))

    assert len(clean_sentences) == len(noisy_sentences), "Clean and Noisy files must have the exact same number of sentences and structure!"

    total_real_sentences = 0
    noisy_count = 0

    with open(args.output, 'w', encoding='utf-8') as f_out:
        for clean_sent, noisy_sent in zip(clean_sentences, noisy_sentences):
            # If it's a separator or DOCSTART, just write it and skip
            if len(clean_sent) == 1 and (clean_sent[0].strip() == "" or clean_sent[0].startswith("-DOCSTART-")):
                f_out.write(clean_sent[0])
                continue
                
            total_real_sentences += 1
            
            # Roll the dice: take from Noisy (Part A) or Clean (Part B)
            if random.random() < args.noise_rate:
                noisy_count += 1
                for line in noisy_sent:
                    f_out.write(line)
            else:
                for line in clean_sent:
                    f_out.write(line)

    print(f"\n🎯 REAL-WORLD NOISE MIXING COMPLETE 🎯")
    print(f"Target Rate: {args.noise_rate * 100}%")
    print(f"Sentences from Real Noisy File: {noisy_count} out of {total_real_sentences} ({(noisy_count/total_real_sentences)*100:.2f}%)")
    print(f"Saved to: {args.output}\n")

if __name__ == '__main__':
    main()
