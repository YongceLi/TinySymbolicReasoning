import json
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
import argparse
from tqdm import tqdm

def main(args):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    tokenizer.padding_side = 'left'
    
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        tokenizer=tokenizer,
        device=0,
        batch_size=2 
    )

    with open("prompt.txt", "r") as f:
        prompt_template = f.read()

    print("Loading data...")
    with open("test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]
    
    if args.debug:
        test_data = test_data[:min(5, len(test_data))]
    
    print("Formatting prompts...")
    formatted_prompts = []
    for example in tqdm(test_data, desc="Formatting prompts"):
        messages = [
            {
                "role": "system",
                "content": "You are a strong symbolic reasoning model."
            },
            {
                "role": "user",
                "content": prompt_template.replace("<statements>", '\n'.join(example["problem"])) \
                                        .replace("<num-characters>", str(example["metadata"]["num_characters"]))
            }
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)
    
    print(f"Generating responses for {len(formatted_prompts)} examples...")
    
    results = []
    batch_size = 2
    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating responses"):
        batch = formatted_prompts[i:i+batch_size]
        batch_results = pipe(
            batch,
            max_new_tokens=8192,
            return_full_text=False
        )
        results.extend(batch_results)
    
    print("Saving results...")
    generation = {}
    for i, result in enumerate(results):
        generation[test_data[i]["id"]] = result[0]["generated_text"]
    
    output_file = "Qwen_generation.json"
    with open(output_file, "w") as f:
        json.dump(generation, f, indent=2)
    
    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
