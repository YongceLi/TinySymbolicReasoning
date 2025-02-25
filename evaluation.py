import os
import json
import re
from tqdm import tqdm
import argparse

def extract_conclusion(response_text):
    # Look for different formats of conclusions
    patterns = [
        # LaTeX format with boxed and text commands (most common in output)
        r'\\\[\s*\\boxed{(?:(?:A|B|C):\s*\\text{(?:truth-teller|liar)}(?:\s*,\s*(?:A|B|C):\s*\\text{(?:truth-teller|liar)})*|\\text{(?:A|B|C):\s*(?:truth-teller|liar)(?:,\s*(?:A|B|C):\s*(?:truth-teller|liar))*})}\s*\\\]',
        
        # Format with conclusion indicators and various styles
        r'(?:Thus|So|Therefore|Hence|The final answer is|Final identities|In summary|Based on this analysis|We conclude|The identities are|Conclusively)(?:[^\n]*\n){0,3}(?:\s*[-\*•]?\s*\**(?:A|B|C):\s*(?:truth-teller|liar)\**(?:\n|$)){1,3}',
        
        # Original format at the end of text with flexible spacing
        r'(?:\s*(?:A|B|C):\s*(?:truth-teller|liar)(?:\n|\s)*){1,3}\s*$',
        
        # Format with "Therefore" or "CONCLUSION" prefix
        r'(?:Therefore|CONCLUSION)[^\n]*\n\s*(?:A|B|C):\s*(?:truth-teller|liar)(?:\n(?:A|B|C):\s*(?:truth-teller|liar))*',
        
        # Bullet point format with optional bold markers
        r'(?:\s*[-\*•]?\s*\**(?:A|B|C):\s*(?:truth-teller|liar)\**(?:\n|$)){1,3}'
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            if matches:
                # Get the last match
                conclusion = matches[-1]
                result = {}
                
                # Handle LaTeX format
                if '\\boxed' in conclusion:
                    conclusion = conclusion.replace('\\[', '').replace('\\]', '')
                    conclusion = conclusion.replace('\\boxed{', '').replace('}', '')
                    conclusion = conclusion.replace('\\text{', '').replace('}', '')
                    # Split on commas and process each part
                    parts = [p.strip() for p in conclusion.split(',')]
                    for part in parts:
                        if ':' in part:
                            char, identity = part.split(':')
                            char = char.strip()
                            identity = identity.strip().lower()
                            result[char] = (identity == 'truth-teller')
                else:
                    # Clean up markdown and other formatting
                    conclusion = conclusion.replace('**', '')
                    # Parse regular format
                    lines = [line for line in conclusion.split('\n') if ':' in line and any(c in line for c in ['A', 'B', 'C'])]
                    for line in lines:
                        char, identity = line.split(':')
                        char = char.strip(' -*•')
                        identity = identity.strip().lower()
                        result[char] = ('truth' in identity)
                
                if len(result) > 0 and len(result) <= 3:
                    return result
                
        except Exception as e:
            print(f"Error processing pattern: {str(e)}")
            continue
    
    print("\nCouldn't extract conclusion from response:")
    print(response_text[-200:])  # Print last 200 chars for debugging
    print("-" * 50)
    return None

def evaluate_outputs(output_folder, test_data, eval_mode='separate'):
    correct = 0
    total = 0
    missing = []
    failed_extraction = []
    
    # Create a mapping of test data indices
    test_data_map = {i: data for i, data in enumerate(test_data)}
    expected_indices = set(range(len(test_data)))  # Should be 0-149
    processed_indices = set()
    
    if eval_mode == 'full':
        # Handle full output file (e.g., Qwen_generation.json)
        try:
            with open(output_folder, 'r') as f:
                outputs = json.load(f)
            
            for idx_str in outputs:
                try:
                    idx = int(idx_str)
                    output = outputs[idx_str]
                    processed_indices.add(idx)
                    total += 1
                    
                    if idx not in test_data_map:
                        print(f"\nWarning: Output index {idx} not in test data")
                        continue
                        
                    model_answer = extract_conclusion(output)
                    if not model_answer:
                        failed_extraction.append(idx)
                        continue
                        
                    true_solution = test_data_map[idx]['solutions'][0]
                    
                    # Check that all three characters are present and match
                    if (set(model_answer.keys()) == set(['A', 'B', 'C']) and 
                        all(model_answer[k] == true_solution[k] for k in ['A', 'B', 'C'])):
                        correct += 1
                    
                except ValueError as e:
                    print(f"\nError processing index {idx_str}: {str(e)}")
                    continue
                
            missing = expected_indices - processed_indices
            
        except Exception as e:
            print(f"\nError processing full output file: {str(e)}")
    
    else:  # separate mode
        for filename in tqdm(os.listdir(output_folder)):
            if not filename.endswith('.json'):
                continue
            
            try:
                idx_match = re.search(r'output_(\d+)\.json', filename)
                if not idx_match:
                    continue
                idx = int(idx_match.group(1))
                processed_indices.add(idx)
                total += 1
                
                if idx not in test_data_map:
                    print(f"\nWarning: File {filename} has index {idx} which is not in test data")
                    continue
                
                with open(os.path.join(output_folder, filename), 'r') as f:
                    output_data = json.load(f)
                    
                # Add check for data point ID matching filename index
                if 'id' in output_data and output_data['id'] != idx:
                    print(f"\nWarning: File {filename} has index {idx} but data point ID is {output_data['id']}")
                    continue
                    
                model_answer = extract_conclusion(output_data['response'])
                if not model_answer:
                    failed_extraction.append(idx)
                    continue
                    
                true_solution = test_data_map[idx]['solutions'][0]
                
                # Check that all three characters are present and match
                if (set(model_answer.keys()) == set(['A', 'B', 'C']) and 
                    all(model_answer[k] == true_solution[k] for k in ['A', 'B', 'C'])):
                    correct += 1
                
            except Exception as e:
                print(f"\nError processing file {filename}: {str(e)}")
                continue
        
        missing = expected_indices - processed_indices
    
    # Print summary
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Missing indices: {sorted(list(missing))}")
    print(f"Failed extraction for indices: {sorted(failed_extraction)}")
    print(f"Total processed: {len(processed_indices)}")
    print(f"Total missing: {len(missing)}")
    print(f"Total failed extraction: {len(failed_extraction)}")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model outputs')
    parser.add_argument('--eval_mode', type=str, default='separate', 
                      choices=['separate', 'full'],
                      help='Evaluation mode: separate for individual files, full for single file')
    parser.add_argument('--eval_data', type=str, required=True,
                      help='Path to evaluation data file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to model outputs (folder for separate mode, file for full mode)')
    
    args = parser.parse_args()
    
    # Load test data
    with open(args.eval_data, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Run evaluation
    accuracy = evaluate_outputs(args.output_path, test_data, eval_mode=args.eval_mode) 
    
    # Usage:
    # For fine-tuned model:
    # python evaluation.py --eval_mode full  --eval_data test.jsonl --output_path Qwen_generation_sft.json
    # For GPT model:
    # python evaluation.py --eval_mode separate  --eval_data test.jsonl --output_path path_to_folder/