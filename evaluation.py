import os
import json
import re
from tqdm import tqdm

def extract_conclusion(response_text):
    # Look for different formats of conclusions
    patterns = [
        # Original format at the end of text
        r'(?:A|B|C): (?:truth-teller|liar)(?:\n(?:A|B|C): (?:truth-teller|liar))*$',
        
        # Format with empty lines after
        r'(?:A|B|C): (?:truth-teller|liar)(?:\n(?:A|B|C): (?:truth-teller|liar))*\s*$',
        
        # Format with "Therefore" or "CONCLUSION" prefix
        r'(?:Therefore|CONCLUSION)[^\n]*\n\s*(?:A|B|C): (?:truth-teller|liar)(?:\n(?:A|B|C): (?:truth-teller|liar))*',
        
        # Numbered conclusion format (as fallback)
        r'\d+\.\s*\*{0,2}Conclusion:\*{0,2}.*?\n(?:\s*[^\n]*\n)*?\s*-\s*\*{0,2}(?:A|B|C):\s*(?:truth-teller|liar|Truth-teller|Liar)\*{0,2}\s*\n(?:\s*-\s*\*{0,2}(?:A|B|C):\s*(?:truth-teller|liar|Truth-teller|Liar)\*{0,2}\s*\n?)*',
        
        # Simple bullet point format (as fallback)
        r'(?:Thus|So|Therefore|Hence|Final),?\s+(?:the\s+)?identities\s*(?:are)?:\s*\n(?:\s*-\s*\*{0,2}(?:A|B|C):\s*(?:truth-teller|liar|Truth-teller|Liar)\*{0,2}\s*\n?)+'
    ]
    
    for pattern in patterns:
        match = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
        if match:
            # Get the last match
            conclusion = match[-1]
            result = {}
            
            # Parse each line
            for line in conclusion.split('\n'):
                if ':' in line:  # Only process lines with character assignments
                    char, identity = line.split(':')
                    # Convert to boolean (True for truth-teller, False for liar)
                    result[char.strip()] = (identity.strip().lower() == 'truth-teller')
            
            if result:  # Only return if we found valid assignments
                return result
    
    print("\nCouldn't extract conclusion from response:")
    print(response_text)
    print("-" * 50)
    return None

def evaluate_outputs(output_folder, test_data):
    correct = 0
    total = 0
    
    # Create a mapping of test data indices
    test_data_map = {i: data for i, data in enumerate(test_data)}
    
    for filename in tqdm(os.listdir(output_folder)):
        if not filename.endswith('.json'):
            continue
            
        # Get the index from filename
        try:
            idx_match = re.search(r'output_(\d+)\.json', filename)
            if not idx_match:
                continue
            idx = int(idx_match.group(1))
            
            # Skip if index not in test data
            if idx not in test_data_map:
                print(f"\nWarning: File {filename} has index {idx} which is not in test data")
                continue
            
            # Load model output
            with open(os.path.join(output_folder, filename), 'r') as f:
                output_data = json.load(f)
                
            # Extract conclusion from response
            model_answer = extract_conclusion(output_data['response'])
            if not model_answer:
                continue
                
            # Get correct solution using the map
            true_solution = test_data_map[idx]['solutions'][0]
            
            # Compare
            if model_answer == true_solution:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"\nError processing file {filename}: {str(e)}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    return accuracy

if __name__ == "__main__":
    # Load test data
    with open('test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Evaluate outputs
    accuracy = evaluate_outputs('output/gpt-4o-zero-CoT', test_data) 