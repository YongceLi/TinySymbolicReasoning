import re
from tqdm import tqdm
import argparse
import json
from sympy import symbols, Not, And, Or, Equivalent, simplify_logic
from sympy.logic.boolalg import Implies

def parse_expression(expr_str, symbol_map):
    """
    Convert a string logical expression to a Sympy expression.
    
    Replacements:
      - '∧' -> '&'
      - '∨' or ' v ' -> '|'
      - '¬' -> '~'
    Uses regex to convert:
      - 'X ↔ Y' -> 'Equivalent(X, Y)'
      - 'X → Y' -> 'Implies(X, Y)'
    by repeatedly substituting until all ↔ and → are gone.
    """
    # 1) Basic text replacements
    expr_str = expr_str.replace("∧", " & ")
    expr_str = expr_str.replace("∨", " | ")
    expr_str = expr_str.replace(" v ", " | ")
    expr_str = expr_str.replace("¬", "~")
    
    # 2) Regex pattern allowing single vars, ~var, or a parenthesized group
    pattern = r'(\w+|~\s*\w+|\(.*?\))'
    
    # 3) Keep substituting ↔ and → until none remain
    old_str = None
    while old_str != expr_str:
        old_str = expr_str
        expr_str = re.sub(
            rf'{pattern}\s*↔\s*{pattern}',
            r'Equivalent(\1, \2)',
            expr_str
        )
        expr_str = re.sub(
            rf'{pattern}\s*→\s*{pattern}',
            r'Implies(\1, \2)',
            expr_str
        )
    
    # 4) Safelist for eval
    allowed = {
        "__builtins__": None,
        "And": And,
        "Or": Or,
        "Not": Not,
        "Equivalent": Equivalent,
        "Implies": Implies,
        **symbol_map
    }
    return eval(expr_str, allowed)


def compare_logical_expressions(expr1_str, expr2_str, vars_list=None):
    """Check if two logical expressions (in string form) are equivalent."""
    if vars_list is None:
        vars_list = ["A", "B", "C"]  # or whatever variables you need

    symbol_map = {var: symbols(var) for var in vars_list}
    
    expr1 = parse_expression(expr1_str, symbol_map)
    expr2 = parse_expression(expr2_str, symbol_map)
    
    simplified_expr1 = simplify_logic(expr1, form='dnf', force=True)
    simplified_expr2 = simplify_logic(expr2, form='dnf', force=True)

    return simplified_expr1.equals(simplified_expr2)

def extract_steps(text):
    think_content = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not think_content:
        return []
    steps = re.findall(r'step\s*\d+:\s*(.*?)(?=\s*step\s*\d+:|$)', think_content.group(1), re.DOTALL)
    return [step.strip() for step in steps]

def calculate_reasoning_accuracy(model_output):
    false_count = 0
    total_count = 0
    extracted_steps = extract_steps(model_output)
    for i in range(len(extracted_steps) - 1):
        total_count += 1
        try:
            result = compare_logical_expressions(extracted_steps[i], extracted_steps[i + 1])
            if result == False:
                false_count += 1
        except:
            false_count += 1
    return false_count / total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model reasoning outputs')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to evaluation output file')
    parser.add_argument('--data', type=str, required=True, help='Path to train or test data file')
    
    
    args = parser.parse_args()
    
    with open(args.output_path, 'r') as f:
        output = json.load(f)
    with open(args.data, 'r') as f:
        dataset = f.readlines()

    false_count = 0
    total_count = 0
    for key in tqdm(output):
        extracted_steps = extract_steps(output[key])
        for i in range(len(extracted_steps) - 1):
            total_count += 1
            try:
                result = compare_logical_expressions(extracted_steps[i], extracted_steps[i + 1])
                if result == False:
                    false_count += 1
            except:
                false_count += 1
    print(f"Total sequence pair reasoning: {total_count}, False: {false_count}\nAccuracy: {(total_count - false_count) / total_count:.2%}, False Rate: {false_count / total_count:.2%}")
    gt_count = 0
    gt_false_count = 0
    for key in tqdm(output):
        groundtruth = json.loads(dataset[int(key)])['symbolic_reasoning'].split("\n")[0]
        extracted_steps = extract_steps(output[key])
        for i in range(len(extracted_steps)):
            gt_count += 1
            try:
                result = compare_logical_expressions(extracted_steps[i], groundtruth)
                if result == False:
                    gt_false_count += 1
            except:
                gt_false_count += 1
    print(f"Total reasoning: {gt_count}, False: {gt_false_count}\nAccuracy: {(gt_count - gt_false_count) / gt_count:.2%}, False Rate: {gt_false_count / gt_count:.2%}")