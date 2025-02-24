import json

with open("prompt.txt", "r") as f:
    prompt = f.read()

content_template = """
We can represent the problem as the following symbolic logic.
<problem_logic>

Let's use symbolic reasoning to solve the problem.
<think><symbolic_reasoning></think>
<answer><solutions></answer>
"""

with open("train.jsonl", "r") as f:
    train_data = f.readlines()

sft_train = []
for data in train_data:
    data = json.loads(data)
    curr_data = {}
    curr_data["messages"] = []
    curr_data["messages"].append(
        {
            "role": "system",
            "content": "You are a strong symbolic reasoning model."
        }
    )
    prompt_text = prompt.replace("<statements>", '\n'.join(data["problem"])).replace("<num-characters>", str(data["metadata"]["num_characters"]))
    curr_data["messages"].append(
        {
            "role": "user",
            "content": prompt_text
        }
    )
    problem_logic = '\n'.join(data["problem_logic"])
    symbolic_reasoning = '\n'.join([f"step {i + 1}: " + data["symbolic_reasoning"].split("\n")[i] for i in range(len(data["symbolic_reasoning"].split("\n")))])
    solutions = ""
    for key in data["solutions"][0]:
        if data["solutions"][0][key]:
            solutions += f"{key}: truth-teller\n"
        else:
            solutions += f"{key}: liar\n"
    content = content_template.replace("<problem_logic>", problem_logic).replace("<symbolic_reasoning>", symbolic_reasoning).replace("<solutions>", solutions)
    curr_data["messages"].append(
        {
            "role": "assistant",
            "content": content
        }
    )
    sft_train.append(curr_data)

with open("sft_train.json", "w") as f:
    json.dump(sft_train, f, indent=2)