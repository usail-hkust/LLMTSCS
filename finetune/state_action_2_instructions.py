import argparse
from utils.my_utils import load_json, getPrompt, state2text, dump_json

def main(input_file, output_file):
    state_action_data = load_json(input_file)
    instructions = []

    for data in state_action_data:
        state = data['state']
        instruction = getPrompt(state2text(state))[-1]['content']
        output = data['prompt'][-1]['content'] if 'content' in data['prompt'][-1] else data['prompt'][-1]

        instructions.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    dump_json(instructions, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process state-action data and generate instructions.")
    parser.add_argument("--input_file", help="Input file path")
    parser.add_argument("--output_file", help="Output file path")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
