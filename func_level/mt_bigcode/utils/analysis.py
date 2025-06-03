import json
import argparse
from collections import defaultdict, Counter


def analyze_dataset(dataset):
    total_instances = len(dataset)
    total_turn_instances = 0
    turn_counts = Counter()  # Count the number of each turn
    example_lengths = []     # Record the number of turns for each instance
    mt_ids_by_turns = defaultdict(list)  # Record mt_id by turn (here is actually the task name)

    # Iterate through the dataset to count
    for instance in dataset:
        task_name = next(iter(instance.keys()))  # Get the task name, such as "BigCodeBench/17"
        turns_list = instance[task_name]         # Get all turns of this task

        num_turns = len(turns_list)

        # If there is a "turn" field, you can consider taking the maximum value as the turn
        if turns_list and 'turn' in turns_list[0]:
            num_turns = max(t['turn'] for t in turns_list)

        turn_counts[num_turns] += 1
        total_turn_instances += num_turns
        example_lengths.append(num_turns)
        mt_ids_by_turns[num_turns].append(task_name)

    # Output the statistics
    print(f"Total tasks: {total_instances}")
    print(f"Total turn instances: {total_turn_instances}")
    print("\nTurn distribution statistics:")
    for turns, count in sorted(turn_counts.items()):
        percentage = (count / total_instances) * 100
        task_names = mt_ids_by_turns[turns]
        print(f"{turns} turns have {count} tasks ({percentage:.2f}%): [{', '.join(task_names)}]")

    # Statistics for other information
    if example_lengths:
        min_turns = min(example_lengths)
        max_turns = max(example_lengths)
        avg_turns = sum(example_lengths) / total_instances
    else:
        min_turns = max_turns = avg_turns = 0

    print("\nOther statistics:")
    print(f"Minimum turns: {min_turns}")
    print(f"Maximum turns: {max_turns}")
    print(f"Average turns: {avg_turns:.2f}")


def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found, please check if the path is correct.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON format.")
        exit(1)


def main():
    # Set the command line argument parser
    parser = argparse.ArgumentParser(description="Analyze the statistics of the multi-turn code generation dataset.")
    parser.add_argument("--file_path", type=str, help="The path of the JSON file containing decompose",default="./logs/decompose.json")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_json_file(args.file_path)

    # Analyze the dataset
    analyze_dataset(dataset)


if __name__ == "__main__":
    main()