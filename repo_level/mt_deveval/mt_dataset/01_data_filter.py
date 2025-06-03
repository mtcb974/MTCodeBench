import json
import argparse
from typing import List, Dict, Set, Optional

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return dictionary list"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved filtered data to: {file_path}")

def filter_cross_file_samples(data: List[Dict]) -> List[Dict]:
    """Filter out samples with cross_file in dependency"""
    filtered_data = []
    empty_dependency_count = 0
    
    for entry in data:
        dependency = entry.get('dependency', {})
        if not dependency.get('cross_file'):
            filtered_data.append(entry)
        else:
            empty_dependency_count += 1
    
    print(f"Filter out {empty_dependency_count} samples with cross_file in dependency")
    print(f"Remaining samples: {len(filtered_data)}")
    return filtered_data

def filter_standalone_samples(data: List[Dict]) -> List[Dict]:
    """Filter out samples with empty dependency"""
    filtered_data = []
    empty_dependency_count = 0
    
    for entry in data:
        dependency = entry.get('dependency', {})
        if dependency.get('intra_class') or dependency.get('intra_file') or dependency.get('cross_file'):
            filtered_data.append(entry)
        else:
            empty_dependency_count += 1
    
    print(f"Filter out {empty_dependency_count} samples with empty dependency")
    print(f"Remaining samples: {len(filtered_data)}")
    return filtered_data

def filter_existing_samples(data: List[Dict], filter_file: str) -> List[Dict]:
    """Filter out samples that are the same as the given JSON file"""
    try:
        with open(filter_file, 'r', encoding='utf-8') as f:
            filter_data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Filter file {filter_file} not found, skip this filter step")
        return data
    
    # Create a set of identifiers for filtering
    filter_ids = {entry.get('namespace') for entry in filter_data if 'namespace' in entry}
    
    filtered_data = []
    removed_count = 0
    
    for entry in data:
        if entry.get('namespace') not in filter_ids:
            filtered_data.append(entry)
        else:
            removed_count += 1
    
    print(f"Filter out {removed_count} samples that already exist")
    print(f"Remaining samples: {len(filtered_data)}")
    return filtered_data

# New filter function
def filter_body_position_length(data: List[Dict], filter_length: int) -> List[Dict]:
    """Keep samples with body lines greater than the given value"""
    filtered_data = []
    removed_count_function = 0
    removed_count_test = 0

    for entry in data:
        gt_code = entry.get('gt', '')
        test_codes = entry.get('test_codes',[])

        if not gt_code or len(test_codes) == 0:
            # If there is no gt field or test_codes, keep the sample
            filtered_data.append(entry)
            continue
            
        # If the length of test_codes is 1, check the number of lines
        if len(test_codes) == 1:
            test_code_lines = test_codes[0].split('\n')
            test_code_non_empty_lines = len([line for line in test_code_lines if line.strip()])
            if test_code_non_empty_lines < 10:
                removed_count_test += 1
                continue
            
        test_codes = '\n'.join(test_codes)
            
        # Delete function signature
        lines = gt_code.split('\n')
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                body_start = i + 1
                break
                
        # Get function body
        body_lines = lines[body_start:]
        
        # Delete comment lines
        body_lines = [line for line in body_lines if not line.strip().startswith('#')]
        
        # Calculate the number of non-empty lines
        non_empty_lines = len([line for line in body_lines if line.strip()])
        
        if non_empty_lines >= filter_length:
            filtered_data.append(entry)
        else:
            removed_count_function += 1

    print(f"Filter out {removed_count_function} samples with function body lines < {filter_length}, {removed_count_test} samples with test lines < {filter_length}")
    print(f"Remaining samples: {len(filtered_data)}")
    return filtered_data


def analyze_data(data: List[Dict]) -> None:
    """Analyze the dataset and print statistics"""
    lengths_with_args = []
    lengths_without_args = []
    body_position_lengths = []
    
    dependency_stats = {
        'total_samples': len(data),
        'samples_with_dependency': 0,
        'intra_class_count': 0,
        'intra_file_count': 0,
        'cross_file_count': 0
    }

    for entry in data:
        # Analyze dependency data
        dependency = entry.get('dependency', {})
        has_dependency = False
        
        for dep_type in ['intra_class', 'intra_file', 'cross_file']:
            if dependency.get(dep_type):
                has_dependency = True
                dependency_stats[f"{dep_type}_count"] += 1
                
        if has_dependency:
            dependency_stats['samples_with_dependency'] += 1

        # Calculate the length of functionality description
        requirement = entry.get('requirement', {})
        functionality = requirement.get('Functionality', '')
        arguments = requirement.get('Arguments', [])
        
        lengths_without_args.append(len(functionality))
        
        prompt_parts = [functionality]
        if arguments:
            prompt_parts.append(', '.join(map(str, arguments)))
        lengths_with_args.append(len(' '.join(prompt_parts)))

        # Calculate the length of body_position interval
        body_position = entry.get('body_position', [])
        if isinstance(body_position, list) and len(body_position) == 2:
            body_position_lengths.append(body_position[1] - body_position[0])

    def print_stats(title: str, data: List[float]) -> None:
        """Print statistics"""
        if not data:
            print(f"{title}: No data")
            return

        min_val = min(data)
        max_val = max(data)
        avg_val = sum(data) / len(data)

        print(f"\n--- {title} ---")
        print(f"Minimum value: {min_val}")
        print(f"Maximum value: {max_val}")
        print(f"Average value: {avg_val:.2f}")

        bin_size = (max_val - min_val) / 8
        bins = [0] * 8

        for val in data:
            idx = min(int((val - min_val) / bin_size), 7)
            bins[idx] += 1

        print("\nDistribution (8 bins):")
        for i, count in enumerate(bins):
            lower = min_val + i * bin_size
            upper = min_val + (i + 1) * bin_size
            print(f"[{lower:.1f}, {upper:.1f}): {count} samples")

    # Print statistics
    print_stats("Functionality + Arguments", lengths_with_args)
    print_stats("Functionality Only", lengths_without_args)
    print_stats("Body Position Interval Length", body_position_lengths)
    
    # Print dependency statistics
    print("\n--- Dependency ---")
    print(f"Total samples: {dependency_stats['total_samples']}")
    print(f"Samples with at least one non-empty dependency: {dependency_stats['samples_with_dependency']} "
          f"(Ratio: {dependency_stats['samples_with_dependency']/dependency_stats['total_samples']*100:.2f}%)")
    print(f"Samples with non-empty intra_class: {dependency_stats['intra_class_count']}")
    print(f"Samples with non-empty intra_file: {dependency_stats['intra_file_count']}")
    print(f"Samples with non-empty cross_file: {dependency_stats['cross_file_count']}")

def main():
    parser = argparse.ArgumentParser(description="Dataset filtering and analysis tool")
    parser.add_argument("--input_file", help="Input JSONL file path")
    parser.add_argument("--filter_file", help="Filter JSONL file path")
    parser.add_argument("--output_file", help="Output file path")
    parser.add_argument("--filter_line", type=int, help="Filter out samples with body_position interval length equal to this value")
    parser.add_argument("--filter_cross_file", action="store_true", help="Filter out samples with cross_file in dependency")
    
    args = parser.parse_args()

    args.input_file = "./backup/shard_multi_agent.jsonl"
    args.filter_file = "./backup/failed_samples.jsonl"
    args.output_file = "./0526_testdata4paper.jsonl"

    data = load_jsonl(args.input_file)
    print(f"The number of original samples: {len(data)}")

    data = filter_standalone_samples(data)

    if args.filter_cross_file:
        data = filter_cross_file_samples(data)

    if args.filter_file:
        data = filter_existing_samples(data, args.filter_file)

    if args.filter_line is not None:
        data = filter_body_position_length(data, args.filter_line)

    analyze_data(data)

    save_jsonl(data, args.output_file)

if __name__ == "__main__":
    main()