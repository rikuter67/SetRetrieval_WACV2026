import os
import json

def aggregate_annotations(scenes_dir, output_path):
    aggregated = []
    for scene_id in os.listdir(scenes_dir):
        scene_path = os.path.join(scenes_dir, scene_id)
        if os.path.isdir(scene_path):
            annotation_file = os.path.join(scene_path, 'annotation.json')
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    aggregated.append(data)
            else:
                print(f"Warning: {annotation_file} does not exist.")
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=4)
    print(f"Aggregated annotations saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate annotation.json files from all scenes into one file.')
    parser.add_argument('--scenes_dir', type=str, required=True, help='Path to scenes directory containing scene subdirectories.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the aggregated annotations.json file.')
    args = parser.parse_args()
    
    aggregate_annotations(args.scenes_dir, args.output_path)