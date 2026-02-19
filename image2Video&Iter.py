import cv2
import json
import argparse
import os
from tqdm import tqdm

def create_video(json_path, output_path, fps):
    # 1. Load the JSON data
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("JSON file is empty.")
        return

    # 2. Setup VideoWriter using the first image for dimensions
    first_img_path = data[0]['image_record_rgb']
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        print(f"Error: Could not read the first image to set resolution: {first_img_path}")
        return

    h, w, _ = first_frame.shape
    # 'mp4v' is highly compatible; 'avc1' is a good alternative for smaller files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing {len(data)} frames into {output_path} at {fps} FPS...")

    # 3. Iterate through frames
    for entry in tqdm(data, desc="Building Video"):
        img_path = entry['image_record_rgb']
        iteration = entry.get('iteration', 'N/A')

        frame = cv2.imread(img_path)
        if frame is None:
            continue # Skip missing files

        # 4. Add the Iteration Text (Green, Top-Left)
        # Text settings: (Image, String, (X,Y), Font, Scale, BGR_Color, Thickness)
        cv2.putText(
            frame, 
            f"Iteration: {iteration}", 
            (30, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            (0, 255, 0), 
            3, 
            cv2.LINE_AA
        )

        video_writer.write(frame)

    video_writer.release()
    print(f"\nDone! Video saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image-iteration JSON to an annotated video.")
    
    # Required arguments
    parser.add_argument("--input_json", type=str, required=True, help="Path to the JSON file containing image paths.")
    parser.add_argument("--output_dir", type=str, default="output.mp4", help="Name/Path of the output video file.")
    
    # Optional arguments
    parser.add_argument("--fps", type=int, default=10, help="Frame rate for the video (default: 10).")

    args = parser.parse_args()
    create_video(args.input_json, args.output_dir, args.fps)