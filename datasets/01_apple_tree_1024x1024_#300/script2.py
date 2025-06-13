import json

# output path
output_path = "transforms.json"
# Read the JSON file and load its content into a Python dictionary

with open('transforms copy.json', 'r') as file:
    data = json.load(file)

    """
    To save from overwriting the original file, we can create a new file with a different name:
    "camera_angle_x": 0.9500215649604797,
    "camera_angle_y": 0.6605947017669678,
    "fl_x": 995.5555555555555,
    "fl_y": 995.5555555555555,
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "cx": 512.0,
    "cy": 512.0,
    "w": 1024.0,
    "h": 1024.0,
    "aabb_scale: 16,
    """

    for i,frame in enumerate(data["frames"]):
        # Removing semantic_path key from each frame
        data["frames"][i].pop("semantic_path", None)

    for i, frame in enumerate(data["frames"]):
        # Obtaining file_path of each frame
        file_path = frame["file_path"]
        # Replacing images to output in the string
        file_path = file_path.replace("images", "output")

        # Changing file_path of each frame"
        
        data["frames"][i].update({"file_path": file_path})

    # Saving the modified data to a new JSON file
    with open(output_path, 'w') as output_file:
        json.dump(data, output_file, indent=4)
        print(f"Modified data saved to {output_path}")


