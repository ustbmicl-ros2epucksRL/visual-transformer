import os

# Define the path for the temporary split directory and the file
split_dir = "train/temp_eval_split"
file_path = os.path.join(split_dir, "traj_names.txt")
trajectory_name = "no1vc_9_0"

# Ensure the directory exists
os.makedirs(split_dir, exist_ok=True)

# Create the file with explicit UTF-8 encoding (the most compatible format)
try:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(trajectory_name + "\n")
    print(f"Successfully created '{file_path}' with correct UTF-8 encoding.")
except Exception as e:
    print(f"An error occurred: {e}")
