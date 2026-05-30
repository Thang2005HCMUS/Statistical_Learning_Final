import kagglehub

# Define where you want the dataset to be saved
custom_folder = "./data"

# Download directly to the specified folder
path = kagglehub.dataset_download("flap1812/fisheye8k", output_dir=custom_folder)

print("Path to dataset files:", path)