from inference import get_model

# Load model
# This downloads the weights locally and caches them
model = get_model(
    model_id="indian-vehicle-classification-lhwbt/3",
    api_key="G6YCLRL0kx2uJGzFhRhi"
)

print("\nModel loaded successfully!\n")

# Print all classes
print("ALL MODEL CLASSES:")
print("=" * 50)

for class_id, class_name in enumerate(model.class_names):
    print(f"{class_id}: {class_name}")

print("=" * 50)
print(f"\nTotal classes: {len(model.class_names)}")


from roboflow import Roboflow

# Initialize with your API key
rf = Roboflow(api_key="G6YCLRL0kx2uJGzFhRhi")

# Get the specific project and version you were using
project = rf.workspace().project("indian-vehicle-classification-lhwbt")
dataset = project.version(3).download("yolov11") # This downloads the weights!

print("Model downloaded successfully!")
