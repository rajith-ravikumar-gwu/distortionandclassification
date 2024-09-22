import os
import subprocess
import matplotlib.pyplot as plt
from efficient80 import get_top1_accuracy
from PIL import Image

# Define paths
image_path1 = r'C:\Users\rajit\Pictures\tank.jpg'
image_path2 = r'C:\Users\rajit\Pictures\golden.jpg'
output_dir = r'C:\Users\rajit\Pictures\merged_images'
os.makedirs(output_dir, exist_ok=True)

# Define transparency levels
transparency_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Generate merged images with different transparency levels
for transparency in transparency_levels:
	output_path = os.path.join(output_dir, f'goldentank_{transparency:.1f}effnet.jpg')
	subprocess.run(['python', 'merge.py', image_path1, image_path2, str(transparency), output_path])

# Analyze accuracy for different transparency levels
accuracies = []
for transparency in transparency_levels:
	image_path = os.path.join(output_dir, f'goldentank_{transparency:.1f}effnet.jpg')
	label, accuracy = get_top1_accuracy(image_path)
	accuracies.append(accuracy)
	print(f"Transparency {transparency:.1f}: {label} - {accuracy:.2f}%")

# Plot the accuracy levels
plt.figure(figsize=(10, 6))
plt.plot(transparency_levels, accuracies, marker='o')
plt.title('Top-1 Accuracy vs. Transparency Level')
plt.xlabel('Transparency Level')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(True)
plt.show()