import sys
from PIL import Image

def merge_images(image_path1, image_path2, transparency, output_path):
    """
    Merge two images with a specified degree of transparency and save the output.

    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    :param transparency: Degree of transparency (0 to 1). 0 means the second image is fully transparent,
                         1 means it is fully opaque.
    :param output_path: Path to save the merged image.
    """
    # Open the two images
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    
    # Resize image2 to match image1 size
    image2 = image2.resize(image1.size)
    
    # Adjust the transparency of image2 and merge the two images
    blended_image = Image.blend(image1, image2, transparency)

    # Save the result in JPEG format
    blended_image.save(output_path, format="JPEG")
    print(f"Merged image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge.py <image_path1> <image_path2> <transparency> <output_path>")
        sys.exit(1)
    
    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    transparency = float(sys.argv[3])
    output_path = sys.argv[4]
    
    merge_images(image_path1, image_path2, transparency, output_path)