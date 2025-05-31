import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from converter import *

def choose_image_file():
    Tk().withdraw()
    return filedialog.askopenfilename(title="Select an image",
                                      filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])

def show_images(original, pixel_art, palette):
    _, indexes, counts = np.unique(pixel_art[:, :, 0], return_index=True, return_counts=True)
    unique = pixel_art.reshape(-1, 3)[indexes]
    usage_percent = dict(zip(unique[:,1], counts / counts.sum() * 100))

    # Sort palette by brightness
    brightness = palette.mean(axis=1)
    sorted_indices = np.argsort(brightness)
    palette_sorted = palette[sorted_indices]
    usage_sorted = [usage_percent[palette_sorted[i,1]] for i in sorted_indices]

    # Plot original and pixel art
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Original")
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Pixel Art")
    ax2.imshow(cv2.cvtColor(pixel_art, cv2.COLOR_BGR2RGB))
    ax2.axis('off')

    # Palette
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.set_title("Color Palette with RGB & Usage")
    ax3.axis('off')

    for i, (color, percent) in enumerate(zip(palette_sorted, usage_sorted)):
        r, g, b = map(int, color)
        hex_color = '#%02x%02x%02x' % (r, g, b)
        ax3.add_patch(patches.Rectangle((i * 1.1, 0), 1, 1, color=hex_color))
        label = f"RGB: ({r},{g},{b}) — {percent:.1f}%"
        ax3.text(i * 1.1 + 0.05, -0.3, label, fontsize=10, rotation=45, ha='left', va='top')

    ax3.set_xlim(0, len(palette_sorted) * 1.1)
    ax3.set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()

def save_outputs(pixel_art, palette_sorted, numbered,  is_gray='y', output_dir='pixel_art_output'):
    os.makedirs(output_dir, exist_ok=True)
    if is_gray == 'y':
        img = cv2.cvtColor(pixel_art, cv2.COLOR_BGR2GRAY)
    else:
        img = pixel_art

    cv2.imwrite(os.path.join(output_dir, 'instructions.png'), numbered)
    cv2.imwrite(os.path.join(output_dir, 'pixel_art.png'), img)
    np.savetxt(os.path.join(output_dir, 'palette_rgb.csv'), palette_sorted, fmt='%d', delimiter=',', header='R,G,B')
    print(f"\n✅ Saved pixel art and palette to: {output_dir}/")

def apply_gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to the input image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


if __name__ == "__main__":
    image_path = choose_image_file()
    if not image_path:
        print("No file selected. Exiting.")
        exit()
    # image_path = 'Image2Pixel/img1.jpg'
    num_pixels = 32 # int(input("Enter number of pixels in width (e.g., 40): "))
    num_colors = 5 # int(input("Enter number of colors (e.g., 6): "))
    use_mosaic = 'n' # input("Apply grayscale mosaic effect like Mozabrick? (y/n): ").strip().lower()
    original = cv2.imread(image_path)
    denoise_method = 'nlm' # input("Denoising method? (none/gaussian/bilateral/nlm): ").strip().lower()
    img = denoise_image(original, method=denoise_method)
    img = apply_gamma_correction(img, 0.85)

    if use_mosaic == 'y':
        orginal, pixel_art, palette, shape = create_mosaic(img, num_tiles_wide=num_pixels, num_tones=num_colors)
        is_gray = 'y'

    if use_mosaic != 'y':
        mode = input("Convert to grayscale? (y/n): ").strip().lower()
        is_gray = (mode == 'y')

        original, pixel_art, palette, shape = convert_to_pixel_art(
            original, num_pixels, num_colors, is_gray=is_gray
        )

    show_images(original, pixel_art, palette)

    save = input("Save output image and palette? (y/n): ").strip().lower()
    if save == 'y':
        numbered = create_numbered_img(pixel_art, shape)
        brightness = palette.mean(axis=1)
        sorted_indices = np.argsort(brightness)
        palette_sorted = palette[sorted_indices]
        save_outputs(pixel_art, palette_sorted, numbered, is_gray)
