import cv2
import numpy as np
from PIL.ImImagePlugin import number
from sklearn.cluster import KMeans

def create_numbered_img(pixel_art, output_size):
    """
    Creates a visualization with numbers replacing gray values.

    Parameters:
    - tone_map: 2D array of tone indices (e.g., 0–5)
    - output_size: (width, height) tuple for final upscaled image

    Returns:
    - image_with_numbers: An image with numbers drawn on each tile
    """


    h, w, c = pixel_art.shape
    tile_w = 50
    tile_h = 50

    # Create blank white image
    img = np.ones((tile_h * h, tile_w * w, 3), dtype=np.uint8) * 255

    numbers = pixel_art[:,:,0]
    _, indexes, counts = np.unique(pixel_art[:, :, 0], return_index=True, return_counts=True)
    unique = pixel_art.reshape(-1, 3)[indexes]
    for i, level in enumerate(unique[:, 0]):
        numbers[np.where(pixel_art[:,:,0] == level)] = i

    for i in range(h):
        for j in range(w):
            tone_number = int(numbers[i, j])  # Start from 1
            x = j * tile_w
            y = i * tile_h

            # Draw tile border (optional)
            cv2.rectangle(img, (x, y), (x + tile_w, y + tile_h), (200, 200, 200), 1)

            # Draw number
            cv2.putText(
                img,
                str(tone_number),
                (x + tile_w // 4, y + int(tile_h // 1.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4 * min(tile_w, tile_h) / 20,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )

    return img

def create_mosaic(image, num_tiles_wide=96, num_tones=6):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute output mosaic size
    h, w = gray.shape
    aspect_ratio = h / w
    num_tiles_high = int(num_tiles_wide * aspect_ratio)

    # Resize to tile resolution
    small = cv2.resize(gray, (num_tiles_wide, num_tiles_high), interpolation=cv2.INTER_AREA)

    # Quantize grayscale to num_tones
    bins = np.linspace(0, 256, num_tones + 1, endpoint=True)
    digitized = np.digitize(small, bins) - 1
    levels = (255 * digitized / (num_tones - 1)).astype(np.uint8)
    palette = cv2.cvtColor(np.unique(levels), cv2.COLOR_GRAY2BGR).squeeze()
    mosaic = cv2.cvtColor(levels, cv2.COLOR_GRAY2BGR)

    return image, mosaic, palette, small.shape

def resize_image(img, width_pixels):
    aspect_ratio = img.shape[0] / img.shape[1]
    new_height = int(width_pixels * aspect_ratio)
    return cv2.resize(img, (width_pixels, new_height), interpolation=cv2.INTER_NEAREST)

def denoise_image(img, method='none'):
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (3, 3), sigmaX=1)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'nlm':
        return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    else:
        return img

def quantize_colors_lab(image, n_colors):
    h, w, c = image.shape
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    reshaped = image_lab.reshape(-1, c)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    labels = kmeans.fit_predict(reshaped)
    palette_lab = kmeans.cluster_centers_.astype("uint8")
    palette_bgr = cv2.cvtColor(palette_lab[np.newaxis, :, :], cv2.COLOR_Lab2BGR)[0]
    quantized = palette_bgr[labels].reshape((h, w, c))
    return quantized, palette_bgr

def convert_to_pixel_art(img, num_pixels, num_colors, is_gray=False):
    if img is None:
        raise ValueError("Failed to read image file.")

    if is_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    small = resize_image(img, num_pixels)
    pixel_art, palette = quantize_colors_lab(small, num_colors)
    return img, pixel_art, palette, small.shape[:2]

def quantize_to_custom_palette(image, palette_bgr):
    """
    Quantizes an image to a given color palette.

    Parameters:
    - image: input image in BGR format (np.ndarray)
    - palette_bgr: array of BGR colors (shape: [N_colors, 3])

    Returns:
    - quantized: output image with same shape as input, colors from palette
    - palette_bgr: reordered palette (same as input)
    """
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    palette_lab = cv2.cvtColor(palette_bgr[np.newaxis, :, :], cv2.COLOR_BGR2Lab)[0]

    # Compute distances to palette
    distances = np.linalg.norm(img_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :], axis=2)
    closest = np.argmin(distances, axis=1)

    # Map to palette
    quantized_flat = palette_bgr[closest]
    quantized = quantized_flat.reshape(image.shape)
    return quantized, palette_bgr
