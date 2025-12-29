from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
import pathlib


# Tensor to PIL
def tensor2pil(image):
    """
    Convert a PyTorch tensor to a PIL Image.

    Args:
        image (torch.Tensor): Input tensor to be converted, expected to be in range [0, 1].

    Returns:
        PIL.Image: The converted image in RGB mode.
    """
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# Convert PIL to Tensor
def pil2tensor(image):
    """
    Convert a PIL Image to a PyTorch tensor.

    Args:
        image (PIL.Image): Input image to be converted.

    Returns:
        torch.Tensor: The converted tensor in range [0, 1] with an added batch dimension.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def remove_background(image, torchscript_jit="default", mode="fast"):
    """
    Remove the background from an image using a pre-trained model.

    Args:
        image (PIL.Image): Input image with background to be removed.
        torchscript_jit (str, optional): Whether to use TorchScript JIT. Defaults to "default".
        mode (str, optional): Processing mode. Can be "fast" or other modes supported by the remover.
                             Defaults to "fast".

    Returns:
        numpy.ndarray: Image with transparent background in RGBA format.
    """
    if (torchscript_jit == "default"):
        remover = Remover()
    else:
        remover = Remover(jit=True, mode=mode)

    removed = remover.process(image, type='rgba')
    return removed
    # mid = remover.process(tensor2pil(img), type='rgba')
    # out = pil2tensor(mid)
    # img_list.append(out)
    # img_stack = torch.cat(img_list, dim=0)
    # mask = img_stack[:, :, :, 3]
    # return (img_stack, mask)


def process_image(img_path, torchscript_jit="default", mode="fast"):
    """
    Process an image to remove its background and save the result.

    This function loads an image, removes its background using the specified method,
    and saves the result to a new file in a 'bg_removed' subdirectory.

    Args:
        img_path (str): Path to the input image file.
        torchscript_jit (str, optional): Whether to use TorchScript JIT. Defaults to "default".
        mode (str, optional): Processing mode for background removal. Defaults to "fast".

    Note:
        The processed image is saved in a 'bg_removed' subdirectory with 'bgrem_' prefix.
    """
    img = Image.open(img_path)
    bgremoved = remove_background(img, torchscript_jit, mode)
    save_bg_removed_image(img_path, bgremoved)


def save_bg_removed_image(orig_image_path, processed_image: Image):
    """
    Saves a processed image to a "bg_removed" subdirectory, creating it if necessary.

    Args:
        orig_image_path: Path to the original image file.
        processed_image: PIL Image object representing the processed image.
    """

    orig_path = pathlib.Path(orig_image_path)
    orig_dir = orig_path.parent
    bg_removed_dir = orig_dir / "bg_removed"

    # Create the "bg_removed" directory if it doesn't exist
    bg_removed_dir.mkdir(parents=True, exist_ok=True)

    # Create the new file name
    orig_filename_base = orig_path.stem  # Filename without extension
    new_filename = f"bgrem_{orig_filename_base}.png"
    new_filepath = bg_removed_dir / new_filename

    # Save the processed image
    processed_image.save(new_filepath)
    print(f"Saved processed image to: {new_filepath}")
