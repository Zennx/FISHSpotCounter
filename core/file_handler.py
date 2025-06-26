import os

def get_mask_path(image_path, masks_dir):
    """
    Given an image_path and masks_dir, return the corresponding mask path if the channel is accepted (Ch1 or Ch5).
    Image: SAMPLE_POPULATION_IMAGEID_CHANNELID.ome.tif
    Mask:  SAMPLE_POPULATION_IMAGEID_CHANNELID.dmask.pgm
    Returns None if channel is not Ch1 or Ch5.
    """
    base = os.path.basename(image_path)
    # Remove .ome.tif or .tif or .tiff
    for ext in [".ome.tif", ".tif", ".tiff"]:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    # Channel filter: only accept Ch1 or Ch5
    if ("Ch1" not in base) or ("Ch5" not in base):
        return None
    mask_name = base + ".dmask.pgm"
    mask_path = os.path.join(masks_dir, mask_name)
    return mask_path
