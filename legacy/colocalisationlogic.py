        
        
        # --- Masking logic --- line 211
        mask_img = masks[i]
        if mask_img is not None and blobs is not None and len(blobs) > 0:
            # Assume mask is binary (nonzero = inside mask)
            filtered_blobs = []
            for blob in blobs:
                y, x, r = blob
                y_int, x_int = int(round(y)), int(round(x))
                if 0 <= y_int < mask_img.shape[0] and 0 <= x_int < mask_img.shape[1]:
                    if mask_img[y_int, x_int] > 0:
                        filtered_blobs.append(blob)
            blobs = np.array(filtered_blobs)