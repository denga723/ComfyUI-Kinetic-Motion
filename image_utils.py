import torch
import numpy as np
from PIL import Image

class CropAndResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "crop_position": (["center", "top", "bottom", "left", "right"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "JR_Nodes/Image"

    def process(self, image, width, height, crop_position):
        # image is (Batch, H, W, C) tensor
        # Process batch
        results = []
        for img_tensor in image:
            # Convert tensor to PIL
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Target dimensions
            target_w, target_h = width, height
            src_w, src_h = img.size
            
            # Calculate scale to preserve aspect ratio while filling target
            scale = max(target_w / src_w, target_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            
            # Resize
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Crop
            left = (new_w - target_w) / 2
            top = (new_h - target_h) / 2
            
            if crop_position == "top":
                top = 0
            elif crop_position == "bottom":
                top = new_h - target_h
            elif crop_position == "left":
                left = 0
            elif crop_position == "right":
                left = new_w - target_w
            # else center (default)
            
            right = left + target_w
            bottom = top + target_h
            
            img_cropped = img_resized.crop((int(left), int(top), int(right), int(bottom)))
            
            # Convert back to tensor
            img_out = np.array(img_cropped).astype(np.float32) / 255.0
            results.append(torch.from_numpy(img_out))
            
        return (torch.stack(results),)

class ImageTileSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["rows_cols", "pixel_size"],),
                "rows": ("INT", {"default": 2, "min": 1, "max": 64}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 64}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_aspect_ratio": (["source", "1:1", "16:9", "9:16", "4:3", "3:4"],),
                "fit_strategy": (["crop", "pad"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "split"
    CATEGORY = "JR_Nodes/Image"

    def split(self, image, mode, rows, cols, tile_width, tile_height, tile_aspect_ratio, fit_strategy):
        results = []
        
        for img_tensor in image:
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            original_w, original_h = img.size
            
            # 1. Determine dimensions
            if mode == "rows_cols":
                # Determine target tile AR
                target_ar = original_w / original_h
                if tile_aspect_ratio == "1:1": target_ar = 1.0
                elif tile_aspect_ratio == "16:9": target_ar = 16/9
                elif tile_aspect_ratio == "9:16": target_ar = 9/16
                elif tile_aspect_ratio == "4:3": target_ar = 4/3
                elif tile_aspect_ratio == "3:4": target_ar = 3/4
                
                # Calculate required total size to satisfy rows/cols and AR
                # We want: total_w / cols = tile_w
                #          total_h / rows = tile_h
                #          tile_w / tile_h = target_ar
                # So: (total_w / cols) / (total_h / rows) = target_ar
                #     (total_w * rows) / (total_h * cols) = target_ar
                
                # We need to map the original image into this grid.
                # Strategy: 
                # 1. Treat the whole image as covering the grid.
                # 2. Resizing/Padding/Cropping the original image to match the grid aspect ratio.
                # Grid AR = (cols * tile_w) / (rows * tile_h) ... wait, tile_w/tile_h is fixed by target_ar.
                # Grid AR = (cols / rows) * target_ar
                
                grid_ar = (cols / rows) * target_ar
                
                # Resize/Crop/Pad source to match Grid AR
                src_ar = original_w / original_h
                
                if fit_strategy == "crop":
                    # Crop logic (Aspect Fill behavior from before)
                    scale = max(1, 1) # Dummy, we just need the rect
                    # If src is wider than grid, crop width
                    if src_ar > grid_ar:
                        new_h = original_h
                        new_w = int(original_h * grid_ar)
                    else:
                        new_w = original_w
                        new_h = int(original_w / grid_ar)
                        
                    # Center crop source to new_w, new_h
                    left = (original_w - new_w) // 2
                    top = (original_h - new_h) // 2
                    img = img.crop((left, top, left + new_w, top + new_h))
                    
                elif fit_strategy == "pad":
                    # Pad logic (Fit Inside)
                    if src_ar > grid_ar:
                        # Src is wider, pad height
                        target_h_padded = int(original_w / grid_ar)
                        target_w_padded = original_w
                    else:
                        # Src is taller, pad width
                        target_w_padded = int(original_h * grid_ar)
                        target_h_padded = original_h
                        
                    new_img = Image.new("RGBA", (target_w_padded, target_h_padded), (0, 0, 0, 0))
                    paste_left = (target_w_padded - original_w) // 2
                    paste_top = (target_h_padded - original_h) // 2
                    new_img.paste(img, (paste_left, paste_top))
                    img = new_img.convert("RGB") # Convert back to RGB for now, or keep RGBA if supported
                
                # Now we have an image that matches the Grid AR exactly.
                # Split it.
                current_w, current_h = img.size
                step_w = current_w // cols
                step_h = current_h // rows
                
                for r in range(rows):
                    for c in range(cols):
                        left = c * step_w
                        top = r * step_h
                        tile = img.crop((left, top, left + step_w, top + step_h))
                        if tile.mode == "RGBA":
                            tile = tile.convert("RGB")
                        results.append(torch.from_numpy(np.array(tile).astype(np.float32) / 255.0))
            
            elif mode == "pixel_size":
                # Split into chunks of tile_width x tile_height
                # Pad or Crop the potential edges?
                # User said: "crop or pad to ensure all tiles are whole"
                
                # required_w = ceil(original_w / tile_w) * tile_w
                import math
                req_cols = math.ceil(original_w / tile_width)
                req_rows = math.ceil(original_h / tile_height)
                
                target_w = req_cols * tile_width
                target_h = req_rows * tile_height
                
                if fit_strategy == "pad":
                    new_img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                    # Center the image? Or top-left?
                    # Usually tiling implies starting from top-left.
                    new_img.paste(img, (0, 0))
                    img = new_img.convert("RGB")
                elif fit_strategy == "crop":
                    # Crop to nearest multiple... wait, usually implies shrinking.
                    # If we crop, we lose data.
                    # "Crop to size": means strict enforcement.
                    # Let's assume we crop to floor multiple.
                    target_w = (original_w // tile_width) * tile_width
                    target_h = (original_h // tile_height) * tile_height
                    # Center crop
                    left = (original_w - target_w) // 2
                    top = (original_h - target_h) // 2
                    img = img.crop((left, top, left + target_w, top + target_h))
                    
                    req_cols = target_w // tile_width
                    req_rows = target_h // tile_height

                for r in range(req_rows):
                    for c in range(req_cols):
                        left = c * tile_width
                        top = r * tile_height
                        tile = img.crop((left, top, left + tile_width, top + tile_height))
                        if tile.mode == "RGBA":
                            tile = tile.convert("RGB")
                        
                        # Ensure tile is exactly the size (it should be)
                        results.append(torch.from_numpy(np.array(tile).astype(np.float32) / 255.0))
                        
        if not results:
             return (torch.zeros((1, 64, 64, 3)),) # Fallback

        return (torch.stack(results),)

class ImageBatchSeamProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1}),
                "cols": ("INT", {"default": 2, "min": 1}),
                # seam_width and seam_height removed, using source size
                "mask_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("col_seams", "row_seams", "corner_seams")
    FUNCTION = "provide_seams"
    CATEGORY = "JR_Nodes/Image"

    def provide_seams(self, images, rows, cols, mask_ratio):
        # images: [B, H, W, C]
        if images.shape[0] != rows * cols:
            print(f"Warning: Image batch size {images.shape[0]} does not match rows*cols {rows*cols}")
        
        tile_height = images.shape[1]
        tile_width = images.shape[2]
        seam_width = tile_width
        seam_height = tile_height

        
        def get_tile(r, c):
             idx = r * cols + c
             if idx < len(images):
                 return images[idx]
             return images[0] # Fallback
        
        def to_rgba(img_tensor):
            if img_tensor.shape[2] == 4:
                return img_tensor
            alpha = torch.ones((img_tensor.shape[0], img_tensor.shape[1], 1), device=img_tensor.device)
            return torch.cat((img_tensor, alpha), dim=2)
            
        col_seams = []
        row_seams = []
        corner_seams = []
        
        # Col Seams
        for r in range(rows):
            for c in range(cols - 1):
                left_tile = to_rgba(get_tile(r, c))
                right_tile = to_rgba(get_tile(r, c+1))
                
                half_w = seam_width // 2
                
                left_part = left_tile[:, -half_w:, :]
                right_part = right_tile[:, :seam_width - half_w, :] 
                
                seam = torch.cat((left_part, right_part), dim=1) 
                
                # Mask
                # 1.0 = All transparent
                # 0.0 = All opaque
                seam[..., 3] = 1.0 # Reset alpha
                
                current_w = seam.shape[1]
                mask_w = int(current_w * mask_ratio)
                if mask_w > 0:
                    start = (current_w - mask_w) // 2
                    end = start + mask_w
                    seam[:, start:end, 3] = 0.0
                
                col_seams.append(seam)
        
        # Row seams
        for r in range(rows - 1):
            for c in range(cols):
                top_tile = to_rgba(get_tile(r, c))
                bottom_tile = to_rgba(get_tile(r+1, c))
                
                half_h = seam_height // 2
                
                top_part = top_tile[-half_h:, :, :]
                bottom_part = bottom_tile[:seam_height - half_h, :, :]
                
                seam = torch.cat((top_part, bottom_part), dim=0)
                
                seam[..., 3] = 1.0
                current_h = seam.shape[0]
                mask_h = int(current_h * mask_ratio)
                
                if mask_h > 0:
                    start = (current_h - mask_h) // 2
                    end = start + mask_h
                    seam[start:end, :, 3] = 0.0
                
                row_seams.append(seam)

        # Corner Seams
        for r in range(rows - 1):
            for c in range(cols - 1):
                tl = to_rgba(get_tile(r, c))
                tr = to_rgba(get_tile(r, c+1))
                bl = to_rgba(get_tile(r+1, c))
                br = to_rgba(get_tile(r+1, c+1))
                
                hw = seam_width // 2
                hh = seam_height // 2
                
                p_tl = tl[-hh:, -hw:, :]
                p_tr = tr[-hh:, :seam_width-hw, :]
                p_bl = bl[:seam_height-hh, -hw:, :]
                p_br = br[:seam_height-hh, :seam_width-hw, :]
                
                r1 = torch.cat((p_tl, p_tr), dim=1)
                r2 = torch.cat((p_bl, p_br), dim=1)
                seam = torch.cat((r1, r2), dim=0)
                
                seam[..., 3] = 1.0
                h, w = seam.shape[0], seam.shape[1]
                
                mask_h = int(h * mask_ratio)
                mask_w = int(w * mask_ratio)
                
                if mask_h > 0 and mask_w > 0:
                     start_h = (h - mask_h) // 2
                     end_h = start_h + mask_h
                     start_w = (w - mask_w) // 2
                     end_w = start_w + mask_w
                     seam[start_h:end_h, start_w:end_w, 3] = 0.0
                
                corner_seams.append(seam)
        
        def stack_or_empty(l):
            if not l: return torch.zeros((1, seam_height, seam_width, 4))
            return torch.stack(l)
            
        return (stack_or_empty(col_seams), stack_or_empty(row_seams), stack_or_empty(corner_seams))




class ImageTileComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rows": ("INT", {"default": 2, "min": 1}),
                "cols": ("INT", {"default": 2, "min": 1}),
                "mask_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100}),
                "alpha_growth": ("INT", {"default": 0, "min": -100, "max": 100}),
            },
            "optional": {
                "composited_image": ("IMAGE",),
                "base_tiles": ("IMAGE",),
                "col_seams": ("IMAGE",),
                "row_seams": ("IMAGE",),
                "corner_seams": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "debug_tiles")
    FUNCTION = "composite"
    CATEGORY = "JR_Nodes/Image"

    def composite(self, rows, cols, mask_ratio, feather, alpha_growth, composited_image=None, base_tiles=None, col_seams=None, row_seams=None, corner_seams=None):
        from PIL import ImageFilter, ImageDraw
        
        def process_seam(img_tensor, mode):
            # img_tensor: [H, W, C]
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode != "RGBA": img = img.convert("RGBA")
            
            w, h = img.size
            mask = Image.new("L", (w, h), 0) # Initialize Black (Transparent)
            draw = ImageDraw.Draw(mask)
            
            # Calculate Opaque Region (White)
            # Inverted logic from SeamProvider: The "Seam" is the part we want to KEEP (Opaque).
            
            if mode == "col":
                # Mask width only
                mask_w = int(w * mask_ratio)
                if mask_w > 0:
                    start = (w - mask_w) // 2
                    draw.rectangle((start, 0, start + mask_w, h), fill=255)
            elif mode == "row":
                # Mask height only
                mask_h = int(h * mask_ratio)
                if mask_h > 0:
                    start = (h - mask_h) // 2
                    draw.rectangle((0, start, w, start + mask_h), fill=255)
            elif mode == "corner":
                # Mask box
                mask_w = int(w * mask_ratio)
                mask_h = int(h * mask_ratio)
                if mask_w > 0 and mask_h > 0:
                    start_w = (w - mask_w) // 2
                    start_h = (h - mask_h) // 2
                    draw.rectangle((start_w, start_h, start_w + mask_w, start_h + mask_h), fill=255)
            
            # Growth (Dilation/Erosion)
            if alpha_growth != 0:
                if alpha_growth > 0:
                    mask = mask.filter(ImageFilter.MaxFilter(alpha_growth * 2 + 1))
                else:
                    mask = mask.filter(ImageFilter.MinFilter(abs(alpha_growth) * 2 + 1))
            
            # Feather (Blur)
            if feather > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(feather))
                
            img.putalpha(mask)
            return img

        # 1. Determine Canvas Dimensions & Initialize
        canvas = None
        ref_w, ref_h = 0, 0
        
        if composited_image is not None:
            # Use existing image as base
            t = composited_image[0] # Take first frame
            i = 255. * t.cpu().numpy()
            canvas = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGBA")
            full_w, full_h = canvas.size
            ref_w = full_w // cols
            ref_h = full_h // rows
            
        elif base_tiles is not None:
             # Infer from base tiles
             ref_tile_tensor = base_tiles[0]
             ref_h, ref_w = ref_tile_tensor.shape[0], ref_tile_tensor.shape[1]
             full_w = ref_w * cols
             full_h = ref_h * rows
             canvas = Image.new("RGBA", (full_w, full_h), (0, 0, 0, 255))
             
        elif col_seams is not None:
             # Infer from col seams (height is ref_h, width is ref_w)
             # Seam width might be same as tile width in this logic
             s = col_seams[0]
             ref_h, ref_w = s.shape[0], s.shape[1]
             full_w = ref_w * cols
             full_h = ref_h * rows
             canvas = Image.new("RGBA", (full_w, full_h), (0, 0, 0, 255))
        
        if canvas is None:
             # Fallback if nothing is connected or inferred
             return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64, 4)))

        # 2. Paste Base Tiles
        if base_tiles is not None:
            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    if idx < len(base_tiles):
                        t = base_tiles[idx]
                        i = 255. * t.cpu().numpy()
                        tile_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGBA")
                        # If we have a composite image, we paste ON TOP? Or UNDER?
                        # Specification says: "input is for passing in an image that has already been stitched"
                        # Usually "composited_image" is the background if we are adding seams.
                        # If we are adding base_tiles to a composited image... that's weird.
                        # Let's assume layering order: Composite -> Base -> Seams.
                        canvas.paste(tile_img, (c * ref_w, r * ref_h), tile_img) # Use alpha verify? tile usually opaque
        
        debug_list = []

        # 3. Process and Paste Col Seams
        if col_seams is not None:
            seam_idx = 0
            for r in range(rows):
                for c in range(cols - 1):
                    if seam_idx < len(col_seams):
                        seam_img = process_seam(col_seams[seam_idx], "col")
                        
                        debug_list.append(torch.from_numpy(np.array(seam_img).astype(np.float32) / 255.0))

                        w, h = seam_img.size
                        x = (c + 1) * ref_w - (w // 2)
                        y = r * ref_h + (ref_h - h) // 2
                        canvas.paste(seam_img, (x, y), seam_img)
                        seam_idx += 1

        # 4. Process and Paste Row Seams
        if row_seams is not None:
            seam_idx = 0
            for r in range(rows - 1):
                for c in range(cols):
                    if seam_idx < len(row_seams):
                        seam_img = process_seam(row_seams[seam_idx], "row")
                        debug_list.append(torch.from_numpy(np.array(seam_img).astype(np.float32) / 255.0))
                        
                        w, h = seam_img.size
                        x = c * ref_w + (ref_w - w) // 2
                        y = (r + 1) * ref_h - (h // 2)
                        canvas.paste(seam_img, (x, y), seam_img)
                        seam_idx += 1

        # 5. Process and Paste Corner Seams
        if corner_seams is not None:
            seam_idx = 0
            for r in range(rows - 1):
                for c in range(cols - 1):
                    if seam_idx < len(corner_seams):
                        seam_img = process_seam(corner_seams[seam_idx], "corner")
                        debug_list.append(torch.from_numpy(np.array(seam_img).astype(np.float32) / 255.0))
                        
                        w, h = seam_img.size
                        x = (c + 1) * ref_w - (w // 2)
                        y = (r + 1) * ref_h - (h // 2)
                        canvas.paste(seam_img, (x, y), seam_img)
                        seam_idx += 1
        
        # Final Output
        final_image = canvas.convert("RGB")
        final_tensor = torch.from_numpy(np.array(final_image).astype(np.float32) / 255.0).unsqueeze(0)
        
        if not debug_list:
             debug_batch = torch.zeros((1, 64, 64, 4))
        else:
             debug_batch = torch.stack(debug_list)
             
        return (final_tensor, debug_batch)

NODE_CLASS_MAPPINGS = {
    "CropAndResize": CropAndResize,
    "ImageTileSplitter": ImageTileSplitter,
    "ImageBatchSeamProvider": ImageBatchSeamProvider,
    "ImageTileComposite": ImageTileComposite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropAndResize": "📐JR Crop & Resize",
    "ImageTileSplitter": "⏹️JR Batch Tiles",
    "ImageBatchSeamProvider": "🪡JR Batch Seams",
    "ImageTileComposite": "🍔JR Composite Tiles"
}
