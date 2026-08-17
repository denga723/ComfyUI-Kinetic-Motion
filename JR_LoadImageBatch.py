import torch
import os
import json
import glob
import folder_paths
from PIL import Image, ImageOps
import numpy as np
from aiohttp import web
from server import PromptServer

# ---------------------------------------------------------------------------
# Media extensions
# ---------------------------------------------------------------------------
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


# ---------------------------------------------------------------------------
# Custom API route: list files in a folder (for JS thumbnails)
# ---------------------------------------------------------------------------
@PromptServer.instance.routes.get("/jr/list_folder")
async def list_folder_handler(request):
    """Return JSON list of media files in a given folder path."""
    folder = request.query.get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        return web.json_response({"files": [], "error": f"Not a valid folder: {folder}"})

    files = []
    for entry in sorted(os.listdir(folder)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in MEDIA_EXTS:
            files.append(os.path.join(folder, entry).replace("\\", "/"))
    return web.json_response({"files": files})


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class JR_LoadImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_data": ("STRING", {"default": "[]", "multiline": False}),
                "upload_anchor": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["guess", "16:9", "1:1", "9:16"],),
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "file_paths": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Comma-separated or JSON list of absolute file paths (e.g. from Batch Image Saver)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "file_paths")
    FUNCTION = "load_batch_images"
    CATEGORY = "JR_Nodes/Experiments"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, True)
    INPUT_IS_LIST = False

    @staticmethod
    def _get_target_size(first_w, first_h, aspect_ratio, all_sizes):
        """Calculate target dimensions based on chosen aspect ratio."""
        RATIOS = {"16:9": (16, 9), "1:1": (1, 1), "9:16": (9, 16)}

        if aspect_ratio == "guess":
            avg_ratio = sum(w / h for w, h in all_sizes) / len(all_sizes)
            aspect_ratio = min(RATIOS, key=lambda k: abs(RATIOS[k][0] / RATIOS[k][1] - avg_ratio))

        rw, rh = RATIOS[aspect_ratio]
        target_ratio = rw / rh

        if first_w / first_h > target_ratio:
            target_h = first_h
            target_w = int(first_h * target_ratio)
        else:
            target_w = first_w
            target_h = int(first_w / target_ratio)

        return target_w, target_h

    @staticmethod
    def _parse_file_paths(file_paths_str):
        """Parse file_paths from various formats: JSON list, comma-sep, newline-sep, or single path."""
        if not file_paths_str or not file_paths_str.strip():
            return []

        s = file_paths_str.strip()

        # Try JSON array first
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [p.strip() for p in parsed if isinstance(p, str) and p.strip()]
            except json.JSONDecodeError:
                pass

        # Try comma-separated (only if commas are present and it's not a single Windows path like C:\...)
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) > 1:
                return parts

        # Try newline-separated
        if "\n" in s:
            return [p.strip() for p in s.split("\n") if p.strip()]

        # Single path
        return [s] if s else []

    @staticmethod
    def _scan_folder(folder_path):
        """Scan a folder for media files, return list of absolute paths."""
        if not folder_path or not folder_path.strip() or not os.path.isdir(folder_path.strip()):
            return []

        folder = folder_path.strip()
        result = []
        for entry in sorted(os.listdir(folder)):
            ext = os.path.splitext(entry)[1].lower()
            if ext in MEDIA_EXTS:
                result.append(os.path.join(folder, entry))
        return result

    def load_batch_images(self, image_data="[]", aspect_ratio="guess",
                          folder_path="", file_paths=None, **kwargs):
        # --- Collect all file paths from all sources ---
        all_paths = []

        # Source 1: Uploaded images (image_data widget — ComfyUI input folder refs)
        try:
            uploaded_list = json.loads(image_data)
        except json.JSONDecodeError:
            print(f"JR_LoadImageBatch: Error decoding JSON: {image_data}")
            uploaded_list = []

        for image_name in uploaded_list:
            try:
                all_paths.append(folder_paths.get_annotated_filepath(image_name))
            except Exception:
                all_paths.append(image_name)

        # Source 2: Input port file_paths
        if file_paths is not None:
            # Handle if it comes as a list (INPUT_IS_LIST = True on upstream)
            if isinstance(file_paths, list):
                for fp in file_paths:
                    all_paths.extend(self._parse_file_paths(str(fp)))
            else:
                all_paths.extend(self._parse_file_paths(str(file_paths)))

        # Source 3: Folder scan
        all_paths.extend(self._scan_folder(folder_path))

        # Deduplicate while preserving order
        seen = set()
        unique_paths = []
        for p in all_paths:
            norm = os.path.normpath(p)
            if norm not in seen:
                seen.add(norm)
                unique_paths.append(p)

        if not unique_paths:
            return (torch.zeros((1, 64, 64, 3)), torch.ones((1, 64, 64)), [])

        # --- First pass: load all PIL images and collect sizes ---
        pil_images = []
        file_paths_list = []

        for image_path in unique_paths:
            ext = os.path.splitext(image_path)[1].lower()
            is_video = ext in VIDEO_EXTS

            i = None
            if is_video:
                try:
                    import cv2
                    cap = cv2.VideoCapture(image_path)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            i = Image.fromarray(frame)
                        cap.release()
                except ImportError:
                    print(f"JR_LoadImageBatch: cv2 not found, cannot load video {image_path}")
                except Exception as e:
                    print(f"JR_LoadImageBatch: Error loading video {image_path}: {e}")

            if i is None:
                try:
                    i = Image.open(image_path)
                    i = ImageOps.exif_transpose(i)
                except Exception:
                    continue

            pil_images.append(i)
            file_paths_list.append(image_path)

        if not pil_images:
            return (torch.zeros((1, 64, 64, 3)), torch.ones((1, 64, 64)), [])

        # --- Compute target size ---
        all_sizes = [(img.size[0], img.size[1]) for img in pil_images]
        first_w, first_h = all_sizes[0]
        target_w, target_h = self._get_target_size(first_w, first_h, aspect_ratio, all_sizes)

        # --- Second pass: crop-to-fill each image to target ---
        images = []
        masks = []
        for img in pil_images:
            src_w, src_h = img.size

            has_alpha = img.mode in ("RGBA", "LA", "PA")
            if has_alpha:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")

            if src_w != target_w or src_h != target_h:
                scale = max(target_w / src_w, target_h / src_h)
                new_w = int(src_w * scale)
                new_h = int(src_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                img = img.crop((int(left), int(top), int(left + target_w), int(top + target_h)))

            if has_alpha:
                r, g, b, a = img.split()
                rgb = Image.merge("RGB", (r, g, b))
                alpha_np = np.array(a).astype(np.float32) / 255.0
                masks.append(torch.from_numpy(alpha_np)[None,])
            else:
                rgb = img
                masks.append(torch.ones((1, target_h, target_w)))

            image_np = np.array(rgb).astype(np.float32) / 255.0
            images.append(torch.from_numpy(image_np)[None,])

        image_batch = torch.cat(images, dim=0)
        mask_batch = torch.cat(masks, dim=0)
        
        # Preview support similar to ComfyUI's PreviewImage
        import folder_paths
        import random
        
        results = []
        output_dir = folder_paths.get_temp_directory()
        filename_prefix = "jr_batch_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
        counter = 1
        for (batch_number, image) in enumerate(image_batch):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, img.size[0], img.size[1])
            
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=1)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp"
            })
            counter += 1

        if results:
            return { "ui": { "images": results }, "result": (image_batch, mask_batch, file_paths_list) }

        return (image_batch, mask_batch, file_paths_list)

NODE_CLASS_MAPPINGS = {
    "JR_LoadImageBatch": JR_LoadImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JR_LoadImageBatch": "📦JR Load Image Batch"
}
