import os
import sys
import tempfile
import urllib.request
import urllib.error
import math
import random
import uuid
import time
import asyncio
import copy
import base64
import json
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Union
import numpy as np
from PIL import Image
import cv2
import torch
import aiohttp
import folder_paths
import google.auth
from google.auth.transport.requests import Request

def tensor_to_base64(tensor):
    # tensor is [H, W, C] in ComfyUI standard
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    i = 255. * tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_tensor(b64_str):
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image)[None,]
        return tensor
    except Exception as e:
        print(f"Failed to decode base64 image: {e}")
        return None

# Optional: define a helper to get GCP access token
def get_gcp_token():
    credentials, project_id = google.auth.default()
    credentials.refresh(Request())
    return credentials.token, project_id


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

ANY_TYPE = AnyType("*")

@dataclass
class GeminiPayload:
    text: str = ""
    images: List[torch.Tensor] = field(default_factory=list)
    videos: List[str] = field(default_factory=list) # Store file paths

@dataclass
class GeminiStream:
    payloads: List[GeminiPayload] = field(default_factory=list)

def get_image_batch_length(raw_val):
    if raw_val is None:
        return 0
    if isinstance(raw_val, str):
        if raw_val.strip('\'"').lower().endswith((".mp4", ".mov", ".webm")):
            return 1
        return 0
    if isinstance(raw_val, torch.Tensor):
        if len(raw_val.shape) == 3:
            return 1
        elif len(raw_val.shape) >= 4:
            shape = raw_val.shape
            num_images = 1
            for dim in shape[:-3]:
                num_images *= dim
            return num_images
    elif isinstance(raw_val, GeminiStream):
        return len(raw_val.payloads)
    elif isinstance(raw_val, list):
        length = 0
        for t in raw_val:
            length += get_image_batch_length(t)
        return length
    return 0

def get_video_file_path(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, str) and (val.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv", ".gif")) or os.path.exists(val.strip('\'"'))):
        return val.strip('\'"')
    if hasattr(val, "get_stream_source") and callable(val.get_stream_source):
        try:
            src = val.get_stream_source()
            if isinstance(src, str) and (os.path.exists(src) or src.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))):
                return src
        except Exception:
            pass
    if hasattr(val, "_VideoFromFile__file"):
        f = getattr(val, "_VideoFromFile__file")
        if isinstance(f, str) and (os.path.exists(f) or f.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))):
            return f
    for attr in ["file", "path", "file_path", "video_path", "filename"]:
        if hasattr(val, attr) and isinstance(getattr(val, attr), str):
            f = getattr(val, attr)
            if os.path.exists(f) or f.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")):
                return f
    return None

def _parse_input_to_stream(inp: Any, split_batches=True) -> GeminiStream:
    stream = GeminiStream()
    if isinstance(inp, GeminiStream):
        return inp
    v_path = get_video_file_path(inp)
    if v_path:
        stream.payloads.append(GeminiPayload(videos=[v_path]))
        return stream
    if isinstance(inp, str):
        stream.payloads.append(GeminiPayload(text=inp))
    elif isinstance(inp, torch.Tensor):
        if len(inp.shape) == 3:
            stream.payloads.append(GeminiPayload(images=[inp.unsqueeze(0)]))
        elif len(inp.shape) >= 4:
            if split_batches:
                shape = inp.shape
                H, W, C = shape[-3], shape[-2], shape[-1]
                flat_batch = inp.reshape(-1, H, W, C)
                for i in range(flat_batch.shape[0]):
                    stream.payloads.append(GeminiPayload(images=[flat_batch[i].unsqueeze(0)]))
            else:
                stream.payloads.append(GeminiPayload(images=[inp]))
    elif isinstance(inp, list):
        for item in inp:
            stream.payloads.extend(_parse_input_to_stream(item, split_batches=split_batches).payloads)
    return stream



class GeminiAuthConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "project_id": ("STRING", {"default": "creativelab-prototypes"}),
                "location": (["us-central1", "us", "eu", "global", "us-east1", "us-west1", "europe-west1", "europe-west4", "asia-northeast1", "asia-southeast1"],),
                "service_account_json_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False})
            }
        }
    RETURN_TYPES = ("GEMINI_AUTH",)
    RETURN_NAMES = ("auth_config",)
    FUNCTION = "setup"
    CATEGORY = "Gemini Enterprise/Config"
    
    def setup(self, project_id, location, service_account_json_path, api_key=""):
        if service_account_json_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json_path
        return ({"project_id": project_id, "location": location[0] if isinstance(location, list) else location, "api_key": api_key},)
# Define Model Node
class GeminiProModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream": (ANY_TYPE,),
                "model_name": (["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", "gemini-1.5-pro", "gemini-1.5-flash"],)
            },
            "optional": {
                "auth_config": ("GEMINI_AUTH",)
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "GEMINI_RESPONSE")
    RETURN_NAMES = ("text", "images", "response")
    FUNCTION = "generate"
    CATEGORY = "Gemini Enterprise/Models"
    INPUT_IS_LIST = True

    def generate(self, stream, model_name, auth_config=None):
        model_name = model_name[0]
        unified_stream = _parse_input_to_stream(stream)
        
        project_id = auth_config[0].get("project_id", "creativelab-prototypes") if auth_config else "creativelab-prototypes"
        location = auth_config[0].get("location", "us-central1") if auth_config else "us-central1"
        key = (auth_config[0].get("api_key", "") if auth_config else "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()

        token = None
        if not key:
            try:
                token, default_project = get_gcp_token()
                project_id = project_id or default_project
            except Exception as e:
                raise RuntimeError(
                    f"Authentication Failed: No Google Cloud Application Default Credentials (ADC) or API Key found ({e}).\n\n"
                    "To fix this, choose one of the following:\n"
                    "1. In the GeminiAuthConfig node, enter your Gemini API Key in the 'api_key' field.\n"
                    "2. In the GeminiAuthConfig node, enter the path to your Service Account JSON key file in 'service_account_json_path'.\n"
                    "3. In your terminal, run: `gcloud auth application-default login`"
                )
            
        if not project_id:
            project_id = "creativelab-prototypes"

        if key:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        else:
            if location in ["us", "eu"]:
                hostname = f"aiplatform.{location}.rep.googleapis.com"
            elif location == "global":
                hostname = "aiplatform.googleapis.com"
            else:
                hostname = f"{location}-aiplatform.googleapis.com"
            endpoint = f"https://{hostname}/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_name}:generateContent"

        async def fetch_gemini(payload: GeminiPayload, session: aiohttp.ClientSession):
            headers = {
                "Content-Type": "application/json"
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"
            # Multimodal payload structure for Gemini REST API
            parts = []
            print(f"DEBUG: fetch_gemini called with payload having {len(payload.images)} images, {len(payload.videos)} videos, and text: {payload.text!r}")
            for vid_path in payload.videos:
                try:
                    vid_clean = vid_path.strip('\'"')
                    if os.path.exists(vid_clean):
                        import base64
                        with open(vid_clean, "rb") as f:
                            b64_video = base64.b64encode(f.read()).decode("utf-8")
                        mime_type = "video/mp4"
                        if vid_clean.lower().endswith(".webm"):
                            mime_type = "video/webm"
                        elif vid_clean.lower().endswith(".mov"):
                            mime_type = "video/quicktime"
                        parts.append({
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": b64_video
                            }
                        })
                except Exception as e:
                    print(f"Error encoding video file for Gemini Pro: {e}")

            for img_tensor in payload.images:
                try:
                    # If multiple images are packed in one tensor [N, H, W, C]
                    if len(img_tensor.shape) == 4 and img_tensor.shape[0] > 1:
                        print(f"DEBUG: Encoding batch of {img_tensor.shape[0]} images")
                        for i in range(img_tensor.shape[0]):
                            b64_img = tensor_to_base64(img_tensor[i:i+1])
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": b64_img
                                }
                            })
                    else:
                        print(f"DEBUG: Encoding single image tensor of shape {img_tensor.shape}")
                        b64_img = tensor_to_base64(img_tensor)
                        parts.append({
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": b64_img
                            }
                        })
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    
            if payload.text:
                parts.append({"text": payload.text})
            elif len(parts) == 0:
                parts.append({"text": " "})

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ]
            }
            print(f"DEBUG: Sending data to Gemini API: {len(parts)} parts")
            
            try:
                async with session.post(endpoint, headers=headers, json=data) as response:
                    if response.status == 200:
                        resp_json = await response.json()
                        text_content = ""
                        image_tensors = []
                        try:
                            for candidate in resp_json.get("candidates", []):
                                content = candidate.get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        text_content += part["text"]
                                    if "inlineData" in part:
                                        mime = part["inlineData"].get("mimeType", "")
                                        b64 = part["inlineData"].get("data", "")
                                        if b64:
                                            t = base64_to_tensor(b64)
                                            if t is not None:
                                                image_tensors.append(t)
                            return (text_content, image_tensors, resp_json)
                        except (KeyError, IndexError) as e:
                            return (f"Unexpected response format: {resp_json}", [], resp_json)
                    else:
                        error_text = await response.text()
                        return (f"API Error {response.status}: {error_text}", [], {"error": error_text})
            except Exception as e:
                return (f"Request Error: {e}", [], {"error": str(e)})

        async def process_batch():
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_gemini(p, session) for p in unified_stream.payloads]
                return await asyncio.gather(*tasks)

        import threading
        
        results = [None]
        
        def run_async():
            # Wait, asyncio.new_event_loop() requires closing if not used anymore, but threading handles it OK for now
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results[0] = loop.run_until_complete(process_batch())
            finally:
                loop.close()

        t = threading.Thread(target=run_async)
        t.start()
        t.join()

        all_texts = []
        all_images = []
        all_responses = []
        
        for res in results[0]:
            if res[0]:
                all_texts.append(res[0])
            
            # Pack images for this response into a single batch tensor
            res_imgs = res[1]
            if res_imgs:
                shape = res_imgs[0].shape
                if all(img.shape == shape for img in res_imgs):
                    final_image = torch.cat(res_imgs, dim=0)
                else:
                    import torch.nn.functional as F
                    H, W = shape[1], shape[2]
                    resized_images = []
                    for img in res_imgs:
                        if img.shape != shape:
                            img_perm = img.permute(0, 3, 1, 2)
                            img_resized = F.interpolate(img_perm, size=(H, W), mode="bilinear")
                            img = img_resized.permute(0, 2, 3, 1)
                        resized_images.append(img)
                    final_image = torch.cat(resized_images, dim=0)
                all_images.append(final_image)
            else:
                all_images.append(torch.zeros((1, 512, 512, 3), dtype=torch.float32))

            all_responses.append(res[2])
        
        final_text = "\n\n".join(all_texts)
        final_image_tensor = torch.cat(all_images, dim=0) if all_images else None
            
        return (final_text, final_image_tensor, all_responses)

class GeminiOmniModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream": (ANY_TYPE,),
            },
            "optional": {
                "gcs_bucket": ("STRING", {"default": ""}),
                "auth_config": ("GEMINI_AUTH",)
            },
            "hidden": {
                "omni_config": ("STRING", {"default": "{}"}),
            }
        }
        
    RETURN_TYPES = ("STRING", "IMAGE", "GEMINI_RESPONSE")
    RETURN_NAMES = ("text", "images", "response")
    FUNCTION = "generate"
    CATEGORY = "Gemini Enterprise/Models"
    INPUT_IS_LIST = True

    def generate(self, stream, gcs_bucket="", auth_config=None, omni_config="{}"):
        import urllib.request
        import urllib.error
        import tempfile
        import cv2
        import torch
        import os
        import numpy as np
        import json

        gcs_bucket = gcs_bucket[0] if isinstance(gcs_bucket, list) else gcs_bucket
        if isinstance(omni_config, list):
            omni_config = omni_config[0]
            
        config = {}
        try:
            if omni_config and isinstance(omni_config, str):
                config = json.loads(omni_config)
        except Exception as e:
            print(f"[GeminiOmniModel] Error parsing omni_config: {e}")
            
        model_name = config.get("model_name", "gemini-omni-flash-preview")
        # Ensure only supported video generation models are passed to the Interactions API
        VALID_OMNI_MODELS = ["gemini-omni-flash-preview", "veo-2.0-generate-001"]
        if not model_name or model_name not in VALID_OMNI_MODELS:
            print(f"[GeminiOmniModel] Notice: '{model_name}' is not supported on the video Interactions endpoint. Auto-defaulting to 'gemini-omni-flash-preview'.")
            model_name = "gemini-omni-flash-preview"

        ar = config.get("aspect_ratio", "16:9")
        task = config.get("task", "text_to_video")
        duration = int(config.get("duration", 3))
        delivery = config.get("delivery", "base64")
        prefix_text = config.get("prefix_text", "")
        suffix_text = config.get("suffix_text", "")
        
        unified_stream = _parse_input_to_stream(stream, split_batches=False)
        
        project_id = auth_config[0].get("project_id", "creativelab-prototypes") if auth_config else "creativelab-prototypes"
        location = auth_config[0].get("location", "us-central1") if auth_config else "us-central1"
        key = (auth_config[0].get("api_key", "") if auth_config else "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()

        token = None
        if not key:
            try:
                token, default_project = get_gcp_token()
                project_id = project_id or default_project
            except Exception as e:
                raise RuntimeError(
                    f"Authentication Failed: No Google Cloud Application Default Credentials (ADC) or API Key found ({e}).\n\n"
                    "To fix this, choose one of the following:\n"
                    "1. In the GeminiAuthConfig node, enter your Gemini API Key in the 'api_key' field.\n"
                    "2. In the GeminiAuthConfig node, enter the path to your Service Account JSON key file in 'service_account_json_path'.\n"
                    "3. In your terminal, run: `gcloud auth application-default login`"
                )

        if not project_id:
            project_id = "creativelab-prototypes"

        all_texts = []
        video_tensors = []
        all_responses = []

        async def fetch_omni(payload, session):
            inputs_payload = []
            
            # Add videos from file paths if present
            for vid_path in payload.videos:
                try:
                    vid_clean = vid_path.strip('\'"')
                    if os.path.exists(vid_clean):
                        import base64
                        import subprocess
                        import tempfile
                        
                        # Ensure standard mp4 encoding for Google API compatibility
                        if not vid_clean.lower().endswith(".mp4"):
                            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
                                tmp_mp4 = tmp_f.name
                            converted = False
                            if os.path.exists("/usr/bin/avconvert"):
                                try:
                                    res = subprocess.run(["/usr/bin/avconvert", "--source", vid_clean, "--output", tmp_mp4, "--preset", "PresetHighestQuality"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    if res.returncode == 0 and os.path.exists(tmp_mp4) and os.path.getsize(tmp_mp4) > 0:
                                        vid_clean = tmp_mp4
                                        converted = True
                                except Exception:
                                    pass
                            if not converted:
                                try:
                                    cap = cv2.VideoCapture(vid_clean)
                                    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    out = cv2.VideoWriter(tmp_mp4, fourcc, fps, (w, h))
                                    while True:
                                        ret, frame = cap.read()
                                        if not ret:
                                            break
                                        out.write(frame)
                                    cap.release()
                                    out.release()
                                    if os.path.exists(tmp_mp4) and os.path.getsize(tmp_mp4) > 0:
                                        vid_clean = tmp_mp4
                                except Exception:
                                    pass

                        with open(vid_clean, "rb") as f:
                            b64_video = base64.b64encode(f.read()).decode("utf-8")
                        inputs_payload.append({
                            "type": "video",
                            "data": b64_video,
                            "mime_type": "video/mp4"
                        })
                except Exception as e:
                    print(f"Error encoding video file for Omni: {e}")

            # Add images if present
            for img_tensor in payload.images:
                try:
                    if len(img_tensor.shape) == 4 and img_tensor.shape[0] > 1:
                        # Multiple individual reference images in batch
                        for i in range(img_tensor.shape[0]):
                            b64_img = tensor_to_base64(img_tensor[i:i+1])
                            inputs_payload.append({
                                "type": "image",
                                "data": b64_img,
                                "mime_type": "image/jpeg"
                            })
                    else:
                        b64_img = tensor_to_base64(img_tensor)
                        inputs_payload.append({
                            "type": "image",
                            "data": b64_img,
                            "mime_type": "image/jpeg"
                        })
                except Exception as e:
                    print(f"Error encoding image for Omni: {e}")

            has_video_input = any(p.get("type") == "video" for p in inputs_payload)

            # Process explicit image & video reference roles
            image_roles = config.get("image_roles", {})
            sources_clauses = []
            guiding_instructions = []
            ref_clauses = []

            img_count = 0
            vid_count = 0
            first_frame_id = None
            ref_img_ids = []
            vid_ids = []

            if task == "video_editing" or task == "edit":
                # Video-to-Video mode: Video1 is the primary source video
                img_count = 0
                vid_count = 0
                for p in inputs_payload:
                    if p.get("type") == "video":
                        vid_count += 1
                        vid_id = f"Video{vid_count}"
                        vid_ids.append(vid_id)
                    elif p.get("type") == "image":
                        img_count += 1
                        img_id = f"Image{img_count}"
                        ref_img_ids.append(img_id)
                        ref_idx = len(ref_clauses)
                        ref_clauses.append(f"<IMAGE_REF_{ref_idx}>@{img_id}")

                if vid_ids:
                    sources_clauses.append(f"[# Sources @{vid_ids[0]}]")
                    guiding_instructions.append(f"Perform video style transfer: Strictly preserve the camera movement, motion trajectory, actions, structure, and timing from {vid_ids[0]}.")

                if ref_clauses:
                    ref_str = " ".join(ref_clauses)
                    sources_clauses.append(f"[# References {ref_str}]")
                    ref_ids_str = ", ".join(ref_img_ids)
                    guiding_instructions.append(f"Re-render and style the scene according to the visual appearance, color palette, texture, and aesthetic style of {ref_ids_str}.")
            elif task != "text_to_video":
                # Image-to-Video mode: Image1 is the first frame, Video1 is motion reference
                for p in inputs_payload:
                    if p.get("type") == "image":
                        img_count += 1
                        img_id = f"Image{img_count}"
                        default_role = "reference"
                        role = image_roles.get(img_id, default_role)
                        
                        if role == "first-frame" and not first_frame_id:
                            first_frame_id = img_id
                            sources_clauses.append(f"[# Sources <FIRST_FRAME>@{img_id}]")
                            guiding_instructions.append(f"Use {img_id} as the initial starting frame.")
                        else:
                            ref_idx = len(ref_img_ids)
                            ref_img_ids.append(img_id)
                            ref_clauses.append(f"<IMAGE_REF_{ref_idx}>@{img_id}")
                    elif p.get("type") == "video":
                        vid_count += 1
                        vid_id = f"Video{vid_count}"
                        vid_ids.append(vid_id)
                        v_ref_idx = len(vid_ids) - 1
                        ref_clauses.append(f"<VIDEO_REF_{v_ref_idx}>@{vid_id}")

                if ref_clauses:
                    ref_str = " ".join(ref_clauses)
                    sources_clauses.append(f"[# References {ref_str}]")

                if ref_img_ids:
                    ref_ids_str = ", ".join(ref_img_ids)
                    guiding_instructions.append(f"Use {ref_ids_str} as visual style, texture, and character references.")

                if vid_ids:
                    vid_ids_str = ", ".join(vid_ids)
                    guiding_instructions.append(f"Replicate and match the exact motion trajectory, camera movement, character actions, physical dynamics, and animation timing from {vid_ids_str}.")

            # Add text prompt (duration + prefix + payload.text + suffix)
            combined_text_parts = []
            if sources_clauses:
                combined_text_parts.extend(sources_clauses)
            if duration:
                combined_text_parts.append(f"[0-{duration}s]")
            if prefix_text:
                combined_text_parts.append(prefix_text)
            if payload.text:
                combined_text_parts.append(payload.text)
            if suffix_text:
                combined_text_parts.append(suffix_text)
            if guiding_instructions:
                combined_text_parts.extend(guiding_instructions)
                
            if combined_text_parts:
                final_text = " ".join(combined_text_parts)
                inputs_payload.append({
                    "type": "text",
                    "text": final_text
                })

            mapped_task = task
            if task == "video_editing":
                mapped_task = "edit"
            
            response_format = {
                "type": "video"
            }
            if ar and mapped_task != "edit" and mapped_task != "video_editing":
                response_format["aspect_ratio"] = ar
                
            if delivery == "uri":
                import uuid
                if gcs_bucket:
                    response_format["delivery"] = "uri"
                    obj_name = f"omni_{uuid.uuid4().hex}.mp4"
                    response_format["gcs_uri"] = f"gs://{gcs_bucket}/{obj_name}"
                else:
                    print("WARNING: URI delivery requested but no GCS bucket provided. Falling back to base64.")

            req_body = {
                "model": model_name,
                "input": inputs_payload,
                "response_format": response_format,
                "generation_config": {
                    "video_config": {
                        "task": mapped_task
                    }
                }
            }

            import json
            log_body = {
                "model": req_body["model"],
                "input": [{"type": p.get("type"), "mime_type": p.get("mime_type"), "data": "<base64_hidden>" if "data" in p else p.get("text") or p.get("uri")} for p in req_body.get("input", [])],
                "response_format": req_body.get("response_format"),
                "generation_config": req_body.get("generation_config")
            }
            print(f"OMNI REQUEST PAYLOAD: {json.dumps(log_body, indent=2)}")

            if model_name == "veo-2.0-generate-001":
                # Google Vertex AI Veo 2.0 Video Generation Pipeline
                veo_location = location if location not in ["global", "us", "eu"] else "us-central1"
                veo_url = f"https://{veo_location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{veo_location}/publishers/google/models/veo-2.0-generate-001:predictLongRunning"
                
                prompt_text = final_text or payload.text or ""
                instance = {"prompt": prompt_text}
                
                # Image-to-video / starting frame support for Veo
                if payload.images:
                    try:
                        img_tensor = payload.images[0]
                        b64_img = tensor_to_base64(img_tensor)
                        instance["image"] = {
                            "bytesBase64Encoded": b64_img,
                            "mimeType": "image/jpeg"
                        }
                    except Exception as e:
                        print(f"[GeminiOmniModel] Veo image encoding error: {e}")
                
                veo_params = {
                    "aspectRatio": ar if ar in ["16:9", "9:16", "1:1"] else "16:9",
                    "sampleCount": 1,
                    "durationSeconds": max(5, min(duration, 8)),
                    "personGeneration": "allow_adult",
                    "fps": 24
                }
                if gcs_bucket and delivery == "uri":
                    veo_params["storageUri"] = f"gs://{gcs_bucket}"
                    
                veo_body = {
                    "instances": [instance],
                    "parameters": veo_params
                }
                
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                elif key:
                    veo_url += f"?key={key}"
                else:
                    return (None, None, None, {"error": "Authentication failed. No token or key."})
                    
                try:
                    async with session.post(veo_url, headers=headers, json=veo_body) as response:
                        if response.status == 200:
                            op_json = await response.json()
                            op_name = op_json.get("name")
                            if not op_name:
                                return ("[Veo 2 Error]: No operation returned.", None, None, op_json)
                            
                            poll_url = f"https://{veo_location}-aiplatform.googleapis.com/v1/{op_name}"
                            done = False
                            video_uri = None
                            video_b64 = None
                            poll_json = op_json
                            for _ in range(60): # poll up to 5 minutes
                                await asyncio.sleep(5)
                                async with session.get(poll_url, headers=headers) as poll_resp:
                                    if poll_resp.status == 200:
                                        poll_json = await poll_resp.json()
                                        if poll_json.get("done"):
                                            done = True
                                            if "error" in poll_json:
                                                return (f"[Veo 2 Error]: {poll_json['error']}", None, None, poll_json)
                                            resp_val = poll_json.get("response", {})
                                            vids = resp_val.get("generatedVideos", []) or resp_val.get("videos", [])
                                            for v in vids:
                                                vid_obj = v.get("video", {})
                                                if "bytesBase64Encoded" in vid_obj:
                                                    video_b64 = vid_obj["bytesBase64Encoded"]
                                                elif "uri" in vid_obj:
                                                    video_uri = vid_obj["uri"]
                                                elif "gcsUri" in vid_obj:
                                                    video_uri = vid_obj["gcsUri"]
                                            break
                            if not done:
                                return ("[Veo 2 Error]: Video generation timed out after 5 minutes.", None, None, poll_json)
                            return (f"[Generated video with Veo 2: {prompt_text}]", video_b64, video_uri, poll_json)
                        else:
                            error_text = await response.text()
                            print(f"VEO API ERROR {response.status}: {error_text}")
                            user_msg = f"[Veo 2.0 API Error ({response.status})]: {error_text}"
                            if response.status == 404:
                                user_msg = (
                                    f"[Veo 2.0 Notice (404)]:\n"
                                    f"• Model 'veo-2.0-generate-001' is not enabled or allowlisted for project '{project_id}' in region '{veo_location}'.\n"
                                    f"• Recommendation: Select 'gemini-omni-flash-preview' in the Gemini Omni Model node (active and verified on your project)."
                                )
                            return (user_msg, None, None, {"error": error_text, "code": response.status, "friendly_message": user_msg})
                except Exception as e:
                    print(f"VEO INTERNAL EXCEPTION: {str(e)}")
                    return (f"[Error]: {str(e)}", None, None, {"error": str(e)})

            # Default: Gemini Omni Flash Video via Global Interactions API
            headers = {"Content-Type": "application/json"}
            if token:
                url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/global/interactions"
                headers["Authorization"] = f"Bearer {token}"
            elif key:
                url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={key}"
            else:
                print("OMNI AUTH ERROR: No token or key in fetch_omni")
                return (None, None, None, {"error": "Authentication failed. No token or key."})

            try:
                async with session.post(url, headers=headers, json=req_body) as response:
                    if response.status == 200:
                        resp_json = await response.json()
                        video_uri = None
                        video_b64 = None
                        text_parts = []
                        steps = resp_json.get("steps", [])
                        for step in steps:
                            for content in step.get("content", []):
                                if "text" in content:
                                    text_parts.append(content.get("text", ""))
                                if content.get("type") == "video":
                                    if "uri" in content:
                                        video_uri = content.get("uri")
                                    if "data" in content:
                                        video_b64 = content.get("data")
                        
                        combined_text = "\n".join(text_parts) if text_parts else f"[Generated video for prompt: {payload.text}]"
                        return (combined_text, video_b64, video_uri, resp_json)
                    else:
                        error_text = await response.text()
                        print(f"OMNI API ERROR {response.status}: {error_text}")
                        user_msg = f"[API Error ({response.status})]: {error_text}"
                        if "unsupported model interaction" in error_text.lower():
                            user_msg = (
                                f"[Gemini Omni Model Error]: The selected model '{model_name}' does not support video generation/editing via the Interactions API.\n"
                                f"• For Video Generation & Stylization (Video-to-Video / Image-to-Video), please select 'gemini-omni-flash-preview'.\n"
                                f"• If you want text analysis / multimodal reasoning with '{model_name}', use the 'Gemini Execution Node' instead."
                            )
                        return (user_msg, None, None, {"error": error_text, "code": response.status, "friendly_message": user_msg})
            except Exception as e:
                print(f"OMNI INTERNAL EXCEPTION: {str(e)}")
                return (f"[Error]: {str(e)}", None, None, {"error": str(e)})

        async def process_batch():
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_omni(p, session) for p in unified_stream.payloads]
                return await asyncio.gather(*tasks)

        import threading
        results = [None]
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results[0] = loop.run_until_complete(process_batch())
            finally:
                loop.close()

        t = threading.Thread(target=run_async)
        t.start()
        t.join()

        all_texts = []
        video_tensors = []
        all_responses = []
        
        for res_tuple in results[0]:
            if not res_tuple:
                continue
            combined_text, video_b64, video_uri, resp_json = res_tuple
            
            if combined_text:
                all_texts.append(combined_text)
            if resp_json:
                all_responses.append(resp_json)
                
            tensor_appended = False
            if video_uri or video_b64:
                import tempfile
                import cv2
                import urllib.request
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    
                    if video_b64:
                        import base64
                        tmp_file.write(base64.b64decode(video_b64))
                    elif video_uri:
                        v_headers = {}
                        if token:
                            v_headers["Authorization"] = f"Bearer {token}"
                        try:
                            v_req = urllib.request.Request(video_uri, headers=v_headers)
                            with urllib.request.urlopen(v_req) as v_resp:
                                tmp_file.write(v_resp.read())
                        except Exception as e:
                            print(f"Failed to download video URI: {e}")

                try:
                    cap = cv2.VideoCapture(tmp_path)
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_float = frame_rgb.astype("float32") / 255.0
                        frames.append(frame_float)
                    cap.release()
                    
                    if frames:
                        tensor = torch.from_numpy(np.stack(frames, axis=0))
                        video_tensors.append(tensor)
                        tensor_appended = True
                except Exception as e:
                    print(f"Failed to decode video: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
            if not tensor_appended:
                # Fallback if no frames were read or error occurred
                h = 720 if ar == "16:9" else 1280
                w = 1280 if ar == "16:9" else 720
                if "error" in resp_json:
                    import math
                    f_count = 24
                    fallback = torch.zeros((f_count, h, w, 3))
                    for f in range(f_count):
                        color_val = (math.sin(f / f_count * math.pi * 2) + 1) / 2
                        fallback[f, :, :, 0] = color_val # Pulsing Red
                        fallback[f, :, :, 1] = 0.2
                        fallback[f, :, :, 2] = 1.0 - color_val
                    video_tensors.append(fallback)
                else:
                    video_tensors.append(torch.zeros((1, h, w, 3)))

        final_text = "\n\n".join(all_texts)
        if not video_tensors:
            h = 720 if ar == "16:9" else 1280
            w = 1280 if ar == "16:9" else 720
            final_image_tensor = torch.zeros((1, h, w, 3))
        else:
            final_image_tensor = torch.cat(video_tensors, dim=0)

        return (final_text, final_image_tensor, all_responses)


class GeminiVideoCombine:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of image frames from Gemini Omni / video node"}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "tooltip": "Playback frame rate (FPS)"}),
                "format": (["mp4", "animated_webp", "gif", "video/mp4", "image/webp", "image/gif"], {"default": "mp4"}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "Save output to ComfyUI output directory"}),
                "filename_prefix": ("STRING", {"default": "GeminiVideo", "tooltip": "Prefix for saved video/animation files"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "Gemini Enterprise/Video"
    DESCRIPTION = "Combines image frames into a playable video/animation with inline preview."

    def combine_video(self, images, frame_rate=24, format="mp4", save_output=True, filename_prefix="GeminiVideo"):
        if images is None or len(images.shape) < 4 or images.shape[0] == 0:
            return {"ui": {"images": []}, "result": (images, "")}

        output_dir = self.output_dir if save_output else self.temp_dir
        type_str = "output" if save_output else "temp"

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )

        h, w = images.shape[1], images.shape[2]
        num_frames = images.shape[0]

        pil_frames = []
        for idx in range(num_frames):
            frame_np = (images[idx].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame_np))

        duration_ms = max(1, int(1000.0 / max(1, frame_rate)))
        saved_file_path = ""
        ui_results = []

        if format in ["animated_webp", "image/webp"]:
            file_name = f"{filename}_{counter:05d}_.webp"
            saved_file_path = os.path.join(full_output_folder, file_name)
            pil_frames[0].save(
                saved_file_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                quality=92,
                method=4
            )
            ui_results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": type_str,
                "format": "image/webp"
            })
        elif format == "gif":
            file_name = f"{filename}_{counter:05d}_.gif"
            saved_file_path = os.path.join(full_output_folder, file_name)
            pil_frames[0].save(
                saved_file_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0
            )
            ui_results.append({
                "filename": file_name,
                "subfolder": subfolder,
                "type": type_str,
                "format": "image/gif"
            })
        else:  # mp4
            file_name = f"{filename}_{counter:05d}_.mp4"
            saved_file_path = os.path.join(full_output_folder, file_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(saved_file_path, fourcc, float(frame_rate), (w, h))
            for idx in range(num_frames):
                frame_bgr = cv2.cvtColor((images[idx].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()

            preview_webp_name = f"{filename}_{counter:05d}_preview.webp"
            preview_path = os.path.join(full_output_folder, preview_webp_name)
            pil_frames[0].save(
                preview_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                quality=85,
                method=3
            )
            ui_results.append({
                "filename": preview_webp_name,
                "subfolder": subfolder,
                "type": type_str,
                "format": "image/webp"
            })

        return {
            "ui": {"images": ui_results},
            "result": (images, saved_file_path)
        }


class GeminiMultimodalPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "response": (ANY_TYPE,)
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "Gemini Enterprise/Preview"

    def preview(self, response):
        return {"ui": {"gemini_response": response if isinstance(response, list) else [response]}}

class GeminiJobBatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "jobs": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "hidden": {
                "stream_config": ("STRING", {"default": "[]"}),
            }
        }

    RETURN_TYPES = ("GEMINI_STREAM",)
    RETURN_NAMES = ("stream",)
    FUNCTION = "process_jobs"
    CATEGORY = "Gemini Enterprise/Routing"
    OUTPUT_NODE = True

    @staticmethod
    def _get_target_size(first_w, first_h, aspect_ratio, all_sizes):
        RATIOS = {"16:9": (16, 9), "1:1": (1, 1), "9:16": (9, 16)}

        if aspect_ratio == "guess":
            avg_ratio = sum(w / max(h, 1) for w, h in all_sizes) / len(all_sizes)
            aspect_ratio = min(RATIOS, key=lambda k: abs(RATIOS[k][0] / RATIOS[k][1] - avg_ratio))

        rw, rh = RATIOS[aspect_ratio]
        target_ratio = rw / rh

        # Ensure first_h and first_w are non-zero
        first_w = max(first_w, 1)
        first_h = max(first_h, 1)

        if first_w / first_h > target_ratio:
            target_h = first_h
            target_w = int(first_h * target_ratio)
        else:
            target_w = first_w
            target_h = int(first_w / target_ratio)

        return max(target_w, 1), max(target_h, 1)

    @staticmethod
    def _load_video_frames(filepath):
        import cv2
        import torch
        import numpy as np
        
        filepath = filepath.strip('\'"')
        cap = cv2.VideoCapture(filepath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if not frames:
            return None
            
        frames_np = np.array(frames).astype(np.float32) / 255.0
        return torch.from_numpy(frames_np)

    def _process_image_stream(self, tensor_batch, config, target_len):
        import torch
        import torch.nn.functional as F
        import random
        import os

        mode = config.get("mode", "images")

        v_path = get_video_file_path(tensor_batch)
        if v_path and mode != "images":
            return [v_path for _ in range(target_len)]
        if v_path and mode == "images":
            tensor_batch = self._load_video_frames(v_path)
            if tensor_batch is None:
                return [None] * target_len

        if mode == "video" and isinstance(tensor_batch, torch.Tensor) and len(tensor_batch.shape) >= 4 and tensor_batch.shape[0] > 1:
            return [tensor_batch for _ in range(target_len)]

        # Gather all images as a list of 3D [H, W, C] tensors
        tensors = []
        if isinstance(tensor_batch, torch.Tensor):
            if len(tensor_batch.shape) == 3:
                tensors.append(tensor_batch)
            elif len(tensor_batch.shape) >= 4:
                shape = tensor_batch.shape
                H, W, C = shape[-3], shape[-2], shape[-1]
                flat_batch = tensor_batch.reshape(-1, H, W, C)
                for i in range(flat_batch.shape[0]):
                    tensors.append(flat_batch[i])
        elif isinstance(tensor_batch, list):
            for t in tensor_batch:
                if isinstance(t, torch.Tensor):
                    if len(t.shape) == 3:
                        tensors.append(t)
                    elif len(t.shape) >= 4:
                        shape = t.shape
                        H, W, C = shape[-3], shape[-2], shape[-1]
                        flat_batch = t.reshape(-1, H, W, C)
                        for i in range(flat_batch.shape[0]):
                            tensors.append(flat_batch[i])

        if not tensors:
            return []

        if mode == "video":
            full_video_tensor = torch.stack(tensors, dim=0)
            return [full_video_tensor for _ in range(target_len)]

        aspect = config.get("aspect", "guess")
        cycle = config.get("cycle", "iterate")
        imgs_per_job = int(config.get("imgs_per_job", 1))
        idx_offset = int(config.get("index", 0))
        if target_len == 1 and imgs_per_job == 1 and len(tensors) > 1 and cycle != "fixed":
            imgs_per_job = len(tensors)

        all_sizes = [(t.shape[1], t.shape[0]) for t in tensors] 
        first_w, first_h = all_sizes[0]
        target_w, target_h = self._get_target_size(first_w, first_h, aspect, all_sizes)

        processed_images = []
        for img in tensors:
            src_h, src_w, _ = img.shape
            
            if src_w != target_w or src_h != target_h:
                img_t = img.permute(2, 0, 1).unsqueeze(0) 
                scale = max(target_w / max(src_w, 1), target_h / max(src_h, 1))
                new_w = int(src_w * scale)
                new_h = int(src_h * scale)
                img_t = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
                
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                img_t = img_t[:, :, top:top+target_h, left:left+target_w]
                img = img_t.squeeze(0).permute(1, 2, 0) 
            
            processed_images.append(img)
            
        N = len(processed_images)
        job_images = []
        
        for job_i in range(target_len):
            job_imgs = []
            for j in range(imgs_per_job):
                if cycle == "fixed":
                    idx = idx_offset + j
                    if idx >= N: idx = N - 1 
                elif cycle == "random":
                    idx = random.randint(0, N - 1)
                else: 
                    idx = (job_i * imgs_per_job + j) % N
                job_imgs.append(processed_images[idx].unsqueeze(0))
            
            if job_imgs:
                job_images.append(torch.cat(job_imgs, dim=0))
            else:
                job_images.append(None)
                
        return job_images

    def process_jobs(self, stream_config="[]", jobs=0, **kwargs):
        jobs = jobs[0] if isinstance(jobs, list) and jobs else (jobs if isinstance(jobs, int) else 0)
        import json
        import os
        import io
        import base64
        import torch
        import random
        
        stream_config_str = stream_config[0] if isinstance(stream_config, list) and stream_config else stream_config
        if not isinstance(stream_config_str, str):
            stream_config_str = "[]"
            
        try:
            stream_config = json.loads(stream_config_str)
        except Exception:
            stream_config = []
            
        streams = {}
        for key, value in kwargs.items():
            import re
            m = re.match(r"^(?:text|image|video|input)_stream_(\d+)$", key)
            if m:
                idx = int(m.group(1))
                stype = "IMAGE" if (key.startswith("image_") or key.startswith("video_")) else "STRING"
                # If tensor or video, ensure stype is IMAGE
                if isinstance(value, torch.Tensor) or get_video_file_path(value) is not None:
                    stype = "IMAGE"
                streams[idx] = {"val": value, "type": stype}
                    
        processed_text_streams = []
        processed_image_streams = {}
        
        if not stream_config and streams:
            for idx in sorted(streams.keys()):
                stream_config.append({
                    "id": idx,
                    "type": streams[idx]["type"],
                    "delim": "\\n",
                    "prefix": "",
                    "suffix": "",
                    "muted": False
                })

        for config in stream_config:
            if config.get("muted", False):
                continue
                
            idx = int(config.get("id", 0))
            if idx not in streams:
                continue
                
            config["type"] = streams[idx]["type"]
            
            raw_val = streams[idx]["val"]
            stype = config.get("type", "STRING")
            
            if stype == "STRING":
                delim = config.get("delim", "\\n")
                if delim == "\\n":
                    delim = "\n"
                
                prefix = config.get("prefix", "")
                suffix = config.get("suffix", "")
                
                input_string = ""
                if raw_val is not None:
                    if isinstance(raw_val, str):
                        input_string = raw_val
                    elif isinstance(raw_val, GeminiStream):
                        input_string = "\n".join([p.text for p in raw_val.payloads if p.text])
                    elif isinstance(raw_val, list) and len(raw_val) > 0 and isinstance(raw_val[0], str):
                        input_string = "\n".join(raw_val)
                        
                if delim == "":
                    parts = [f"{prefix}{input_string}{suffix}"] if input_string else []
                else:
                    parts = [f"{prefix}{p}{suffix}" for p in input_string.split(delim) if p]
                    
                processed_text_streams.append(parts)

        max_text_len = max([len(p) for p in processed_text_streams if p] + [0])
        max_img_len = 0
        import math
        for config in stream_config:
            if config.get("muted", False):
                continue
            idx = int(config.get("id", 0))
            if idx not in streams:
                continue
            if config.get("type", "STRING") == "IMAGE":
                raw_val = streams[idx]["val"]
                N = get_image_batch_length(raw_val)
                imgs_per_job = int(config.get("imgs_per_job", 1))
                cycle = config.get("cycle", "iterate")
                if cycle == "iterate" and imgs_per_job > 0:
                    max_img_len = max(max_img_len, math.ceil(N / imgs_per_job))
                else:
                    max_img_len = max(max_img_len, 1)
                    
        max_len = max(max_text_len, max_img_len)
        
        target_len = jobs if jobs > 0 else max_len
        if target_len == 0:
            target_len = 1
        
        for config in stream_config:
            if config.get("muted", False):
                continue
                
            idx = int(config.get("id", 0))
            if idx not in streams:
                continue
                
            raw_val = streams[idx]["val"]
            stype = config.get("type", "STRING")
            
            if stype == "IMAGE":
                print(f"DEBUG: Processing IMAGE stream for idx {idx}, raw_val type={type(raw_val)}")
                job_imgs = self._process_image_stream(raw_val, config, target_len)
                print(f"DEBUG: Processed image stream for idx {idx}, got {len(job_imgs)} job_imgs")
                processed_image_streams[idx] = job_imgs

        merged_stream = GeminiStream()
        for i in range(target_len):
            combined_string = ""
            for parts in processed_text_streams:
                if not parts:
                    continue
                part = parts[i % len(parts)]
                combined_string += part
            
            job_img_list = []
            job_vid_list = []
            for idx, img_stream in processed_image_streams.items():
                if i < len(img_stream) and img_stream[i] is not None:
                    val = img_stream[i]
                    c = next((x for x in stream_config if int(x.get("id", -1)) == int(idx)), {})
                    is_vid_stream = (c.get("mode") == "video")
                    v_path = get_video_file_path(val)
                    
                    if v_path:
                        job_vid_list.append(v_path)
                    elif is_vid_stream and isinstance(val, torch.Tensor) and len(val.shape) >= 4 and val.shape[0] > 1:
                        import tempfile
                        import cv2
                        import numpy as np
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                        fps = float(c.get("fps", 24.0))
                        frames = (val.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        h, w = frames.shape[1], frames.shape[2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
                        for f_idx in range(frames.shape[0]):
                            out.write(cv2.cvtColor(frames[f_idx], cv2.COLOR_RGB2BGR))
                        out.release()
                        job_vid_list.append(tmp_path)
                    elif isinstance(val, torch.Tensor):
                        job_img_list.append(val)
            
            print(f"DEBUG: Merged payload for job {i} has {len(job_img_list)} images, {len(job_vid_list)} videos, and text: {combined_string!r}")
            merged_p = GeminiPayload(text=combined_string, images=job_img_list, videos=job_vid_list)
            merged_stream.payloads.append(merged_p)
            
        import folder_paths
        import os
        from PIL import Image
        import numpy as np
        import random
        import shutil
        
        results = []
        max_idx = max(processed_image_streams.keys()) if processed_image_streams else 0
        ui_streams = [None] * (max_idx + 1)
        output_dir = folder_paths.get_temp_directory()
        filename_prefix = "gemini_batch_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
        counter = 1
        for idx, img_stream in processed_image_streams.items():
            ui_streams[idx] = []
            
            c = next((x for x in stream_config if int(x.get("id", -1)) == int(idx)), {})
            mode = c.get("mode", "images")
            fps = int(c.get("fps", 24))
            
            for i in range(target_len):
                job_imgs_info = []
                if i < len(img_stream) and img_stream[i] is not None:
                    val = img_stream[i]
                    v_path = get_video_file_path(val)
                    if v_path:
                        file_prefix = f"{filename_prefix}_{idx}_{i}"
                        ext = os.path.splitext(v_path)[1] or ".mp4"
                        file_name = f"{file_prefix}_{counter:05d}_{ext}"
                        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(file_prefix, output_dir, 512, 512)
                        full_path = os.path.join(full_output_folder, file_name)
                        try:
                            shutil.copyfile(v_path, full_path)
                            img_dict = {
                                "filename": file_name,
                                "subfolder": subfolder,
                                "type": "temp",
                                "is_video": True
                            }
                            results.append(img_dict)
                            job_imgs_info.append(img_dict)
                            counter += 1
                        except Exception as e:
                            print(f"Failed to copy video for preview: {e}")
                    elif mode == "video" and isinstance(val, torch.Tensor) and val.shape[0] > 1:
                        import cv2
                        file_prefix = f"{filename_prefix}_{idx}_{i}"
                        file_name = f"{file_prefix}_{counter:05d}_.mp4"
                        full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(file_prefix, output_dir, val.shape[2], val.shape[1])
                        
                        full_path = os.path.join(full_output_folder, file_name)
                        frames_np = (val.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        
                        h, w, _ = frames_np.shape[1], frames_np.shape[2], frames_np.shape[3]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(full_path, fourcc, float(fps), (w, h))
                        for f in frames_np:
                            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                        writer.release()
                        
                        img_dict = {
                            "filename": file_name,
                            "subfolder": subfolder,
                            "type": "temp",
                            "is_video": True
                        }
                        results.append(img_dict)
                        job_imgs_info.append(img_dict)
                        counter += 1
                    elif isinstance(val, torch.Tensor):
                        batch_tensor = val
                        for b in range(batch_tensor.shape[0]):
                            image = batch_tensor[b]
                            i_val = 255. * image.cpu().numpy()
                            img = Image.fromarray(np.clip(i_val, 0, 255).astype(np.uint8))
                            
                            full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, img.size[0], img.size[1])
                            
                            file = f"{filename}_{counter:05d}_.png"
                            img.save(os.path.join(full_output_folder, file), compress_level=1)
                            img_dict = {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": "temp"
                            }
                            results.append(img_dict)
                            job_imgs_info.append(img_dict)
                            counter += 1
                ui_streams[idx].append(job_imgs_info)

        if results:
            return { "ui": { "b_images": results, "streams": ui_streams }, "result": (merged_stream,) }

        return (merged_stream,)

class MediaPipePoseExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_or_images": (ANY_TYPE,),
            },
            "optional": {
                "render_mode": (["brush_motion_trails", "fluid_ribbons", "skeleton_only"], {"default": "brush_motion_trails"}),
                "trail_length": ("INT", {"default": 14, "min": 0, "max": 60, "step": 1}),
                "background": (["black", "white", "original"], {"default": "black"}),
                "model_complexity": (["full", "heavy", "lite"], {"default": "full"}),
                "min_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("skeleton_frames", "skeleton_video_file")
    FUNCTION = "extract_pose"
    CATEGORY = "gemini"

    def extract_pose(self, video_or_images, render_mode="brush_motion_trails", trail_length=14, background="black", model_complexity="full", min_detection_confidence=0.5, min_tracking_confidence=0.5, fps=24):
        import os
        import urllib.request
        import numpy as np
        import cv2
        import torch
        import tempfile
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, PoseLandmarksConnections

        # Download or locate model
        model_dir = os.path.join(folder_paths.base_path, "models", "mediapipe")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"pose_landmarker_{model_complexity}.task")
        
        if not os.path.exists(model_path):
            urls = {
                "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            }
            url = urls.get(model_complexity, urls["full"])
            print(f"[MediaPipe] Downloading {model_complexity} pose model from {url}...")
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        frames_rgb = []
        v_path = get_video_file_path(video_or_images)
        
        if v_path and os.path.exists(v_path):
            cap = cv2.VideoCapture(v_path)
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_fps and vid_fps > 0:
                fps = int(round(vid_fps))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        elif isinstance(video_or_images, torch.Tensor):
            t = video_or_images
            if len(t.shape) == 3:
                t = t.unsqueeze(0)
            np_frames = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for i in range(np_frames.shape[0]):
                frames_rgb.append(np_frames[i])
        elif isinstance(video_or_images, list):
            for item in video_or_images:
                if isinstance(item, torch.Tensor):
                    t = item
                    if len(t.shape) == 3:
                        t = t.unsqueeze(0)
                    np_frames = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    for i in range(np_frames.shape[0]):
                        frames_rgb.append(np_frames[i])

        if not frames_rgb:
            raise ValueError("[MediaPipePoseExtractor] No frames could be decoded from input video/images.")

        # Keypoint trail palettes for brush trajectories
        TRAIL_COLORS = {
            15: (255, 95, 160),  # Left wrist: Vivid Rose
            16: (0, 230, 255),   # Right wrist: Cyan
            13: (255, 175, 20),  # Left elbow: Amber Gold
            14: (60, 230, 90),   # Right elbow: Jade Green
            27: (160, 60, 255),  # Left ankle: Violet
            28: (255, 230, 0),   # Right ankle: Lemon
            0:  (240, 245, 255)  # Head: Ice White
        }

        LIMB_PAIRS = [
            (11, 13), (13, 15), # Left arm
            (12, 14), (14, 16), # Right arm
            (11, 12),           # Shoulders
            (11, 23), (12, 24), # Torso
            (23, 24),           # Hips
            (23, 25), (25, 27), # Left leg
            (24, 26), (26, 28)  # Right leg
        ]

        skeleton_frames = []
        history = []
        
        for rgb_frame in frames_rgb:
            h, w, _ = rgb_frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            current_pts = {}
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                lm = detection_result.pose_landmarks[0]
                for i in range(len(lm)):
                    current_pts[i] = (int(lm[i].x * w), int(lm[i].y * h))
            
            history.append(current_pts)
            if len(history) > max(1, trail_length):
                history.pop(0)

            if background == "white":
                canvas = np.ones_like(rgb_frame) * 255
            elif background == "original":
                canvas = rgb_frame.copy()
            else: # black
                canvas = np.zeros_like(rgb_frame)
                
            if render_mode == "skeleton_only":
                if detection_result.pose_landmarks:
                    for pose_landmarks in detection_result.pose_landmarks:
                        drawing_utils.draw_landmarks(
                            canvas,
                            pose_landmarks,
                            PoseLandmarksConnections.POSE_LANDMARKS,
                            drawing_styles.get_default_pose_landmarks_style()
                        )
            elif render_mode == "brush_motion_trails":
                # 1. Motion trails for sweeping joints
                for pt_idx, color in TRAIL_COLORS.items():
                    pts_seq = [h_pts[pt_idx] for h_pts in history if pt_idx in h_pts]
                    if len(pts_seq) > 1:
                        for t_i in range(len(pts_seq) - 1):
                            alpha = (t_i + 1) / len(pts_seq)
                            thickness = int(max(2, alpha * 20))
                            c_faded = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                            cv2.line(canvas, pts_seq[t_i], pts_seq[t_i + 1], c_faded, thickness, cv2.LINE_AA)
                            cv2.circle(canvas, pts_seq[t_i + 1], int(thickness // 2), c_faded, -1, cv2.LINE_AA)

                # 2. Dynamic painterly ribbon limbs
                if current_pts:
                    for (p1, p2) in LIMB_PAIRS:
                        if p1 in current_pts and p2 in current_pts:
                            pt1, pt2 = current_pts[p1], current_pts[p2]
                            cv2.line(canvas, pt1, pt2, (180, 210, 255), 14, cv2.LINE_AA)
                            cv2.line(canvas, pt1, pt2, (255, 255, 255), 6, cv2.LINE_AA)
            elif render_mode == "fluid_ribbons":
                # Flowing ribbons tracing entire body structure across time
                for (p1, p2) in LIMB_PAIRS:
                    for t_i in range(len(history) - 1):
                        h1 = history[t_i]
                        h2 = history[t_i + 1]
                        if p1 in h1 and p1 in h2:
                            alpha = (t_i + 1) / len(history)
                            c = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
                            cv2.line(canvas, h1[p1], h2[p1], c, int(max(1, alpha * 8)), cv2.LINE_AA)
                if current_pts:
                    for (p1, p2) in LIMB_PAIRS:
                        if p1 in current_pts and p2 in current_pts:
                            cv2.line(canvas, current_pts[p1], current_pts[p2], (255, 255, 255), 4, cv2.LINE_AA)

            skeleton_frames.append(canvas)

        h, w, _ = skeleton_frames[0].shape
        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for sf in skeleton_frames:
            if sf.shape[1] != w or sf.shape[0] != h:
                sf = cv2.resize(sf, (w, h))
            out_writer.write(cv2.cvtColor(sf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        out_tensor = torch.from_numpy(np.stack(skeleton_frames, axis=0).astype(np.float32) / 255.0)
        
        print(f"[MediaPipePoseExtractor] Extracted {len(skeleton_frames)} skeleton frames @ {fps}fps to {tmp_mp4_path}")
        return (out_tensor, tmp_mp4_path)

class KineticMotionCurveExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_or_images": (ANY_TYPE,),
            },
            "optional": {
                "spline_type": (["catmull_rom_spline", "bezier_spline", "linear"], {"default": "catmull_rom_spline"}),
                "trail_window": ("INT", {"default": 20, "min": 2, "max": 60, "step": 1}),
                "stroke_base_thickness": ("INT", {"default": 18, "min": 2, "max": 50, "step": 1}),
                "speed_to_width_factor": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.1}),
                "speed_to_brightness_factor": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1}),
                "dense_optical_flow": (["enable", "disable"], {"default": "enable"}),
                "temporal_smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "model_complexity": (["full", "heavy", "lite"], {"default": "full"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "max_resolution": (["720p (Fastest)", "1080p (Standard)", "540p (Draft)", "1440p", "Original (No Limit)"], {"default": "720p (Fastest)", "tooltip": "Automatically limit processing resolution to prevent 4K/UHD bottlenecks and excessive memory usage"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "KINETIC_MOTION_DATA")
    RETURN_NAMES = (
        "motion_representation",
        "stage_pipeline_grid",
        "1_segmentation_mask",
        "2_pose_keypoints",
        "3_optical_flow",
        "4_bezier_splines",
        "motion_video_file",
        "kinetic_motion_data"
    )
    FUNCTION = "extract_motion_representation"
    CATEGORY = "gemini"

    def _catmull_rom(self, pts, num_samples=8):
        if len(pts) < 2: return pts
        if len(pts) == 2: return pts
        pts = [pts[0]] + list(pts) + [pts[-1]]
        curve = []
        for i in range(len(pts) - 3):
            p0, p1, p2, p3 = np.array(pts[i], dtype=float), np.array(pts[i+1], dtype=float), np.array(pts[i+2], dtype=float), np.array(pts[i+3], dtype=float)
            for t in np.linspace(0, 1, num_samples, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3)
                curve.append((int(round(pt[0])), int(round(pt[1]))))
        curve.append(pts[-2])
        return curve

    def _draw_hud(self, img, text, subtext=""):
        h, w, _ = img.shape
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (12, 14, 18), -1)
        cv2.line(overlay, (0, 36), (w, 36), (0, 230, 255), 1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 1, cv2.LINE_AA)
        if subtext:
            tw = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
            cv2.putText(img, subtext, (max(10, w - tw - 10), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 230), 1, cv2.LINE_AA)

    def extract_motion_representation(self, video_or_images, spline_type="catmull_rom_spline", trail_window=20, stroke_base_thickness=18, speed_to_width_factor=1.8, speed_to_brightness_factor=1.2, dense_optical_flow="enable", temporal_smoothing=0.6, model_complexity="full", fps=24, max_resolution="720p (Fastest)"):
        import os
        import urllib.request
        import numpy as np
        import cv2
        import torch
        import tempfile
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import drawing_utils, drawing_styles, PoseLandmarksConnections

        model_dir = os.path.join(folder_paths.base_path, "models", "mediapipe")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"pose_landmarker_{model_complexity}.task")
        
        if not os.path.exists(model_path):
            urls = {
                "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            }
            url = urls.get(model_complexity, urls["full"])
            print(f"[KineticMotion] Downloading {model_complexity} pose model from {url}...")
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        frames_rgb = []
        v_path = get_video_file_path(video_or_images)
        
        if v_path and os.path.exists(v_path):
            cap = cv2.VideoCapture(v_path)
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_fps and vid_fps > 0:
                fps = int(round(vid_fps))
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        elif isinstance(video_or_images, torch.Tensor):
            t = video_or_images
            if len(t.shape) == 3: t = t.unsqueeze(0)
            np_frames = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for i in range(np_frames.shape[0]):
                frames_rgb.append(np_frames[i])
        elif isinstance(video_or_images, list):
            for item in video_or_images:
                if isinstance(item, torch.Tensor):
                    t = item
                    if len(t.shape) == 3: t = t.unsqueeze(0)
                    np_frames = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    for i in range(np_frames.shape[0]):
                        frames_rgb.append(np_frames[i])

        if not frames_rgb:
            raise ValueError("[KineticMotionCurveExtractor] No frames could be decoded from input.")

        # Resolution limiter (prevents 4K / UHD video bottlenecks)
        orig_h, orig_w, _ = frames_rgb[0].shape
        res_limits = {
            "540p (Draft)": 960,
            "720p (Fastest)": 1280,
            "1080p (Standard)": 1920,
            "1440p": 2560,
            "Original (No Limit)": 0
        }
        max_dim = res_limits.get(max_resolution, 1280)
        if max_dim > 0 and max(orig_w, orig_h) > max_dim:
            scale = float(max_dim) / float(max(orig_w, orig_h))
            target_w = int(round(orig_w * scale))
            target_h = int(round(orig_h * scale))
            target_w -= (target_w % 2)
            target_h -= (target_h % 2)
            print(f"[KineticMotionCurveExtractor] Auto-downscaling input video from {orig_w}x{orig_h} (4K/UHD) to {target_w}x{target_h} ({max_resolution})")
            frames_rgb = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames_rgb]

        TRAIL_PALETTE = {
            15: (255, 60, 140),   # Left wrist (Rose)
            16: (0, 240, 255),    # Right wrist (Cyan)
            13: (255, 170, 30),   # Left elbow (Amber)
            14: (50, 255, 120),   # Right elbow (Emerald)
            27: (170, 70, 255),   # Left ankle (Violet)
            28: (255, 230, 20),   # Right ankle (Gold)
            0:  (240, 250, 255),  # Head (Luminous White)
            11: (100, 180, 255),  # Left shoulder
            12: (100, 180, 255),  # Right shoulder
            23: (255, 120, 180),  # Left hip
            24: (255, 120, 180),  # Right hip
        }

        KINEMATIC_SWEEPS = [
            (15, 13), (16, 14), # Forearms
            (27, 25), (28, 26), # Lower legs
            (11, 12), (23, 24), # Shoulders & Pelvis
        ]

        smoothed_landmarks = {}
        history = []
        history_vels = []
        history_accels = []
        masks_list = []
        flow_list = []
        flow_particles = []
        prev_gray = None
        prev_gray_small = None

        final_frames = []
        grid_frames = []
        stage1_frames = []
        stage2_frames = []
        stage3_frames = []
        stage4_frames = []

        h, w, _ = frames_rgb[0].shape

        for frame_idx, rgb_frame in enumerate(frames_rgb):
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_img)

            # Stage 1: Human Segmentation Mask
            s1_canvas = np.zeros_like(rgb_frame)
            mask_bool = None
            if detection_result.segmentation_masks and len(detection_result.segmentation_masks) > 0:
                mask = detection_result.segmentation_masks[0].numpy_view()
                mask_bool = (mask.squeeze() > 0.4)
                mask_3c = np.repeat(mask, 3, axis=2)
                s1_canvas = (rgb_frame * mask_3c).astype(np.uint8)
                mask_u8 = (mask.squeeze() * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(s1_canvas, contours, -1, (0, 230, 255), 2)
            masks_list.append(mask_bool)
            s1_annotated = s1_canvas.copy()
            self._draw_hud(s1_annotated, "1. Human Segmentation", "Silhouette Isolated")
            stage1_frames.append(s1_annotated)

            # Stage 2: MediaPipe 3D Pose Keypoints
            curr_pts = {}
            s2_canvas = np.zeros_like(rgb_frame)
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                lm = detection_result.pose_landmarks[0]
                for i in range(len(lm)):
                    raw_pt = np.array([lm[i].x * w, lm[i].y * h], dtype=float)
                    if i in smoothed_landmarks and temporal_smoothing > 0:
                        alpha = 1.0 - np.clip(temporal_smoothing, 0.0, 0.95)
                        smoothed_pt = alpha * raw_pt + (1.0 - alpha) * smoothed_landmarks[i]
                    else:
                        smoothed_pt = raw_pt
                    smoothed_landmarks[i] = smoothed_pt
                    curr_pts[i] = (int(round(smoothed_pt[0])), int(round(smoothed_pt[1])))

                drawing_utils.draw_landmarks(
                    s2_canvas,
                    lm,
                    PoseLandmarksConnections.POSE_LANDMARKS,
                    drawing_styles.get_default_pose_landmarks_style()
                )

            history.append(curr_pts)

            # Instantaneous Velocity & Acceleration
            curr_vels = {}
            curr_accels = {}
            if len(history) >= 2:
                prev_p = history[-2]
                for k, pt in curr_pts.items():
                    if k in prev_p:
                        v = float(np.hypot(pt[0] - prev_p[k][0], pt[1] - prev_p[k][1]))
                        curr_vels[k] = v
                        if len(history_vels) > 0 and k in history_vels[-1]:
                            curr_accels[k] = abs(v - history_vels[-1][k])
            history_vels.append(curr_vels)
            history_accels.append(curr_accels)

            recent_history = history[-max(2, trail_window):]

            s2_annotated = s2_canvas.copy()
            self._draw_hud(s2_annotated, "2. MediaPipe Pose Keypoints", "33 3D Joints Tracked")
            stage2_frames.append(s2_annotated)

            # Stage 3: Dense Optical Flow Velocity (Multi-Scale Accelerated)
            s3_canvas = np.zeros_like(rgb_frame)
            flow = None
            if dense_optical_flow == "enable":
                # Scale down for fast Farneback computation (10x-15x faster on CPU)
                max_flow_dim = 480
                flow_scale = min(1.0, max_flow_dim / max(h, w)) if max(h, w) > max_flow_dim else 1.0
                small_w, small_h = int(w * flow_scale), int(h * flow_scale)
                gray_small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA) if flow_scale < 1.0 else gray

                if prev_gray_small is not None:
                    flow_small = cv2.calcOpticalFlowFarneback(prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    if flow_scale < 1.0:
                        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / flow_scale)
                    else:
                        flow = flow_small

                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros_like(rgb_frame)
                    hsv[..., 1] = 255
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    s3_canvas = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                prev_gray_small = gray_small
            flow_list.append(flow)
            prev_gray = gray
            s3_annotated = s3_canvas.copy()
            self._draw_hud(s3_annotated, "3. Dense Optical Flow", "Direction & Velocity Field")
            stage3_frames.append(s3_annotated)

            # Stage 4: Converted Bezier / Spline Curves
            s4_canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Optical flow particle streamlines
            if flow is not None and curr_pts:
                for pt_idx in [15, 16, 27, 28, 0, 13, 14]:
                    if pt_idx in curr_pts:
                        px, py = curr_pts[pt_idx]
                        if 0 <= px < w and 0 <= py < h:
                            fx, fy = flow[py, px]
                            if np.hypot(fx, fy) > 1.5:
                                color = TRAIL_PALETTE.get(pt_idx, (200, 200, 255))
                                flow_particles.append([float(px), float(py), float(fx), float(fy), 0, color])
                
                new_particles = []
                for p in flow_particles:
                    px, py, fx, fy, age, color = p
                    if age < 8 and 0 <= int(px) < w and 0 <= int(py) < h:
                        npx, npy = px + fx * 1.5, py + fy * 1.5
                        alpha = max(0.1, 1.0 - (age / 8.0))
                        c_p = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                        cv2.line(s4_canvas, (int(px), int(py)), (int(npx), int(npy)), c_p, int(max(1, 4 * alpha)), cv2.LINE_AA)
                        new_particles.append([npx, npy, fx * 0.9, fy * 0.9, age + 1, color])
                flow_particles = new_particles[-150:]

            # Spline Curves with Velocity Mapping
            for pt_idx, base_color in TRAIL_PALETTE.items():
                raw_pts = [h_dict[pt_idx] for h_dict in recent_history if pt_idx in h_dict]
                if len(raw_pts) >= 2:
                    if spline_type == "catmull_rom_spline":
                        spline_pts = self._catmull_rom(raw_pts, num_samples=6)
                    else:
                        spline_pts = raw_pts

                    if len(spline_pts) >= 2:
                        speeds = []
                        for s_i in range(len(spline_pts) - 1):
                            dx = spline_pts[s_i+1][0] - spline_pts[s_i][0]
                            dy = spline_pts[s_i+1][1] - spline_pts[s_i][1]
                            speeds.append(np.hypot(dx, dy))
                        avg_spd = max(0.1, np.mean(speeds))

                        for s_i in range(len(spline_pts) - 1):
                            progress = (s_i + 1) / len(spline_pts)
                            local_spd = speeds[s_i]
                            
                            spd_factor = np.clip(local_spd / (avg_spd + 1e-5), 0.5, 3.0) if speed_to_width_factor > 0 else 1.0
                            thickness = int(max(2, round(progress * stroke_base_thickness * (1.0 + (speed_to_width_factor - 1.0) * (spd_factor - 1.0)))))
                            
                            brightness = np.clip(progress * (0.8 + 0.4 * (spd_factor if speed_to_brightness_factor > 0 else 1.0)), 0.1, 1.0)
                            seg_color = (
                                int(np.clip(base_color[0] * brightness, 0, 255)),
                                int(np.clip(base_color[1] * brightness, 0, 255)),
                                int(np.clip(base_color[2] * brightness, 0, 255))
                            )
                            cv2.line(s4_canvas, spline_pts[s_i], spline_pts[s_i+1], seg_color, thickness, cv2.LINE_AA)
                            cv2.circle(s4_canvas, spline_pts[s_i+1], int(thickness // 2), seg_color, -1, cv2.LINE_AA)

            # Kinematic Ribbon Sweeps
            if curr_pts:
                for (p1, p2) in KINEMATIC_SWEEPS:
                    if p1 in curr_pts and p2 in curr_pts:
                        pt1, pt2 = curr_pts[p1], curr_pts[p2]
                        cv2.line(s4_canvas, pt1, pt2, (220, 235, 255), 8, cv2.LINE_AA)
                        cv2.line(s4_canvas, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)

            final_frames.append(s4_canvas)

            # Annotated Stage 4 for preview
            s4_annotated = s4_canvas.copy()
            self._draw_hud(s4_annotated, "4. Converted Splines", "Velocity -> Stroke Width")
            stage4_frames.append(s4_annotated)

            # 2x2 Comparison Grid
            half_w, half_h = w // 2, h // 2
            r1 = np.hstack([cv2.resize(s1_annotated, (half_w, half_h)), cv2.resize(s2_annotated, (half_w, half_h))])
            r2 = np.hstack([cv2.resize(s3_annotated, (half_w, half_h)), cv2.resize(s4_annotated, (half_w, half_h))])
            grid_canvas = np.vstack([r1, r2])
            grid_frames.append(grid_canvas)

        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for rf in final_frames:
            if rf.shape[1] != w or rf.shape[0] != h:
                rf = cv2.resize(rf, (w, h))
            out_writer.write(cv2.cvtColor(rf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        t_final = torch.from_numpy(np.stack(final_frames, axis=0).astype(np.float32) / 255.0)
        t_grid = torch.from_numpy(np.stack(grid_frames, axis=0).astype(np.float32) / 255.0)
        t_s1 = torch.from_numpy(np.stack(stage1_frames, axis=0).astype(np.float32) / 255.0)
        t_s2 = torch.from_numpy(np.stack(stage2_frames, axis=0).astype(np.float32) / 255.0)
        t_s3 = torch.from_numpy(np.stack(stage3_frames, axis=0).astype(np.float32) / 255.0)
        t_s4 = torch.from_numpy(np.stack(stage4_frames, axis=0).astype(np.float32) / 255.0)

        # Kinetic Motion Data Bundle for KineticMotionToBrushRenderer
        motion_data = {
            "frames_rgb": frames_rgb,
            "history_pts": history,
            "history_vels": history_vels,
            "history_accels": history_accels,
            "flow_list": flow_list,
            "masks_list": masks_list,
            "fps": fps,
            "width": w,
            "height": h,
            "num_frames": len(frames_rgb)
        }

        print(f"[KineticMotionCurveExtractor] Rendered {len(final_frames)} multi-stage frames @ {fps}fps to {tmp_mp4_path}")
        return (t_final, t_grid, t_s1, t_s2, t_s3, t_s4, tmp_mp4_path, motion_data)


class KineticMotionToBrushRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_input": (ANY_TYPE, {"tooltip": "Kinetic Motion Data from Google Kinetic Motion Extractor or video tensor"}),
            },
            "optional": {
                "trajectory_history_length": ("INT", {"default": 24, "min": 2, "max": 120, "step": 1, "tooltip": "Progressive trajectory temporal history length (frames)"}),
                "temporal_decay": ("FLOAT", {"default": 0.88, "min": 0.50, "max": 0.99, "step": 0.01, "tooltip": "Canvas decay rate per frame (0.88 = smooth physical paint fade)"}),
                "stroke_base_width": ("INT", {"default": 22, "min": 2, "max": 80, "step": 1, "tooltip": "Base brush ribbon stroke width"}),
                "min_stroke_width": ("INT", {"default": 3, "min": 1, "max": 30, "step": 1, "tooltip": "Minimum tapered stroke width"}),
                "max_stroke_width": ("INT", {"default": 48, "min": 4, "max": 120, "step": 1, "tooltip": "Maximum explosive stroke width"}),
                "velocity_influence": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Velocity modulation on stroke width and energy"}),
                "acceleration_influence": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Acceleration burst scaling on sharp directional shifts"}),
                "brush_head_size": ("INT", {"default": 14, "min": 2, "max": 50, "step": 1, "tooltip": "Radius of active leading brush head"}),
                "optical_flow_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Intensity of secondary optical flow particle streamers"}),
                "particle_density": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Density of momentum particles"}),
                "glow_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05, "tooltip": "Soft luminescence bloom around active brush ribbons"}),
                "anchor_weight_wrists": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Weight for wrist gestures (primary dance marks)"}),
                "anchor_weight_feet": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Weight for ankle/feet floor movements"}),
                "anchor_weight_elbows_knees": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Weight for elbow and knee articulation"}),
                "anchor_weight_torso_head": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Weight for torso center of mass and head"}),
                "brush_color_mode": (["luminous_white", "kinetic_spectrum", "warm_amber", "cool_cyan"], {"default": "luminous_white"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("brush_stroke_frames", "brush_stroke_video_file")
    FUNCTION = "render_brush_strokes"
    CATEGORY = "gemini"
    DESCRIPTION = "Converts motion trajectories and splines into abstract, progressive dynamic brushstrokes on canvas."

    def _catmull_rom(self, pts, num_samples=8):
        if len(pts) < 2: return pts
        if len(pts) == 2: return pts
        pts = [pts[0]] + list(pts) + [pts[-1]]
        curve = []
        for i in range(len(pts) - 3):
            p0 = np.array(pts[i], dtype=float)
            p1 = np.array(pts[i+1], dtype=float)
            p2 = np.array(pts[i+2], dtype=float)
            p3 = np.array(pts[i+3], dtype=float)
            for t in np.linspace(0, 1, num_samples, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3)
                curve.append((int(round(pt[0])), int(round(pt[1]))))
        curve.append(pts[-2])
        return curve

    def render_brush_strokes(self, motion_input, trajectory_history_length=24, temporal_decay=0.88, stroke_base_width=22, min_stroke_width=3, max_stroke_width=48, velocity_influence=1.8, acceleration_influence=0.8, brush_head_size=14, optical_flow_strength=0.4, particle_density=0.5, glow_strength=0.6, anchor_weight_wrists=2.2, anchor_weight_feet=1.5, anchor_weight_elbows_knees=1.0, anchor_weight_torso_head=0.8, brush_color_mode="luminous_white", fps=24):
        import os
        import numpy as np
        import cv2
        import torch
        import tempfile

        # Unpack motion_input
        motion_data = None
        if isinstance(motion_input, dict) and "history_pts" in motion_input:
            motion_data = motion_input
        elif isinstance(motion_input, tuple) and len(motion_input) > 0 and isinstance(motion_input[-1], dict):
            motion_data = motion_input[-1]
        
        if motion_data is None:
            # Fallback: run extractor directly
            extractor = KineticMotionCurveExtractor()
            _, _, _, _, _, _, _, motion_data = extractor.extract_motion_representation(motion_input, fps=fps)

        history_pts = motion_data["history_pts"]
        history_vels = motion_data.get("history_vels", [])
        history_accels = motion_data.get("history_accels", [])
        flow_list = motion_data.get("flow_list", [])
        masks_list = motion_data.get("masks_list", [])
        fps = motion_data.get("fps", fps)
        w = motion_data["width"]
        h = motion_data["height"]
        num_frames = motion_data["num_frames"]

        # Motion Anchor Hierarchical Weighting
        ANCHOR_WEIGHTS = {
            15: anchor_weight_wrists,  # Left wrist
            16: anchor_weight_wrists,  # Right wrist
            27: anchor_weight_feet,    # Left ankle
            28: anchor_weight_feet,    # Right ankle
            13: anchor_weight_elbows_knees, # Left elbow
            14: anchor_weight_elbows_knees, # Right elbow
            25: anchor_weight_elbows_knees, # Left knee
            26: anchor_weight_elbows_knees, # Right knee
            0:  anchor_weight_torso_head,   # Head
            11: anchor_weight_torso_head * 0.7, # Left shoulder
            12: anchor_weight_torso_head * 0.7, # Right shoulder
            23: anchor_weight_torso_head,   # Left hip
            24: anchor_weight_torso_head    # Right hip
        }

        # Color Palette Profiles
        PALETTES = {
            "luminous_white": {15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.95, 1.0), 28: (0.95, 0.95, 1.0), "default": (0.9, 0.9, 0.9)},
            "kinetic_spectrum": {15: (1.0, 0.25, 0.55), 16: (0.0, 0.95, 1.0), 27: (0.65, 0.3, 1.0), 28: (1.0, 0.9, 0.1), "default": (0.7, 0.85, 1.0)},
            "warm_amber": {15: (1.0, 0.7, 0.1), 16: (1.0, 0.35, 0.1), 27: (1.0, 0.85, 0.4), 28: (1.0, 0.5, 0.2), "default": (0.9, 0.6, 0.2)},
            "cool_cyan": {15: (0.1, 0.95, 1.0), 16: (0.2, 0.6, 1.0), 27: (0.5, 0.4, 1.0), 28: (0.0, 0.8, 0.9), "default": (0.4, 0.7, 0.95)}
        }
        palette = PALETTES.get(brush_color_mode, PALETTES["luminous_white"])

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        rendered_frames = []
        flow_particles = []

        decay = np.clip(temporal_decay, 0.50, 0.99)
        max_particles = int(180 * particle_density)

        for t in range(num_frames):
            # 1. Temporal Persistence Canvas Decay
            canvas = canvas * decay
            stroke_layer = np.zeros((h, w, 3), dtype=np.float32)

            curr_pts = history_pts[t]
            mask = masks_list[t] if t < len(masks_list) else None
            flow = flow_list[t] if t < len(flow_list) else None

            # 2. Secondary Optical Flow Particle Streamlines
            if optical_flow_strength > 0 and flow is not None and curr_pts and max_particles > 0:
                for p_idx in [15, 16, 27, 28, 0, 13, 14]:
                    if p_idx in curr_pts:
                        px, py = curr_pts[p_idx]
                        if 0 <= px < w and 0 <= py < h:
                            if mask is None or (0 <= py < mask.shape[0] and 0 <= px < mask.shape[1] and mask[py, px]):
                                fx, fy = flow[py, px]
                                if np.hypot(fx, fy) > 1.2:
                                    color = palette.get(p_idx, palette["default"])
                                    flow_particles.append([float(px), float(py), float(fx), float(fy), 0, color])
                
                new_particles = []
                for p in flow_particles:
                    px, py, fx, fy, age, color = p
                    if age < 10 and 0 <= int(px) < w and 0 <= int(py) < h:
                        npx, npy = px + fx * 1.5, py + fy * 1.5
                        alpha = max(0.05, 1.0 - (age / 10.0)) * optical_flow_strength
                        c_p = (color[0] * alpha, color[1] * alpha, color[2] * alpha)
                        cv2.line(stroke_layer, (int(px), int(py)), (int(npx), int(npy)), c_p, int(max(1, 3 * alpha)), cv2.LINE_AA)
                        new_particles.append([npx, npy, fx * 0.88, fy * 0.88, age + 1, color])
                flow_particles = new_particles[-max_particles:]

            # 3. Progressive Spline Ribbon Strokes (strictly past window [t - history, t])
            window_start = max(0, t - trajectory_history_length)

            for p_idx, anchor_w in ANCHOR_WEIGHTS.items():
                if anchor_w <= 0.01: continue
                
                pts_seq = [history_pts[fi][p_idx] for fi in range(window_start, t + 1) if p_idx in history_pts[fi]]
                if len(pts_seq) >= 2:
                    spline_pts = self._catmull_rom(pts_seq, num_samples=6)
                    if len(spline_pts) >= 2:
                        speeds = []
                        for s_i in range(len(spline_pts) - 1):
                            dx = spline_pts[s_i+1][0] - spline_pts[s_i][0]
                            dy = spline_pts[s_i+1][1] - spline_pts[s_i][1]
                            speeds.append(np.hypot(dx, dy))
                        avg_spd = max(0.1, np.mean(speeds))

                        base_c = palette.get(p_idx, palette["default"])

                        # Draw Progressive Spline Ribbon
                        for s_i in range(len(spline_pts) - 1):
                            progress = (s_i + 1) / len(spline_pts) # 0 at oldest tail -> 1 at active brush head
                            local_spd = speeds[s_i]
                            spd_factor = np.clip(local_spd / (avg_spd + 1e-5), 0.4, 3.0) if velocity_influence > 0 else 1.0
                            
                            # Stroke width modulation
                            raw_w = progress * stroke_base_width * anchor_w * (1.0 + (velocity_influence - 1.0) * (spd_factor - 1.0) * 0.5)
                            w_clamped = int(np.clip(round(raw_w), min_stroke_width, max_stroke_width))
                            
                            # Luminescence & color gradient
                            lum = np.clip(progress * (0.6 + 0.4 * spd_factor), 0.1, 1.0)
                            seg_c = (base_c[0] * lum, base_c[1] * lum, base_c[2] * lum)
                            
                            cv2.line(stroke_layer, spline_pts[s_i], spline_pts[s_i+1], seg_c, w_clamped, cv2.LINE_AA)
                            cv2.circle(stroke_layer, spline_pts[s_i+1], w_clamped // 2, seg_c, -1, cv2.LINE_AA)

                        # Active Leading Brush Head (at current position P(t))
                        head_pt = spline_pts[-1]
                        head_spd = speeds[-1] if speeds else 1.0
                        head_spd_ratio = np.clip(head_spd / (avg_spd + 1e-5), 0.5, 3.0)
                        
                        head_r = int(np.clip(brush_head_size * anchor_w * (0.7 + 0.3 * head_spd_ratio), 3, max_stroke_width))
                        head_c = (base_c[0] * 1.0, base_c[1] * 1.0, base_c[2] * 1.0)
                        cv2.circle(stroke_layer, head_pt, head_r, head_c, -1, cv2.LINE_AA)

            # 4. Composite active strokes onto decaying canvas
            canvas = np.maximum(canvas, stroke_layer)

            # 5. Glow Bloom Post-Processing
            if glow_strength > 0:
                blurred = cv2.GaussianBlur(canvas, (21, 21), 0)
                frame_comp = np.clip(canvas + glow_strength * 0.5 * blurred, 0.0, 1.0)
            else:
                frame_comp = np.clip(canvas, 0.0, 1.0)

            rendered_frames.append((frame_comp * 255.0).astype(np.uint8))

        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for rf in rendered_frames:
            if rf.shape[1] != w or rf.shape[0] != h:
                rf = cv2.resize(rf, (w, h))
            out_writer.write(cv2.cvtColor(rf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        out_tensor = torch.from_numpy(np.stack(rendered_frames, axis=0).astype(np.float32) / 255.0)
        print(f"[KineticMotionToBrushRenderer] Rendered {len(rendered_frames)} abstract brushstroke frames @ {fps}fps to {tmp_mp4_path}")
        return (out_tensor, tmp_mp4_path)


class TAPNetKineticPointTracker:
    """
    DeepMind TAPNet / Lagrangian Point Tracker for ComfyUI.
    Extracts and tracks dense persistent physical query points strictly on the dancer
    with sub-pixel forward-backward verification and occlusion confidence.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_or_images": (ANY_TYPE, {"tooltip": "Video file path, video loader output, or IMAGE batch tensor [B, H, W, C]"}),
            },
            "optional": {
                "mask_or_images": (ANY_TYPE, {"tooltip": "Human segmentation mask or KINETIC_MOTION_DATA bundle from Extractor"}),
                "num_points": ("INT", {"default": 128, "min": 16, "max": 1024, "step": 16, "tooltip": "Number of persistent physical surface points to track"}),
                "grid_sampling": (["dancer_silhouette", "mask_weighted", "salient_features", "uniform_grid"], {"default": "dancer_silhouette", "tooltip": "Strategy for query point initialization"}),
                "trail_history": ("INT", {"default": 24, "min": 2, "max": 60, "step": 1, "tooltip": "Temporal history trail length in frames"}),
                "point_radius": ("INT", {"default": 5, "min": 1, "max": 16, "step": 1, "tooltip": "Visualized point marker radius in pixels"}),
                "trail_thickness": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1, "tooltip": "Visualized point trajectory trail line thickness"}),
                "color_scheme": (["radiant_red", "emerald_green", "electric_blue", "hot_pink", "luminous_white", "golden_amber", "track_spectrum", "velocity_heat", "cyan_amber"], {"default": "radiant_red", "tooltip": "Color palette for tracked points and ribbons"}),
                "occlusion_threshold": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Visibility confidence cutoff for occlusion detection"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "Output playback frame rate"}),
                "max_resolution": (["720p (Fastest)", "1080p (Standard)", "540p (Draft)", "1440p", "Original (No Limit)"], {"default": "720p (Fastest)", "tooltip": "Automatically limit processing resolution to prevent 4K/UHD bottlenecks and excessive memory usage"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "TAPNET_TRACK_DATA")
    RETURN_NAMES = ("point_tracking_preview", "video_path", "tapnet_track_data")
    FUNCTION = "track_points"
    CATEGORY = "kinetic_motion"

    def track_points(
        self,
        video_or_images: Any,
        mask_or_images: Optional[Any] = None,
        num_points: int = 128,
        grid_sampling: str = "dancer_silhouette",
        trail_history: int = 24,
        point_radius: int = 5,
        trail_thickness: int = 3,
        color_scheme: str = "radiant_red",
        occlusion_threshold: float = 0.40,
        fps: int = 24,
        max_resolution: str = "720p (Fastest)"
    ):
        # 1. Read input frames (reuse already processed frames if available)
        frames = []
        if isinstance(mask_or_images, dict) and "frames_rgb" in mask_or_images and len(mask_or_images["frames_rgb"]) > 0:
            frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in mask_or_images["frames_rgb"]]
        else:
            video_path = get_video_file_path(video_or_images)
            if video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cap.release()
            elif isinstance(video_or_images, torch.Tensor):
                t = video_or_images.cpu().numpy()
                if len(t.shape) == 4:
                    for fi in range(t.shape[0]):
                        frames.append((np.clip(t[fi], 0.0, 1.0) * 255.0).astype(np.uint8))
                elif len(t.shape) == 3:
                    frames.append((np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8))

        if not frames:
            raise ValueError("[TAPNetKineticPointTracker] Could not load video frames from input.")

        # Resolution limiter (prevents 4K / UHD bottlenecks)
        orig_h, orig_w = frames[0].shape[:2]
        res_limits = {
            "540p (Draft)": 960,
            "720p (Fastest)": 1280,
            "1080p (Standard)": 1920,
            "1440p": 2560,
            "Original (No Limit)": 0
        }
        max_dim = res_limits.get(max_resolution, 1280)
        if max_dim > 0 and max(orig_w, orig_h) > max_dim:
            scale = float(max_dim) / float(max(orig_w, orig_h))
            target_w = int(round(orig_w * scale))
            target_h = int(round(orig_h * scale))
            target_w -= (target_w % 2)
            target_h -= (target_h % 2)
            print(f"[TAPNetKineticPointTracker] Auto-downscaling input video from {orig_w}x{orig_h} (4K/UHD) to {target_w}x{target_h} ({max_resolution})")
            frames = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames]

        num_frames = len(frames)
        h, w = frames[0].shape[:2]

        # 2. Extract Human Mask / Dancer Silhouette
        mask0 = None
        history_keypoints = None

        if isinstance(mask_or_images, dict):
            masks_list = mask_or_images.get("masks_list", [])
            if masks_list and masks_list[0] is not None:
                mask0 = (masks_list[0].astype(np.uint8) * 255)
            history_pts = mask_or_images.get("history_pts", [])
            if history_pts and len(history_pts) > 0:
                history_keypoints = history_pts[0]
        elif isinstance(mask_or_images, torch.Tensor):
            mt = mask_or_images.cpu().numpy()
            if len(mt.shape) == 4:
                mf = mt[0]
                if len(mf.shape) == 3:
                    mf_gray = np.mean(mf, axis=-1)
                    mask0 = (mf_gray > 0.08).astype(np.uint8) * 255
            elif len(mt.shape) == 3:
                mask0 = (np.mean(mt, axis=-1) > 0.08).astype(np.uint8) * 255

        if mask0 is not None and mask0.shape[:2] != (h, w):
            mask0 = cv2.resize(mask0, (w, h))

        if mask0 is None and num_frames >= 2:
            g0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
            g1 = cv2.cvtColor(frames[min(2, num_frames - 1)], cv2.COLOR_RGB2GRAY)
            diff = cv2.absdiff(g0, g1)
            _, diff_thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask0 = cv2.dilate(diff_thresh, kernel, iterations=3)

        first_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        # 3. Strictly Seed Query Points Exclusively on Dancer
        query_pts = []

        if history_keypoints:
            for pt_idx in [15, 16, 27, 28, 13, 14, 25, 26, 11, 12, 0]:
                if pt_idx in history_keypoints:
                    kp = history_keypoints[pt_idx]
                    if 0 <= kp[0] < w and 0 <= kp[1] < h:
                        query_pts.append((float(kp[0]), float(kp[1])))

        if mask0 is not None and np.sum(mask0 > 25) > 100:
            feat_pts = cv2.goodFeaturesToTrack(
                first_frame_gray,
                maxCorners=num_points * 3,
                qualityLevel=0.005,
                minDistance=max(4, min(w, h) // 45),
                mask=mask0
            )
            if feat_pts is not None:
                for pt in feat_pts:
                    query_pts.append((float(pt[0][0]), float(pt[0][1])))

            if len(query_pts) < num_points:
                y_idx, x_idx = np.where(mask0 > 25)
                if len(x_idx) > 0:
                    needed = num_points - len(query_pts)
                    chosen_idx = np.random.choice(len(x_idx), size=min(needed * 2, len(x_idx)), replace=False)
                    for c_i in chosen_idx:
                        query_pts.append((float(x_idx[c_i]), float(y_idx[c_i])))
                        if len(query_pts) >= num_points: break
        else:
            grid_cols = int(np.ceil(np.sqrt(num_points * (w / (h + 1e-5)))))
            grid_rows = int(np.ceil(num_points / grid_cols))
            xs = np.linspace(w * 0.2, w * 0.8, grid_cols)
            ys = np.linspace(h * 0.2, h * 0.8, grid_rows)
            for y in ys:
                for x in xs:
                    query_pts.append((float(x), float(y)))
                    if len(query_pts) >= num_points: break
                if len(query_pts) >= num_points: break

        query_pts = query_pts[:num_points]
        N = len(query_pts)

        # 4. Multi-frame Lagrangian Point Tracking
        trajectories = np.zeros((N, num_frames, 2), dtype=np.float32)
        visibilities = np.ones((N, num_frames), dtype=np.float32)
        velocities = np.zeros((N, num_frames), dtype=np.float32)

        for i, pt in enumerate(query_pts):
            trajectories[i, 0] = [pt[0], pt[1]]
            visibilities[i, 0] = 1.0

        prev_gray = first_frame_gray
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        current_pts = np.array(query_pts, dtype=np.float32).reshape(-1, 1, 2)

        for fi in range(1, num_frames):
            curr_gray = cv2.cvtColor(frames[fi], cv2.COLOR_RGB2GRAY)
            
            next_pts, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, current_pts, None, **lk_params
            )
            back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, prev_gray, next_pts, None, **lk_params
            )
            # Get optical flow (reuse from Extractor if available, else compute fast downscaled flow)
            cached_flow = None
            if isinstance(mask_or_images, dict) and "flow_list" in mask_or_images:
                f_list = mask_or_images["flow_list"]
                if fi - 1 < len(f_list) and f_list[fi - 1] is not None:
                    cached_flow = f_list[fi - 1]

            if cached_flow is not None:
                flow = cached_flow
            else:
                max_f_dim = 480
                f_scale = min(1.0, max_f_dim / max(h, w)) if max(h, w) > max_f_dim else 1.0
                if f_scale < 1.0:
                    sw, sh = int(w * f_scale), int(h * f_scale)
                    g_prev_s = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
                    g_curr_s = cv2.resize(curr_gray, (sw, sh), interpolation=cv2.INTER_AREA)
                    f_s = cv2.calcOpticalFlowFarneback(g_prev_s, g_curr_s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow = cv2.resize(f_s, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / f_scale)
                else:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            for i in range(N):
                pt_prev = current_pts[i, 0]
                pt_next = next_pts[i, 0]
                pt_back = back_pts[i, 0]
                fwd_ok = status_fwd[i, 0] == 1
                bwd_ok = status_bwd[i, 0] == 1
                
                fb_dist = np.hypot(pt_prev[0] - pt_back[0], pt_prev[1] - pt_back[1])
                
                if fwd_ok and bwd_ok and fb_dist < 4.5:
                    nx = float(np.clip(pt_next[0], 0, w - 1))
                    ny = float(np.clip(pt_next[1], 0, h - 1))
                    vis = float(np.clip(1.0 - (fb_dist / 4.5) * 0.4, 0.5, 1.0))
                else:
                    ix = int(np.clip(round(pt_prev[0]), 0, w - 1))
                    iy = int(np.clip(round(pt_prev[1]), 0, h - 1))
                    fx, fy = flow[iy, ix]
                    nx = float(np.clip(pt_prev[0] + fx, 0, w - 1))
                    ny = float(np.clip(pt_prev[1] + fy, 0, h - 1))
                    vis = 0.2

                spd = float(np.hypot(nx - pt_prev[0], ny - pt_prev[1]))
                trajectories[i, fi] = [nx, ny]
                visibilities[i, fi] = vis
                velocities[i, fi] = spd
                current_pts[i, 0] = [nx, ny]

            prev_gray = curr_gray

        # 5. Render Visualization on Pure Black Canvas with Chosen Color Palette
        vis_frames = []
        for fi in range(num_frames):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            start_t = max(0, fi - trail_history)

            for i in range(N):
                # Determine point color based on color_scheme
                if color_scheme == "emerald_green":
                    base_rgb = (30, 255, 90)
                elif color_scheme == "electric_blue":
                    base_rgb = (20, 180, 255)
                elif color_scheme == "hot_pink":
                    base_rgb = (255, 40, 180)
                elif color_scheme == "luminous_white":
                    base_rgb = (245, 248, 255)
                elif color_scheme == "golden_amber":
                    base_rgb = (255, 175, 30)
                elif color_scheme == "cyan_amber":
                    prog = float(i) / float(max(1, N - 1))
                    base_rgb = (int(20 + 235 * prog), int(180 - 10 * prog), int(255 - 225 * prog))
                elif color_scheme == "velocity_heat":
                    v = velocities[i, fi]
                    v_norm = np.clip(v / 8.0, 0.0, 1.0)
                    heat_hsv = np.uint8([[[int((1.0 - v_norm) * 120), 240, 255]]])
                    base_rgb = tuple(int(x) for x in cv2.cvtColor(heat_hsv, cv2.COLOR_HSV2RGB)[0][0])
                elif color_scheme == "track_spectrum":
                    hue = int((i * 180 / max(1, N)) % 180)
                    hsv = np.uint8([[[hue, 230, 255]]])
                    base_rgb = tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0])
                else: # radiant_red default
                    base_rgb = (255, 30, 60)

                pts = [tuple(map(int, trajectories[i, t])) for t in range(start_t, fi + 1)]
                
                # Trailing Filaments
                for k in range(len(pts) - 1):
                    v_k = visibilities[i, start_t + k]
                    if v_k >= occlusion_threshold:
                        alpha = float(k + 1) / float(len(pts))
                        seg_c = (int(base_rgb[0] * alpha), int(base_rgb[1] * alpha), int(base_rgb[2] * alpha))
                        cv2.line(canvas, pts[k], pts[k+1], seg_c, trail_thickness, cv2.LINE_AA)

                # Active Leading Head Marker
                curr_p = tuple(map(int, trajectories[i, fi]))
                curr_vis = visibilities[i, fi]
                if curr_vis >= occlusion_threshold:
                    cv2.circle(canvas, curr_p, point_radius + 2, base_rgb, -1, cv2.LINE_AA)
                    cv2.circle(canvas, curr_p, max(1, point_radius // 2), (255, 245, 245), -1, cv2.LINE_AA)

            # Luminescence bloom
            bloom = cv2.GaussianBlur(canvas, (17, 17), 0)
            comp = cv2.addWeighted(canvas, 1.0, bloom, 0.6, 0)
            vis_frames.append(comp)

        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for vf in vis_frames:
            if vf.shape[1] != w or vf.shape[0] != h:
                vf = cv2.resize(vf, (w, h))
            out_writer.write(cv2.cvtColor(vf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        out_tensor = torch.from_numpy(np.stack(vis_frames, axis=0).astype(np.float32) / 255.0)

        tapnet_data = {
            "trajectories": trajectories,
            "visibilities": visibilities,
            "velocities": velocities,
            "num_points": N,
            "fps": fps,
            "width": w,
            "height": h,
            "num_frames": num_frames
        }

        print(f"[TAPNetKineticPointTracker] Tracked {N} dancer points across {num_frames} frames ({color_scheme} palette)")
        return (out_tensor, tmp_mp4_path, tapnet_data)


class KineticTAPNetBrushFusionRenderer:
    """
    Hybrid Kinetic + TAPNet Physical Brush Fusion Renderer.
    Combines Macro MediaPipe Skeletal ribbons with Micro TAPNet surface filaments,
    kinetic particle embers, occlusion pen-lift mechanics, optical flow streamers,
    and automatic dynamic prompt generation for Gemini Omni.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_input": (ANY_TYPE, {"tooltip": "KINETIC_MOTION_DATA from KineticMotionCurveExtractor"}),
            },
            "optional": {
                "tapnet_tracks": (ANY_TYPE, {"tooltip": "TAPNET_TRACK_DATA from TAPNetKineticPointTracker"}),
                "skeletal_brush_weight": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Intensity multiplier for MediaPipe macro skeletal brushstrokes"}),
                "tapnet_filament_weight": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Intensity multiplier for TAPNet micro surface filament ribbons"}),
                "tapnet_particle_density": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Density of trailing kinetic particle embers from tracked surface points"}),
                "temporal_decay": ("FLOAT", {"default": 0.88, "min": 0.50, "max": 0.99, "step": 0.01, "tooltip": "Canvas temporal fade/decay rate per frame"}),
                "stroke_base_width": ("INT", {"default": 24, "min": 2, "max": 80, "step": 1, "tooltip": "Base width for skeletal ribbons"}),
                "filament_width": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "Width for TAPNet surface filaments"}),
                "velocity_influence": ("FLOAT", {"default": 1.8, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Velocity modulation scaling on stroke width and energy"}),
                "optical_flow_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Intensity of optical flow streamers"}),
                "glow_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Luminescence bloom intensity"}),
                "occlusion_pen_lift": (["enable", "disable"], {"default": "enable", "tooltip": "Naturally lift virtual brush and fade filaments when points become occluded"}),
                "brush_color_mode": ([
                    "green_kinetic_red_tapnet",
                    "white_kinetic_colorful_tapnet",
                    "white_kinetic_pinkred_tapnet",
                    "white_kinetic_red_tapnet",
                    "white_kinetic_green_tapnet",
                    "white_kinetic_blue_tapnet",
                    "white_kinetic_amber_tapnet",
                    "white_kinetic_white_tapnet",
                    "colorful_kinetic_white_tapnet",
                    "green_kinetic_colorful_tapnet",
                    "gold_kinetic_cyan_tapnet",
                    "pink_kinetic_cyan_tapnet",
                    "blue_kinetic_amber_tapnet",
                    "amber_kinetic_amber_tapnet",
                    "cyan_kinetic_cyan_tapnet",
                    "colorful_kinetic_colorful_tapnet",
                    "kinetic_spectrum",
                    "luminous_white",
                    "warm_amber",
                    "cool_cyan"
                ], {"default": "green_kinetic_red_tapnet", "tooltip": "Color palette preset structure: <kinetic_color>_kinetic_<tapnet_color>_tapnet"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "Target frame rate"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("fused_brush_video", "video_path", "dynamic_prompt")
    FUNCTION = "render_fused_brush_strokes"
    CATEGORY = "kinetic_motion"

    def _catmull_rom(self, points: List[tuple], num_samples: int = 6) -> List[tuple]:
        if len(points) < 2:
            return points
        if len(points) == 2:
            p0, p1 = points
            return [(int(p0[0] + (p1[0]-p0[0])*t), int(p0[1] + (p1[1]-p0[1])*t)) for t in np.linspace(0, 1, num_samples)]
        
        pts = [points[0]] + list(points) + [points[-1]]
        curve = []
        for i in range(len(pts) - 3):
            p0, p1, p2, p3 = np.array(pts[i]), np.array(pts[i+1]), np.array(pts[i+2]), np.array(pts[i+3])
            for t in np.linspace(0, 1, num_samples, endpoint=(i == len(pts)-4)):
                t2, t3 = t*t, t*t*t
                pt = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3)
                curve.append((int(round(pt[0])), int(round(pt[1]))))
        return curve

    def _generate_dynamic_prompt(self, brush_color_mode: str) -> str:
        prompts_map = {
            "green_kinetic_red_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the vibrant neon green kinetic ribbons as dynamic, sweeping strokes of rich emerald and cadmium green impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single radiant glowing red tracking dot and filament as an autonomous droplet of glistening, thick crimson and vermilion oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the green structural ribbons and red surface filaments self-paint across the scene, blending rich green and crimson paint "
                "against the pure dark void with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_colorful_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the pure luminous white kinetic ribbons as dynamic, sweeping strokes of thick titanium white impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single radiant multi-colored tracking dot and filament as an autonomous droplet of glistening rainbow oil paint "
                "(crimson, cobalt, gold, emerald, violet) that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the white structural ribbons and colorful surface filaments self-paint across the scene against the pure dark void "
                "with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_pinkred_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the pure luminous white kinetic ribbons as dynamic, sweeping strokes of thick titanium white impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single glowing pink and red tracking dot and filament as an autonomous droplet of glistening magenta, crimson, and rose oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the white structural ribbons and pink-red surface filaments self-paint across the scene against the pure dark void "
                "with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_red_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the pure luminous white kinetic ribbons as dynamic, sweeping strokes of thick titanium white impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single glowing red tracking dot and filament as an autonomous droplet of glistening crimson and vermilion oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the white structural ribbons and red surface filaments self-paint across the scene against the pure dark void "
                "with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_green_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the pure luminous white kinetic ribbons as dynamic, sweeping strokes of thick titanium white impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single vibrant green tracking dot and filament as an autonomous droplet of glistening emerald and viridian oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the white structural ribbons and emerald green surface filaments self-paint across the scene against the pure dark void "
                "with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_blue_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the pure luminous white kinetic ribbons as dynamic, sweeping strokes of thick titanium white impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single electric blue tracking dot and filament as an autonomous droplet of glistening ultramarine, cerulean, and azure oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the white structural ribbons and blue surface filaments self-paint across the scene against the pure dark void "
                "with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_amber_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the pure luminous white kinetic ribbons as dynamic, sweeping strokes of thick titanium white impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single golden amber tracking dot and filament as an autonomous droplet of glistening cadmium gold and warm ochre oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the white structural ribbons and golden surface filaments self-paint across the scene against the pure dark void "
                "with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "white_kinetic_white_tapnet": (
                "Masterpiece self-moving animated monochrome oil painting on deep black canvas. "
                "Interpret both the kinetic ribbons and tracking dots as pure luminous strokes and droplets of thick titanium white impasto oil paint "
                "with visible palette-knife textures, wet brush ridges, and heavy physical relief. "
                "As the dancer moves, the white calligraphy ribbons and delicate white surface filaments self-paint across the scene against the pure dark void "
                "with dramatic studio lighting, ultra-detailed fine art."
            ),
            "colorful_kinetic_white_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the dynamic kinetic ribbons as sweeping multi-colored strokes of thick rainbow impasto oil paint "
                "(emerald, cobalt, crimson, amber, violet) with visible palette-knife textures and heavy physical relief. "
                "Interpret every single tracking dot and filament as an autonomous droplet of glistening, thick titanium white oil paint "
                "that blooms and swirls into fine wet white filaments with luminous glowing embers. "
                "As the dancer moves, the colorful structural ribbons and delicate white surface filaments self-paint across the scene "
                "against the pure dark void with dramatic raking studio light illuminating every 3D wet paint ridge, ultra-detailed fine art."
            ),
            "green_kinetic_colorful_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the vibrant green kinetic ribbons as dynamic, sweeping strokes of rich emerald and jade impasto oil paint. "
                "Interpret every single tracking dot and filament as an autonomous droplet of glistening multi-colored rainbow oil paint. "
                "As the dancer moves, the green structural ribbons and rainbow surface filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
            "gold_kinetic_cyan_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the rich golden amber kinetic ribbons as dynamic, sweeping strokes of warm gold, cadmium yellow, and ochre impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single electric cyan tracking dot and filament as an autonomous droplet of glistening turquoise, teal, and cyan oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the golden structural ribbons and cyan surface filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
            "pink_kinetic_cyan_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the hot magenta pink kinetic ribbons as dynamic, sweeping strokes of vivid magenta, rose, and pink impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single electric cyan tracking dot and filament as an autonomous droplet of glistening turquoise, aqua, and cyan oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the pink structural ribbons and cyan surface filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
            "blue_kinetic_amber_tapnet": (
                "Masterpiece self-moving animated oil painting on deep black canvas. "
                "Interpret the deep cobalt blue kinetic ribbons as dynamic, sweeping strokes of rich ultramarine and navy impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "Interpret every single golden amber tracking dot and filament as an autonomous droplet of glistening cadmium gold and warm amber oil paint "
                "that blooms and swirls into fine wet filaments with glowing embers. "
                "As the dancer moves, the blue structural ribbons and golden surface filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
            "amber_kinetic_amber_tapnet": (
                "Masterpiece self-moving animated warm oil painting on deep black canvas. "
                "Interpret both the kinetic ribbons and tracking dots as rich strokes and droplets of warm golden amber, flame orange, and ochre impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "As the dancer moves, the fiery ribbons and warm filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
            "cyan_kinetic_cyan_tapnet": (
                "Masterpiece self-moving animated cool oil painting on deep black canvas. "
                "Interpret both the kinetic ribbons and tracking dots as electric strokes and droplets of cyan, turquoise, and deep indigo impasto oil paint "
                "with visible palette-knife textures and heavy physical relief. "
                "As the dancer moves, the cool luminous ribbons and cyan filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
            "colorful_kinetic_colorful_tapnet": (
                "Masterpiece self-moving animated multi-color oil painting on deep black canvas. "
                "Interpret the kinetic ribbons and all tracking dots as a vibrant spectrum of glistening impasto oil paint (crimson, gold, emerald, cobalt, violet) "
                "with rich 3D palette-knife textures, glistening wet paint ridges, and dynamic glowing particle embers. "
                "As the dancer moves, the rainbow ribbons and multi-colored filaments self-paint across the scene against the pure dark void, ultra-detailed fine art."
            ),
        }
        # Fallback aliases
        alias_map = {
            "luminous_white": "white_kinetic_white_tapnet",
            "kinetic_spectrum": "colorful_kinetic_colorful_tapnet",
            "warm_amber": "amber_kinetic_amber_tapnet",
            "cool_cyan": "cyan_kinetic_cyan_tapnet"
        }
        effective_mode = alias_map.get(brush_color_mode, brush_color_mode)
        return prompts_map.get(effective_mode, prompts_map["green_kinetic_red_tapnet"])

    def render_fused_brush_strokes(
        self,
        motion_input: Any,
        tapnet_tracks: Optional[Any] = None,
        skeletal_brush_weight: float = 1.2,
        tapnet_filament_weight: float = 1.2,
        tapnet_particle_density: float = 0.6,
        temporal_decay: float = 0.88,
        stroke_base_width: int = 24,
        filament_width: int = 5,
        velocity_influence: float = 1.8,
        optical_flow_strength: float = 0.4,
        glow_strength: float = 0.6,
        occlusion_pen_lift: str = "enable",
        brush_color_mode: str = "green_kinetic_red_tapnet",
        fps: int = 24
    ):
        if not isinstance(motion_input, dict):
            raise ValueError("[KineticTAPNetBrushFusionRenderer] Expected KINETIC_MOTION_DATA dictionary from extractor.")

        history_pts = motion_input.get("history_pts", [])
        optical_flow_history = motion_input.get("optical_flow_history", [])
        num_frames = motion_input.get("num_frames", len(history_pts))
        w = motion_input.get("width", 720)
        h = motion_input.get("height", 1280)
        fps = motion_input.get("fps", fps)

        # Dynamic Color Palette Presets
        is_tapnet_rainbow = False

        if brush_color_mode in ["white_kinetic_white_tapnet", "luminous_white"]:
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (0.95, 0.98, 1.0)
            }
            filament_base = (0.95, 0.98, 1.0)
        elif brush_color_mode == "colorful_kinetic_white_tapnet":
            # Rainbow Skeletal Ribbons + Pure Luminous White Filaments
            skeletal_palette = {
                15: (1.0, 0.18, 0.45), 16: (0.05, 0.85, 0.95), 27: (0.7, 0.2, 1.0), 28: (1.0, 0.8, 0.1),
                11: (0.2, 0.9, 0.4), 12: (1.0, 0.45, 0.1), "default": (0.9, 0.9, 0.95)
            }
            filament_base = (0.96, 0.98, 1.0)
        elif brush_color_mode == "white_kinetic_colorful_tapnet":
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (1.0, 1.0, 1.0)
            }
            is_tapnet_rainbow = True
            filament_base = (1.0, 1.0, 1.0)
        elif brush_color_mode in ["white_kinetic_pinkred_tapnet", "white_kinetic_pink_tapnet"]:
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (1.0, 1.0, 1.0)
            }
            filament_base = (1.0, 0.15, 0.45)
        elif brush_color_mode == "white_kinetic_red_tapnet":
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (1.0, 1.0, 1.0)
            }
            filament_base = (1.0, 0.12, 0.22)
        elif brush_color_mode == "white_kinetic_green_tapnet":
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (1.0, 1.0, 1.0)
            }
            filament_base = (0.1, 1.0, 0.35)
        elif brush_color_mode == "white_kinetic_blue_tapnet":
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (1.0, 1.0, 1.0)
            }
            filament_base = (0.05, 0.8, 1.0)
        elif brush_color_mode == "white_kinetic_amber_tapnet":
            skeletal_palette = {
                15: (1.0, 1.0, 1.0), 16: (1.0, 1.0, 1.0), 27: (0.95, 0.98, 1.0), 28: (0.95, 0.98, 1.0),
                "default": (1.0, 1.0, 1.0)
            }
            filament_base = (1.0, 0.72, 0.1)
        elif brush_color_mode == "green_kinetic_red_tapnet":
            skeletal_palette = {
                15: (0.1, 1.0, 0.35), 16: (0.1, 1.0, 0.35), 27: (0.05, 0.95, 0.4), 28: (0.05, 0.95, 0.4),
                13: (0.15, 0.9, 0.3), 14: (0.15, 0.9, 0.3), 25: (0.1, 0.85, 0.35), 26: (0.1, 0.85, 0.35),
                "default": (0.1, 1.0, 0.35)
            }
            filament_base = (1.0, 0.12, 0.22)
        elif brush_color_mode == "green_kinetic_colorful_tapnet":
            skeletal_palette = {
                15: (0.1, 1.0, 0.35), 16: (0.1, 1.0, 0.35), 27: (0.05, 0.95, 0.4), 28: (0.05, 0.95, 0.4),
                "default": (0.1, 1.0, 0.35)
            }
            is_tapnet_rainbow = True
            filament_base = (1.0, 1.0, 1.0)
        elif brush_color_mode == "gold_kinetic_cyan_tapnet":
            skeletal_palette = {
                15: (1.0, 0.75, 0.15), 16: (1.0, 0.75, 0.15), 27: (1.0, 0.6, 0.1), 28: (1.0, 0.6, 0.1),
                "default": (1.0, 0.75, 0.15)
            }
            filament_base = (0.05, 0.85, 1.0)
        elif brush_color_mode == "pink_kinetic_cyan_tapnet":
            skeletal_palette = {
                15: (1.0, 0.15, 0.65), 16: (1.0, 0.15, 0.65), 27: (0.9, 0.1, 0.8), 28: (0.9, 0.1, 0.8),
                "default": (1.0, 0.15, 0.65)
            }
            filament_base = (0.05, 0.9, 1.0)
        elif brush_color_mode == "blue_kinetic_amber_tapnet":
            skeletal_palette = {
                15: (0.1, 0.4, 1.0), 16: (0.1, 0.4, 1.0), 27: (0.05, 0.3, 0.9), 28: (0.05, 0.3, 0.9),
                "default": (0.1, 0.4, 1.0)
            }
            filament_base = (1.0, 0.72, 0.1)
        elif brush_color_mode in ["amber_kinetic_amber_tapnet", "warm_amber"]:
            skeletal_palette = {
                15: (1.0, 0.35, 0.1), 16: (1.0, 0.7, 0.1), 27: (1.0, 0.2, 0.3), 28: (1.0, 0.85, 0.2),
                "default": (1.0, 0.55, 0.15)
            }
            filament_base = (1.0, 0.65, 0.2)
        elif brush_color_mode in ["cyan_kinetic_cyan_tapnet", "cool_cyan"]:
            skeletal_palette = {
                15: (0.0, 0.9, 1.0), 16: (0.1, 0.5, 1.0), 27: (0.4, 0.2, 1.0), 28: (0.0, 1.0, 0.8),
                "default": (0.1, 0.7, 0.95)
            }
            filament_base = (0.1, 0.85, 1.0)
        else: # colorful_kinetic_colorful_tapnet / kinetic_spectrum
            skeletal_palette = {
                15: (1.0, 0.18, 0.45), 16: (0.05, 0.85, 0.95), 27: (0.7, 0.2, 1.0), 28: (1.0, 0.8, 0.1),
                11: (0.2, 0.9, 0.4), 12: (1.0, 0.45, 0.1), "default": (0.9, 0.9, 0.95)
            }
            is_tapnet_rainbow = True
            filament_base = (1.0, 0.15, 0.25)

        ANCHOR_WEIGHTS = {
            15: 1.6, 16: 1.6, 27: 1.4, 28: 1.4, 13: 1.0, 14: 1.0,
            25: 1.0, 26: 1.0, 11: 0.8, 12: 0.8, 0: 0.7
        }

        tap_trajectories = None
        tap_visibilities = None
        tap_velocities = None
        tap_N = 0
        if isinstance(tapnet_tracks, dict) and "trajectories" in tapnet_tracks:
            tap_trajectories = tapnet_tracks["trajectories"]
            tap_visibilities = tapnet_tracks.get("visibilities", None)
            tap_velocities = tapnet_tracks.get("velocities", None)
            tap_N = tapnet_tracks.get("num_points", tap_trajectories.shape[0])

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        rendered_frames = []
        window_len = 24

        for t in range(num_frames):
            canvas *= float(temporal_decay)

            if optical_flow_strength > 0 and t < len(optical_flow_history) and optical_flow_history[t] is not None:
                flow = optical_flow_history[t]
                fh, fw = flow.shape[:2]
                if fh == h and fw == w:
                    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
                    map_x = grid_x - flow[..., 0] * optical_flow_strength * 0.5
                    map_y = grid_y - flow[..., 1] * optical_flow_strength * 0.5
                    canvas = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            stroke_layer = np.zeros((h, w, 3), dtype=np.float32)

            # 1. Macro Skeletal Ribbons
            if skeletal_brush_weight > 0.01:
                win_start = max(0, t - window_len + 1)
                for p_idx, anchor_w in ANCHOR_WEIGHTS.items():
                    pts_seq = [history_pts[fi][p_idx] for fi in range(win_start, t + 1) if fi < len(history_pts) and p_idx in history_pts[fi]]
                    if len(pts_seq) >= 2:
                        spline_pts = self._catmull_rom(pts_seq, num_samples=6)
                        if len(spline_pts) >= 2:
                            speeds = [np.hypot(spline_pts[si+1][0]-spline_pts[si][0], spline_pts[si+1][1]-spline_pts[si][1]) for si in range(len(spline_pts)-1)]
                            avg_spd = max(0.1, np.mean(speeds))
                            base_c = skeletal_palette.get(p_idx, skeletal_palette["default"])

                            for si in range(len(spline_pts) - 1):
                                progress = float(si + 1) / float(len(spline_pts))
                                spd_factor = np.clip(speeds[si] / (avg_spd + 1e-5), 0.4, 3.0)
                                raw_w = progress * stroke_base_width * anchor_w * skeletal_brush_weight * (1.0 + (velocity_influence - 1.0) * (spd_factor - 1.0) * 0.5)
                                w_clamped = int(np.clip(round(raw_w), 2, 60))
                                lum = np.clip(progress * (0.6 + 0.4 * spd_factor), 0.1, 1.0)
                                seg_c = (float(base_c[0] * lum), float(base_c[1] * lum), float(base_c[2] * lum))
                                cv2.line(stroke_layer, spline_pts[si], spline_pts[si+1], seg_c, w_clamped, cv2.LINE_AA)
                                cv2.circle(stroke_layer, spline_pts[si+1], w_clamped // 2, seg_c, -1, cv2.LINE_AA)

                            # Leading Brush Head
                            head_pt = spline_pts[-1]
                            head_r = int(np.clip(14 * anchor_w * skeletal_brush_weight, 3, 40))
                            cv2.circle(stroke_layer, head_pt, head_r, (float(base_c[0]), float(base_c[1]), float(base_c[2])), -1, cv2.LINE_AA)
                            cv2.circle(stroke_layer, head_pt, max(1, head_r // 3), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)

            # 2. Micro Surface Filaments & Embers
            if tap_trajectories is not None and tapnet_filament_weight > 0.01 and t < tap_trajectories.shape[1]:
                tap_win_start = max(0, t - window_len + 1)
                for i in range(tap_N):
                    vis_t = float(tap_visibilities[i, t]) if tap_visibilities is not None else 1.0
                    if occlusion_pen_lift == "enable" and vis_t < 0.35:
                        continue

                    p_history = [tuple(map(int, tap_trajectories[i, fi])) for fi in range(tap_win_start, t + 1)]
                    if len(p_history) >= 2:
                        fil_pts = self._catmull_rom(p_history, num_samples=4)
                        if len(fil_pts) >= 2:
                            if is_tapnet_rainbow:
                                hue_angle = (i * 360 / max(1, tap_N)) % 360
                                f_hsv = np.uint8([[[int(hue_angle / 2), 230, 255]]])
                                f_rgb = tuple(float(x) / 255.0 for x in cv2.cvtColor(f_hsv, cv2.COLOR_HSV2RGB)[0][0])
                            else:
                                f_rgb = filament_base

                            for fi_k in range(len(fil_pts) - 1):
                                prog = float(fi_k + 1) / float(len(fil_pts))
                                fw = max(1, int(round(filament_width * tapnet_filament_weight * prog * vis_t)))
                                fil_c = (float(f_rgb[0] * prog * vis_t), float(f_rgb[1] * prog * vis_t), float(f_rgb[2] * prog * vis_t))
                                cv2.line(stroke_layer, fil_pts[fi_k], fil_pts[fi_k+1], fil_c, fw, cv2.LINE_AA)

                            # Leading Point Head Marker
                            curr_head = fil_pts[-1]
                            head_rad = max(2, int(filament_width * 1.3))
                            cv2.circle(stroke_layer, curr_head, head_rad + 1, (float(f_rgb[0]), float(f_rgb[1]), float(f_rgb[2])), -1, cv2.LINE_AA)
                            cv2.circle(stroke_layer, curr_head, max(1, head_rad // 2), (1.0, 0.95, 0.95), -1, cv2.LINE_AA)

                            # Dynamic Particle Embers
                            if tapnet_particle_density > 0.1 and tap_velocities is not None:
                                p_spd = tap_velocities[i, t]
                                if p_spd > 3.5 and np.random.rand() < tapnet_particle_density:
                                    ember_offset = np.random.uniform(-10, 10, size=2).astype(int)
                                    ember_pt = (curr_head[0] + ember_offset[0], curr_head[1] + ember_offset[1])
                                    if 0 <= ember_pt[0] < w and 0 <= ember_pt[1] < h:
                                        cv2.circle(stroke_layer, ember_pt, 2, (float(f_rgb[0]), float(f_rgb[1]), float(f_rgb[2])), -1, cv2.LINE_AA)

            canvas = np.maximum(canvas, stroke_layer)

            if glow_strength > 0:
                blurred = cv2.GaussianBlur(canvas, (21, 21), 0)
                frame_comp = np.clip(canvas + glow_strength * 0.55 * blurred, 0.0, 1.0)
            else:
                frame_comp = np.clip(canvas, 0.0, 1.0)

            rendered_frames.append((frame_comp * 255.0).astype(np.uint8))

        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp_mp4_path = tmp_f.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(tmp_mp4_path, fourcc, float(fps), (w, h))
        for rf in rendered_frames:
            if rf.shape[1] != w or rf.shape[0] != h:
                rf = cv2.resize(rf, (w, h))
            out_writer.write(cv2.cvtColor(rf, cv2.COLOR_RGB2BGR))
        out_writer.release()

        out_tensor = torch.from_numpy(np.stack(rendered_frames, axis=0).astype(np.float32) / 255.0)
        
        # Generate matched dynamic prompt based on the chosen color mode
        dynamic_prompt = self._generate_dynamic_prompt(brush_color_mode)

        print(f"[KineticTAPNetBrushFusionRenderer] Rendered {len(rendered_frames)} fused brush frames ({brush_color_mode}) to {tmp_mp4_path}")
        return (out_tensor, tmp_mp4_path, dynamic_prompt)


class KineticDualComparisonViewer:
    """
    Dual Video Comparison & Side-by-Side Synchronizer.
    Combines Stage 6 Fused Kinetic Brushstrokes and Final Stylized Omni Artwork
    into a unified side-by-side comparison video with live inline preview.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stage6_fused_brushstrokes": ("IMAGE", {"tooltip": "Stage 6 Fused Kinetic + TAPNet Brushstrokes video"}),
                "final_stylized_artwork": ("IMAGE", {"tooltip": "Final Stylized Artwork video from Gemini Omni"}),
            },
            "optional": {
                "layout": (["side_by_side_horizontal", "stacked_vertical"], {"default": "side_by_side_horizontal"}),
                "show_labels": (["enable", "disable"], {"default": "enable"}),
                "label_left": ("STRING", {"default": "Stage 6: Fused Brushstrokes"}),
                "label_right": ("STRING", {"default": "Final Stylized Artwork"}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 60}),
                "format": (["mp4", "animated_webp", "gif"], {"default": "mp4"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "Dual_Kinetic_Comparison"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison_video", "video_path")
    OUTPUT_NODE = True
    FUNCTION = "create_comparison"
    CATEGORY = "kinetic_motion"
    DESCRIPTION = "Combines Stage 6 Fused Brushstrokes and Final Stylized Video into a unified side-by-side comparison video."

    def create_comparison(
        self,
        stage6_fused_brushstrokes: torch.Tensor,
        final_stylized_artwork: torch.Tensor,
        layout: str = "side_by_side_horizontal",
        show_labels: str = "enable",
        label_left: str = "Stage 6: Fused Brushstrokes",
        label_right: str = "Final Stylized Artwork",
        frame_rate: int = 24,
        format: str = "mp4",
        save_output: bool = True,
        filename_prefix: str = "Dual_Kinetic_Comparison"
    ):
        if stage6_fused_brushstrokes is None or final_stylized_artwork is None:
            raise ValueError("[KineticDualComparisonViewer] Both stage6 and final stylized inputs are required.")

        num_frames = min(stage6_fused_brushstrokes.shape[0], final_stylized_artwork.shape[0])
        h1, w1 = stage6_fused_brushstrokes.shape[1], stage6_fused_brushstrokes.shape[2]
        h2, w2 = final_stylized_artwork.shape[1], final_stylized_artwork.shape[2]

        target_h = max(h1, h2)
        target_w = int(w1 * (target_h / h1))
        target_w2 = int(w2 * (target_h / h2))

        comp_frames = []
        pil_frames = []

        for fi in range(num_frames):
            f1_np = (stage6_fused_brushstrokes[fi].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            f2_np = (final_stylized_artwork[fi].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            f1_resized = cv2.resize(f1_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            f2_resized = cv2.resize(f2_np, (target_w2, target_h), interpolation=cv2.INTER_LANCZOS4)

            if show_labels == "enable":
                for f_img, lbl in [(f1_resized, label_left), (f2_resized, label_right)]:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.55
                    thick = 1
                    (txt_w, txt_h), baseline = cv2.getTextSize(lbl, font, scale, thick)
                    pad = 8
                    box_coords = ((12, 12), (12 + txt_w + pad * 2, 12 + txt_h + pad * 2))
                    sub_img = f_img[box_coords[0][1]:box_coords[1][1], box_coords[0][0]:box_coords[1][0]]
                    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
                        dark_rect = np.zeros_like(sub_img)
                        cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 0, sub_img)
                        f_img[box_coords[0][1]:box_coords[1][1], box_coords[0][0]:box_coords[1][0]] = sub_img
                        cv2.rectangle(f_img, box_coords[0], box_coords[1], (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(f_img, lbl, (12 + pad, 12 + txt_h + pad - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

            if layout == "stacked_vertical":
                comb = np.concatenate([f1_resized, f2_resized], axis=0)
            else:
                comb = np.concatenate([f1_resized, f2_resized], axis=1)

            comp_frames.append(comb)
            pil_frames.append(Image.fromarray(comb))

        out_tensor = torch.from_numpy(np.stack(comp_frames, axis=0).astype(np.float32) / 255.0)

        output_dir = self.output_dir if save_output else self.temp_dir
        type_str = "output" if save_output else "temp"
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, comp_frames[0].shape[1], comp_frames[0].shape[0]
        )

        duration_ms = max(1, int(1000.0 / max(1, frame_rate)))
        h_c, w_c = comp_frames[0].shape[:2]
        if w_c % 2 != 0: w_c -= 1
        if h_c % 2 != 0: h_c -= 1

        ui_results = []
        if format in ["animated_webp", "image/webp"]:
            file_name = f"{filename}_{counter:05d}_.webp"
            saved_path = os.path.join(full_output_folder, file_name)
            pil_frames[0].save(saved_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0, quality=90)
            ui_results.append({"filename": file_name, "subfolder": subfolder, "type": type_str, "format": "image/webp"})
        elif format in ["gif", "image/gif"]:
            file_name = f"{filename}_{counter:05d}_.gif"
            saved_path = os.path.join(full_output_folder, file_name)
            pil_frames[0].save(saved_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0)
            ui_results.append({"filename": file_name, "subfolder": subfolder, "type": type_str, "format": "image/gif"})
        else: # mp4
            file_name = f"{filename}_{counter:05d}_.mp4"
            saved_path = os.path.join(full_output_folder, file_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(saved_path, fourcc, float(frame_rate), (w_c, h_c))
            for cf in comp_frames:
                if cf.shape[1] != w_c or cf.shape[0] != h_c:
                    cf = cv2.resize(cf, (w_c, h_c))
                out.write(cv2.cvtColor(cf, cv2.COLOR_RGB2BGR))
            out.release()

            preview_webp_name = f"{filename}_{counter:05d}_preview.webp"
            preview_path = os.path.join(full_output_folder, preview_webp_name)
            pil_frames[0].save(preview_path, save_all=True, append_images=pil_frames[1:], duration=duration_ms, loop=0, quality=85)
            ui_results.append({"filename": preview_webp_name, "subfolder": subfolder, "type": type_str, "format": "image/webp"})

        print(f"[KineticDualComparisonViewer] Saved side-by-side comparison to {saved_path}")
        return {
            "ui": {"images": ui_results},
            "result": (out_tensor, saved_path)
        }


NODE_CLASS_MAPPINGS = {
    "GeminiProModel": GeminiProModel,
    "GeminiOmniModel": GeminiOmniModel,
    "GeminiAuthConfig": GeminiAuthConfig,
    "GeminiMultimodalPreview": GeminiMultimodalPreview,
    "GeminiJobBatcher": GeminiJobBatcher,
    "GeminiVideoCombine": GeminiVideoCombine,
    "KineticVideoCombine": GeminiVideoCombine,
    "MediaPipePoseExtractor": MediaPipePoseExtractor,
    "KineticMotionCurveExtractor": KineticMotionCurveExtractor,
    "KineticMotionToBrushRenderer": KineticMotionToBrushRenderer,
    "TAPNetKineticPointTracker": TAPNetKineticPointTracker,
    "KineticTAPNetBrushFusionRenderer": KineticTAPNetBrushFusionRenderer,
    "TAPNetBrushFusionRenderer": KineticTAPNetBrushFusionRenderer,
    "KineticDualComparisonViewer": KineticDualComparisonViewer,
    "DualVideoComparisonViewer": KineticDualComparisonViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiProModel": "Gemini Execution Node",
    "GeminiOmniModel": "Gemini Omni Model",
    "GeminiAuthConfig": "Gemini Auth Config",
    "GeminiMultimodalPreview": "Gemini Multimodal Preview",
    "GeminiJobBatcher": "Gemini Job Batcher",
    "GeminiVideoCombine": "Gemini Video Combine & Preview",
    "KineticVideoCombine": "Kinetic Video Combine & Preview",
    "MediaPipePoseExtractor": "Google MediaPipe Pose Extractor",
    "KineticMotionCurveExtractor": "Google Kinetic Motion Curve Extractor",
    "KineticMotionToBrushRenderer": "Kinetic Motion-to-Brush Renderer",
    "TAPNetKineticPointTracker": "TAPNet Point Tracker (Tracking Any Point)",
    "KineticTAPNetBrushFusionRenderer": "Kinetic + TAPNet Brush Fusion Renderer",
    "TAPNetBrushFusionRenderer": "TAPNet Brush Fusion Renderer",
    "KineticDualComparisonViewer": "Kinetic Dual Comparison Viewer (Stage 6 + Final Art)",
    "DualVideoComparisonViewer": "Dual Video Comparison Viewer"
}
