import os
import json
import folder_paths
from server import PromptServer
from aiohttp import web

# Global register for wireless shortcuts / collections
SHORTCUT_STORE = {
    "default": ""
}

class VisualFileBrowserNode:
    """
    A visual file browser node that lists outputs and allows setting active collections/shortcuts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_name": ("STRING", {"default": ""}),
                "auto_refresh": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "passthrough": ("*",),
            }
        }

    RETURN_TYPES = ("STRING", "*")
    RETURN_NAMES = ("folder_path", "passthrough")
    FUNCTION = "process"
    CATEGORY = "Gemini/FileBrowser"

    def process(self, folder_name="", auto_refresh=True, passthrough=None):
        base_output = folder_paths.get_output_directory()
        target_dir = os.path.join(base_output, folder_name) if folder_name else base_output
        
        # Broadcast folder update event to UI via WebSocket
        try:
            PromptServer.instance.send_sync("filebrowser-refresh", {
                "folder_name": folder_name,
                "full_path": target_dir
            })
        except Exception as e:
            pass

        return (target_dir, passthrough)


class OutputFolderBroadcasterNode:
    """
    Broadcaster node that generates a subfolder path and broadcasts it to output/save nodes and UI.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subfolder": ("STRING", {"default": "session_01"}),
                "prefix": ("STRING", {"default": "output"}),
            },
            "optional": {
                "passthrough": ("*",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "*")
    RETURN_NAMES = ("filename_prefix", "full_folder_path", "passthrough")
    FUNCTION = "broadcast"
    CATEGORY = "Gemini/FileBrowser"

    def broadcast(self, subfolder, prefix, passthrough=None):
        base_output = folder_paths.get_output_directory()
        target_dir = os.path.join(base_output, subfolder) if subfolder else base_output
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        prefix_path = f"{subfolder}/{prefix}" if subfolder else prefix

        try:
            PromptServer.instance.send_sync("filebrowser-folder-changed", {
                "subfolder": subfolder,
                "prefix": prefix,
                "full_path": target_dir
            })
        except Exception as e:
            pass

        return (prefix_path, target_dir, passthrough)


class WirelessGetterNode:
    """
    Wireless getter node that dynamically resolves shortcut keys/collections set by the FileBrowser.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shortcut_key": ("STRING", {"default": "active_selection"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path_or_value", "shortcut_key")
    FUNCTION = "get_value"
    CATEGORY = "Gemini/FileBrowser"

    def get_value(self, shortcut_key):
        val = SHORTCUT_STORE.get(shortcut_key, "")
        return (val, shortcut_key)


# API Endpoints
try:
    if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
        @PromptServer.instance.routes.get("/gemini/filebrowser/list")
        async def api_list_files(request):
            folder = request.query.get("folder", "")
            base_dir = folder_paths.get_output_directory()
            target_path = os.path.normpath(os.path.join(base_dir, folder)) if folder else base_dir

            if not target_path.startswith(os.path.normpath(base_dir)):
                return web.json_response({"error": "Invalid folder path"}, status=400)

            files = []
            if os.path.exists(target_path):
                for root, _, filenames in os.walk(target_path):
                    for f in filenames:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm')):
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, base_dir)
                            subf = os.path.dirname(rel_p).replace("\\", "/")
                            files.append({
                                "filename": f,
                                "subfolder": subf,
                                "url": f"/view?filename={f}&subfolder={subf}&type=output",
                                "relative_path": rel_p.replace("\\", "/")
                            })

            # Sort newest first
            files.sort(key=lambda x: x["filename"], reverse=True)

            return web.json_response({"files": files, "shortcuts": SHORTCUT_STORE})


        @PromptServer.instance.routes.post("/gemini/filebrowser/shortcut")
        async def api_set_shortcut(request):
            try:
                data = await request.json()
                key = data.get("key")
                value = data.get("value")
                if key:
                    SHORTCUT_STORE[key] = value
                    try:
                        PromptServer.instance.send_sync("filebrowser-shortcuts-updated", SHORTCUT_STORE)
                    except Exception:
                        pass
                    return web.json_response({"status": "ok", "shortcuts": SHORTCUT_STORE})
                return web.json_response({"error": "Missing key"}, status=400)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)
except Exception:
    pass


NODE_CLASS_MAPPINGS = {
    "VisualFileBrowserNode": VisualFileBrowserNode,
    "OutputFolderBroadcasterNode": OutputFolderBroadcasterNode,
    "WirelessGetterNode": WirelessGetterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualFileBrowserNode": "Visual FileBrowser & Asset Manager",
    "OutputFolderBroadcasterNode": "Output Folder Broadcaster",
    "WirelessGetterNode": "Wireless Collection Getter",
}
