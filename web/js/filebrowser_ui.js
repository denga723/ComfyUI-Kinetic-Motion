import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "Gemini.VisualFileBrowser",

    async nodeCreated(node) {
        if (node.comfyClass === "VisualFileBrowserNode") {
            node.setSize([380, 420]);

            // Container element
            const container = document.createElement("div");
            container.style.cssText = `
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                background: #18181c;
                border-radius: 8px;
                padding: 8px;
                box-sizing: border-box;
                font-family: system-ui, -apple-system, sans-serif;
                color: #e0e0e0;
            `;

            // Header Controls
            const header = document.createElement("div");
            header.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 8px;
            `;

            const folderInput = document.createElement("input");
            folderInput.type = "text";
            folderInput.placeholder = "Subfolder (blank for root)...";
            folderInput.style.cssText = `
                flex: 1;
                background: #282830;
                border: 1px solid #3f3f4a;
                color: #fff;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
            `;

            const refreshBtn = document.createElement("button");
            refreshBtn.innerText = "🔄";
            refreshBtn.title = "Refresh Gallery";
            refreshBtn.style.cssText = `
                background: #3b82f6;
                border: none;
                color: white;
                padding: 4px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            `;

            header.appendChild(folderInput);
            header.appendChild(refreshBtn);
            container.appendChild(header);

            // Active Shortcut Bar
            const shortcutBar = document.createElement("div");
            shortcutBar.style.cssText = `
                font-size: 11px;
                color: #a0a0b0;
                margin-bottom: 6px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            `;
            shortcutBar.innerHTML = `Active Shortcut (<b>active_selection</b>): <span id="shortcut_val" style="color: #60a5fa;">None</span>`;
            container.appendChild(shortcutBar);

            // Grid Container
            const grid = document.createElement("div");
            grid.style.cssText = `
                flex: 1;
                overflow-y: auto;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
                gap: 8px;
                padding-right: 4px;
            `;
            container.appendChild(grid);

            // Fetch and render images
            async function loadGallery() {
                const folder = folderInput.value.trim();
                try {
                    const res = await fetch(`/gemini/filebrowser/list?folder=${encodeURIComponent(folder)}`);
                    const data = await res.json();
                    
                    grid.innerHTML = "";
                    if (!data.files || data.files.length === 0) {
                        grid.innerHTML = `<div style="grid-column: 1/-1; color: #666; font-size: 12px; text-align: center; margin-top: 20px;">No images found</div>`;
                        return;
                    }

                    const activeVal = data.shortcuts?.active_selection || "";
                    const shortcutValSpan = container.querySelector("#shortcut_val");
                    if (shortcutValSpan) shortcutValSpan.innerText = activeVal ? activeVal.split('/').pop() : "None";

                    data.files.forEach(file => {
                        const card = document.createElement("div");
                        card.style.cssText = `
                            position: relative;
                            aspect-ratio: 1;
                            border-radius: 6px;
                            overflow: hidden;
                            border: 2px solid ${activeVal === file.relative_path ? "#3b82f6" : "transparent"};
                            cursor: pointer;
                            background: #111;
                        `;

                        const img = document.createElement("img");
                        img.src = file.url;
                        img.style.cssText = "width: 100%; height: 100%; object-fit: cover;";
                        card.appendChild(img);

                        // Click to set as active wireless shortcut
                        card.onclick = async () => {
                            await api.fetchApi("/gemini/filebrowser/shortcut", {
                                method: "POST",
                                body: JSON.stringify({
                                    key: "active_selection",
                                    value: file.relative_path
                                })
                            });
                            loadGallery();
                        };

                        grid.appendChild(card);
                    });
                } catch (e) {
                    console.error("[VisualFileBrowser] Error loading gallery:", e);
                }
            }

            refreshBtn.onclick = () => loadGallery();
            folderInput.onchange = () => loadGallery();

            // Add custom LiteGraph widget
            const widget = node.addCustomWidget({
                name: "filebrowser_ui",
                type: "DOM",
                element: container,
                draw(ctx, node, widget_width, y, widget_height) {
                    // Handled automatically by DOM widget in ComfyUI
                }
            });

            // WebSocket event listeners
            api.addEventListener("filebrowser-refresh", () => loadGallery());
            api.addEventListener("filebrowser-folder-changed", (e) => {
                if (e.detail?.subfolder !== undefined) {
                    folderInput.value = e.detail.subfolder;
                }
                loadGallery();
            });
            api.addEventListener("filebrowser-shortcuts-updated", (e) => {
                const shortcutValSpan = container.querySelector("#shortcut_val");
                if (shortcutValSpan && e.detail?.active_selection) {
                    shortcutValSpan.innerText = e.detail.active_selection.split('/').pop();
                }
            });

            // Initial load
            setTimeout(() => loadGallery(), 200);
        }
    }
});
