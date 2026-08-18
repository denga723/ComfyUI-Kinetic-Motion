import { app } from "/scripts/app.js";

const DEFAULT_CONFIG = {
    model_name: "gemini-omni-flash-preview",
    task: "video_editing",
    duration: 8,
    aspect_ratio: "1:1",
    delivery: "base64",
    prefix_text: "",
    suffix_text: "",
    image_roles: {
        "Image1": "reference",
        "Image2": "reference",
        "Image3": "reference",
        "Image4": "reference"
    }
};

function isNodeOutputtingVideo(node, outputSlot, visited = new Set()) {
    if (!node || visited.has(node.id)) return false;
    visited.add(node.id);

    if (node.geminiJobHtmls && Array.isArray(node.geminiJobHtmls)) {
        for (const html of node.geminiJobHtmls) {
            if (typeof html === "string" && html.includes("<video")) return true;
        }
    }

    if (node.outputs && node.outputs[outputSlot]) {
        const outType = String(node.outputs[outputSlot].type || "").toUpperCase();
        if (outType === "VIDEO") return true;
    }

    const nodeType = String(node.type || node.comfyClass || "").toLowerCase();
    if (nodeType.includes("video") || nodeType.includes("vhs")) return true;

    if (nodeType.includes("geminijobbatcher") || nodeType.includes("jobbatcher")) {
        const cfgWidget = node.widgets?.find(w => w.name === "stream_config");
        if (cfgWidget && cfgWidget.value) {
            try {
                const cfg = JSON.parse(cfgWidget.value);
                if (Array.isArray(cfg)) {
                    for (const item of cfg) {
                        if (item.mode === "video") return true;
                    }
                }
            } catch (e) {}
        }

        if (node.inputs) {
            for (const input of node.inputs) {
                if (input.link != null) {
                    const inLink = app.graph?.links?.[input.link];
                    if (inLink) {
                        const parentNode = app.graph?.getNodeById?.(inLink.origin_id);
                        if (isNodeOutputtingVideo(parentNode, inLink.origin_slot, visited)) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    if (node.widgets) {
        for (const w of node.widgets) {
            const val = String(w.value || "").toLowerCase();
            if (val.endsWith(".mp4") || val.endsWith(".mov") || val.endsWith(".webm")) {
                return true;
            }
        }
    }

    return false;
}

function checkIfVideoConnected(node) {
    if (!node.inputs || !node.inputs.length) return false;
    
    const streamInput = node.inputs.find(i => i.name === "stream" || i.type === "GEMINI_STREAM" || i.type === "*");
    if (!streamInput || streamInput.link == null) return false;
    
    const link = app.graph?.links?.[streamInput.link];
    if (!link) return false;
    
    const originNode = app.graph?.getNodeById?.(link.origin_id);
    if (!originNode) return false;
    
    return isNodeOutputtingVideo(originNode, link.origin_slot);
}

app.registerExtension({
    name: "GeminiEnterprise.OmniUX",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiOmniModel") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                const self = this;

                // Hidden config widget for serialization
                let configW = this.widgets?.find(w => w.name === "omni_config");
                if (!configW) {
                    configW = this.addWidget("string", "omni_config", JSON.stringify(DEFAULT_CONFIG));
                }
                configW.type = "hidden";
                configW.computeSize = () => [0,0];

                const parseConfig = () => {
                    try {
                        let parsed = JSON.parse(configW.value);
                        return Object.assign({}, DEFAULT_CONFIG, parsed);
                    } catch(e) {
                        return Object.assign({}, DEFAULT_CONFIG);
                    }
                };

                const saveConfig = (cfg) => {
                    configW.value = JSON.stringify(cfg);
                    app.graph?.setDirtyCanvas(true, false);
                };

                let config = parseConfig();

                // Build inline DOM widget
                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%", height: "100%", boxSizing: "border-box",
                    backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "4px",
                    padding: "4px", display: "flex", flexDirection: "column", gap: "4px",
                    color: "#e0e0e0", fontFamily: "sans-serif", fontSize: "10px",
                    overflow: "hidden"
                });
                
                const topRow = document.createElement("div");
                Object.assign(topRow.style, { display: "flex", alignItems: "center", gap: "6px", width: "100%" });
                
                const bottomRow = document.createElement("div");
                Object.assign(bottomRow.style, { display: "flex", alignItems: "center", gap: "6px", width: "100%" });

                const createField = (label, element, flex = "0 1 auto") => {
                    const row = document.createElement("div");
                    Object.assign(row.style, { display: "flex", alignItems: "center", gap: "2px", flex: flex });
                    const lbl = document.createElement("span");
                    lbl.innerText = label;
                    lbl.style.whiteSpace = "nowrap";
                    lbl.style.color = "#aaa";
                    row.appendChild(lbl);
                    Object.assign(element.style, {
                        backgroundColor: "#111", color: "#eee", flex: 1,
                        border: "1px solid #444", borderRadius: "2px", padding: "1px 2px",
                        fontSize: "10px", boxSizing: "border-box", minWidth: "0"
                    });
                    row.appendChild(element);
                    return { row, lbl };
                };

                const modelSel = document.createElement("select");
                modelSel.innerHTML = `
                    <option value="gemini-omni-flash-preview">gemini-omni-flash-preview (Omni Video)</option>
                    <option value="veo-2.0-generate-001">veo-2.0-generate-001 (Google Veo 2)</option>
                `;
                modelSel.value = config.model_name || "gemini-omni-flash-preview";
                modelSel.onchange = (e) => { config.model_name = e.target.value; saveConfig(config); };
                topRow.appendChild(createField("model:", modelSel, "1 1 auto").row);

                const taskSel = document.createElement("select");
                taskSel.innerHTML = `
                    <option value="video_editing">video_editing</option>
                    <option value="image_to_video">image_to_video</option>
                    <option value="text_to_video">text_to_video</option>
                `;
                taskSel.value = config.task || "video_editing";
                taskSel.onchange = (e) => { config.task = e.target.value; saveConfig(config); };
                topRow.appendChild(createField("task:", taskSel, "1 1 auto").row);

                const durInp = document.createElement("input");
                durInp.type = "number";
                durInp.min = "3"; durInp.max = "10"; durInp.step = "1";
                durInp.value = config.duration || 8;
                durInp.onchange = (e) => { config.duration = parseInt(e.target.value) || 8; saveConfig(config); };
                durInp.style.width = "25px";
                const durField = createField("dur(s):", durInp, "0 0 auto");
                topRow.appendChild(durField.row);

                const arSel = document.createElement("select");
                arSel.innerHTML = `
                    <option value="1:1">1:1</option>
                    <option value="16:9">16:9</option>
                    <option value="9:16">9:16</option>
                `;
                arSel.value = config.aspect_ratio || "1:1";
                arSel.onchange = (e) => { config.aspect_ratio = e.target.value; saveConfig(config); };
                arSel.style.width = "40px";
                const arField = createField("AR:", arSel, "0 0 auto");
                topRow.appendChild(arField.row);

                const delSel = document.createElement("select");
                delSel.innerHTML = `
                    <option value="base64">base64</option>
                    <option value="uri">uri</option>
                `;
                delSel.value = config.delivery || "base64";
                delSel.onchange = (e) => { config.delivery = e.target.value; saveConfig(config); };
                delSel.style.width = "50px";
                topRow.appendChild(createField("del:", delSel, "0 0 auto").row);

                const preInp = document.createElement("input");
                preInp.type = "text";
                preInp.value = config.prefix_text || "";
                preInp.oninput = (e) => { config.prefix_text = e.target.value; saveConfig(config); };
                bottomRow.appendChild(createField("prefix:", preInp, "1 1 50%").row);

                const sufInp = document.createElement("input");
                sufInp.type = "text";
                sufInp.value = config.suffix_text || "";
                sufInp.oninput = (e) => { config.suffix_text = e.target.value; saveConfig(config); };
                bottomRow.appendChild(createField("suffix:", sufInp, "1 1 50%").row);
                
                container.appendChild(topRow);
                container.appendChild(bottomRow);

                const syncUIFromConfig = () => {
                    config = parseConfig();
                    if (!config.task) config.task = "video_editing";
                    modelSel.value = config.model_name || "gemini-omni-flash-preview";
                    taskSel.value = config.task || "video_editing";
                    durInp.value = config.duration != null ? config.duration : 8;
                    arSel.value = config.aspect_ratio || "1:1";
                    delSel.value = config.delivery || "base64";
                    preInp.value = config.prefix_text || "";
                    sufInp.value = config.suffix_text || "";
                };

                try {
                    const uiWidget = this.addDOMWidget("omni_params_ui", "HTML", container, { serialize: false, hideOnZoom: false });
                    uiWidget.computeSize = function() {
                        return [450, 48];
                    };
                    
                    let previewIdx = this.widgets.findIndex(w => w.name === "execution_preview");
                    if (previewIdx !== -1) {
                        this.widgets.splice(this.widgets.indexOf(uiWidget), 1);
                        this.widgets.splice(previewIdx, 0, uiWidget);
                    }
                } catch(e) {}

                // Hook configure to synchronize UI elements when loading saved workflow
                const origOnConfigure = this.onConfigure;
                this.onConfigure = function() {
                    if (origOnConfigure) origOnConfigure.apply(this, arguments);
                    syncUIFromConfig();
                };

                // Keep controls active and synchronizable
                durInp.disabled = false;
                durInp.style.opacity = "1";
                durInp.style.backgroundColor = "#111";
                durField.lbl.style.color = "#aaa";

                arSel.disabled = false;
                arSel.style.opacity = "1";
                arSel.style.backgroundColor = "#111";
                arField.lbl.style.color = "#aaa";
            };
        }
    }
});
