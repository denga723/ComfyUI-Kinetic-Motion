import { app } from "/scripts/app.js";

app.registerExtension({
    name: "comfyui-gemini-enterprise.MultimodalPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiMultimodalPreview") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                // Create or find container
                let container = null;
                for (let w of this.widgets || []) {
                    if (w.name === "gemini_preview") {
                        container = w.element;
                        break;
                    }
                }

                if (!container) {
                    container = document.createElement("div");
                    container.style.cssText = `
                        overflow-y: auto;
                        max-height: 400px;
                        background: #222;
                        color: #ddd;
                        padding: 10px;
                        font-family: sans-serif;
                        font-size: 14px;
                        border-radius: 4px;
                        white-space: pre-wrap;
                        box-sizing: border-box;
                        width: 100%;
                    `;

                    // Add widget
                    this.addDOMWidget("gemini_preview", "HTML", container, {
                        serialize: false,
                        hideOnZoom: false,
                    });
                }

                // Parse the response message
                if (message && message.gemini_response && message.gemini_response.length > 0) {
                    container.innerHTML = ""; // Clear existing

                    message.gemini_response.forEach((resp, index) => {
                        const jobDiv = document.createElement("div");
                        jobDiv.style.marginBottom = "15px";
                        if (message.gemini_response.length > 1) {
                            jobDiv.style.borderBottom = "1px solid #444";
                            jobDiv.style.paddingBottom = "10px";
                            const header = document.createElement("div");
                            header.style.color = "#888";
                            header.style.fontSize = "12px";
                            header.style.marginBottom = "5px";
                            header.textContent = `--- Job ${index + 1} ---`;
                            jobDiv.appendChild(header);
                        }

                        if (resp.error) {
                            const errNode = document.createElement("div");
                            errNode.style.color = "red";
                            errNode.textContent = `Error: ${resp.error}`;
                            jobDiv.appendChild(errNode);
                        } else {
                            try {
                                const parts = resp.candidates[0].content.parts;
                                for (let part of parts) {
                                    if (part.text) {
                                        const textNode = document.createElement("div");
                                        textNode.textContent = part.text;
                                        textNode.style.marginBottom = "10px";
                                        jobDiv.appendChild(textNode);
                                    }
                                    if (part.inlineData) {
                                        const imgNode = document.createElement("img");
                                        imgNode.src = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                                        imgNode.style.maxWidth = "100%";
                                        imgNode.style.borderRadius = "4px";
                                        imgNode.style.marginTop = "5px";
                                        jobDiv.appendChild(imgNode);
                                    }
                                }
                            } catch (e) {
                                const errNode = document.createElement("div");
                                errNode.style.color = "orange";
                                errNode.textContent = `Could not parse response format.`;
                                jobDiv.appendChild(errNode);
                            }
                        }
                        container.appendChild(jobDiv);
                    });
                }
                
                this.onResize?.(this.size);
            };
        }
    }
});
