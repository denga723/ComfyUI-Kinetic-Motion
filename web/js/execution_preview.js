import { app } from "/scripts/app.js";

app.registerExtension({
    name: "GeminiEnterprise.ExecutionPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiProModel" || nodeData.name === "GeminiOmniModel") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            const origComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function() {
                let size = origComputeSize ? origComputeSize.apply(this, arguments) : [400, 200];
                size[0] = Math.max(size[0], 500); // ensure it's wide enough for preview
                return size;
            };

            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                const self = this;
                
                self.currentPreviewIndex = 0;
                
                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%",
                    display: "flex",
                    flexDirection: "column",
                    gap: "4px",
                    padding: "4px",
                    boxSizing: "border-box",
                    backgroundColor: "#1a1a1a",
                    border: "1px solid #333",
                    borderRadius: "4px",
                    marginTop: "8px",
                    fontFamily: "sans-serif",
                    color: "#e0e0e0",
                    fontSize: "12px"
                });
                
                // Header with pagination controls
                const header = document.createElement("div");
                Object.assign(header.style, {
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    borderBottom: "1px solid #444",
                    paddingBottom: "4px"
                });
                
                const title = document.createElement("strong");
                title.style.color = "#ffffff";
                title.innerText = "Incoming Job Preview";
                
                const controls = document.createElement("div");
                Object.assign(controls.style, {
                    display: "flex",
                    gap: "8px",
                    alignItems: "center"
                });
                
                const btnStyle = `
                    background: #333; border: 1px solid #555; color: #fff; 
                    border-radius: 3px; cursor: pointer; padding: 2px 8px; font-size: 11px;
                `;
                
                const prevBtn = document.createElement("button");
                prevBtn.innerText = "◄ Prev";
                prevBtn.style.cssText = btnStyle;
                
                const pageLabel = document.createElement("span");
                pageLabel.innerText = "0 / 0";
                
                const nextBtn = document.createElement("button");
                nextBtn.innerText = "Next ►";
                nextBtn.style.cssText = btnStyle;
                
                controls.appendChild(prevBtn);
                controls.appendChild(pageLabel);
                controls.appendChild(nextBtn);
                
                header.appendChild(title);
                header.appendChild(controls);
                container.appendChild(header);
                
                // Inputs panel area for explicit reference roles
                const inputsPanel = document.createElement("div");
                Object.assign(inputsPanel.style, {
                    padding: "4px 6px",
                    backgroundColor: "#161616",
                    border: "1px solid #333",
                    borderRadius: "3px",
                    marginTop: "4px",
                    display: "none"
                });
                container.appendChild(inputsPanel);

                // Content area
                const contentArea = document.createElement("div");
                Object.assign(contentArea.style, {
                    padding: "6px",
                    backgroundColor: "#222",
                    borderRadius: "3px",
                    minHeight: "60px",
                    maxHeight: "150px",
                    overflowY: "auto",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    marginTop: "4px"
                });
                container.appendChild(contentArea);
                
                try {
                    const previewWidget = this.addDOMWidget("execution_preview", "HTML", container, { serialize: false, hideOnZoom: false });
                    previewWidget.computeSize = function() {
                        return [500, 180];
                    };
                } catch (e) {}

                let upstreamHtmls = [];
                let currentConfigStr = "";
                
                const highlightPromptTags = (text) => {
                    if (!text) return "";
                    let escaped = String(text)
                        .replace(/&/g, "&amp;")
                        .replace(/</g, "&lt;")
                        .replace(/>/g, "&gt;");
                        
                    const imgTagStyle = "display: inline-block; background: #2e64de; color: #ffffff; font-weight: bold; font-size: 9px; padding: 1px 5px; border-radius: 3px; margin: 0 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); vertical-align: middle;";
                    const vidTagStyle = "display: inline-block; background: #8e24aa; color: #ffffff; font-weight: bold; font-size: 9px; padding: 1px 5px; border-radius: 3px; margin: 0 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); vertical-align: middle;";
                    const srcTagStyle = "display: inline-block; background: #2e7d32; color: #ffffff; font-weight: bold; font-size: 9px; padding: 1px 5px; border-radius: 3px; margin: 0 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); vertical-align: middle;";
                    
                    return escaped
                        .replace(/\b(Video\d+|<VIDEO_REF_\d+>)\b/gi, (m) => `<span style="${vidTagStyle}">${m}</span>`)
                        .replace(/\b(Image\d+|<IMAGE_REF_\d+>)\b/gi, (m) => `<span style="${imgTagStyle}">${m}</span>`)
                        .replace(/(<FIRST_FRAME>)/gi, (m) => `<span style="${srcTagStyle}">${m}</span>`);
                };

                const updateUI = () => {
                    if (upstreamHtmls.length === 0) {
                        inputsPanel.style.display = "none";
                        contentArea.innerHTML = "<div style='color:#666; font-style:italic;'>[No incoming stream]</div>";
                        pageLabel.innerText = "0 / 0";
                        prevBtn.disabled = true;
                        nextBtn.disabled = true;
                        prevBtn.style.opacity = "0.5";
                        nextBtn.style.opacity = "0.5";
                        return;
                    }
                    
                    if (self.currentPreviewIndex >= upstreamHtmls.length) {
                        self.currentPreviewIndex = upstreamHtmls.length - 1;
                    }
                    if (self.currentPreviewIndex < 0) {
                        self.currentPreviewIndex = 0;
                    }
                    
                    pageLabel.innerText = `${self.currentPreviewIndex + 1} / ${upstreamHtmls.length}`;
                    
                    let html = upstreamHtmls[self.currentPreviewIndex] || "<div style='color:red;'>Error parsing job</div>";
                    
                    if (self.type === "GeminiOmniModel" || self.comfyClass === "GeminiOmniModel") {
                        let config = {};
                        try {
                            if (currentConfigStr) config = JSON.parse(currentConfigStr);
                        } catch (e) {}
                        if (!config || typeof config !== "object") config = {};
                        if (!config.image_roles || typeof config.image_roles !== "object") config.image_roles = {};
                        
                        let pre = config.prefix_text || "";
                        let suf = config.suffix_text || "";
                        let task = config.task || "image_to_video";
                        
                        let ar = config.aspect_ratio || "16:9";
                        let durStr = config.duration ? `[0-${config.duration}s] ` : "";
                        let combinedPre = durStr + (pre ? pre + " " : "");
                        let combinedSuf = suf ? " " + suf : "";

                        let tempDiv = document.createElement("div");
                        tempDiv.innerHTML = html;

                        let imgEls = tempDiv.querySelectorAll("img");
                        let vidEls = tempDiv.querySelectorAll("video");
                        let imageCount = imgEls.length;
                        let videoCount = vidEls.length;

                        if (imageCount === 0 && videoCount === 0) {
                            const streamInput = self.inputs?.find(i => i.name === "stream");
                            if (streamInput && streamInput.link) {
                                const link = app.graph.links[streamInput.link];
                                if (link) {
                                    const originNode = app.graph.getNodeById(link.origin_id);
                                    if (originNode) {
                                        if (originNode.type === "GeminiJobBatcher" || originNode.comfyClass === "GeminiJobBatcher") {
                                            let batcherConfig = [];
                                            try {
                                                let configW = originNode.widgets?.find(w => w.name === "stream_config");
                                                if (configW && configW.value) batcherConfig = JSON.parse(configW.value);
                                            } catch (e) {}
                                            for (let c of batcherConfig) {
                                                if (c.muted) continue;
                                                if (c.mode === "video") {
                                                    videoCount += 1;
                                                } else if (c.type === "IMAGE") {
                                                    let imgsPerJob = parseInt(c.imgs_per_job) || 1;
                                                    imageCount += imgsPerJob;
                                                }
                                            }
                                        } else {
                                            imageCount = 1;
                                        }
                                    }
                                }
                            }
                        }

                        const isTextToVideo = (task === "text_to_video");

                        if (imageCount > 0 || videoCount > 0) {
                            inputsPanel.style.display = "block";
                            inputsPanel.innerHTML = "";

                            let titleEl = document.createElement("div");
                            titleEl.style.cssText = "font-size: 10px; font-weight: bold; color: #888; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center;";
                            let parts = [];
                            if (imageCount > 0) parts.push(`${imageCount} image(s)`);
                            if (videoCount > 0) parts.push(`${videoCount} video motion ref(s)`);
                            let titleNotice = isTextToVideo ? "<span style='font-size: 9px; color: #e57373; font-style: italic;'>[Inputs disabled in T2V mode]</span>" : `<span style="font-size: 9px; font-weight: normal; color: #aaa;">${parts.join(", ")}</span>`;
                            titleEl.innerHTML = `<span>INPUTS PANEL</span>${titleNotice}`;
                            inputsPanel.appendChild(titleEl);

                            if (isTextToVideo) {
                                inputsPanel.style.opacity = "0.45";
                                inputsPanel.style.pointerEvents = "none";
                            } else {
                                inputsPanel.style.opacity = "1.0";
                                inputsPanel.style.pointerEvents = "auto";
                            }

                            // 1. Render Video Motion References
                            for (let v = 0; v < videoCount; v++) {
                                let vidId = "Video" + (v + 1);
                                let vRow = document.createElement("div");
                                vRow.style.cssText = "display:flex; align-items:center; justify-content:space-between; padding:2px 4px; background:#1e142b; border:1px solid #4a148c; border-radius:3px; margin-bottom:2px; font-size:10px;";

                                let vLabel = document.createElement("span");
                                vLabel.style.cssText = "background:#8e24aa; color:#fff; font-weight:bold; padding:1px 5px; border-radius:3px; font-size:9px;";
                                vLabel.innerText = vidId;

                                let vBadge = document.createElement("span");
                                vBadge.style.cssText = "background:#4a148c; color:#ce93d8; font-weight:bold; border:1px solid #7b1fa2; border-radius:3px; padding:1px 6px; font-size:9px;";
                                vBadge.innerText = "motion-reference";

                                vRow.appendChild(vLabel);
                                vRow.appendChild(vBadge);
                                inputsPanel.appendChild(vRow);
                            }

                            // 2. Render Image Inputs
                            let hasFirstFrame = false;
                            for (let i = 0; i < imageCount; i++) {
                                let imgId = "Image" + (i + 1);
                                let currentRole = config.image_roles[imgId] || (i === 0 ? "first-frame" : "reference");
                                if (currentRole === "first-frame") {
                                    if (hasFirstFrame) {
                                        config.image_roles[imgId] = "reference";
                                    } else {
                                        hasFirstFrame = true;
                                        config.image_roles[imgId] = "first-frame";
                                    }
                                } else {
                                    config.image_roles[imgId] = "reference";
                                }
                            }

                            for (let i = 0; i < imageCount; i++) {
                                let imgId = "Image" + (i + 1);
                                let role = config.image_roles[imgId] || "reference";

                                let row = document.createElement("div");
                                row.style.cssText = "display:flex; align-items:center; justify-content:space-between; padding:2px 4px; background:#222; border:1px solid #333; border-radius:3px; margin-bottom:2px; font-size:10px;";

                                let label = document.createElement("span");
                                label.style.cssText = "background:#2e64de; color:#fff; font-weight:bold; padding:1px 5px; border-radius:3px; font-size:9px;";
                                label.innerText = imgId;

                                let btnGroup = document.createElement("div");
                                btnGroup.style.cssText = "display:flex; gap:3px;";

                                let ffBtn = document.createElement("button");
                                ffBtn.innerText = "first-frame";
                                let ffActive = (role === "first-frame");
                                ffBtn.style.cssText = ffActive
                                    ? "background: #4caf50; color: #fff; font-weight: bold; border: 1px solid #66bb6a; border-radius: 3px; padding: 1px 5px; font-size: 9px; cursor: pointer;"
                                    : "background: #252525; color: #888; border: 1px solid #444; border-radius: 3px; padding: 1px 5px; font-size: 9px; cursor: pointer;";
                                
                                ffBtn.onclick = (e) => {
                                    e.preventDefault();
                                    for (let k = 1; k <= imageCount; k++) {
                                        config.image_roles["Image" + k] = (k === (i + 1)) ? "first-frame" : "reference";
                                    }
                                    const configWidget = self.widgets?.find(w => w.name === "omni_config");
                                    if (configWidget) {
                                        configWidget.value = JSON.stringify(config);
                                        currentConfigStr = configWidget.value;
                                    }
                                    updateUI();
                                };

                                let refBtn = document.createElement("button");
                                refBtn.innerText = "reference";
                                let refActive = (role === "reference");
                                refBtn.style.cssText = refActive
                                    ? "background: #2196f3; color: #fff; font-weight: bold; border: 1px solid #42a5f5; border-radius: 3px; padding: 1px 5px; font-size: 9px; cursor: pointer;"
                                    : "background: #252525; color: #888; border: 1px solid #444; border-radius: 3px; padding: 1px 5px; font-size: 9px; cursor: pointer;";
                                
                                refBtn.onclick = (e) => {
                                    e.preventDefault();
                                    config.image_roles[imgId] = "reference";
                                    const configWidget = self.widgets?.find(w => w.name === "omni_config");
                                    if (configWidget) {
                                        configWidget.value = JSON.stringify(config);
                                        currentConfigStr = configWidget.value;
                                    }
                                    updateUI();
                                };

                                btnGroup.appendChild(ffBtn);
                                btnGroup.appendChild(refBtn);
                                row.appendChild(label);
                                row.appendChild(btnGroup);
                                inputsPanel.appendChild(row);
                            }
                        } else {
                            inputsPanel.style.display = "none";
                        }

                        // Add number badges to preview thumbnails (positioned in top-right corner outside/over thumbnail)
                        imgEls.forEach((img, idx) => {
                            let num = idx + 1;
                            let wrapper = document.createElement("span");
                            wrapper.style.cssText = "position:relative; display:inline-flex; margin: 2px 6px; vertical-align:middle;";
                            img.parentNode.insertBefore(wrapper, img);
                            wrapper.appendChild(img);
                            
                            let badge = document.createElement("span");
                            badge.innerText = num;
                            badge.style.cssText = "position:absolute; top:-4px; right:-6px; background:#2e64de; color:#ffffff; font-size:9px; font-weight:bold; width:14px; height:14px; line-height:14px; text-align:center; border-radius:50%; border:1px solid #ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.5); pointer-events:none; z-index:2;";
                            wrapper.appendChild(badge);
                        });

                        // Update text spans with prefix/suffix and prompt tag highlighting
                        let textSpans = tempDiv.querySelectorAll("span[title]");
                        if (textSpans.length > 0) {
                            let targetSpan = textSpans[textSpans.length - 1];
                            let originalText = targetSpan.innerText;
                            let newText = combinedPre + originalText + combinedSuf;
                            targetSpan.innerHTML = highlightPromptTags(newText);
                            targetSpan.title = newText;
                            html = tempDiv.innerHTML;
                        } else if (combinedPre || combinedSuf) {
                            let newSpan = document.createElement("span");
                            newSpan.style.cssText = "display:inline-block; padding: 2px 4px; margin: 0 4px; border: 1px solid #888; border-radius: 2px; font-size: 9px; color: #ccc; max-width: 300px; background: #222; vertical-align: middle; white-space: pre-wrap; word-break: break-word;";
                            newSpan.innerHTML = highlightPromptTags(combinedPre + combinedSuf.trim());
                            tempDiv.appendChild(newSpan);
                            html = tempDiv.innerHTML;
                        }

                        let tagStyle = "border: 1px solid #4caf50; padding: 1px 4px; border-radius: 2px; color: #4caf50; font-size: 10px;";
                        
                        let wrappedHtml = `
                            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:4px;">
                                ${html}
                                <span style="${tagStyle}">${ar}</span>
                                <span style="${tagStyle}">${task}</span>
                            </div>
                        `;
                        contentArea.innerHTML = wrappedHtml;
                    } else {
                        inputsPanel.style.display = "none";
                        contentArea.innerHTML = html;
                    }
                    
                    prevBtn.disabled = self.currentPreviewIndex === 0;
                    nextBtn.disabled = self.currentPreviewIndex === upstreamHtmls.length - 1;
                    
                    prevBtn.style.opacity = prevBtn.disabled ? "0.5" : "1";
                    nextBtn.style.opacity = nextBtn.disabled ? "0.5" : "1";
                };
                
                prevBtn.onclick = (e) => {
                    e.preventDefault();
                    if (self.currentPreviewIndex > 0) {
                        self.currentPreviewIndex--;
                        updateUI();
                    }
                };
                
                nextBtn.onclick = (e) => {
                    e.preventDefault();
                    if (self.currentPreviewIndex < upstreamHtmls.length - 1) {
                        self.currentPreviewIndex++;
                        updateUI();
                    }
                };
                
                const pollInterval = setInterval(() => {
                    if (!app.graph._nodes.find(n => n.id === self.id)) {
                        clearInterval(pollInterval);
                        return;
                    }
                    
                    let foundHtmls = [];
                    let foundConfigStr = "";
                    
                    const streamInput = self.inputs?.find(inp => inp.name === "stream");
                    if (streamInput && streamInput.link) {
                        const link = app.graph.links[streamInput.link];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode && originNode.geminiJobHtmls) {
                                foundHtmls = originNode.geminiJobHtmls;
                            }
                        }
                    }
                    
                    if (self.type === "GeminiOmniModel" || self.comfyClass === "GeminiOmniModel") {
                        const configW = self.widgets?.find(w => w.name === "omni_config");
                        if (configW) foundConfigStr = configW.value;
                    }
                    
                    // Simple array equality check to see if we need to update
                    let changed = false;
                    if (foundHtmls.length !== upstreamHtmls.length || foundConfigStr !== currentConfigStr) {
                        changed = true;
                    } else {
                        for (let i = 0; i < foundHtmls.length; i++) {
                            if (foundHtmls[i] !== upstreamHtmls[i]) {
                                changed = true;
                                break;
                            }
                        }
                    }
                    
                    if (changed) {
                        upstreamHtmls = foundHtmls;
                        currentConfigStr = foundConfigStr;
                        updateUI();
                        app.graph.setDirtyCanvas(true, false);
                    }
                }, 200);
            };
        }
    }
});
