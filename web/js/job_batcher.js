import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PALETTE = [
    "#C25953", "#CF7E48", "#C9A24D", "#8CCF7B", 
    "#369B97", "#5A8FDC", "#826A8C", "#9E677F"
];

function shuffle(array) {
    let currentIndex = array.length,  randomIndex;
    while (currentIndex != 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }
    return array;
}

const folderCache = {};
const pendingFolders = new Set();

function getFolderFiles(folderPath, onLoaded) {
    if (!folderPath) return [];
    if (folderCache[folderPath]) {
        return folderCache[folderPath];
    }
    if (pendingFolders.has(folderPath)) {
        return [];
    }
    pendingFolders.add(folderPath);
    fetch(`/jr/list_folder?folder=${encodeURIComponent(folderPath)}`)
        .then(r => r.json())
        .then(data => {
            if (data && data.files) {
                folderCache[folderPath] = data.files;
            } else {
                folderCache[folderPath] = [];
            }
            pendingFolders.delete(folderPath);
            if (onLoaded) onLoaded();
        })
        .catch(err => {
            console.error("Error fetching folder files:", err);
            folderCache[folderPath] = [];
            pendingFolders.delete(folderPath);
        });
    return [];
}


function resolveMediaUrl(rawPath, defaultType = "input") {
    if (!rawPath) return { src: "", type: "img", isVideo: false, frameIdx: 0, original: rawPath };
    
    let isVideo = false;
    let type = "img";
    let frameIdx = 0;
    
    let path = String(rawPath).replace(/^["']|["']$/g, '');
    let reqType = defaultType;
    
    let frameMatch = path.match(/\?frame=(\d+)$/);
    if (frameMatch) {
        frameIdx = parseInt(frameMatch[1]);
        path = path.replace(/\?frame=\d+$/, '');
    }
    
    let typeMatch = path.match(/\s*\[(input|temp|output|path)\]\s*$/i);
    if (typeMatch) {
        reqType = typeMatch[1].toLowerCase();
        path = path.replace(/\s*\[(input|temp|output|path)\]\s*$/i, "");
    }
    
    if (path.match(/\.(mp4|mov|webm|mkv|gif)(?:[?&].*)?$/i)) {
        isVideo = true;
        type = "video";
    }
    
    let src = "";
    
    if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("data:") || path.startsWith("blob:")) {
        src = path;
    } else if (path.startsWith("/view") || path.startsWith("view?") || path.startsWith("/vhs/viewvideo") || path.startsWith(api.apiURL(""))) {
        src = path;
        if (src.startsWith("view?")) src = "/" + src;
        if (src.includes("type=path") || src.includes("/vhs/viewvideo")) {
            isVideo = true; type = "video";
        }
    } else {
        let isAbsolute = path.includes(":\\") || path.includes(":/") || path.startsWith("/") || path.startsWith("\\");
        let name = path;
        let subfolder = "";
        
        if (isAbsolute) {
            reqType = "path";
        } else {
            if (path.includes("/") || path.includes("\\")) {
                const parts = path.split(/[\\/]/);
                name = parts.pop();
                subfolder = parts.join("/");
            }
            if (subfolder.includes("clipspace")) reqType = "temp";
        }
        
        if (reqType === "path") {
            if (isVideo) {
                src = api.apiURL(`/vhs/viewvideo?filename=${encodeURIComponent(name)}&type=path`);
            } else {
                src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=path`);
            }
        } else {
            src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=${reqType}&subfolder=${encodeURIComponent(subfolder)}`);
        }
    }
    
    return { src, type, isVideo, frameIdx, original: rawPath };
}

function hasEmptyFilename(url) {
    if (!url) return true;
    try {
        let u = new URL(url, window.location.origin);
        let filename = u.searchParams.get("filename");
        return !filename || filename.trim() === "";
    } catch(e) {
        return false;
    }
}

function traceImageFilenames(node, app, maxDepth = 5, onFolderLoaded = null) {
    if (!node || maxDepth <= 0) return [];
    console.log("[JobBatcher Debug] Tracing node:", node.id, node.type);
    
    let files = [];
    
    // Check if it is a VideoHelperSuite loader node
    if (node.type && node.type.startsWith("VHS_LoadVideo")) {
        console.log("[JobBatcher Debug] Detected VHS Video Loader:", node.type);
        let videoW = node.widgets?.find(widg => widg.name === "video");
        if (videoW && videoW.value) {
            let filename = String(videoW.value).replace(/^["']|["']$/g, '');
            console.log("[JobBatcher Debug] VHS Video widget value:", filename);
            
            // Get total frames in video
            let totalFrames = 0;
            
            // Try to get from video_query
            if (node.video_query && typeof node.video_query.frames === 'number') {
                totalFrames = node.video_query.frames;
            }
            
            // Try to get from nodeOutputs frame_count
            if (!totalFrames && app.nodeOutputs && app.nodeOutputs[node.id]) {
                let output = app.nodeOutputs[node.id];
                if (output.frame_count && Array.isArray(output.frame_count)) {
                    totalFrames = parseInt(output.frame_count[0]);
                } else if (output[1] && Array.isArray(output[1])) {
                    totalFrames = parseInt(output[1][0]);
                }
            }
            console.log("[JobBatcher Debug] Resolved totalFrames:", totalFrames);
            
            // Determine cap, skip, select
            let cap = 0;
            let capW = node.widgets?.find(widg => widg.name === "frame_load_cap");
            if (capW && parseInt(capW.value) > 0) {
                cap = parseInt(capW.value);
            }
            
            let skipFirst = 0;
            let skipW = node.widgets?.find(widg => widg.name === "skip_first_frames");
            if (skipW && parseInt(skipW.value) > 0) {
                skipFirst = parseInt(skipW.value);
            }
            
            let selectNth = 1;
            let selectW = node.widgets?.find(widg => widg.name === "select_every_nth");
            if (selectW && parseInt(selectW.value) > 0) {
                selectNth = parseInt(selectW.value);
            }
            console.log("[JobBatcher Debug] Widgets - cap:", cap, "skipFirst:", skipFirst, "selectNth:", selectNth);
            
            // Calculate actual number of frames to load
            let numFrames = cap;
            if (numFrames === 0) {
                if (totalFrames > 0) {
                    let available = totalFrames - skipFirst;
                    if (available > 0) {
                        numFrames = Math.ceil(available / selectNth);
                    } else {
                        numFrames = 1;
                    }
                } else {
                    numFrames = 16; // default fallback if metadata not loaded yet
                }
            }
            console.log("[JobBatcher Debug] Calculated numFrames:", numFrames);
            
            // Reconstruct the files list with frame queries
            for (let i = 0; i < numFrames; i++) {
                let frameIdx = skipFirst + i * selectNth;
                files.push(filename + "?frame=" + frameIdx);
            }
            
            if (files.length > 0) {
                console.log("[JobBatcher Debug] VHS files generated:", files);
                return [...new Set(files)];
            }
        } else {
            console.log("[JobBatcher Debug] VHS Video widget not found or has no value.");
        }
    }
    
    // Priority 1: Check UI preview images (node.imgs) and VHS videopreview widget
    if (node.imgs && node.imgs.length > 0) {
        for (let img of node.imgs) {
            if (img && img.src && !hasEmptyFilename(img.src)) {
                files.push(img.src);
            }
        }
    }
    let videopreviewW = node.widgets?.find(widg => widg.name === "videopreview");
    if (videopreviewW) {
        console.log("[JobBatcher Debug] Found videopreview widget");
        if (videopreviewW.videoEl && videopreviewW.videoEl.src && !hasEmptyFilename(videopreviewW.videoEl.src)) {
            files.push(videopreviewW.videoEl.src);
        }
        if (videopreviewW.imgEl && videopreviewW.imgEl.src && !hasEmptyFilename(videopreviewW.imgEl.src)) {
            files.push(videopreviewW.imgEl.src);
        }
        if (videopreviewW.value && videopreviewW.value.params && videopreviewW.value.params.filename) {
            let params = videopreviewW.value.params;
            let filename = params.filename || "";
            if (typeof filename === "string") filename = filename.replace(/^["']|["']$/g, '');
            if (filename.trim() !== "") {
                console.log("[JobBatcher Debug] videopreview params:", params);
                let subfolder = params.subfolder || "";
                let type = params.type || "output";
                let format = params.format || "";
                let isVideo = format.split('/')[0] === 'video' || format.split('/')[1] === 'gif' || format === 'folder';
                let src = "";
                if (isVideo) {
                    src = `/vhs/viewvideo?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                } else {
                    src = `/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                }
                files.push(src);
            }
        }
    }
    if (files.length > 0) {
        console.log("[JobBatcher Debug] Priority 1 files found:", files);
        return files;
    }
    
    // Priority 2: Check for outputs from execution (app.nodeOutputs)
    if (app.nodeOutputs && app.nodeOutputs[node.id]) {
        let output = app.nodeOutputs[node.id];
        console.log("[JobBatcher Debug] Found nodeOutputs:", output);
        if (output.images && Array.isArray(output.images)) {
            for (let img of output.images) {
                if (img.filename && img.filename.trim() !== "") {
                    let subfolder = img.subfolder || "";
                    let type = img.type || "output";
                    let src = `/view?filename=${encodeURIComponent(img.filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                    files.push(src);
                }
            }
        }
        if (files.length > 0) {
            console.log("[JobBatcher Debug] Priority 2 files found:", files);
            return files;
        }
    }
    
    // Priority 3: Check widgets
    if (node.type === "JR_LoadImageBatch") {
        let imageDataW = node.widgets?.find(widg => widg.name === "image_data");
        if (imageDataW && imageDataW.value) {
            if (typeof imageDataW.value === 'string') {
                try {
                    let parsed = JSON.parse(imageDataW.value);
                    if (Array.isArray(parsed)) files.push(...parsed);
                } catch(e){}
            } else if (Array.isArray(imageDataW.value)) {
                files.push(...imageDataW.value);
            }
        }
        let filePathsW = node.widgets?.find(widg => widg.name === "file_paths");
        if (filePathsW && filePathsW.value) {
            if (typeof filePathsW.value === 'string') {
                if (filePathsW.value.startsWith("[")) {
                    try {
                        let parsed = JSON.parse(filePathsW.value);
                        if (Array.isArray(parsed)) files.push(...parsed);
                    } catch(e){}
                } else if (filePathsW.value.includes(",")) {
                    files.push(...filePathsW.value.split(",").map(p => p.trim()).filter(Boolean));
                } else if (filePathsW.value.includes("\n")) {
                    files.push(...filePathsW.value.split("\n").map(p => p.trim()).filter(Boolean));
                } else if (filePathsW.value.trim()) {
                    files.push(filePathsW.value.trim());
                }
            } else if (Array.isArray(filePathsW.value)) {
                files.push(...filePathsW.value);
            }
        }
        let folderW = node.widgets?.find(widg => widg.name === "folder_path");
        if (folderW && folderW.value) {
            let folderFiles = getFolderFiles(folderW.value, onFolderLoaded);
            files.push(...folderFiles);
        }
    } else {
        let w = node.widgets?.find(widg => 
            widg.name === "file" || 
            widg.name === "filename" || 
            widg.name === "image" || 
            widg.name === "image_data" || 
            widg.name === "image_dir" || 
            widg.name === "video" ||
            widg.name === "video_file" ||
            widg.name === "value" ||
            widg.name === "string" ||
            widg.name === "text" ||
            widg.name === "path"
        );
        if (w && w.value) {
            if (w.name === "image_data") {
                if (typeof w.value === 'string') {
                    try { 
                        let parsed = JSON.parse(w.value);
                        if (Array.isArray(parsed)) files.push(...parsed);
                    } catch(e){}
                } else if (Array.isArray(w.value)) {
                    files.push(...w.value);
                }
            } else if (w.name === "video") {
                let cap = 16;
                let capW = node.widgets?.find(widg => widg.name === "frame_load_cap");
                if (capW && parseInt(capW.value) > 0) {
                    cap = parseInt(capW.value);
                }
                
                let skipFirst = 0;
                let skipW = node.widgets?.find(widg => widg.name === "skip_first_frames");
                if (skipW && parseInt(skipW.value) > 0) {
                    skipFirst = parseInt(skipW.value);
                }
                
                let selectNth = 1;
                let selectW = node.widgets?.find(widg => widg.name === "select_every_nth");
                if (selectW && parseInt(selectW.value) > 0) {
                    selectNth = parseInt(selectW.value);
                }

                let filename = String(w.value).trim().replace(/^["']|["']$/g, '');
                for (let i = 0; i < cap; i++) {
                    let frameIdx = skipFirst + i * selectNth;
                    files.push(filename + "?frame=" + frameIdx);
                }
            } else {
                files.push(String(w.value).trim());
            }
        }
        
        let folderW = node.widgets?.find(widg => 
            widg.name === "folder_path" || 
            widg.name === "image_dir" || 
            widg.name === "directory" || 
            widg.name === "path"
        );
        if (folderW && folderW.value && typeof folderW.value === "string") {
            let folderFiles = getFolderFiles(folderW.value, onFolderLoaded);
            files.push(...folderFiles);
        }
    }
    
    if (files.length > 0) {
        return files;
    }
    
    // Priority 4: Trace upstream
    if (node.inputs) {
        for (let inp of node.inputs) {
            if ((inp.type === "IMAGE" || inp.type === "VIDEO" || inp.type === "*" || (typeof inp.name === "string" && (inp.name.includes("video") || inp.name.includes("image") || inp.name === "video_or_images" || inp.name === "folder_path" || inp.name === "image_dir" || inp.name === "directory" || inp.name === "path"))) && inp.link) {
                const link = app.graph.links[inp.link];
                if (link) {
                    const prevNode = app.graph.getNodeById(link.origin_id);
                    if (prevNode) {
                        let upstreamFiles = traceImageFilenames(prevNode, app, maxDepth - 1, onFolderLoaded);
                        if (node.type === "RepeatImageBatch") {
                            let amount = 1;
                            let amountW = node.widgets?.find(w => w.name === "amount");
                            if (amountW) amount = parseInt(amountW.value);
                            
                            let amountInp = node.inputs?.find(inp => inp.name === "amount");
                            if (amountInp && amountInp.link) {
                                let link = app.graph.links[amountInp.link];
                                if (link) {
                                    let pNode = app.graph.getNodeById(link.origin_id);
                                    if (pNode && pNode.widgets) {
                                        let valW = pNode.widgets.find(w => w.name === "value" || w.name === "int" || w.name === "amount");
                                        if (valW) amount = parseInt(valW.value);
                                    }
                                }
                            }
                            
                            if (isNaN(amount) || amount < 1) amount = 1;
                            for (let i = 0; i < amount; i++) {
                                files.push(...upstreamFiles);
                            }
                        } else if (node.type === "ImageTileComposite") {
                            // Aggregator node: Combines multiple tiles into 1 image.
                            // Do not pass through all upstream tile paths, as that breaks the job count.
                            // Pushing a single empty string signals 1 dynamic job placeholder.
                            files.push("");
                        } else {
                            files.push(...upstreamFiles);
                        }
                    }
                }
            }
        }
    }
    
    return files;
}

app.registerExtension({
    name: "GeminiEnterprise.JobBatcher",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiJobBatcher") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            const origComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function() {
                let size = origComputeSize ? origComputeSize.apply(this, arguments) : [450, 100];
                size[0] = Math.max(size[0], 450); 
                return size;
            };

            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                const self = this;
                this.availableColors = shuffle([...PALETTE]);
                this.getColor = () => {
                    if (this.availableColors.length === 0) {
                        this.availableColors = shuffle([...PALETTE]);
                    }
                    return this.availableColors.pop();
                };
                
                this.streamColors = {}; 

                const origOnDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    if (origOnDrawForeground) {
                        origOnDrawForeground.apply(this, arguments);
                    }
                    const jobsWidget = this.widgets?.find(w => w.name === "jobs");
                    if (jobsWidget && jobsWidget.value === 0 && jobsWidget.last_y) {
                        ctx.save();
                        ctx.font = "12px sans-serif";
                        ctx.fillStyle = "#888";
                        // The default LiteGraph integer widget draws its value around x=width/2 or similar.
                        // We can just overlay "(guess)" manually at a reasonable offset.
                        ctx.fillText("(guess)", this.size[0] / 2 + 30, jobsWidget.last_y + 14);
                        ctx.restore();
                    }
                };

                let configW = this.widgets?.find(w => w.name === "stream_config");
                if (!configW) {
                    configW = this.addWidget("string", "stream_config", "[]");
                }
                configW.type = "hidden";
                configW.computeSize = () => [0,0];

                const parseConfig = () => {
                    try { 
                        let parsed = JSON.parse(configW.value); 
                        if (!Array.isArray(parsed)) {
                            return [];
                        }
                        return parsed;
                    } catch(e) { 
                        return []; 
                    }
                };
                const saveConfig = (cfg) => {
                    configW.value = JSON.stringify(cfg);
                    self.updateBatchPreview();
                };

                const delimitersContainer = document.createElement("div");
                Object.assign(delimitersContainer.style, {
                    width: "100%",
                    display: "flex",
                    flexDirection: "column",
                    gap: "2px",
                    padding: "2px 4px",
                    boxSizing: "border-box",
                    backgroundColor: "#1a1a1a",
                    border: "1px solid #333",
                    borderRadius: "4px",
                    marginBottom: "4px"
                });

                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%",
                    height: "100%",
                    overflowY: "auto",
                    backgroundColor: "#1a1a1a",
                    color: "#e0e0e0",
                    padding: "2px 4px",
                    boxSizing: "border-box",
                    fontFamily: "monospace",
                    fontSize: "11px",
                    whiteSpace: "pre-wrap",
                    borderRadius: "4px",
                    border: "1px solid #333",
                });
                
                const contentDiv = document.createElement("div");
                container.appendChild(contentDiv);
                
                try {
                    const delimWidget = this.addDOMWidget("delimiters_ui", "HTML", delimitersContainer, { serialize: false, hideOnZoom: false });
                    delimWidget.computeSize = function() {
                        const count = self.inputs ? self.inputs.filter(inp => /^(text|image)_stream_\d+$/.test(inp.name) && inp.link).length : 0;
                        return [450, count > 0 ? (count * 20) + 8 : 0];
                    };

                    const previewWidget = this.addDOMWidget("batch_preview", "HTML", container, { serialize: false, hideOnZoom: false });
                    previewWidget.computeSize = function() {
                        return [450, 100];
                    };
                } catch (e) {}
                
                let draggingRowId = null;

                this.updateDynamicPorts = () => {
                    if (!this.inputs) this.inputs = [];
                    
                    let streamInputs = this.inputs.filter(inp => /^(text|image)_stream_\d+$/.test(inp.name));
                    
                    let connectedIds = streamInputs.filter(inp => inp.link).map(inp => parseInt(inp.name.match(/\d+/)[0]));
                    let maxConnectedId = Math.max(0, ...connectedIds);
                    
                    let nextId = maxConnectedId + 1;
                    for (let inp of streamInputs) {
                        if (!inp.link) {
                            let currentId = parseInt(inp.name.match(/\d+/)[0]);
                            if (connectedIds.includes(currentId)) {
                                inp.name = `text_stream_${nextId}`;
                                nextId++;
                            }
                        }
                    }
                    let maxIndex = 0;
                    
                    // Re-fetch after potential renames
                    streamInputs = this.inputs.filter(inp => /^(text|image)_stream_\d+$/.test(inp.name));
                    
                    let emptyCount = 0;
                    for (let i = 0; i < streamInputs.length; i++) {
                        const inp = streamInputs[i];
                        const idx = parseInt(inp.name.match(/\d+/)[0]);
                        if (idx > maxIndex) maxIndex = idx;
                        if (!inp.link) {
                            emptyCount++;
                            inp.name = `text_stream_${idx}`;
                            inp.color_on = undefined;
                            inp.color_off = undefined;
                            inp.label = "input_stream";
                        } else {
                            if (!this.streamColors[idx]) {
                                this.streamColors[idx] = this.getColor();
                            }
                            inp.color_on = this.streamColors[idx];
                            inp.color_off = this.streamColors[idx];
                            inp.label = inp.name;
                        }
                    }
                    
                    if (emptyCount === 0) {
                        const newIdx = maxIndex + 1;
                        this.streamColors[newIdx] = this.getColor();
                        const inp = this.addInput(`text_stream_${newIdx}`, "*");
                        inp.color_on = this.streamColors[newIdx];
                        inp.color_off = this.streamColors[newIdx];
                        inp.label = "input_stream";
                        
                        this.setSize(this.computeSize());
                    } else if (emptyCount > 1) {
                        for (let i = streamInputs.length - 1; i >= 0; i--) {
                            if (emptyCount <= 1) break;
                            const inp = streamInputs[i];
                            if (!inp.link) {
                                this.removeInput(this.inputs.indexOf(inp));
                                emptyCount--;
                            }
                        }
                        this.setSize(this.computeSize());
                    }

                    let config = parseConfig();
                    let activeIds = this.inputs.filter(inp => /^(text|image|video|input)_stream_\d+$/.test(inp.name) && inp.link).map(inp => parseInt(inp.name.match(/\d+/)[0]));
                    
                    config = config.filter(c => activeIds.includes(c.id));
                    
                    let existingIds = config.map(c => c.id);
                    for (let id of activeIds) {
                        const inp = this.inputs.find(i => (i.name === `text_stream_${id}` || i.name === `image_stream_${id}` || i.name === `video_stream_${id}` || i.name === `input_stream_${id}`));
                        let savedConfig = config.find(x => x.id === id);
                        let streamType = savedConfig ? savedConfig.type : ((inp && (inp.name.startsWith("image_") || inp.name.startsWith("video_"))) ? "IMAGE" : "STRING");
                        let detectedMode = savedConfig?.mode || "images";
                        
                        // Auto-detect based on upstream node
                        if (inp && inp.link) {
                            const link = app.graph.links[inp.link];
                            if (link) {
                                const originNode = app.graph.getNodeById(link.origin_id);
                                if (originNode) {
                                    const nodeTypeStr = (originNode.type || "").toLowerCase();
                                    const isVideoNode = nodeTypeStr.includes("video") || 
                                                        nodeTypeStr.includes("mediapipe") || 
                                                        nodeTypeStr.includes("pose") || 
                                                        nodeTypeStr.includes("skeleton") ||
                                                        nodeTypeStr.includes("kinetic") ||
                                                        nodeTypeStr.includes("motion") ||
                                                        nodeTypeStr.includes("curve") ||
                                                        nodeTypeStr.includes("brush") ||
                                                        nodeTypeStr.includes("renderer") ||
                                                        nodeTypeStr.includes("animatediff");
                                    
                                    if (originNode.outputs) {
                                        const out = originNode.outputs[link.origin_slot];
                                        if (out) {
                                            let outType = (out.type || "").toString().toUpperCase();
                                            if (outType === "IMAGE" || outType.includes("IMAGE") || outType === "VIDEO" || outType.includes("VIDEO") || outType === "*") {
                                                streamType = "IMAGE";
                                                if (outType === "VIDEO" || outType.includes("VIDEO") || isVideoNode) {
                                                    detectedMode = "video";
                                                }
                                            } else if (outType === "STRING" || outType.includes("STRING")) {
                                                streamType = "STRING";
                                            }
                                        }
                                    } else if (isVideoNode) {
                                        streamType = "IMAGE";
                                        detectedMode = "video";
                                    }
                                    
                                    if (originNode.widgets) {
                                        const stringWidgets = originNode.widgets.filter(w => typeof w.value === 'string');
                                        for (let w of stringWidgets) {
                                            let cleanVal = w.value.replace(/^["']|["']$/g, '');
                                            if (cleanVal.match(/\.(mp4|mov|webm)(?:[?&].*)?$/i)) {
                                                streamType = "IMAGE";
                                                detectedMode = "video";
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (inp) {
                            inp.name = streamType === "IMAGE" ? (detectedMode === "video" ? `image_stream_${id}` : `image_stream_${id}`) : `text_stream_${id}`;
                            inp.label = inp.name;
                        }
                        
                        let detectedImgsPerJob = 1;
                        if (streamType === "IMAGE" && detectedMode === "images") {
                            if (inp && inp.link) {
                                const link = app.graph.links[inp.link];
                                if (link) {
                                    const originNode = app.graph.getNodeById(link.origin_id);
                                    if (originNode) {
                                        if (originNode.type === "ImageBatchMulti") {
                                            const countW = originNode.widgets?.find(w => w.name === "inputcount");
                                            if (countW && parseInt(countW.value) > 1) {
                                                detectedImgsPerJob = parseInt(countW.value);
                                            }
                                        }
                                        const upstreamFiles = traceImageFilenames(originNode, app, 5);
                                        if (upstreamFiles.length > 1) {
                                            detectedImgsPerJob = upstreamFiles.length;
                                        }
                                    }
                                }
                            }
                        }

                        if (!existingIds.includes(id)) {
                            config.push({
                                id: id, 
                                type: streamType,
                                prefix: "", 
                                delim: "\\n", 
                                suffix: "", 
                                aspect: "guess",
                                imgs_per_job: detectedImgsPerJob,
                                cycle: "iterate",
                                index: 0,
                                muted: false,
                                mode: streamType === "IMAGE" ? detectedMode : "images",
                                fps: 24
                            });
                        } else {
                            let c = config.find(x => x.id === id);
                            if (c) {
                                c.type = streamType;
                                if (detectedMode === "video") {
                                    c.mode = "video";
                                }
                                if (detectedImgsPerJob > 1 && (!c.imgs_per_job || c.imgs_per_job === 1)) {
                                    c.imgs_per_job = detectedImgsPerJob;
                                }
                            }
                        }
                    }
                    
                    delimitersContainer.innerHTML = "";
                    let activeCount = config.length;
                    
                    // Compute sequential labels (Video1, Image1, Text1) per stream type
                    let videoCounter = 0;
                    let imageCounter = 0;
                    let textCounter = 0;
                    let orderedStreamInputs = [];

                    for (let i = 0; i < config.length; i++) {
                        const c = config[i];
                        const idx = c.id;
                        if (c.type === "IMAGE") {
                            if (c.mode === "video") {
                                videoCounter++;
                                c._displayLabel = `Video${videoCounter}`;
                            } else {
                                imageCounter++;
                                c._displayLabel = `Image${imageCounter}`;
                            }
                        } else {
                            textCounter++;
                            c._displayLabel = `Text${textCounter}`;
                        }
                        const inp = this.inputs.find(inpItem => inpItem.name === `image_stream_${idx}` || inpItem.name === `text_stream_${idx}` || inpItem.name === `video_stream_${idx}`);
                        if (inp) {
                            inp.label = c._displayLabel;
                            orderedStreamInputs.push(inp);
                        }
                    }

                    let emptyStreamInputs = this.inputs.filter(inpItem => /^(text|image|video)_stream_\d+$/.test(inpItem.name) && !inpItem.link);
                    
                    if (emptyStreamInputs.length === 0) {
                        const newIdx = Math.max(0, ...config.map(c => c.id), ...this.inputs.filter(inp => /^(text|image|video)_stream_\d+$/.test(inp.name)).map(inp => parseInt(inp.name.match(/\d+/)[0]) || 0)) + 1;
                        this.streamColors[newIdx] = this.getColor();
                        const newInp = this.addInput(`text_stream_${newIdx}`, "*");
                        newInp.color_on = undefined;
                        newInp.color_off = undefined;
                        newInp.label = "+ Add Stream";
                        emptyStreamInputs.push(newInp);
                    }

                    for (let emptyInp of emptyStreamInputs) {
                        emptyInp.type = "*";
                        emptyInp.label = "+ Add Stream";
                        emptyInp.color_on = undefined;
                        emptyInp.color_off = undefined;
                    }

                    // Reorder this.inputs array: nonStreamInputs + orderedStreamInputs + emptyStreamInputs
                    let nonStreamInputs = this.inputs.filter(inpItem => !/^(text|image|video)_stream_\d+$/.test(inpItem.name));
                    this.inputs = [...nonStreamInputs, ...orderedStreamInputs, ...emptyStreamInputs];

                    // CRITICAL FIX: Synchronize link.target_slot in LiteGraph's link table
                    // When this.inputs array is updated, every link.target_slot MUST match its new index in this.inputs.
                    // Otherwise, LiteGraph will cross-wire or disconnect inputs when new connections are made.
                    if (app.graph && app.graph.links) {
                        for (let slotIdx = 0; slotIdx < this.inputs.length; slotIdx++) {
                            const inputItem = this.inputs[slotIdx];
                            if (inputItem && inputItem.link != null) {
                                const linkObj = app.graph.links[inputItem.link];
                                if (linkObj) {
                                    linkObj.target_slot = slotIdx;
                                }
                            }
                        }
                    }
                    
                    this.setSize(this.computeSize());
                    if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
                    if (app.graph && app.graph.setDirtyCanvas) app.graph.setDirtyCanvas(true, true);

                    for (let i = 0; i < config.length; i++) {
                        const c = config[i];
                        const idx = c.id;
                        const color = this.streamColors[idx] || "#ffffff";
                        const streamLabel = c._displayLabel || `S${idx}`;
                        
                        const row = document.createElement("div");
                        row.draggable = true;
                        Object.assign(row.style, {
                            display: "flex",
                            alignItems: "center",
                            background: "#2a2a2a",
                            padding: "2px",
                            borderRadius: "2px",
                            borderLeft: `3px solid ${color}`,
                            fontSize: "10px",
                            color: "#ccc",
                            fontFamily: "monospace",
                            cursor: "grab",
                            opacity: c.muted ? "0.5" : "1"
                        });

                        row.addEventListener('dragstart', (e) => {
                            draggingRowId = idx;
                            row.style.opacity = '0.4';
                        });
                        row.addEventListener('dragend', (e) => {
                            draggingRowId = null;
                            row.style.opacity = c.muted ? '0.5' : '1';
                        });
                        row.addEventListener('dragover', (e) => {
                            e.preventDefault();
                            row.style.borderTop = "2px solid #8CCF7B";
                        });
                        row.addEventListener('dragleave', (e) => {
                            row.style.borderTop = "none";
                        });
                        row.addEventListener('drop', (e) => {
                            e.preventDefault();
                            row.style.borderTop = "none";
                            if (draggingRowId && draggingRowId !== idx) {
                                let curConfig = parseConfig();
                                const fromIdx = curConfig.findIndex(x => x.id === draggingRowId);
                                const toIdx = curConfig.findIndex(x => x.id === idx);
                                if (fromIdx > -1 && toIdx > -1) {
                                    const [moved] = curConfig.splice(fromIdx, 1);
                                    curConfig.splice(toIdx, 0, moved);
                                    saveConfig(curConfig);
                                    self.updateDynamicPorts();
                                }
                            }
                        });

                        const createInput = (placeholder, value, width, onChange) => {
                            const inp = document.createElement("input");
                            inp.type = "text";
                            inp.placeholder = placeholder;
                            inp.value = value;
                            Object.assign(inp.style, {
                                width: width,
                                background: "#111",
                                border: "1px solid #444",
                                color: "#fff",
                                padding: "1px 2px",
                                borderRadius: "2px",
                                fontSize: "9px",
                                fontFamily: "monospace",
                                outline: "none",
                                marginRight: "2px"
                            });
                            inp.addEventListener("input", onChange);
                            return inp;
                        };
                        
                        row.innerHTML = `<span style="margin: 0 4px; font-weight: bold; cursor: grab;">☰ ${streamLabel}</span>`;
                        
                        const muteLabel = document.createElement("label");
                        Object.assign(muteLabel.style, {
                            display: "flex",
                            alignItems: "center",
                            marginLeft: "auto",
                            cursor: "pointer",
                            fontSize: "9px"
                        });
                        const muteBox = document.createElement("input");
                        muteBox.type = "checkbox";
                        muteBox.checked = c.muted;
                        muteBox.style.marginRight = "2px";
                        muteBox.addEventListener("change", (e) => {
                            c.muted = e.target.checked;
                            saveConfig(config);
                            self.updateDynamicPorts();
                        });
                        muteLabel.appendChild(muteBox);
                        muteLabel.appendChild(document.createTextNode("M"));
                        
                        if (c.type === "IMAGE") {
                            const modeSelect = document.createElement("select");
                            modeSelect.innerHTML = `<option value="images">Images</option><option value="video">Video</option>`;
                            modeSelect.value = c.mode || "images";
                            Object.assign(modeSelect.style, {
                                background: "#111", border: "1px solid #444", color: "#fff", fontSize: "9px", marginRight: "4px"
                            });
                            
                            let filenames = [];
                            const inp = self.inputs.find(i => i.name === `image_stream_${idx}`);
                            if (inp && inp.link) {
                                const link = app.graph.links[inp.link];
                                if (link) {
                                    const originNode = app.graph.getNodeById(link.origin_id);
                                    if (originNode) {
                                        filenames = traceImageFilenames(originNode, app, 5);
                                    }
                                }
                            }
                            
                            let isSingleImage = (filenames.length === 1 && typeof filenames[0] === 'string' && !filenames[0].match(/\.(mp4|mov|webm)$/i));
                            if (isSingleImage) {
                                modeSelect.disabled = true;
                                modeSelect.style.opacity = "0.5";
                            }
                            
                            let mode = c.mode || "images";
                            if (filenames.length > 0 && typeof filenames[0] === 'string' && filenames[0].match(/\.(mp4|mov|webm)(?:\?.*)?$/i) && !c._auto_mode_set) {
                                mode = "video";
                                c.mode = "video";
                                c._auto_mode_set = true;
                                modeSelect.value = "video";
                                saveConfig(config);
                            }
                            
                            modeSelect.addEventListener("change", e => { c.mode = e.target.value; saveConfig(config); renderConfigUI(); });
                            
                            const aspectLabel = document.createElement("span");
                            aspectLabel.innerText = "Aspect:";
                            aspectLabel.style.margin = "0 2px";
                            aspectLabel.style.fontSize = "9px";
                            
                            const aspectSelect = document.createElement("select");
                            aspectSelect.innerHTML = `<option value="guess">guess</option><option value="16:9">16:9</option><option value="9:16">9:16</option><option value="1:1">1:1</option>`;
                            aspectSelect.value = c.aspect || "guess";
                            Object.assign(aspectSelect.style, {
                                background: "#111", border: "1px solid #444", color: "#fff", fontSize: "9px", marginRight: "4px"
                            });
                            aspectSelect.addEventListener("change", e => { c.aspect = e.target.value; saveConfig(config); });
                            
                            const countLabel = document.createElement("span");
                            countLabel.innerText = "Imgs/job:";
                            countLabel.style.margin = "0 2px";
                            countLabel.style.fontSize = "9px";
                            
                            const countInp = createInput("", c.imgs_per_job || 1, "20px", (e) => { c.imgs_per_job = parseInt(e.target.value)||1; saveConfig(config); });
                            
                            const cycleLabel = document.createElement("span");
                            cycleLabel.innerText = "Cycle:";
                            cycleLabel.style.margin = "0 2px";
                            cycleLabel.style.fontSize = "9px";
                            
                            const cycleSelect = document.createElement("select");
                            cycleSelect.innerHTML = `<option value="iterate">iterate</option><option value="random">random</option><option value="fixed">fixed</option>`;
                            cycleSelect.value = c.cycle || "iterate";
                            Object.assign(cycleSelect.style, {
                                background: "#111", border: "1px solid #444", color: "#fff", fontSize: "9px", marginRight: "4px"
                            });
                            
                            const indexInp = createInput("idx", c.index || 0, "20px", (e) => { c.index = parseInt(e.target.value)||0; saveConfig(config); });
                            indexInp.style.display = (c.cycle === "fixed") ? "block" : "none";
                            
                            cycleSelect.addEventListener("change", e => { 
                                c.cycle = e.target.value; 
                                indexInp.style.display = (c.cycle === "fixed") ? "block" : "none";
                                saveConfig(config); 
                            });
                            

                            const infoLabel = document.createElement("span");
                            infoLabel.style.margin = "0 4px";
                            infoLabel.style.fontSize = "8px";
                            infoLabel.style.color = "#aaa";
                            infoLabel.style.whiteSpace = "nowrap";
                            infoLabel.style.maxWidth = "none";
                            
                            row.appendChild(modeSelect);
                            let fCount = filenames.length;
                            let vPath = filenames.length > 0 ? filenames[0] : "";
                            let isVid = (typeof vPath === 'string' && vPath.match(/\.(mp4|mov|webm)(?:[?&].*)?$/i)) || mode === "video";
                            let isImg = typeof vPath === 'string' && vPath.match(/\.(png|jpg|jpeg|webp|gif)(?:[?&].*)?$/i);
                            let baseText = fCount > 1 ? `${fCount} frames` : (isVid ? "Video" : "1 frame");
                            let vText = baseText;
                            
                            if (isVid || isImg) {
                                vText = "Loading info...";
                                let vsrc = "";
                                let cleanPath = vPath.replace(/^["']|["']$/g, '').replace(/\?frame=\d+$/, '');
                                let typeMatch = cleanPath.match(/\s*\[(input|temp|output)\]\s*$/i);
                                let reqType = "input";
                                if (typeMatch) {
                                    reqType = typeMatch[1].toLowerCase();
                                    cleanPath = cleanPath.replace(/\s*\[(input|temp|output)\]\s*$/i, "");
                                }
                                let isAbsolute = cleanPath.includes(":\\") || cleanPath.includes(":/") || cleanPath.startsWith("/") || cleanPath.startsWith("\\");
                                if (cleanPath.startsWith("http") || cleanPath.startsWith("/view") || cleanPath.startsWith("blob:") || cleanPath.startsWith("data:")) {
                                    vsrc = cleanPath;
                                    if (vsrc.startsWith("view?")) vsrc = "/" + vsrc;
                                } else if (isAbsolute) {
                                    vsrc = api.apiURL(`/vhs/viewvideo?filename=${encodeURIComponent(cleanPath)}&type=path`);
                                } else {
                                    let name = cleanPath;
                                    let subfolder = "";
                                    if (cleanPath.includes("/") || cleanPath.includes("\\")) {
                                        const parts = cleanPath.split(/[/\\]/);
                                        name = parts.pop();
                                        subfolder = parts.join("/");
                                    }
                                    vsrc = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=${reqType}&subfolder=${encodeURIComponent(subfolder)}`);
                                }
                                
                                const updateLabel = (w, h, duration) => {
                                    let gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
                                    let divisor = gcd(w, h);
                                    let aspect = `${w/divisor}:${h/divisor}`;
                                    if (w === 1280 && h === 720) aspect = "16:9";
                                    if (w === 720 && h === 1280) aspect = "9:16";
                                    
                                    let estFrames = duration && duration !== Infinity && !isNaN(duration) ? Math.round(duration * 24) : 0;
                                    let frameStr = fCount > 1 ? `${fCount} frames` : (estFrames > 0 ? `${estFrames} frames` : `Video`);
                                    if (!isVid && fCount === 1) frameStr = "1 frame";
                                    
                                    vText = `${frameStr} (${aspect}, ${w}x${h})`;
                                    infoLabel.innerText = vText;
                                    infoLabel.title = vText;
                                };
                                
                                if (isVid) {
                                    let tempVid = document.createElement("video");
                                    tempVid.src = vsrc + (vsrc.includes("?") ? "&" : "?") + "_meta=" + Date.now();
                                    tempVid.preload = "metadata";
                                    let handleLoad = () => updateLabel(tempVid.videoWidth, tempVid.videoHeight, tempVid.duration);
                                    tempVid.onloadedmetadata = handleLoad;
                                    tempVid.ondurationchange = handleLoad;
                                    tempVid.onerror = () => {
                                        vText = baseText;
                                        infoLabel.innerText = vText;
                                        infoLabel.title = vText;
                                    };
                                } else if (isImg) {
                                    let tempImg = new Image();
                                    tempImg.src = vsrc;
                                    tempImg.onload = () => updateLabel(tempImg.naturalWidth, tempImg.naturalHeight, 0);
                                    tempImg.onerror = () => {
                                        vText = baseText;
                                        infoLabel.innerText = vText;
                                        infoLabel.title = vText;
                                    };
                                }
                            }
                            infoLabel.innerText = vText;
                            infoLabel.title = vText;
                            
                            const imageWidgets = document.createElement("span");
                            imageWidgets.style.display = (mode === "video") ? "none" : "inline";
                            imageWidgets.appendChild(aspectLabel);
                            imageWidgets.appendChild(aspectSelect);
                            imageWidgets.appendChild(countLabel);
                            imageWidgets.appendChild(countInp);
                            imageWidgets.appendChild(cycleLabel);
                            imageWidgets.appendChild(cycleSelect);
                            imageWidgets.appendChild(indexInp);
                            
                            const videoWidgets = document.createElement("span");
                            videoWidgets.style.display = (mode === "video") ? "inline" : "none";
                            videoWidgets.appendChild(infoLabel);
                            
                            row.appendChild(imageWidgets);
                            row.appendChild(videoWidgets);
                            row.appendChild(muteLabel);
                        } else {
                            const delimInp = createInput("delim", c.delim, "25px", (e) => { c.delim = e.target.value; saveConfig(config); });
                            const prefixInp = createInput("pre", c.prefix, "75px", (e) => { c.prefix = e.target.value; saveConfig(config); });
                            const suffixInp = createInput("suf", c.suffix, "75px", (e) => { c.suffix = e.target.value; saveConfig(config); });
                            
                            row.appendChild(delimInp);
                            row.appendChild(prefixInp);
                            row.appendChild(suffixInp);
                            row.appendChild(muteLabel);
                        }
                        
                        delimitersContainer.appendChild(row);
                    }
                    
                    if (activeCount === 0) {
                        delimitersContainer.style.display = "none";
                    } else {
                        delimitersContainer.style.display = "flex";
                    }
                    
                    configW.value = JSON.stringify(config);
                };

                this.updateDynamicPorts();
                
                this.updateBatchPreview = () => {
                    let hasDynamic = false;
                    let processedStreams = [];
                    
                    let config = parseConfig();
                    
                    for (let c of config) {
                        if (c.muted) continue;
                        
                        const idx = c.id;
                        const color = this.streamColors[idx] || "#ffffff";
                        
                        const inp = self.inputs.find(i => i.name === `text_stream_${idx}` || i.name === `image_stream_${idx}`);
                        if (!inp || !inp.link) continue;
                        
                        if (c.type === "IMAGE") {
                            let filenames = [];
                            
                            let cookedStream = null;
                            if (app.nodeOutputs && app.nodeOutputs[self.id] && app.nodeOutputs[self.id].streams) {
                                cookedStream = app.nodeOutputs[self.id].streams[idx];
                            }
                            
                            const link = app.graph.links[inp.link];
                            if (link) {
                                const originNode = app.graph.getNodeById(link.origin_id);
                                if (originNode) {
                                    filenames = traceImageFilenames(originNode, app, 5, () => {
                                        this.updateBatchPreview();
                                    });
                                }
                            }
                            
                            // Map filenames to media objects
                            let mediaFiles = filenames.map(f => resolveMediaUrl(f));
                            
                            let mode = c.mode || "images";
                            let imgsPerJob = parseInt(c.imgs_per_job) || 1;
                            let htmlParts = [];
                            
                            if (mode === "video") {
                                imgsPerJob = mediaFiles.length > 0 ? mediaFiles.length : 1;
                            }
                            
                            let aspect = c.aspect || "guess";
                            let widthStyle = (aspect === "16:9") ? "width: 39px;" : ((aspect === "9:16") ? "width: 12px;" : ((aspect === "1:1") ? "width: 22px;" : "width: auto;"));
                            let boxW = (aspect === "16:9") ? 39 : ((aspect === "9:16") ? 12 : 22);
                            if (mode === "video") {
                                widthStyle = "width: 39px;";
                                boxW = 39;
                            }

                            const jobsWidget = self.widgets?.find(w => w.name === "jobs");
                            const jobsCount = jobsWidget ? (parseInt(jobsWidget.value) || 0) : 0;
                            
                            // If user manually changed jobs count and it doesn't match the cached execution, ignore the cache
                            if (cookedStream && jobsCount > 0 && cookedStream.length !== jobsCount) {
                                cookedStream = null;
                            }
                            if (cookedStream && jobsCount === 0) {
                                // For auto-detect mode, if we haven't recalculated natural length yet, 
                                // we can't fully trust cookedStream, but we'll let it pass for now.
                                // However, if they just switched TO 0, it's safer to invalidate.
                            }
                            
                            if (cookedStream) {
                                let targetLen = jobsCount > 0 ? jobsCount : cookedStream.length;
                                for (let job_i = 0; job_i < targetLen; job_i++) {
                                    let jobHtml = [];
                                    let imgs = cookedStream[job_i] || [];
                                    if (imgs.length === 0) {
                                        jobHtml.push(`<div style="display:inline-block; width: 22px; height: 22px; background:#444; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; text-align:center; line-height:22px; font-size: 8px; border-radius:2px; color: #888;">IMG</div>`);
                                    } else {
                                        for (let img of imgs) {
                                            let src = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder || "")}`);
                                            let isVid = img.is_video || (img.filename && img.filename.match(/\.(mp4|mov|webm)(?:\?.*)?$/i));
                                            if (isVid) {
                                                jobHtml.push(`<video src="${src}" autoplay loop muted style="height: 22px; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; object-fit: cover; border-radius: 2px;"></video>`);
                                            } else {
                                                jobHtml.push(`<img src="${src}" style="height: 22px; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; object-fit: cover; border-radius: 2px;" />`);
                                            }
                                        }
                                    }
                                    htmlParts.push(jobHtml.join(""));
                                }
                                processedStreams.push({ color, type: "IMAGE", parts: htmlParts, naturalLength: cookedStream.length });
                                continue;
                            }

                            if (mediaFiles.length === 0) {
                                for (let j=0; j<imgsPerJob; j++) {
                                    let txt = mode === "video" ? "VID" : "IMG";
                                    htmlParts.push(`<div style="display:inline-block; width: ${boxW}px; min-width: 22px; height: 22px; background:#444; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; text-align:center; line-height:22px; font-size: 8px; border-radius:2px; color: #888;">${txt}</div>`);
                                }
                                processedStreams.push({ color, type: "IMAGE", parts: [htmlParts.join("")], naturalLength: 1 });
                                hasDynamic = true;
                            } else {
                                let parts = [];
                                let N = mediaFiles.length;
                                let naturalLength = (c.cycle === "iterate") ? Math.ceil(N / imgsPerJob) : 1;
                                if (mode === "video") naturalLength = 1;
                                if (naturalLength < 1) naturalLength = 1;
                                
                                for (let job_i = 0; job_i < 1000; job_i++) { 
                                    let jobHtml = [];
                                    let videoFrames = [];
                                    let videoFps = parseInt(c.fps) || 24;
                                    for (let j = 0; j < imgsPerJob; j++) {
                                        let cycle = c.cycle || "iterate";
                                        let idxOffset = parseInt(c.index) || 0;
                                        let fileIdx = 0;
                                        if (mode === "video") {
                                            fileIdx = j;
                                        } else if (cycle === "fixed") {
                                            fileIdx = Math.min(idxOffset + j, N - 1);
                                        } else if (cycle === "random") {
                                            fileIdx = Math.floor(Math.random() * N); 
                                        } else {
                                            fileIdx = (job_i * imgsPerJob + j) % N;
                                        }
                                        
                                        let mf = mediaFiles[fileIdx];
                                        
                                        if (!mf || !mf.src) {
                                            jobHtml.push(`<div style="display:inline-block; ${widthStyle} min-width: 22px; height: 22px; background:#444; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; text-align:center; line-height:22px; font-size: 8px; border-radius:2px; color: #888;">IMG</div>`);
                                            continue;
                                        }

                                        let src = mf.src;
                                        let safeFallback = `<div style="display:inline-block; ${widthStyle} min-width: 22px; height: 22px; background:#333; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; text-align:center; line-height:22px; font-size: 8px; border-radius:2px; color: #aaa; overflow:hidden;">VID</div>`;
                                        
                                        if (mode === "video") {
                                            if (mf.isVideo) {
                                                let uSrc = src + (src.includes("?") ? "&" : "?") + "_vid=" + Date.now() + "_" + job_i;
                                                // Only add to jobHtml if we haven't already added a video player for this job (video mode is 1 player)
                                                if (j === 0) {
                                                    jobHtml.push(`<span style="position:relative;"><span class="vid-fallback" style="display:none;">${safeFallback}</span><video src="${uSrc}" alt="" style="${widthStyle} min-width: 22px; height: 22px; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; object-fit: cover; border-radius: 2px;" autoplay loop muted onerror="this.style.display='none'; this.previousSibling.style.display='inline-block';"></video></span>`);
                                                }
                                            } else {
                                                videoFrames.push({ type: "img", src: src });
                                            }
                                        } else { // mode === "images"
                                            if (mf.isVideo) {
                                                let timeOffset = mf.frameIdx / 24.0;
                                                let tSrc = src + `#t=${timeOffset}`;
                                                jobHtml.push(`<span style="position:relative;"><span class="vid-fallback" style="display:none;">${safeFallback}</span><video src="${tSrc}" style="${widthStyle} min-width: 22px; height: 22px; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; object-fit: cover; border-radius: 2px;" disablePictureInPicture muted preload="metadata" onerror="this.style.display='none'; this.previousSibling.style.display='inline-block';"></video></span>`);
                                            } else {
                                                jobHtml.push(`<img src="${src}" alt="" style="${widthStyle} min-width: 22px; height: 22px; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; object-fit: cover; border-radius: 2px; color: transparent;" />`);
                                            }
                                        }
                                    }
                                    
                                    if (mode === "video" && videoFrames.length > 0 && videoFrames[0].type === "img") {
                                        let frames = videoFrames.map(f => f.src);
                                        let framesStr = JSON.stringify(frames).replace(/"/g, '&quot;');
                                        let delay = Math.floor(1000 / videoFps);
                                        jobHtml.push(`<img src="${frames[0]}" alt="" style="${widthStyle} min-width: 22px; height: 22px; border: 1px solid ${color}; vertical-align: middle; margin: 0 4px; object-fit: cover; border-radius: 2px; color: transparent;" onload="if(!this.intervalSet){ this.intervalSet=true; this.idx=0; this.frames=JSON.parse('${framesStr}'); this.timer = setInterval(()=>{ if(!this.isConnected){ clearInterval(this.timer); return; } this.idx=(this.idx+1)%this.frames.length; this.src=this.frames[this.idx]; }, ${delay}); }" />`);
                                    }
                                    
                                    parts.push(jobHtml.join(""));
                                }
                                processedStreams.push({ color, type: "IMAGE", parts, naturalLength });
                            }
                            continue;
                        }

                        let delim = c.delim || "";
                        if (delim === "\\n") delim = "\n";
                        
                        let inputText = "";
                        const link = app.graph.links[inp.link];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode && originNode.widgets) {
                                const textWidget = originNode.widgets.find(widg => widg.type === "customtext" || widg.name === "text" || widg.name === "string" || widg.name === "value");
                                if (textWidget) {
                                    inputText = textWidget.value || "";
                                }
                            }
                        }
                        
                        if (!inputText) {
                            hasDynamic = true;
                            inputText = "[Dynamic]";
                        }
                        
                        let parts = delim === "" ? [inputText] : inputText.split(delim);
                        parts = parts.map(p => {
                            return `${c.prefix}${p}${c.suffix}`;
                        });
                        processedStreams.push({ color, type: "STRING", parts, naturalLength: parts.length });
                    }

                    if (processedStreams.length === 0) {
                        if (hasDynamic) {
                            contentDiv.innerHTML = "<div style='color:#888; font-style:italic; padding:2px;'>[Connected to dynamic node.<br/>Preview unavailable until execution.]</div>";
                        } else {
                            contentDiv.innerHTML = "<div style='color:#666; padding:2px;'>[No jobs detected]</div>";
                        }
                        return;
                    }

                    const jobsWidget = self.widgets?.find(w => w.name === "jobs");
                    const jobsCount = jobsWidget ? (parseInt(jobsWidget.value) || 0) : 0;
                    
                    if (self.lastJobsCount !== undefined && self.lastJobsCount !== jobsCount) {
                        if (app.nodeOutputs && app.nodeOutputs[self.id]) {
                            app.nodeOutputs[self.id].streams = null; 
                        }
                    }
                    self.lastJobsCount = jobsCount;
                    
                    let maxLen = Math.max(...processedStreams.map(s => s.naturalLength || 1), 0);
                    let targetLen = jobsCount > 0 ? jobsCount : maxLen;

                    let html = `<strong style="display:block; margin-bottom: 2px; color: #ffffff;">${targetLen} jobs</strong>`;
                    self.geminiJobHtmls = [];
                    
                    for (let i = 0; i < targetLen; i++) {
                        let innerHtml = "";
                        let rowHtml = `<div style='margin-bottom:1px; padding:1px 3px; background:#2a2a2a; border-radius:2px; border-left: 2px solid #555;'><span style="color:#888; font-size:9px; margin-right:4px;">#${i+1}</span>`;
                        for (let stream of processedStreams) {
                            if (stream.parts.length === 0) continue;
                            let part = stream.parts[i % stream.parts.length];
                            if (stream.type === "IMAGE") {
                                let span = `<span style="color: ${stream.color}; display: inline-flex; align-items: center; vertical-align: middle;">${part}</span>`;
                                rowHtml += span;
                                innerHtml += span;
                            } else {
                                let partStr = typeof part === 'string' ? part.trim() : "";
                                let mf = resolveMediaUrl(partStr);
                                if (mf.isVideo) {
                                    let uSrc = mf.src + (mf.src.includes("?") ? "&" : "?") + "_vid=" + Date.now() + "_" + i;
                                    let span = `<span style="color: ${stream.color}; display: inline-flex; align-items: center; vertical-align: middle; margin: 2px;"><video src="${uSrc}" autoplay loop muted style="max-height: 60px; max-width: 100px; border-radius: 2px; border: 1px solid #444;" onerror="this.style.display='none';"></video></span>`;
                                    rowHtml += span;
                                    innerHtml += span;
                                } else {
                                    let escaped = partStr.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                                    let tagHighlighted = escaped
                                        .replace(/\b(Video\d+)\b/gi, (match) => {
                                            return `<span style="display: inline-block; background: #8e24aa; color: #ffffff; font-weight: bold; font-size: 9px; padding: 1px 5px; border-radius: 3px; margin: 0 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); vertical-align: middle;">${match}</span>`;
                                        })
                                        .replace(/\b(Image\d+)\b/gi, (match) => {
                                            return `<span style="display: inline-block; background: #2e64de; color: #ffffff; font-weight: bold; font-size: 9px; padding: 1px 5px; border-radius: 3px; margin: 0 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); vertical-align: middle;">${match}</span>`;
                                        });
                                    let span = `<span style="color: ${stream.color}; display:inline-block; padding: 2px 4px; margin: 0 4px; border: 1px solid ${stream.color}; border-radius: 2px; font-size: 9px; color: #ccc; max-width: 300px; background: #222; vertical-align: middle; white-space: pre-wrap; word-break: break-word;" title="${escaped}">${tagHighlighted}</span>`;
                                    rowHtml += span;
                                    innerHtml += span;
                                }
                            }
                        }
                        rowHtml += `</div>`;
                        html += rowHtml;
                        self.geminiJobHtmls.push(innerHtml);
                    }
                    self.geminiTargetLen = targetLen;
                    contentDiv.innerHTML = html;
                };
let lastState = "";
                const pollInterval = setInterval(() => {
                    if (!app.graph._nodes.find(n => n.id === self.id)) {
                        clearInterval(pollInterval);
                        return;
                    }
                    
                    let currentState = "";
                    let streamInputs = self.inputs ? self.inputs.filter(inp => /^(text|image)_stream_\d+$/.test(inp.name)) : [];
                    let config = null;
                    try { config = JSON.parse(self.widgets.find(w => w.name === "stream_config").value || "[]"); } catch (e) { config = []; }
                    let needsConfigUpdate = false;
                    
                    for (let inp of streamInputs) {
                        if (inp.link) {
                            const link = app.graph.links[inp.link];
                            if (link) {
                                const originNode = app.graph.getNodeById(link.origin_id);
                                if (originNode) {
                                    if (inp.name.startsWith("image_stream_")) {
                                        let files = traceImageFilenames(originNode, app, 5);
                                        currentState += JSON.stringify(files) + "|";
                                    }
                                    
                                    let detectedType = "STRING";
                                    let detectedMode = "images";
                                    
                                    if (originNode.outputs) {
                                        const out = originNode.outputs[link.origin_slot];
                                        if (out) {
                                            let outType = out.type;
                                            if (outType === "IMAGE" || (typeof outType === "string" && outType.includes("IMAGE"))) {
                                                detectedType = "IMAGE";
                                            }
                                        }
                                    }
                                    
                                    if (originNode.widgets) {
                                        const textWidget = originNode.widgets.find(w => w.type === "customtext" || w.name === "text" || w.name === "string" || w.name === "value" || w.name === "path" || w.name === "video");
                                        if (textWidget) currentState += textWidget.value + "|";
                                        
                                        if (detectedType === "STRING") {
                                            const stringWidgets = originNode.widgets.filter(w => typeof w.value === 'string');
                                            for (let w of stringWidgets) {
                                                let cleanVal = w.value.replace(/^["']|["']$/g, '');
                                                if (cleanVal.match(/\.(mp4|mov|webm)(?:[?&].*)?$/i)) {
                                                    detectedType = "IMAGE";
                                                    detectedMode = "video";
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    
                                    const idxMatch = inp.name.match(/\d+/);
                                    if (idxMatch) {
                                        const idx = parseInt(idxMatch[0]);
                                        let c = config.find(x => x.id === idx);
                                        if (c) {
                                            if (detectedType === "IMAGE" && detectedMode === "video" && (c.type !== "IMAGE" || c.mode !== "video")) {
                                                c.type = "IMAGE";
                                                c.mode = "video";
                                                needsConfigUpdate = true;
                                            } else if (detectedType === "STRING" && c.type !== "STRING") {
                                                c.type = "STRING";
                                                needsConfigUpdate = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if (needsConfigUpdate) {
                        const configW = self.widgets.find(w => w.name === "stream_config");
                        if (configW) configW.value = JSON.stringify(config);
                        if (self.updateDynamicPorts) self.updateDynamicPorts();
                        return; // updateDynamicPorts calls updateBatchPreview anyway
                    }
                    
                    for (let w of self.widgets || []) {
                        if (w.name === "stream_config" || w.name === "jobs") {
                            currentState += w.value + "|";
                        }
                    }
                    
                    if (app.nodeOutputs && app.nodeOutputs[self.id] && app.nodeOutputs[self.id].streams) {
                        currentState += JSON.stringify(app.nodeOutputs[self.id].streams) + "|";
                    }
                    
                    if (currentState !== lastState) {
                        lastState = currentState;
                        if (self.updateBatchPreview) self.updateBatchPreview();
                        app.graph.setDirtyCanvas(true, false);
                    }
                }, 100);
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                // Backward compatibility fix:
                // Previous versions of this node scrambled the widget order, putting stream_config (string)
                // before jobs (int) when saving the workflow.
                // ComfyUI restores widgets_values sequentially by index.
                // We swap them back if we detect the scrambled order in info.widgets_values.
                if (info && info.widgets_values && info.widgets_values.length >= 2) {
                    if (typeof info.widgets_values[0] === "string" && info.widgets_values[0].startsWith("[")) {
                        let temp = info.widgets_values[0];
                        info.widgets_values[0] = info.widgets_values[1];
                        info.widgets_values[1] = temp;
                    }
                }
                
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                for (let w of this.widgets || []) {
                    if (w.name.startsWith("delimiter_")) {
                        w.type = "converted-widget";
                        w.computeSize = () => [0,0];
                    }
                    if (w.name === "stream_config") {
                        w.type = "hidden";
                        w.computeSize = () => [0,0];
                    }
                }
                
                if (this.updateDynamicPorts) this.updateDynamicPorts();
                if (this.updateBatchPreview) this.updateBatchPreview();
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                if (this.updateDynamicPorts) {
                    this.updateDynamicPorts();
                }
                if (this.updateBatchPreview) {
                    setTimeout(() => {
                        this.updateBatchPreview();
                    }, 50);
                }
            };
        }
    }
});
