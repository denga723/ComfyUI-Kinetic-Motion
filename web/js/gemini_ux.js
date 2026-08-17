import { app } from "/scripts/app.js";

app.registerExtension({
    name: "ComfyUI.GeminiEnterprise.UX",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only modify our Gemini Enterprise nodes
        if (!nodeData.name.startsWith("Gemini") || !nodeData.category.startsWith("Gemini Enterprise")) {
            return;
        }

        // We want to allow wildcard '*' ports to connect to anything.
        // ComfyUI / LiteGraph defaults to strict type checking.
        
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }
            
            this.color = "#2b3d54";
            this.bgcolor = "#1c2836";

            // Iterate inputs and override connect behavior if type is '*'
            if (this.inputs) {
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].type === "*") {
                        // In litegraph, setting type to '*' doesn't automatically mean "allow all".
                        // A common hack is to hook onConnectInput and return true.
                        // We also need to clear the type when disconnected so it can accept anything again.
                        
                        // We do not strict override here because litegraph allows custom connect logic
                        // if we define it in the node.
                    }
                }
            }
        };

        // Override onConnectInput to allow wildcard connections
        const origOnConnectInput = nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function (target_slot, type, output, node, slot) {
            let res = undefined;
            if (origOnConnectInput) {
                res = origOnConnectInput.apply(this, arguments);
            }
            if (!this.inputs[target_slot]) return res;

            if (this.inputs[target_slot].type === "*" || this.inputs[target_slot].name.toLowerCase().includes("stream")) {
                return true; // Allow any type to connect to our wildcard stream ports
            }
            return res;
        };
    }
});
