import { app } from "../../scripts/app.js";

// Dynamic input slot management for the *Multi nodes.
// Adds an "Update inputs" button that rebuilds prefix_1..prefix_N slots to match
// the "inputcount" widget. Mirrors the behaviour of KJNodes' multi nodes.
function setupDynamicInputs(node, { type, prefix, countWidget = "inputcount", slotOptions } = {}) {
    const rebuild = () => {
        if (!node.inputs) node.inputs = [];
        const countW = node.widgets?.find((w) => w.name === countWidget);
        if (!countW) return;
        const target = countW.value;
        const current = node.inputs.filter((i) => i.name?.startsWith(prefix)).length;
        if (target === current) return;
        if (target < current) {
            for (let i = 0; i < current - target; i++) {
                node.removeInput(node.inputs.length - 1);
            }
        } else {
            for (let i = current + 1; i <= target; i++) {
                node.addInput(`${prefix}${i}`, type, slotOptions);
            }
        }
    };
    node.addWidget("button", "Update inputs", null, rebuild);
    const countW = node.widgets?.find((w) => w.name === countWidget);
    if (countW) {
        const origCb = countW.callback;
        countW.callback = function (value, canvas) {
            const r = origCb ? origCb.apply(this, arguments) : undefined;
            if (!canvas) rebuild(); // bare callback = API reload; skip interactive scrub
            return r;
        };
    }
    return rebuild;
}

app.registerExtension({
    name: "ZNGBNodes.dynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        switch (nodeData.name) {
            case "ZNGB_ImageBatchMulti":
                nodeType.prototype.onNodeCreated = function () {
                    setupDynamicInputs(this, { type: "IMAGE", prefix: "image_", slotOptions: { shape: 7 } });
                };
                break;
            case "ZNGB_AudioConcatMulti":
                nodeType.prototype.onNodeCreated = function () {
                    setupDynamicInputs(this, { type: "AUDIO", prefix: "audio_", slotOptions: { shape: 7 } });
                };
                break;
            default:
                break;
        }
    },
});
