let labels = [];
let modelMeta = null;

export async function loadLabels() {
    if (labels.length > 0) return labels;

    try {
        const [labelsRes, metaRes] = await Promise.all([
            fetch(import.meta.env.BASE_URL + 'model/class_names.json'),
            fetch(import.meta.env.BASE_URL + 'model/model_meta.json')
        ]);
        
        labels = await labelsRes.json();
        modelMeta = await metaRes.json();

        // Check meta
        if (modelMeta.num_classes !== labels.length) {
            console.warn(`Mismatch: model_meta num_classes (${modelMeta.num_classes}) != labels length (${labels.length})`);
        }
        if (modelMeta.pipeline_version !== "landmark-v1") {
            console.warn(`Mismatch: pipeline_version is ${modelMeta.pipeline_version}, expected "landmark-v1"`);
        }
        if (modelMeta.crop_padding !== 0.2 || modelMeta.crop_size !== 224) { // CROP_PADDING/CROP_SIZE from handCrop.js
            console.warn(`Mismatch: crop_padding/crop_size mismatch with handCrop.js constants`);
        }
        if (modelMeta.min_detection_confidence !== 0.3 || modelMeta.min_presence_confidence !== 0.3 || modelMeta.num_hands !== 2) {
            console.warn(`Mismatch: MediaPipe confidences or num_hands mismatch with handDetection.js`);
        }
        
    } catch (e) {
        console.error("Failed to load labels or meta:", e);
    }
    
    // Order = sorted folder names (digits 1–9 first, then A–Z), exactly as training saved them.
    return labels;
}

export function getLabels() {
    return labels;
}

export function isDigit(label) {
    return /^[0-9]+$/.test(label);
}

export function isLetter(label) {
    return /^[A-Za-z]+$/.test(label);
}

export function isSpecialClass(label) {
    return label.toLowerCase() === "other";
}

export function getClassIndices(filter) {
    const indices = [];
    labels.forEach((label, index) => {
        if (filter === "both") {
            indices.push(index);
        } else if (filter === "letters" && (isLetter(label) || isSpecialClass(label))) {
            indices.push(index);
        } else if (filter === "numbers" && (isDigit(label) || isSpecialClass(label))) {
            indices.push(index);
        }
    });
    return indices;
}
