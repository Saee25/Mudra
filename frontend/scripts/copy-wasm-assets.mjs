import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const frontendDir = path.resolve(__dirname, '..');
const nodeModulesDir = path.join(frontendDir, 'node_modules');
const publicDir = path.join(frontendDir, 'public');

const mediapipeTaskPath = path.join(publicDir, 'mediapipe', 'hand_landmarker.task');
const mediapipeWasmSrc = path.join(nodeModulesDir, '@mediapipe', 'tasks-vision', 'wasm');
const mediapipeWasmDst = path.join(publicDir, 'mediapipe', 'wasm');

function copyDirWithStats(src, dst) {
    if (!fs.existsSync(src)) {
        console.warn(`Warning: Source directory ${src} not found. Ensure npm install has run.`);
        return;
    }
    
    fs.mkdirSync(dst, { recursive: true });
    const items = fs.readdirSync(src);
    
    for (const item of items) {
        const srcPath = path.join(src, item);
        const dstPath = path.join(dst, item);
        
        const stat = fs.statSync(srcPath);
        if (stat.isDirectory()) {
            copyDirWithStats(srcPath, dstPath);
        } else {
            fs.copyFileSync(srcPath, dstPath);
            const sizeKb = (stat.size / 1024).toFixed(2);
            console.log(`Copied ${item} -> ${dstPath} (${sizeKb} KB)`);
        }
    }
}

console.log('Copying WASM assets to public directory...');
/* 
 * We self-host WASM files rather than using a CDN for two reasons:
 * 1. Version parity: The WASM loaded perfectly matches the @mediapipe/tasks-vision version in package.json.
 * 2. No external dependency: Ensures the app works completely offline and without relying on third-party CDNs.
 */

// 1. Check for the task file
if (!fs.existsSync(mediapipeTaskPath)) {
    console.error(`ERROR: ${mediapipeTaskPath} is missing.`);
    console.error("Please run `python training/copy_model_to_frontend.py` first.");
    process.exit(1);
}

// 2. Copy MediaPipe WASM
console.log('\n--- MediaPipe Assets ---');
copyDirWithStats(mediapipeWasmSrc, mediapipeWasmDst);

// 3. Copy ONNX Runtime Web WASM (only plain WASM, skip jsep/jspi/asyncify)
console.log('\n--- ONNX Runtime Web Assets ---');
const ortWasmSrc = path.join(nodeModulesDir, 'onnxruntime-web', 'dist');
const ortWasmDst = path.join(publicDir, 'ort');

if (!fs.existsSync(ortWasmSrc)) {
    console.warn(`Warning: ${ortWasmSrc} not found. Ensure npm install has run.`);
} else {
    fs.mkdirSync(ortWasmDst, { recursive: true });
    // In onnxruntime-web 1.20+, the core wasm is ort-wasm-simd-threaded.wasm and .mjs
    const filesToCopy = [
        'ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd-threaded.mjs'
    ];
    for (const file of filesToCopy) {
        const srcPath = path.join(ortWasmSrc, file);
        const dstPath = path.join(ortWasmDst, file);
        if (fs.existsSync(srcPath)) {
            fs.copyFileSync(srcPath, dstPath);
            const stat = fs.statSync(srcPath);
            const sizeKb = (stat.size / 1024).toFixed(2);
            console.log(`Copied ${file} -> ${dstPath} (${sizeKb} KB)`);
        }
    }
}

console.log('\nDone!');
