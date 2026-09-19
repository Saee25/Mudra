import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

import fs from 'fs';
import path from 'path';

// Custom plugin to bypass Vite's strict public import check for ORT's .mjs files.
// ONNX Runtime Web dynamically imports .mjs files from the wasmPaths directory.
// Vite blocks dynamic imports of files in the /public directory by default.
// This middleware intercepts requests to /ort/*.mjs and serves them directly.
const serveOrtMjsPlugin = () => {
  return {
    name: 'serve-ort-mjs',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith('/ort/') && req.url.includes('.mjs')) {
          const filePath = path.join(process.cwd(), 'public', req.url.split('?')[0]);
          if (fs.existsSync(filePath)) {
            res.setHeader('Content-Type', 'application/javascript');
            res.end(fs.readFileSync(filePath));
            return;
          }
        }
        next();
      });
    }
  };
};

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    serveOrtMjsPlugin(),
  ],
  optimizeDeps: {
    // Exclude onnxruntime-web from Vite's pre-bundling.
    // ORT uses dynamic imports for WASM files which Vite tries to bundle incorrectly.
    // Since we copy the WASM to public/ and tell ORT where to find it, 
    // excluding it here fixes dev-mode loading issues.
    exclude: ['onnxruntime-web']
  }
})
