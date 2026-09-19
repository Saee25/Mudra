import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  optimizeDeps: {
    // Exclude onnxruntime-web from Vite's pre-bundling.
    // ORT uses dynamic imports for WASM files which Vite tries to bundle incorrectly.
    // Since we copy the WASM to public/ and tell ORT where to find it, 
    // excluding it here fixes dev-mode loading issues.
    exclude: ['onnxruntime-web']
  }
})
