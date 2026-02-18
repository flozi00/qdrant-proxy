import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/admin/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/admin/stats': 'http://localhost:8002',
      '/admin/documents': 'http://localhost:8002',
      '/admin/facts': 'http://localhost:8002',
      '/admin/feedback': 'http://localhost:8002',
      '/admin/maintenance': 'http://localhost:8002',
      '/admin/templates': 'http://localhost:8002',
      '/feedback': 'http://localhost:8002',
      '/documents': 'http://localhost:8002',
      '/kv': 'http://localhost:8002',
      '/mcp-server': 'http://localhost:8002',
    },
  },
})
