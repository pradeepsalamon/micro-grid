import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// export default defineConfig({
//   plugins: [react()],
//   server: {
//     proxy: {
//       '/api/telemetry': {
//         target: 'http://localhost/api',
//         changeOrigin: true
//       },
//       '/forecast': {
//         target: 'http://localhost/api',
//         changeOrigin: true
//       },
//       '/weather': {
//         target: 'http://localhost/api',
//         changeOrigin: true
//       }
//     }
//   }
// })



export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/telemetry': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/forecast': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/weather': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
