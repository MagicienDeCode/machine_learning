import axios, { type InternalAxiosRequestConfig } from 'axios'
import { useAuthStore } from '@/stores/auth'

// Circular with stores/auth.ts by design (common axios+Pinia pattern): safe because both sides
// only touch the other module's exports inside function bodies, never at module-evaluation time.

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  // Needed so the httpOnly refresh_token cookie is sent/received on /api/auth/* calls.
  withCredentials: true,
})

apiClient.interceptors.request.use((config) => {
  const auth = useAuthStore()
  if (auth.accessToken) {
    config.headers.Authorization = `Bearer ${auth.accessToken}`
  }
  return config
})

interface RetryableConfig extends InternalAxiosRequestConfig {
  _retried?: boolean
}

// Several requests can 401 at once when the access token expires; share one in-flight refresh.
let refreshInFlight: Promise<boolean> | null = null

apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const auth = useAuthStore()
    const config = error.config as RetryableConfig | undefined
    const isAuthEndpoint = config?.url?.startsWith('/api/auth/')

    if (error.response?.status === 401 && config && !config._retried && !isAuthEndpoint) {
      config._retried = true
      refreshInFlight ??= auth.refresh().finally(() => {
        refreshInFlight = null
      })
      const refreshed = await refreshInFlight
      if (refreshed) {
        config.headers.Authorization = `Bearer ${auth.accessToken}`
        return apiClient(config)
      }
    }
    return Promise.reject(error)
  },
)

/** The backend gives picture URLs relative to itself (`/static/...`). */
export function assetUrl(path: string | null): string | null {
  return path ? `${import.meta.env.VITE_API_BASE_URL}${path}` : null
}

export default apiClient
