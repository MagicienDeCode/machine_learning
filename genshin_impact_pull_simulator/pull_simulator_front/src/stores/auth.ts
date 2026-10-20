import { defineStore } from 'pinia'
import apiClient from '@/api/client'
import type { AccessTokenResponse, UserView } from '@/api/types'

interface AuthState {
  accessToken: string | null
  user: UserView | null
  /** Set once tryRestoreSession() has run, so app boot only ever attempts one silent refresh. */
  initialized: boolean
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    accessToken: null,
    user: null,
    initialized: false,
  }),

  getters: {
    isAuthenticated: (state) => state.accessToken !== null && state.user !== null,
  },

  actions: {
    clearSession() {
      this.accessToken = null
      this.user = null
    },

    async fetchMe() {
      const { data } = await apiClient.get<UserView>('/api/users/me')
      this.user = data
      return data
    },

    async login(email: string, password: string) {
      const { data } = await apiClient.post<AccessTokenResponse>('/api/auth/login', {
        email,
        password,
      })
      this.accessToken = data.access_token
      await this.fetchMe()
    },

    /** Silent: used at app boot (restore a session from the refresh cookie) and by the axios
     *  interceptor on a 401. Never throws -- returns whether a session came back. */
    async refresh(): Promise<boolean> {
      try {
        const { data } = await apiClient.post<AccessTokenResponse>('/api/auth/refresh')
        this.accessToken = data.access_token
        if (!this.user) {
          await this.fetchMe()
        }
        return true
      } catch {
        this.clearSession()
        return false
      }
    },

    async logout() {
      try {
        await apiClient.post('/api/auth/logout')
      } finally {
        this.clearSession()
      }
    },

    async tryRestoreSession() {
      if (this.initialized) {
        return
      }
      await this.refresh()
      this.initialized = true
    },
  },
})
