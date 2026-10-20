import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import i18n from './i18n'
import { useAuthStore } from './stores/auth'
import 'ant-design-vue/dist/reset.css'
import './assets/theme.css'

const app = createApp(App)

app.use(createPinia())
app.use(i18n)

// Restore a session from the httpOnly refresh cookie (if any) before the router's first
// navigation, so the auth guard sees the final auth state instead of a "logged out" flash.
const authStore = useAuthStore()
authStore.tryRestoreSession().finally(() => {
  app.use(router)
  app.mount('#app')
})
