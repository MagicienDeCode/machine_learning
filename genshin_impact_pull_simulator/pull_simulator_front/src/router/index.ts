import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import BannersPage from '../views/BannersPage.vue'
import HistoryPage from '../views/HistoryPage.vue'
import InventoryPage from '../views/InventoryPage.vue'
import LoginPage from '../views/LoginPage.vue'
import StatsPage from '../views/StatsPage.vue'
import WishPage from '../views/WishPage.vue'

declare module 'vue-router' {
  interface RouteMeta {
    guestOnly?: boolean
  }
}

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'banners', component: BannersPage },
    { path: '/banners/:code', name: 'wish', component: WishPage, props: true },
    { path: '/history', name: 'history', component: HistoryPage },
    { path: '/stats', name: 'stats', component: StatsPage },
    { path: '/inventory', name: 'inventory', component: InventoryPage },
    { path: '/login', name: 'login', component: LoginPage, meta: { guestOnly: true } },
    { path: '/:pathMatch(.*)*', redirect: { name: 'banners' } },
  ],
})

// Every page but the login needs a session: wishes, wallet and pity are per player.
router.beforeEach((to) => {
  const auth = useAuthStore()
  if (to.meta.guestOnly) {
    return auth.isAuthenticated ? { name: 'banners' } : true
  }
  if (!auth.isAuthenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
})

export default router
