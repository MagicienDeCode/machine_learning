<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { Button as AButton, Select as ASelect } from 'ant-design-vue'
import { useAuthStore } from '@/stores/auth'
import { useWalletStore } from '@/stores/wallet'
import { LOCALES, type Locale } from '@/i18n'
import { useDisplay } from '@/utils/display'
import CurrencyIcon from './CurrencyIcon.vue'
import ShopModal from './ShopModal.vue'

const { t, locale } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const walletStore = useWalletStore()
const { number } = useDisplay()
const shopOpen = ref(false)

const localeLabels: Record<Locale, string> = { zh: '中文', en: 'English', fr: 'Français' }
const localeOptions = LOCALES.map((value) => ({ value, label: localeLabels[value] }))

function changeLocale(value: Locale) {
  locale.value = value
  try {
    localStorage.setItem('locale', value)
  } catch {
    // storage blocked: the choice only lasts this visit
  }
}

watch(
  () => auth.isAuthenticated,
  (loggedIn) => {
    if (loggedIn) {
      walletStore.load()
    } else {
      walletStore.clear()
    }
  },
  { immediate: true },
)

async function logout() {
  await auth.logout()
  router.push({ name: 'login' })
}

const links = [
  { name: 'banners', label: 'nav.banners' },
  { name: 'history', label: 'nav.history' },
  { name: 'stats', label: 'nav.stats' },
  { name: 'inventory', label: 'nav.inventory' },
]
</script>

<template>
  <header class="header">
    <div class="header-inner">
      <router-link :to="{ name: 'banners' }" class="brand">✦ {{ t('app.title') }}</router-link>

      <nav v-if="auth.isAuthenticated" class="nav">
        <router-link
          v-for="link in links"
          :key="link.name"
          :to="{ name: link.name }"
          class="nav-link"
        >
          {{ t(link.label) }}
        </router-link>
      </nav>

      <div class="right">
        <template v-if="auth.isAuthenticated && walletStore.wallet">
          <span class="chip" :title="t('currency.primogems')">
            <CurrencyIcon currency="primogem" /> {{ number(walletStore.wallet.primogems) }}
          </span>
          <span class="chip" :title="t('currency.fates')">
            <CurrencyIcon currency="fate" /> {{ number(walletStore.wallet.intertwined_fates) }}
          </span>
          <AButton type="primary" size="small" @click="shopOpen = true">{{
            t('header.shop')
          }}</AButton>
        </template>
        <ASelect
          :value="locale"
          size="small"
          :options="localeOptions"
          style="width: 100px"
          @change="(v) => changeLocale(v as Locale)"
        />
        <AButton v-if="auth.isAuthenticated" size="small" @click="logout">
          {{ t('header.logout') }}
        </AButton>
      </div>
    </div>
    <ShopModal v-model:open="shopOpen" />
  </header>
</template>

<style scoped>
.header {
  background: rgba(21, 24, 38, 0.92);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(6px);
}

.header-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}

.brand {
  color: var(--gold);
  font-weight: 700;
  font-size: 18px;
  text-decoration: none;
  white-space: nowrap;
}

.nav {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.nav-link {
  color: var(--text-dim);
  text-decoration: none;
  padding: 4px 10px;
  border-radius: 6px;
}

.nav-link.router-link-exact-active {
  color: var(--text);
  background: var(--bg-card);
}

.right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.chip {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 10px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
</style>
