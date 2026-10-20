<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { Button as AButton, Spin as ASpin, Switch as ASwitch, message } from 'ant-design-vue'
import { listBanners, makeWish } from '@/api/gacha'
import { extractErrorMessage, isInsufficientFunds } from '@/api/errors'
import type { BannerView, WishView } from '@/api/types'
import BannerArt from '@/components/BannerArt.vue'
import BannerDetailsDrawer from '@/components/BannerDetailsDrawer.vue'
import CurrencyIcon from '@/components/CurrencyIcon.vue'
import ItemCard from '@/components/ItemCard.vue'
import PityPanel from '@/components/PityPanel.vue'
import WishOverlay from '@/components/WishOverlay.vue'
import { useWalletStore } from '@/stores/wallet'

const props = defineProps<{ code: string }>()

const { t } = useI18n()
const walletStore = useWalletStore()
const banner = ref<BannerView | null>(null)
const autoBuy = ref(true)
const busy = ref(false)
const lastWish = ref<WishView | null>(null)
const detailsOpen = ref(false)

const fates = computed(() => walletStore.wallet?.intertwined_fates ?? 0)

onMounted(async () => {
  banner.value = (await listBanners()).find((b) => b.code === props.code) ?? null
  if (!walletStore.pity) {
    await walletStore.load()
  }
})

async function wish(count: 1 | 10) {
  busy.value = true
  try {
    const result = await makeWish(props.code, count, autoBuy.value)
    walletStore.applyWish(result)
    lastWish.value = result
  } catch (error) {
    message.error(
      isInsufficientFunds(error)
        ? t('wish.notEnough')
        : extractErrorMessage(error, t('wish.failed')),
    )
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <ASpin v-if="!banner" />
  <template v-else>
    <router-link :to="{ name: 'banners' }" class="back">← {{ t('wish.back') }}</router-link>
    <div class="layout">
      <div class="main">
        <BannerArt :banner="banner" />
        <div class="actions">
          <label class="auto-buy">
            <ASwitch v-model:checked="autoBuy" size="small" />
            {{ t('wish.autoBuy') }}
          </label>
          <AButton @click="detailsOpen = true">{{ t('wish.details') }}</AButton>
          <AButton size="large" :loading="busy" @click="wish(1)">
            {{ t('wish.one') }}
            <span class="cost"><CurrencyIcon currency="fate" :size="16" /> 1</span>
          </AButton>
          <AButton type="primary" size="large" :loading="busy" @click="wish(10)">
            {{ t('wish.ten') }}
            <span class="cost"><CurrencyIcon currency="fate" :size="16" /> 10</span>
          </AButton>
        </div>
        <p class="fates"><CurrencyIcon currency="fate" /> {{ t('currency.fates') }}: {{ fates }}</p>

        <h3>{{ t('wish.featured') }}</h3>
        <div class="featured">
          <ItemCard :item="banner.featured5" />
          <ItemCard v-for="item in banner.featured4" :key="item.slug" :item="item" />
        </div>
      </div>
      <aside class="side">
        <PityPanel v-if="walletStore.pity" :pity="walletStore.pity" />
      </aside>
    </div>
    <WishOverlay :wish="lastWish" @close="lastWish = null" />
    <BannerDetailsDrawer v-model:open="detailsOpen" :code="code" />
  </template>
</template>

<style scoped>
.back {
  display: inline-block;
  margin-bottom: 12px;
  text-decoration: none;
}

.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 20px;
}

@media (max-width: 860px) {
  .layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

.actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  flex-wrap: wrap;
}

.auto-buy {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-right: auto;
  color: var(--text-dim);
  font-size: 13px;
}

.cost {
  opacity: 0.75;
  margin-left: 4px;
}

.fates {
  text-align: right;
  color: var(--text-dim);
  margin: 6px 0 0;
}

h3 {
  margin: 20px 0 8px;
}

.featured {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
</style>
