<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { Spin as ASpin } from 'ant-design-vue'
import { listBanners } from '@/api/gacha'
import type { BannerView } from '@/api/types'
import BannerArt from '@/components/BannerArt.vue'

const { t } = useI18n()
const banners = ref<BannerView[] | null>(null)

onMounted(async () => {
  banners.value = await listBanners()
})
</script>

<template>
  <h1 class="page-title">{{ t('banners.title') }}</h1>
  <p class="page-subtitle">{{ t('banners.subtitle') }}</p>
  <ASpin v-if="!banners" />
  <div v-else class="grid">
    <router-link
      v-for="banner in banners"
      :key="banner.code"
      :to="{ name: 'wish', params: { code: banner.code } }"
      class="banner-link"
    >
      <BannerArt :banner="banner" />
    </router-link>
  </div>
</template>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 480px), 1fr));
  gap: 16px;
}

.banner-link {
  display: block;
  transition: transform 0.15s ease;
}

.banner-link:hover {
  transform: translateY(-3px);
}
</style>
