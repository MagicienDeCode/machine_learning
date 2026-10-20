<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { assetUrl } from '@/api/client'
import type { BannerView } from '@/api/types'
import { useDisplay } from '@/utils/display'

// No official banner art is downloaded: the banner is drawn from the featured 5★ wish splash.
const props = defineProps<{ banner: BannerView }>()

const { t } = useI18n()
const { nameOf } = useDisplay()
const splash = computed(
  () => assetUrl(props.banner.banner_image_url) ?? assetUrl(props.banner.featured5.splash_url),
)
const element = computed(() => (props.banner.featured5.element ?? '').toLowerCase())
</script>

<template>
  <div class="banner" :class="`element-${element}`">
    <img v-if="splash" class="splash" :src="splash" :alt="banner.featured5.name" />
    <div class="shade" />
    <div class="text">
      <div class="version">
        {{ t('banners.version', { version: banner.version, phase: banner.phase }) }}
      </div>
      <div class="banner-name">{{ nameOf(banner) }}</div>
      <div class="featured5">{{ nameOf(banner.featured5) }}</div>
      <div class="featured4">
        <img
          v-for="item in banner.featured4"
          :key="item.slug"
          :src="assetUrl(item.icon_url) ?? undefined"
          :alt="item.name"
          :title="nameOf(item)"
        />
      </div>
      <div class="dates">{{ banner.start_date }} → {{ banner.end_date }}</div>
    </div>
  </div>
</template>

<style scoped>
.banner {
  position: relative;
  aspect-ratio: 16 / 8;
  border-radius: 14px;
  overflow: hidden;
  background: linear-gradient(120deg, #2a2f4a, #3b3354);
  border: 1px solid var(--border);
}

.element-anemo {
  background: linear-gradient(120deg, #1d3b3a, #3d7d6f);
}
.element-geo {
  background: linear-gradient(120deg, #3b2f1a, #8a6a2c);
}
.element-electro {
  background: linear-gradient(120deg, #2c1f45, #6b4aa3);
}
.element-dendro {
  background: linear-gradient(120deg, #1f3a1e, #5c8a33);
}
.element-hydro {
  background: linear-gradient(120deg, #172a4a, #3769a8);
}
.element-pyro {
  background: linear-gradient(120deg, #4a1d17, #b3522e);
}
.element-cryo {
  background: linear-gradient(120deg, #1d3348, #6aa3c8);
}

.splash {
  position: absolute;
  right: -6%;
  top: 0;
  height: 100%;
  width: 80%;
  object-fit: cover;
  object-position: center 20%;
}

.shade {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    rgba(10, 12, 22, 0.9) 0%,
    rgba(10, 12, 22, 0.55) 45%,
    transparent 75%
  );
}

.text {
  position: absolute;
  inset: 0;
  padding: 5% 6%;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-width: 60%;
}

.version {
  color: var(--text-dim);
  font-size: 12px;
}

.banner-name {
  font-size: clamp(16px, 2.4vw, 26px);
  font-weight: 700;
  color: var(--rarity-5-light);
  line-height: 1.2;
}

.featured5 {
  font-size: clamp(13px, 1.6vw, 18px);
}

.featured4 {
  display: flex;
  gap: 6px;
  margin-top: auto;
}

.featured4 img {
  width: clamp(32px, 4.5vw, 52px);
  aspect-ratio: 1;
  border-radius: 8px;
  background: linear-gradient(160deg, #4b3a72, var(--rarity-4));
  object-fit: cover;
}

.dates {
  color: var(--text-dim);
  font-size: 11px;
}
</style>
