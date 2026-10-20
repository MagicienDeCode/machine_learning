<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { assetUrl } from '@/api/client'

export type Currency = 'primogem' | 'fate' | 'starglitter' | 'stardust'

// The icons are downloaded by the backend (app/tools/download_assets.py) into static/currency/.
// Until they are there, a coloured glyph stands in.
const props = withDefaults(defineProps<{ currency: Currency; size?: number }>(), { size: 18 })

const { t } = useI18n()
const failed = ref(false)

const GLYPHS: Record<Currency, string> = {
  primogem: '◆',
  fate: '✧',
  starglitter: '✦',
  stardust: '✧',
}
const LABELS: Record<Currency, string> = {
  primogem: 'currency.primogems',
  fate: 'currency.fates',
  starglitter: 'currency.starglitter',
  stardust: 'currency.stardust',
}

const src = computed(() => assetUrl(`/static/currency/${props.currency}.png`) ?? undefined)
const label = computed(() => t(LABELS[props.currency]))
</script>

<template>
  <img
    v-if="!failed"
    class="currency-icon"
    :src="src"
    :alt="label"
    :title="label"
    :width="size"
    :height="size"
    @error="failed = true"
  />
  <span
    v-else
    class="currency-glyph"
    :class="currency"
    :title="label"
    :style="{ fontSize: `${size * 0.8}px` }"
    >{{ GLYPHS[currency] }}</span
  >
</template>

<style scoped>
.currency-icon {
  display: inline-block;
  vertical-align: -0.25em;
  object-fit: contain;
}

.currency-glyph {
  display: inline-block;
  width: 1.25em;
  text-align: center;
}

.primogem {
  color: #8fd3ff;
}
.fate {
  color: #f4a7d8;
}
.starglitter {
  color: #f4d38a;
}
.stardust {
  color: #a9c9f0;
}
</style>
