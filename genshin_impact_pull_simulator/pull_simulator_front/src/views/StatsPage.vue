<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { Empty as AEmpty, Spin as ASpin } from 'ant-design-vue'
import { getStats } from '@/api/gacha'
import type { StatsView } from '@/api/types'
import { useDisplay } from '@/utils/display'

const { t } = useI18n()
const { number } = useDisplay()
const stats = ref<StatsView | null>(null)

onMounted(async () => {
  stats.value = await getStats()
})

const dash = (value: number | null) => (value === null ? '–' : String(value))

const tiles = computed(() => {
  const s = stats.value
  if (!s) return []
  return [
    { label: 'stats.totalPulls', value: number(s.total_pulls) },
    { label: 'stats.primogems', value: number(s.primogems_equivalent) },
    { label: 'stats.fiveStars', value: String(s.five_star_count), tone: 5 },
    { label: 'stats.featuredFiveStars', value: String(s.featured_five_star_count), tone: 5 },
    { label: 'stats.fourStars', value: String(s.four_star_count), tone: 4 },
    { label: 'stats.avgFive', value: dash(s.average_five_star_pity), tone: 5 },
    { label: 'stats.avgFour', value: dash(s.average_four_star_pity), tone: 4 },
    { label: 'stats.won', value: String(s.fifty_fifty.won) },
    { label: 'stats.lost', value: String(s.fifty_fifty.lost) },
    { label: 'stats.guaranteed', value: String(s.fifty_fifty.guaranteed) },
    { label: 'stats.luckiest', value: dash(s.luckiest_five_star_pity), tone: 5 },
    { label: 'stats.unluckiest', value: dash(s.unluckiest_five_star_pity), tone: 5 },
  ]
})
</script>

<template>
  <h1 class="page-title">{{ t('stats.title') }}</h1>
  <ASpin v-if="!stats" />
  <AEmpty v-else-if="stats.total_pulls === 0" :description="t('stats.empty')" />
  <template v-else>
    <div class="hero panel">
      <div class="hero-label">{{ t('stats.pullsPerFeatured') }}</div>
      <div class="hero-value">{{ dash(stats.pulls_per_featured_five_star) }}</div>
    </div>
    <div class="tiles">
      <div v-for="tile in tiles" :key="tile.label" class="tile panel">
        <div class="label">{{ t(tile.label) }}</div>
        <div class="value" :class="tile.tone ? `rarity-${tile.tone}-text` : ''">
          {{ tile.value }}
        </div>
      </div>
    </div>
  </template>
</template>

<style scoped>
.hero {
  text-align: center;
  margin: 16px 0;
  padding: 24px;
}

.hero-label {
  color: var(--text-dim);
}

.hero-value {
  font-size: 56px;
  font-weight: 700;
  color: var(--rarity-5-light);
  font-variant-numeric: tabular-nums;
}

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  gap: 12px;
}

.label {
  color: var(--text-dim);
  font-size: 13px;
}

.value {
  font-size: 26px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
</style>
