<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { Drawer as ADrawer, Spin as ASpin } from 'ant-design-vue'
import { getBannerDetails } from '@/api/gacha'
import type { BannerDetailsView, ItemView } from '@/api/types'
import { useDisplay } from '@/utils/display'
import ItemCard from './ItemCard.vue'

const props = defineProps<{ code: string }>()
const open = defineModel<boolean>('open', { required: true })

const { t, tm, rt } = useI18n()
const { percent } = useDisplay()
const details = ref<BannerDetailsView | null>(null)

watch(open, async (isOpen) => {
  if (isOpen && details.value?.banner.code !== props.code) {
    details.value = await getBannerDetails(props.code)
  }
})

function sections(d: BannerDetailsView): [string, ItemView[]][] {
  return [
    ['details.featured5', [d.pool.featured5]],
    ['details.featured4', d.pool.featured4],
    ['details.standard5', d.pool.standard5],
    ['details.off4c', d.pool.off_banner4_characters],
    ['details.off4w', d.pool.off_banner4_weapons],
    ['details.three', d.pool.three_stars],
  ]
}
</script>

<template>
  <ADrawer v-model:open="open" :title="t('details.title')" width="min(720px, 100vw)">
    <ASpin v-if="!details" />
    <template v-else>
      <h3>{{ t('details.rates') }}</h3>
      <p>
        {{
          t('details.fiveBase', {
            base: percent(details.rates.five_star_base),
            total: percent(details.rates.five_star_consolidated),
            hard: details.rates.five_star_hard_pity,
          })
        }}
      </p>
      <p>
        {{
          t('details.fourBase', {
            base: percent(details.rates.four_star_base),
            total: percent(details.rates.four_star_consolidated),
            hard: details.rates.four_star_hard_pity,
          })
        }}
      </p>

      <h3>{{ t('details.rules') }}</h3>
      <ul>
        <li v-for="(rule, i) in tm('details.ruleLines')" :key="i">{{ rt(rule) }}</li>
      </ul>

      <h3>{{ t('details.pool') }}</h3>
      <div v-for="[label, items] in sections(details)" :key="label" class="section">
        <h4>{{ t(label) }} ({{ items.length }})</h4>
        <div class="grid">
          <ItemCard v-for="item in items" :key="item.slug" :item="item" size="small" />
        </div>
      </div>
    </template>
  </ADrawer>
</template>

<style scoped>
h3 {
  margin: 16px 0 6px;
}

h4 {
  margin: 12px 0 6px;
  color: var(--text-dim);
}

ul {
  padding-left: 18px;
  color: var(--text-dim);
}

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
</style>
