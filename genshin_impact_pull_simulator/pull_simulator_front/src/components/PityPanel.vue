<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { Progress as AProgress, Tag as ATag } from 'ant-design-vue'
import type { PityView } from '@/api/types'
import { useDisplay } from '@/utils/display'

const props = defineProps<{ pity: PityView }>()

const { t } = useI18n()
const { percent } = useDisplay()
const pity5Percent = computed(() => Math.round((props.pity.pity5 / 90) * 100))
const pity4Percent = computed(() => Math.min(100, Math.round((props.pity.pity4 / 10) * 100)))
</script>

<template>
  <div class="panel pity">
    <h3>{{ t('pity.title') }}</h3>
    <div class="line">
      <span>{{ t('pity.pity5') }}</span>
      <strong class="rarity-5-text">{{ pity.pity5 }} / 90</strong>
    </div>
    <AProgress :percent="pity5Percent" :show-info="false" stroke-color="#f0c46a" size="small" />
    <div class="line">
      <span>{{ t('pity.pity4') }}</span>
      <strong class="rarity-4-text">{{ pity.pity4 }} / 10</strong>
    </div>
    <AProgress :percent="pity4Percent" :show-info="false" stroke-color="#c3a2f0" size="small" />
    <div class="line">
      <span>{{ t('pity.guaranteed5') }}</span>
      <ATag :color="pity.guaranteed5 ? 'gold' : 'default'">
        {{ pity.guaranteed5 ? t('pity.guaranteed') : t('pity.fiftyFifty') }}
      </ATag>
    </div>
    <div class="line">
      <span>{{ t('pity.guaranteed4') }}</span>
      <ATag :color="pity.guaranteed4 ? 'purple' : 'default'">
        {{ pity.guaranteed4 ? t('pity.guaranteed') : t('pity.fiftyFifty') }}
      </ATag>
    </div>
    <div class="line">
      <span>{{ t('pity.nextRate') }}</span>
      <strong>{{ percent(pity.next_5star_rate) }}</strong>
    </div>
    <div class="line">
      <span>{{ t('pity.toHard') }}</span>
      <strong>{{ pity.pulls_to_hard_pity }}</strong>
    </div>
  </div>
</template>

<style scoped>
h3 {
  margin: 0 0 8px;
}

.line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  font-size: 13px;
  color: var(--text-dim);
}

.line strong {
  color: var(--text);
  font-variant-numeric: tabular-nums;
}

.line strong.rarity-5-text {
  color: var(--rarity-5-light);
}

.line strong.rarity-4-text {
  color: var(--rarity-4-light);
}
</style>
