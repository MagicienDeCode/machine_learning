<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { Button as AButton } from 'ant-design-vue'
import { assetUrl } from '@/api/client'
import type { WishView } from '@/api/types'
import { useDisplay } from '@/utils/display'
import CurrencyIcon from './CurrencyIcon.vue'
import ItemCard from './ItemCard.vue'

// Only plays back what the backend already decided: a meteor in the colour of the best rarity,
// then the results, best first.
const props = defineProps<{ wish: WishView | null }>()
const emit = defineEmits<{ close: [] }>()

const { t } = useI18n()
const { nameOf } = useDisplay()
const phase = ref<'meteor' | 'results'>('meteor')
let timer: ReturnType<typeof setTimeout> | undefined

watch(
  () => props.wish,
  (wish) => {
    clearTimeout(timer)
    if (wish) {
      phase.value = 'meteor'
      timer = setTimeout(() => (phase.value = 'results'), 1600)
    }
  },
  { immediate: true },
)
onBeforeUnmount(() => clearTimeout(timer))

const sorted = computed(() =>
  props.wish
    ? [...props.wish.results].sort(
        (a, b) =>
          b.rarity - a.rarity ||
          Number(b.item.item_type === 'CHARACTER') - Number(a.item.item_type === 'CHARACTER') ||
          a.position - b.position,
      )
    : [],
)

// The splash of the best character pulled, behind the results.
const highlight = computed(() =>
  sorted.value.find((r) => r.item.item_type === 'CHARACTER' && r.rarity >= 4),
)

function skip() {
  clearTimeout(timer)
  phase.value = 'results'
}
</script>

<template>
  <Teleport to="body">
    <div v-if="wish" class="overlay" @click.self="phase === 'meteor' && skip()">
      <div v-if="phase === 'meteor'" class="meteor-stage" @click="skip">
        <div class="meteor" :class="`meteor-${wish.max_rarity}`" />
        <AButton class="skip" size="small" ghost @click.stop="skip">{{ t('result.skip') }}</AButton>
      </div>

      <div v-else class="results">
        <img
          v-if="highlight?.item.splash_url"
          class="backdrop"
          :src="assetUrl(highlight.item.splash_url) ?? undefined"
          alt=""
        />
        <div class="results-inner">
          <h2>{{ t('result.title') }}</h2>
          <div v-if="highlight && highlight.rarity === 5" class="highlight rarity-5-text">
            ★★★★★ {{ nameOf(highlight.item) }}
            <span v-if="highlight.outcome" class="outcome">{{
              t(`outcome.${highlight.outcome}`)
            }}</span>
          </div>
          <div class="grid">
            <ItemCard v-for="result in sorted" :key="result.position" :item="result.item">
              <template #badges>
                <span v-if="result.is_new" class="badge new">{{ t('result.new') }}</span>
                <span v-else-if="result.constellation_after !== null" class="badge cons">
                  {{ t('result.constellation', { n: result.constellation_after }) }}
                </span>
                <span v-else />
                <span v-if="result.pity !== null" class="badge pity">{{ result.pity }}</span>
              </template>
              <div class="meta">
                <span v-if="result.outcome && result.rarity === 5" class="tag">
                  {{ t(`outcome.${result.outcome}`) }}
                </span>
                <span v-if="result.starglitter" class="refund"
                  >+{{ result.starglitter }} <CurrencyIcon currency="starglitter" :size="14"
                /></span>
                <span v-if="result.stardust" class="refund dust"
                  >+{{ result.stardust }} <CurrencyIcon currency="stardust" :size="14"
                /></span>
              </div>
            </ItemCard>
          </div>
          <AButton type="primary" size="large" class="close" @click="emit('close')">
            {{ t('result.close') }}
          </AButton>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: radial-gradient(circle at 50% 40%, #232a4a, #090b14 75%);
  overflow-y: auto;
}

.meteor-stage {
  position: absolute;
  inset: 0;
  cursor: pointer;
}

.meteor {
  position: absolute;
  top: -10%;
  left: 85%;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  animation: fall 1.4s cubic-bezier(0.55, 0.05, 0.6, 0.95) forwards;
}

.meteor::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 260px;
  height: 6px;
  transform-origin: 0 50%;
  transform: translateY(-50%) rotate(-50deg);
  border-radius: 6px;
}

.meteor-3 {
  background: #cfe6ff;
  box-shadow: 0 0 30px 12px var(--rarity-3-light);
}
.meteor-3::after {
  background: linear-gradient(90deg, var(--rarity-3-light), transparent);
}
.meteor-4 {
  background: #f0e2ff;
  box-shadow: 0 0 34px 14px var(--rarity-4-light);
}
.meteor-4::after {
  background: linear-gradient(90deg, var(--rarity-4-light), transparent);
}
.meteor-5 {
  background: #fff3d1;
  box-shadow: 0 0 44px 20px var(--rarity-5-light);
}
.meteor-5::after {
  background: linear-gradient(90deg, var(--rarity-5-light), transparent);
}

@keyframes fall {
  to {
    top: 55%;
    left: 30%;
    transform: scale(2.4);
  }
}

.skip {
  position: absolute;
  top: 20px;
  right: 20px;
}

.results {
  position: relative;
  min-height: 100%;
}

.backdrop {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center 20%;
  opacity: 0.22;
  pointer-events: none;
}

.results-inner {
  position: relative;
  max-width: 1000px;
  margin: 0 auto;
  padding: 48px 16px;
  text-align: center;
  animation: appear 0.4s ease-out;
}

@keyframes appear {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
}

h2 {
  color: var(--text);
  margin: 0 0 8px;
}

.highlight {
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 16px;
}

.outcome {
  font-size: 14px;
  margin-left: 8px;
  color: var(--text-dim);
}

.grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin: 16px 0 28px;
}

.badge {
  font-size: 10px;
  font-weight: 700;
  border-radius: 4px;
  padding: 0 4px;
  line-height: 16px;
}

.new {
  background: #e05a47;
  color: #fff;
}

.cons {
  background: rgba(0, 0, 0, 0.6);
  color: var(--rarity-5-light);
}

.pity {
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 4px;
  padding: 0 4px 6px;
  font-size: 10px;
}

.tag {
  color: var(--rarity-5-light);
}

.refund {
  color: #f4d38a;
}

.refund.dust {
  color: #a9c9f0;
}
</style>
