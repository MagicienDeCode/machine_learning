<script setup lang="ts">
import { computed } from 'vue'
import { assetUrl } from '@/api/client'
import type { ItemView } from '@/api/types'
import { stars, useDisplay } from '@/utils/display'

const props = withDefaults(defineProps<{ item: ItemView; size?: 'small' | 'medium' }>(), {
  size: 'medium',
})

const { nameOf } = useDisplay()
const icon = computed(() => assetUrl(props.item.icon_url))
</script>

<template>
  <div class="card" :class="[`rarity-${item.rarity}`, size]">
    <div class="art">
      <img v-if="icon" :src="icon" :alt="item.name" loading="lazy" />
      <span v-else class="placeholder">{{ item.name.charAt(0) }}</span>
      <div class="badges"><slot name="badges" /></div>
    </div>
    <div class="stars">{{ stars(item.rarity) }}</div>
    <div class="name" :title="nameOf(item)">{{ nameOf(item) }}</div>
    <slot />
  </div>
</template>

<style scoped>
.card {
  width: 112px;
  border-radius: 10px;
  overflow: hidden;
  background: var(--bg-card);
  text-align: center;
  position: relative;
  border: 1px solid var(--border);
}

.card.small {
  width: 84px;
}

.art {
  aspect-ratio: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.rarity-5 .art {
  background: linear-gradient(160deg, #6e4a2e, var(--rarity-5));
}

.rarity-4 .art {
  background: linear-gradient(160deg, #4b3a72, var(--rarity-4));
}

.rarity-3 .art {
  background: linear-gradient(160deg, #31466a, var(--rarity-3));
}

.art img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.placeholder {
  font-size: 36px;
  color: rgba(255, 255, 255, 0.8);
}

.badges {
  position: absolute;
  top: 4px;
  left: 4px;
  right: 4px;
  display: flex;
  justify-content: space-between;
  pointer-events: none;
}

.stars {
  font-size: 11px;
  letter-spacing: -1px;
  margin-top: 2px;
}

.rarity-5 .stars {
  color: var(--rarity-5-light);
}

.rarity-4 .stars {
  color: var(--rarity-4-light);
}

.rarity-3 .stars {
  color: var(--rarity-3-light);
}

.name {
  font-size: 12px;
  padding: 0 4px 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.small .name {
  font-size: 11px;
}
</style>
