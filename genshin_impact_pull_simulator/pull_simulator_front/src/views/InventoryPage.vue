<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { Empty as AEmpty, Segmented as ASegmented, Spin as ASpin } from 'ant-design-vue'
import { getInventory } from '@/api/gacha'
import type { InventoryEntryView, ItemType } from '@/api/types'
import ItemCard from '@/components/ItemCard.vue'

const { t } = useI18n()
const type = ref<ItemType>('CHARACTER')
const entries = ref<InventoryEntryView[] | null>(null)

async function load() {
  entries.value = null
  entries.value = await getInventory(type.value)
}

onMounted(load)
watch(type, load)
</script>

<template>
  <h1 class="page-title">{{ t('inventory.title') }}</h1>
  <ASegmented
    v-model:value="type"
    class="filter"
    :options="[
      { value: 'CHARACTER', label: t('inventory.characters') },
      { value: 'WEAPON', label: t('inventory.weapons') },
    ]"
  />
  <ASpin v-if="!entries" />
  <AEmpty v-else-if="entries.length === 0" :description="t('inventory.empty')" />
  <div v-else class="grid">
    <ItemCard v-for="entry in entries" :key="entry.item.slug" :item="entry.item">
      <template #badges>
        <span v-if="entry.constellation !== null" class="badge">C{{ entry.constellation }}</span>
        <span v-else />
        <span class="badge">{{ t('inventory.copies', { n: entry.copies }) }}</span>
      </template>
    </ItemCard>
  </div>
</template>

<style scoped>
.filter {
  margin: 12px 0 16px;
}

.grid {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.badge {
  font-size: 11px;
  font-weight: 700;
  border-radius: 4px;
  padding: 0 4px;
  background: rgba(0, 0, 0, 0.6);
  color: var(--rarity-5-light);
}
</style>
