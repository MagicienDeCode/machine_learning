<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  Radio as ARadio,
  RadioGroup as ARadioGroup,
  Table as ATable,
  Tag as ATag,
} from 'ant-design-vue'
import { getHistory } from '@/api/gacha'
import type { HistoryEntryView, Page, Rarity } from '@/api/types'
import { stars, useDisplay } from '@/utils/display'

const { t } = useI18n()
const { nameOf, dateTime } = useDisplay()

const rarity = ref<Rarity | 0>(0)
const page = ref(0)
const size = 20
const data = ref<Page<HistoryEntryView> | null>(null)
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    data.value = await getHistory(page.value, size, rarity.value || undefined)
  } finally {
    loading.value = false
  }
}

onMounted(load)

function changeRarity() {
  page.value = 0
  load()
}

const columns = computed(() => [
  { title: t('history.time'), key: 'time', width: 190 },
  { title: t('history.item'), key: 'item' },
  { title: t('history.rarity'), key: 'rarity', width: 100 },
  { title: t('history.pity'), key: 'pity', width: 80 },
  { title: t('history.outcome'), key: 'outcome', width: 130 },
  { title: t('history.banner'), key: 'banner' },
])

const pagination = computed(() => ({
  current: page.value + 1,
  pageSize: size,
  total: data.value?.total ?? 0,
  showSizeChanger: false,
}))

function onChange(p: { current?: number }) {
  page.value = (p.current ?? 1) - 1
  load()
}
</script>

<template>
  <h1 class="page-title">{{ t('history.title') }}</h1>
  <ARadioGroup v-model:value="rarity" button-style="solid" class="filter" @change="changeRarity">
    <ARadio :value="0">{{ t('history.all') }}</ARadio>
    <ARadio :value="5">5★</ARadio>
    <ARadio :value="4">4★</ARadio>
    <ARadio :value="3">3★</ARadio>
  </ARadioGroup>
  <ATable
    :columns="columns"
    :data-source="data?.items ?? []"
    :loading="loading"
    :pagination="pagination"
    row-key="id"
    size="middle"
    :scroll="{ x: 760 }"
    @change="onChange"
  >
    <template #bodyCell="{ column, record }">
      <template v-if="column.key === 'time'">{{ dateTime(record.created_at) }}</template>
      <span v-else-if="column.key === 'item'" :class="`rarity-${record.rarity}-text`">
        {{ nameOf(record.item) }}
      </span>
      <span v-else-if="column.key === 'rarity'" :class="`rarity-${record.rarity}-text`">
        {{ stars(record.rarity) }}
      </span>
      <template v-else-if="column.key === 'pity'">{{ record.pity ?? '' }}</template>
      <template v-else-if="column.key === 'outcome'">
        <ATag
          v-if="record.outcome && record.rarity === 5"
          :color="record.outcome === 'LOST_50_50' ? 'default' : 'gold'"
        >
          {{ t(`outcome.${record.outcome}`) }}
        </ATag>
      </template>
      <template v-else-if="column.key === 'banner'">
        {{ nameOf({ name: record.banner_name, name_zh: record.banner_name_zh }) }}
      </template>
    </template>
  </ATable>
</template>

<style scoped>
.filter {
  margin: 12px 0 16px;
}
</style>
