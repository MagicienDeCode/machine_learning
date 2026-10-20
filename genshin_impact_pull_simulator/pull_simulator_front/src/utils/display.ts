import { useI18n } from 'vue-i18n'
import type { Rarity } from '@/api/types'

interface Named {
  name: string
  name_zh: string | null
}

/** Display helpers that follow the current locale (Chinese names when the UI is in Chinese). */
export function useDisplay() {
  const { locale } = useI18n()

  function nameOf(thing: Named): string {
    return locale.value === 'zh' && thing.name_zh ? thing.name_zh : thing.name
  }

  function percent(rate: number, digits = 1): string {
    return `${(rate * 100).toFixed(digits)} %`
  }

  function number(value: number): string {
    return value.toLocaleString(locale.value)
  }

  function dateTime(iso: string): string {
    // The backend sends naive UTC.
    return new Date(iso.endsWith('Z') ? iso : `${iso}Z`).toLocaleString(locale.value)
  }

  return { nameOf, percent, number, dateTime }
}

export function stars(rarity: Rarity): string {
  return '★'.repeat(rarity)
}
