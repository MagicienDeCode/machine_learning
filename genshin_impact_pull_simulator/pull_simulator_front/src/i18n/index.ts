import { createI18n } from 'vue-i18n'
import en from './locales/en'
import zh from './locales/zh'
import fr from './locales/fr'

export const LOCALES = ['zh', 'en', 'fr'] as const
export type Locale = (typeof LOCALES)[number]

function savedLocale(): Locale {
  try {
    const saved = localStorage.getItem('locale')
    if (saved && (LOCALES as readonly string[]).includes(saved)) {
      return saved as Locale
    }
  } catch {
    // storage blocked: fall back to the default
  }
  return 'zh'
}

const i18n = createI18n({
  legacy: false,
  locale: savedLocale(),
  fallbackLocale: 'en',
  messages: { en, zh, fr },
})

export default i18n
