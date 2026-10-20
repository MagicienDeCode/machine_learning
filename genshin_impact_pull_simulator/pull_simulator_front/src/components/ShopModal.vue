<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  Button as AButton,
  InputNumber as AInputNumber,
  Modal as AModal,
  message,
} from 'ant-design-vue'
import { buyFates, resetProgress, topUp } from '@/api/gacha'
import { extractErrorMessage } from '@/api/errors'
import type { PayWith } from '@/api/types'
import { useWalletStore } from '@/stores/wallet'
import { useDisplay } from '@/utils/display'
import CurrencyIcon, { type Currency } from './CurrencyIcon.vue'

const open = defineModel<boolean>('open', { required: true })

const { t } = useI18n()
const walletStore = useWalletStore()
const { number } = useDisplay()

const TOP_UPS = [1600, 16000, 160000]
const PRICES: Record<PayWith, number> = { PRIMOGEM: 160, STARGLITTER: 5, STARDUST: 75 }
const ICON: Record<PayWith, Currency> = {
  PRIMOGEM: 'primogem',
  STARGLITTER: 'starglitter',
  STARDUST: 'stardust',
}
const counts = ref<Record<PayWith, number>>({ PRIMOGEM: 10, STARGLITTER: 1, STARDUST: 1 })
const busy = ref(false)

async function run(action: () => Promise<void>) {
  busy.value = true
  try {
    await action()
  } catch (error) {
    message.error(extractErrorMessage(error, t('shop.failed')))
  } finally {
    busy.value = false
  }
}

function doTopUp(amount: number) {
  return run(async () => {
    walletStore.setWallet(await topUp(amount))
    message.success(`+${number(amount)} ${t('currency.primogems')}`)
  })
}

function doBuy(payWith: PayWith) {
  return run(async () => {
    const count = counts.value[payWith]
    walletStore.setWallet(await buyFates(count, payWith))
    message.success(`+${count} ${t('currency.fates')}`)
  })
}

function doReset() {
  AModal.confirm({
    title: t('shop.reset'),
    content: t('shop.resetConfirm'),
    okText: t('common.confirm'),
    cancelText: t('common.cancel'),
    onOk: () =>
      run(async () => {
        await resetProgress()
        await walletStore.load()
        message.success(t('shop.done'))
      }),
  })
}
</script>

<template>
  <AModal v-model:open="open" :title="t('shop.title')" :footer="null" width="560px">
    <div v-if="walletStore.wallet" class="balances">
      <span><CurrencyIcon currency="primogem" /> {{ number(walletStore.wallet.primogems) }}</span>
      <span>
        <CurrencyIcon currency="fate" /> {{ number(walletStore.wallet.intertwined_fates) }}
      </span>
      <span><CurrencyIcon currency="starglitter" /> {{ walletStore.wallet.starglitter }}</span>
      <span><CurrencyIcon currency="stardust" /> {{ walletStore.wallet.stardust }}</span>
    </div>

    <section>
      <h3>{{ t('shop.topUp') }}</h3>
      <p class="hint">{{ t('shop.topUpHint') }}</p>
      <div class="row">
        <AButton v-for="amount in TOP_UPS" :key="amount" :loading="busy" @click="doTopUp(amount)">
          +{{ number(amount) }} <CurrencyIcon currency="primogem" :size="16" />
        </AButton>
      </div>
    </section>

    <section>
      <h3>{{ t('shop.buyFates') }}</h3>
      <div class="buy-row">
        <span class="price">
          1 <CurrencyIcon currency="fate" /> = {{ PRICES.PRIMOGEM }}
          <CurrencyIcon currency="primogem" />
        </span>
        <AInputNumber v-model:value="counts.PRIMOGEM" :min="1" :max="1000" />
        <AButton type="primary" :loading="busy" @click="doBuy('PRIMOGEM')">{{
          t('shop.buy')
        }}</AButton>
      </div>
    </section>

    <section>
      <h3>{{ t('shop.bargains') }}</h3>
      <div v-for="payWith in ['STARGLITTER', 'STARDUST'] as const" :key="payWith" class="buy-row">
        <span class="price">
          1 <CurrencyIcon currency="fate" /> = {{ PRICES[payWith] }}
          <CurrencyIcon :currency="ICON[payWith]" />
        </span>
        <AInputNumber
          v-model:value="counts[payWith]"
          :min="1"
          :max="payWith === 'STARDUST' ? 5 : 1000"
        />
        <AButton :loading="busy" @click="doBuy(payWith)">{{ t('shop.buy') }}</AButton>
      </div>
      <p v-if="walletStore.wallet" class="hint">
        {{ t('shop.stardustLeft', { n: walletStore.wallet.stardust_fates_left_this_month }) }}
      </p>
    </section>

    <section class="reset">
      <AButton danger size="small" :loading="busy" @click="doReset">{{ t('shop.reset') }}</AButton>
    </section>
  </AModal>
</template>

<style scoped>
section {
  margin-top: 16px;
}

h3 {
  margin: 0 0 6px;
  font-size: 15px;
}

.hint {
  color: var(--text-dim);
  margin: 0 0 8px;
  font-size: 13px;
}

.balances {
  display: flex;
  gap: 16px;
  align-items: center;
  flex-wrap: wrap;
  color: var(--text-dim);
}

.row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.buy-row {
  display: grid;
  grid-template-columns: 1fr 110px auto;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}

.reset {
  border-top: 1px solid var(--border);
  padding-top: 12px;
  text-align: right;
}
</style>
