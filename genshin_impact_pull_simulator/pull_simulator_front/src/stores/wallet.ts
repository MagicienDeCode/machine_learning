import { defineStore } from 'pinia'
import { getPity, getWallet } from '@/api/gacha'
import type { PityView, WalletView, WishView } from '@/api/types'

/** Wallet and pity as last seen from the backend; shown in the header and on the wish page. */
export const useWalletStore = defineStore('wallet', {
  state: () => ({
    wallet: null as WalletView | null,
    pity: null as PityView | null,
  }),

  actions: {
    async load() {
      const [wallet, pity] = await Promise.all([getWallet(), getPity()])
      this.wallet = wallet
      this.pity = pity
    },

    setWallet(wallet: WalletView) {
      this.wallet = wallet
    },

    applyWish(wish: WishView) {
      this.pity = wish.pity_after
      if (this.wallet) {
        this.wallet = {
          ...this.wallet,
          ...wish.wallet_after,
          total_pulls: this.wallet.total_pulls + wish.count,
        }
      }
    },

    clear() {
      this.wallet = null
      this.pity = null
    },
  },
})
