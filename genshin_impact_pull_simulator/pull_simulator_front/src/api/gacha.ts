import apiClient from '@/api/client'
import type {
  BannerDetailsView,
  BannerView,
  HistoryEntryView,
  InventoryEntryView,
  ItemType,
  Page,
  PayWith,
  PityView,
  Rarity,
  StatsView,
  WalletView,
  WishView,
} from '@/api/types'

export async function listBanners(): Promise<BannerView[]> {
  return (await apiClient.get<BannerView[]>('/api/banners')).data
}

export async function getBannerDetails(code: string): Promise<BannerDetailsView> {
  return (await apiClient.get<BannerDetailsView>(`/api/banners/${code}/details`)).data
}

export async function makeWish(code: string, count: 1 | 10, autoBuy: boolean): Promise<WishView> {
  const body = { count, auto_buy: autoBuy }
  return (await apiClient.post<WishView>(`/api/banners/${code}/wishes`, body)).data
}

export async function getWallet(): Promise<WalletView> {
  return (await apiClient.get<WalletView>('/api/wallet')).data
}

export async function topUp(amount: number): Promise<WalletView> {
  return (await apiClient.post<WalletView>('/api/wallet/top-ups', { amount })).data
}

export async function buyFates(count: number, payWith: PayWith): Promise<WalletView> {
  const body = { count, pay_with: payWith }
  return (await apiClient.post<WalletView>('/api/wallet/fates', body)).data
}

export async function getPity(): Promise<PityView> {
  return (await apiClient.get<PityView>('/api/pity')).data
}

export async function getHistory(
  page: number,
  size: number,
  rarity?: Rarity,
): Promise<Page<HistoryEntryView>> {
  const params = { page, size, rarity }
  return (await apiClient.get<Page<HistoryEntryView>>('/api/wishes/history', { params })).data
}

export async function getStats(): Promise<StatsView> {
  return (await apiClient.get<StatsView>('/api/wishes/stats')).data
}

export async function getInventory(type?: ItemType): Promise<InventoryEntryView[]> {
  const params = { type }
  return (await apiClient.get<InventoryEntryView[]>('/api/inventory', { params })).data
}

export async function resetProgress(): Promise<void> {
  await apiClient.post('/api/users/me/reset')
}
