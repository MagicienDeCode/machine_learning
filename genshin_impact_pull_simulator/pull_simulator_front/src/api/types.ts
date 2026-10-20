// Shapes of the backend's JSON (snake_case, as FastAPI sends it). See spec.md §5.

export interface AccessTokenResponse {
  access_token: string
  access_token_expires_at: string
}

export interface UserView {
  id: number
  email: string
  display_name: string
}

export interface ApiErrorResponse {
  timestamp: string
  status: number
  error: string
  message: string
  path: string
}

export interface Page<T> {
  items: T[]
  page: number
  size: number
  total: number
}

export type Rarity = 3 | 4 | 5
export type ItemType = 'CHARACTER' | 'WEAPON'
export type Outcome = 'WON_50_50' | 'LOST_50_50' | 'GUARANTEED'

export interface ItemView {
  slug: string
  name: string
  name_zh: string | null
  rarity: Rarity
  item_type: ItemType
  element: string | null
  weapon_type: string
  icon_url: string | null
  splash_url: string | null
}

export interface BannerView {
  code: string
  name: string
  name_zh: string | null
  version: string
  phase: number
  slot: number
  start_date: string
  end_date: string
  banner_image_url: string | null
  featured5: ItemView
  featured4: ItemView[]
}

export interface BannerDetailsView {
  banner: BannerView
  rates: {
    five_star_base: number
    five_star_consolidated: number
    five_star_soft_pity_start: number
    five_star_hard_pity: number
    four_star_base: number
    four_star_consolidated: number
    four_star_hard_pity: number
    featured_chance: number
  }
  rules: string[]
  pool: {
    featured5: ItemView
    featured4: ItemView[]
    standard5: ItemView[]
    off_banner4_characters: ItemView[]
    off_banner4_weapons: ItemView[]
    three_stars: ItemView[]
  }
}

export interface WalletView {
  primogems: number
  intertwined_fates: number
  starglitter: number
  stardust: number
  total_primogems_added: number
  total_pulls: number
  stardust_fates_left_this_month: number
}

export type PayWith = 'PRIMOGEM' | 'STARGLITTER' | 'STARDUST'

export interface PityView {
  pool: string
  pity5: number
  pity4: number
  guaranteed5: boolean
  guaranteed4: boolean
  next_5star_rate: number
  pulls_to_hard_pity: number
  pulls_to_soft_pity: number
}

export interface WishResultView {
  position: number
  item: ItemView
  rarity: Rarity
  pity: number | null
  featured: boolean
  outcome: Outcome | null
  is_new: boolean
  constellation_after: number | null
  starglitter: number
  stardust: number
}

export interface WishView {
  wish_id: string
  banner_code: string
  count: number
  max_rarity: Rarity
  fates_bought: number
  results: WishResultView[]
  pity_after: PityView
  wallet_after: Pick<WalletView, 'primogems' | 'intertwined_fates' | 'starglitter' | 'stardust'>
}

export interface HistoryEntryView {
  id: number
  wish_id: string
  banner_code: string
  banner_name: string
  banner_name_zh: string | null
  item: ItemView
  rarity: Rarity
  pity: number | null
  featured: boolean
  outcome: Outcome | null
  created_at: string
}

export interface StatsView {
  total_pulls: number
  fates_spent: number
  primogems_equivalent: number
  five_star_count: number
  four_star_count: number
  average_five_star_pity: number | null
  average_four_star_pity: number | null
  fifty_fifty: { won: number; lost: number; guaranteed: number }
  featured_five_star_count: number
  pulls_per_featured_five_star: number | null
  luckiest_five_star_pity: number | null
  unluckiest_five_star_pity: number | null
}

export interface InventoryEntryView {
  item: ItemView
  copies: number
  constellation: number | null
  first_obtained_at: string
  last_obtained_at: string
}
