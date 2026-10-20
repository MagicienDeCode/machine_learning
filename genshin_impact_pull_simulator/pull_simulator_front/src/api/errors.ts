import { isAxiosError } from 'axios'
import type { ApiErrorResponse } from '@/api/types'

/** Every error response from the backend has the same body (see app/shared/errors.py). */
export function extractErrorMessage(error: unknown, fallback: string): string {
  if (isAxiosError<ApiErrorResponse>(error) && error.response?.data?.message) {
    return error.response.data.message
  }
  return fallback
}

/** True when the backend refused because the wallet does not cover it. */
export function isInsufficientFunds(error: unknown): boolean {
  return isAxiosError(error) && error.response?.status === 402
}
