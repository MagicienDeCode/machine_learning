<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import {
  Alert as AAlert,
  Button as AButton,
  Form as AForm,
  FormItem as AFormItem,
  Input as AInput,
  InputPassword as AInputPassword,
} from 'ant-design-vue'
import { extractErrorMessage } from '@/api/errors'
import { useAuthStore } from '@/stores/auth'

const { t } = useI18n()
const auth = useAuthStore()
const router = useRouter()
const route = useRoute()

const form = reactive({ email: 'traveler@test.local', password: '' })
const loading = ref(false)
const error = ref<string | null>(null)

async function submit() {
  loading.value = true
  error.value = null
  try {
    await auth.login(form.email, form.password)
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
    router.push(redirect)
  } catch (e) {
    error.value = extractErrorMessage(e, t('login.failed'))
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login panel">
    <h1 class="page-title">{{ t('login.title') }}</h1>
    <p class="page-subtitle">{{ t('login.hint') }}</p>
    <AAlert v-if="error" type="error" :message="error" show-icon class="error" />
    <AForm :model="form" layout="vertical" @finish="submit">
      <AFormItem :label="t('login.email')" name="email">
        <AInput v-model:value="form.email" autocomplete="username" />
      </AFormItem>
      <AFormItem :label="t('login.password')" name="password">
        <AInputPassword v-model:value="form.password" autocomplete="current-password" />
      </AFormItem>
      <AButton type="primary" html-type="submit" block :loading="loading">
        {{ t('login.submit') }}
      </AButton>
    </AForm>
  </div>
</template>

<style scoped>
.login {
  max-width: 380px;
  margin: 48px auto;
  padding: 28px;
}

.error {
  margin-bottom: 16px;
}
</style>
