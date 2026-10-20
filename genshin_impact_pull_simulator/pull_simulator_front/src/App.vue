<script setup lang="ts">
import { ConfigProvider, theme } from 'ant-design-vue'
import AppHeader from './components/AppHeader.vue'

const themeConfig = {
  algorithm: theme.darkAlgorithm,
  token: {
    colorPrimary: '#d9a84a',
    colorBgBase: '#151826',
    borderRadius: 8,
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif",
  },
}
</script>

<template>
  <ConfigProvider :theme="themeConfig">
    <div class="app-shell">
      <AppHeader />
      <main class="app-content">
        <router-view />
      </main>
      <footer class="app-footer">{{ $t('app.footer') }}</footer>
    </div>
  </ConfigProvider>
</template>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-content {
  flex: 1;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px 16px 48px;
}

.app-footer {
  text-align: center;
  padding: 16px;
  color: var(--text-dim);
  font-size: 12px;
}
</style>
