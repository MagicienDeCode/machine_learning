# pull_simulator_front

Vue 3 + Vite + ant-design-vue front of the Genshin Impact wish simulator. It only shows what
the backend (`../pull_simulator`) decides: no draw and no rule run here.

- Banners (`/`) — the six Archon banners, drawn from the featured character's splash art.
- Wish (`/banners/:code`) — 1 or 10 pulls, auto-buy of fates with primogems, pity panel, the
  banner's details (rates, rules, item pool). A meteor in the colour of the best rarity, then
  the results.
- History (`/history`), Stats (`/stats`), Characters (`/inventory`).
- Shop (header) — free primogems, fates, Paimon's Bargains, start over.

Interface in Chinese, English or French (header); item and banner names follow it.

It needs the backend running; its URL is `VITE_API_BASE_URL` in `.env`
(`http://localhost:8000`). Login: `traveler@test.local` / `123456`.

```sh
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production build
npm run lint
```
