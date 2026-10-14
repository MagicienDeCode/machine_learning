#!/usr/bin/env python3
"""李想汇率 — a pygame currency converter styled after the 极简汇率 mini app.

Rows are fixed: CNY, EUR, USD, JPY, RUB. Click a row (or use Up/Down) and type an
amount; the other rows convert as you type. Rates come from Sina Finance (real-time)
with open.er-api.com (daily) as fallback, refreshed every minute.

Keys: 0-9 and . to type, Backspace to delete, Up/Down to switch row, R to refresh, Esc to quit.
The arrow button also refreshes.
"""

import math
import os
import threading
import time
import warnings
from functools import lru_cache

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import pygame  # noqa: E402
import pygame.gfxdraw  # noqa: E402

from exchange_rate import fetch_open_er, fetch_sina  # noqa: E402

# (code, Chinese name, symbol), in display order
CURRENCIES = [
    ("CNY", "人民币", "¥"),
    ("EUR", "欧元", "€"),
    ("USD", "美元", "$"),
    ("JPY", "日元", "¥"),
    ("RUB", "俄罗斯卢布", "₽"),
]
# Fetchers quote 1 EUR in these currencies
TARGETS = [code for code, _, _ in CURRENCIES if code != "EUR"]
REFRESH_SECONDS = 60
MAX_DIGITS = 12
MAX_DECIMALS = 4

W, H = 420, 620
PAD = 24
HEADER_H = 100
ROW_H = 90
FLAG = 36

WHITE = (255, 255, 255)
TEXT = (51, 51, 51)
GREY = (154, 154, 154)
LIGHT = (229, 229, 229)
SELECTED_BG = (245, 245, 247)
CARET = (52, 120, 246)

FONT_FILES = {
    "latin": ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc"],
    "cjk": ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Light.ttc"],
    "cjk_bold": ["/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"],
}


@lru_cache(maxsize=None)
def font(kind, size):
    for path in FONT_FILES[kind]:
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("pingfangsc,hiraginosansgb,arialunicodems", size)


# ---------------------------------------------------------------- flags

def _star(cx, cy, r, rot=-math.pi / 2):
    points = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.382
        a = rot + k * math.pi / 5
        points.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return points


def _draw_china(s, n):
    red, yellow = (222, 41, 16), (255, 222, 0)
    s.fill(red)
    bx, by = n * 0.28, n * 0.32
    pygame.draw.polygon(s, yellow, _star(bx, by, n * 0.17))
    for fx, fy in [(0.54, 0.13), (0.66, 0.26), (0.66, 0.43), (0.54, 0.56)]:
        cx, cy = n * fx, n * fy
        # small stars point at the big one
        pygame.draw.polygon(s, yellow, _star(cx, cy, n * 0.06, math.atan2(by - cy, bx - cx)))


def _draw_eu(s, n):
    s.fill((0, 51, 153))
    c = n / 2
    for k in range(12):
        a = k * math.pi / 6
        pygame.draw.polygon(s, (255, 204, 0), _star(c + n * 0.3 * math.cos(a), c + n * 0.3 * math.sin(a), n * 0.065))


def _draw_usa(s, n):
    stripe = n / 13
    for k in range(13):
        color = (178, 34, 52) if k % 2 == 0 else WHITE
        pygame.draw.rect(s, color, (0, round(k * stripe), n, math.ceil(stripe) + 1))
    cw, ch = n * 0.46, stripe * 7
    pygame.draw.rect(s, (60, 59, 110), (0, 0, cw, ch))
    for row in range(4):
        for col in range(5):
            pygame.draw.circle(s, WHITE, (cw * (col + 0.5) / 5, ch * (row + 0.5) / 4), n * 0.022)


def _draw_japan(s, n):
    s.fill((243, 243, 243))
    pygame.draw.circle(s, (188, 0, 45), (n / 2, n / 2), n * 0.25)


def _draw_russia(s, n):
    for k, color in enumerate([WHITE, (0, 57, 166), (213, 43, 30)]):
        pygame.draw.rect(s, color, (0, round(k * n / 3), n, math.ceil(n / 3) + 1))


FLAG_PAINTERS = {"CNY": _draw_china, "EUR": _draw_eu, "USD": _draw_usa, "JPY": _draw_japan, "RUB": _draw_russia}


@lru_cache(maxsize=None)
def flag(code, size=FLAG, supersample=4):
    """Paint the flag at a higher resolution, round the corners, then scale down for anti-aliasing."""
    n = size * supersample
    surf = pygame.Surface((n, n), pygame.SRCALPHA)
    FLAG_PAINTERS[code](surf, n)
    radius = n // 9
    if code in ("JPY", "RUB"):  # light flags need an outline on the white background
        pygame.draw.rect(surf, LIGHT, surf.get_rect(), width=supersample, border_radius=radius)
    mask = pygame.Surface((n, n), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return pygame.transform.smoothscale(surf, (size, size))


# ---------------------------------------------------------------- rates

class RateFeed:
    """Fetches rates on a background thread so the UI never blocks on the network."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rates = None  # {code: units per 1 EUR}
        self.status = "正在获取汇率…"
        self.loading = False
        self.last_fetch = 0.0

    def refresh(self):
        if self.loading:
            return
        self.loading = True
        self.last_fetch = time.monotonic()
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self):
        sources = [("新浪财经 实时", fetch_sina), ("open.er-api 每日", fetch_open_er)]
        try:
            for name, fetch in sources:
                try:
                    as_of, rates = fetch(TARGETS)
                except (OSError, ValueError, KeyError):
                    continue
                with self.lock:
                    self.rates = {"EUR": 1.0, **rates}
                    self.status = f"{name} · {as_of}"
                return
            with self.lock:
                self.status = "获取汇率失败，按 R 重试"
        finally:
            self.loading = False

    def snapshot(self):
        with self.lock:
            return self.rates, self.status


# ---------------------------------------------------------------- formatting

def format_value(value):
    if 0 < value < 1:
        return f"{value:,.4f}"
    return f"{value:,.2f}"


def plain_value(value):
    """Value as editable input text: no separators, no trailing zeros."""
    text = f"{value:.4f}" if 0 < value < 1 else f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_input(text):
    whole, dot, frac = text.partition(".")
    return f"{int(whole or 0):,}{dot}{frac}"


# ---------------------------------------------------------------- app

class App:
    def __init__(self):
        self.feed = RateFeed()
        self.selected = 0  # 100 CNY
        self.text = "100"
        self.fresh = False  # True right after switching rows: next keystroke replaces the value
        self.refresh_button = pygame.Rect(W - PAD - 32, 34, 32, 32)

    # -- state

    def amount(self):
        try:
            return float(self.text)
        except ValueError:
            return 0.0

    def converted(self, rates, index):
        base = CURRENCIES[self.selected][0]
        return self.amount() / rates[base] * rates[CURRENCIES[index][0]]

    def select(self, index):
        index %= len(CURRENCIES)
        if index == self.selected:
            return
        rates, _ = self.feed.snapshot()
        if rates:
            self.text = plain_value(self.converted(rates, index))
        self.selected = index
        self.fresh = True

    def type_char(self, ch):
        if self.fresh:
            self.text, self.fresh = "", False
        whole, dot, frac = self.text.partition(".")
        if ch == ".":
            if not dot:
                self.text = (self.text or "0") + "."
            return
        if len(whole) + len(frac) >= MAX_DIGITS or (dot and len(frac) >= MAX_DECIMALS):
            return
        self.text = ch if self.text == "0" else self.text + ch

    def backspace(self):
        self.text = "" if self.fresh else self.text[:-1]
        self.fresh = False

    # -- events

    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_UP:
                self.select(self.selected - 1)
            elif event.key in (pygame.K_DOWN, pygame.K_TAB):
                self.select(self.selected + 1)
            elif event.key == pygame.K_BACKSPACE:
                self.backspace()
            elif event.key == pygame.K_r:
                self.feed.refresh()
            elif event.unicode and (event.unicode.isdigit() or event.unicode == "."):
                self.type_char(event.unicode)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            if self.refresh_button.collidepoint(x, y):
                self.feed.refresh()
            elif HEADER_H <= y < HEADER_H + ROW_H * len(CURRENCIES):
                self.select((y - HEADER_H) // ROW_H)
        return True

    def tick(self):
        if time.monotonic() - self.feed.last_fetch > REFRESH_SECONDS:
            self.feed.refresh()

    # -- drawing

    def draw(self, screen):
        screen.fill(WHITE)
        rates, status = self.feed.snapshot()
        title = font("cjk_bold", 24).render("李想汇率", True, (17, 17, 17))
        screen.blit(title, (PAD, self.refresh_button.centery - title.get_height() // 2))
        self._draw_refresh_button(screen)
        for i in range(len(CURRENCIES)):
            self._draw_row(screen, i, rates)
        if self.feed.loading and rates:
            status = "正在更新…"
        label = font("cjk", 12).render(status, True, GREY)
        screen.blit(label, label.get_rect(center=(W // 2, H - 30)))

    def _draw_refresh_button(self, screen):
        rect = self.refresh_button
        pygame.draw.rect(screen, WHITE, rect, border_radius=rect.h // 2)
        pygame.draw.rect(screen, LIGHT, rect, width=1, border_radius=rect.h // 2)
        # circular refresh arrow, spinning while loading
        cx, cy = rect.center
        spin = time.monotonic() * 6 if self.feed.loading else 0
        r = 7
        arc = [(cx + r * math.cos(spin + a / 10), cy + r * math.sin(spin + a / 10)) for a in range(5, 58)]
        pygame.draw.lines(screen, TEXT, False, arc, 2)
        end, a = arc[-1], spin + 5.8
        tangent = (-math.sin(a), math.cos(a))
        normal = (math.cos(a), math.sin(a))
        tip = (end[0] + tangent[0] * 4, end[1] + tangent[1] * 4)
        wing1 = (end[0] + normal[0] * 3.5, end[1] + normal[1] * 3.5)
        wing2 = (end[0] - normal[0] * 3.5, end[1] - normal[1] * 3.5)
        pygame.gfxdraw.aapolygon(screen, [tip, wing1, wing2], TEXT)
        pygame.gfxdraw.filled_polygon(screen, [tip, wing1, wing2], TEXT)

    def _draw_row(self, screen, i, rates):
        code, name, symbol = CURRENCIES[i]
        top = HEADER_H + i * ROW_H
        selected = i == self.selected
        if selected:
            pygame.draw.rect(screen, SELECTED_BG, (0, top, W, ROW_H))

        mid = top + ROW_H // 2
        screen.blit(flag(code), (PAD, mid - FLAG // 2))
        label = font("latin", 20).render(code, True, TEXT)
        label_x = PAD + FLAG + 16
        screen.blit(label, (label_x, mid - label.get_height() // 2))

        if selected:
            number = format_input(self.text)
        elif rates:
            number = format_value(self.converted(rates, i))
        else:
            number = "—"
        right = W - PAD - 6
        room = right - (label_x + label.get_width() + 40)
        for size in (30, 26, 22, 18, 15):
            num = font("latin", size).render(number, True, TEXT if selected else GREY)
            if num.get_width() <= room:
                break
        num_rect = num.get_rect(right=right, centery=mid - 8)
        screen.blit(num, num_rect)
        if selected and time.monotonic() % 1 < 0.5:
            pygame.draw.rect(screen, CARET, (right + 3, num_rect.y + 4, 2, num_rect.h - 8))

        # The CJK font lacks ₽, so render the symbol with the Latin font
        sym = font("latin", 11).render(symbol, True, GREY)
        sym_rect = sym.get_rect(right=right, top=num_rect.bottom + 2)
        sub = font("cjk", 11).render(name, True, GREY)
        screen.blit(sub, sub.get_rect(right=sym_rect.x - 3, centery=sym_rect.centery))
        screen.blit(sym, sym_rect)


def main():
    pygame.init()
    pygame.display.set_caption("李想汇率")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    app = App()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or not app.handle(event):
                running = False
        app.tick()
        app.draw(screen)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
