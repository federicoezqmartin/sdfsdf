"""Margaritas: animación de arte ASCII de un ramo de margaritas en un florero.

Cae un chorro de agua en el florero, crecen los tallos con sus hojas, se
abren las margaritas una por una, se mecen y sueltan algunos pétalos, y al
final todo se disuelve y vuelve a empezar. Todo se calcula con fórmulas en
NumPy; no se usa ninguna imagen ni video de origen.

Uso:
    pip install numpy pygame
    python margaritas.py

Teclas: ESPACIO pausa, R reinicia, ESC sale.
"""

import argparse
import os

import numpy as np
import pygame

# Rampa de caracteres ordenada de menos a más "tinta".
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
ROWS = 120
FONT_SIZE = 10
FPS = 30
DURATION = 18.0  # segundos; después vuelve a empezar
ASPECT = 9 / 16  # ventana vertical, como un video de celular
MOUTH_Y = 0.42  # altura de la boca del florero
BASE_Y = 0.95  # apoyo del florero sobre la mesa


def smooth(a, b, x):
    """Smoothstep: 0 antes de a, 1 después de b, transición suave entre ambos."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash(ix, iy):
    h = np.sin(ix * 127.1 + iy * 311.7) * 43758.5453
    return h - np.floor(h)


def noise(x, y):
    """Value noise 2D en [0, 1]."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    a, b = _hash(ix, iy), _hash(ix + 1, iy)
    c, d = _hash(ix, iy + 1), _hash(ix + 1, iy + 1)
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def rgb(r, g, b):
    return np.array([r, g, b], np.float32)


def bezier(p0, p1, p2, s):
    """Punto de una curva de Bézier cuadrática para los parámetros s (array)."""
    s = np.asarray(s, np.float32)[..., None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2


# Ramo: (x, y, radio) del centro de cada margarita.
BOUQUET = [
    (0.00, -0.66, 0.200), (-0.30, -0.47, 0.180), (0.31, -0.45, 0.185),
    (-0.13, -0.17, 0.190), (0.17, -0.14, 0.180), (-0.40, -0.02, 0.150),
    (0.41, 0.01, 0.150),
]


class Daisy:
    def __init__(self, i, x, y, radius, rng):
        self.x, self.y, self.radius = x, y, radius
        self.base = np.array([np.clip(x * 0.18, -0.08, 0.08), MOUTH_Y + 0.02], np.float32)
        self.top = np.array([x, y], np.float32)
        self.ctrl = np.array([self.base[0] + (x - self.base[0]) * 0.15,
                              (MOUTH_Y + y) / 2 + 0.06], np.float32)
        # Las de los costados miran hacia afuera: se ven "achatadas".
        self.facing = np.arctan2(y - MOUTH_Y, x) + np.pi / 2
        self.squash = 0.92 - 0.38 * min(abs(x) / 0.45, 1.0) + rng.uniform(-0.05, 0.05)
        self.n_petals = int(rng.integers(18, 23))
        self.petal_len = rng.uniform(0.86, 1.08, self.n_petals).astype(np.float32)
        self.petal_off = rng.uniform(0, 1)
        self.spin = rng.uniform(-0.4, 0.4)
        self.phase = rng.uniform(0, 2 * np.pi)
        self.t_grow = 2.6 + 0.33 * i + rng.uniform(0, 0.25)
        self.t_bloom = self.t_grow + 2.1 + rng.uniform(0, 0.5)
        self.leaf_side = 1 if i % 2 else -1
        self.stem_green = rgb(0.22, 0.58, 0.20) * rng.uniform(0.85, 1.1)

    def curve(self, t):
        """Puntos de control del tallo en el instante t (con el balanceo)."""
        sway = 0.014 * np.sin(1.1 * t + self.phase) * float(smooth(self.t_grow, self.t_grow + 2.5, t))
        top = self.top + np.array([sway, 0.0], np.float32)
        ctrl = self.ctrl + np.array([sway * 0.4, 0.0], np.float32)
        return self.base, ctrl, top


class Scene:
    def __init__(self, cols, rows, cell_w, cell_h):
        self.cols, self.rows = cols, rows
        self.w, self.h = cols * cell_w, rows * cell_h
        self.cell_w, self.cell_h = cell_w, cell_h
        xs = (np.arange(cols) + 0.5) * cell_w
        ys = (np.arange(rows) + 0.5) * cell_h
        u, v = np.meshgrid((xs - self.w / 2) / (self.h / 2), (ys - self.h / 2) / (self.h / 2))
        self.u, self.v = u.astype(np.float32), v.astype(np.float32)
        self.half_w = self.w / self.h  # u va de -half_w a +half_w

        rng = np.random.default_rng(11)
        self.daisies = [Daisy(i, x, y, r, rng) for i, (x, y, r) in enumerate(BOUQUET)]
        rng.shuffle(self.daisies)  # orden de crecimiento al azar
        for i, d in enumerate(self.daisies):
            d.t_grow = 2.6 + 0.45 * i + rng.uniform(0, 0.25)
            d.t_bloom = d.t_grow + 2.1 + rng.uniform(0, 0.5)
        # Para dibujar: las de más abajo quedan adelante.
        self.draw_order = sorted(self.daisies, key=lambda d: d.y)

        self.cell_jitter = rng.random((rows, cols)).astype(np.float32)
        n = 28
        self.petal_src = rng.integers(0, len(self.daisies), n)
        self.petal_t0 = np.sort(rng.uniform(10.0, 15.5, n))
        self.petal_drift = rng.uniform(-0.08, 0.08, n)
        self.petal_phase = rng.uniform(0, 2 * np.pi, n)
        self.petal_speed = rng.uniform(0.10, 0.18, n)

    # -- utilidades de dibujo ---------------------------------------------

    def _box(self, x0, y0, x1, y1):
        """Rebanadas de filas/columnas que cubren un rectángulo en (u, v)."""
        c0 = int(np.floor((x0 * self.h / 2 + self.w / 2) / self.cell_w)) - 1
        c1 = int(np.ceil((x1 * self.h / 2 + self.w / 2) / self.cell_w)) + 1
        r0 = int(np.floor((y0 * self.h / 2 + self.h / 2) / self.cell_h)) - 1
        r1 = int(np.ceil((y1 * self.h / 2 + self.h / 2) / self.cell_h)) + 1
        return (slice(max(r0, 0), max(min(r1, self.rows), 0)),
                slice(max(c0, 0), max(min(c1, self.cols), 0)))

    def _cell(self, x, y):
        c = int((x * self.h / 2 + self.w / 2) / self.cell_w)
        r = int((y * self.h / 2 + self.h / 2) / self.cell_h)
        return r, c

    @staticmethod
    def _paint(buf_l, buf_c, box, mask, lum, col):
        """Pinta lum/col sobre los buffers donde mask (0..1) lo indique."""
        m = mask[..., None]
        buf_l[box] = buf_l[box] * (1 - mask) + lum * mask
        buf_c[box] = buf_c[box] * (1 - m) + col * m

    # -- capas -------------------------------------------------------------

    def _table(self, L, C):
        box = self._box(-self.half_w, BASE_Y - 0.01, self.half_w, 1.0)
        u, v = self.u[box], self.v[box]
        grain = noise(u * 3.0, v * 40.0)
        shadow = np.exp(-((u + 0.05) / 0.30) ** 2)
        lum = (0.10 + 0.08 * grain) * (1 - 0.6 * shadow)
        mask = smooth(BASE_Y - 0.005, BASE_Y + 0.01, v)
        self._paint(L, C, box, mask, lum,
                    np.broadcast_to(rgb(0.55, 0.36, 0.22), u.shape + (3,)))

    def _water(self, L, C, t):
        if not 0.4 < t < 3.6:
            return
        head = -1.05 + (t - 0.4) * 1.9
        tail = -1.05 + max(t - 2.4, 0.0) * 1.9
        box = self._box(-0.04, -1.0, 0.04, MOUTH_Y)
        u, v = self.u[box], self.v[box]
        wobble = 0.004 * np.sin(v * 35.0 + t * 20.0)
        d = np.abs(u - wobble)
        on = (v > tail) & (v < np.minimum(head, MOUTH_Y))
        mask = on * (1 - smooth(0.006, 0.014, d))
        lum = 0.55 + 0.35 * noise(v * 30.0 - t * 25.0, u * 50.0)
        self._paint(L, C, box, mask.astype(np.float32), lum,
                    np.broadcast_to(rgb(0.45, 0.75, 1.0), u.shape + (3,)))

    def _stem(self, L, C, d, t):
        g = float(smooth(d.t_grow, d.t_grow + 2.2, t))
        if g <= 0.0:
            return
        p0, p1, p2 = d.curve(t)
        pts = bezier(p0, p1, p2, np.linspace(0.0, g, 24))
        box = self._box(pts[:, 0].min() - 0.03, pts[:, 1].min() - 0.03,
                        pts[:, 0].max() + 0.03, pts[:, 1].max() + 0.03)
        u, v = self.u[box], self.v[box]
        a, b = pts[:-1], pts[1:]
        ab = b - a
        px = u[..., None] - a[:, 0]
        py = v[..., None] - a[:, 1]
        k = np.clip((px * ab[:, 0] + py * ab[:, 1]) / ((ab ** 2).sum(1) + 1e-9), 0.0, 1.0)
        dist = np.hypot(px - k * ab[:, 0], py - k * ab[:, 1]).min(-1)
        width = 0.0085
        mask = 1 - smooth(width * 0.7, width * 1.3, dist)
        lum = 0.45 + 0.35 * (1 - dist / width) + 0.1 * noise(u * 80, v * 80)
        self._paint(L, C, box, mask, np.clip(lum, 0, 1),
                    np.broadcast_to(d.stem_green, u.shape + (3,)))

        # Hojas: una o dos por tallo, se despliegan cuando el tallo las pasa.
        for s_leaf, side, length in ((0.42, d.leaf_side, 0.11), (0.62, -d.leaf_side, 0.08)):
            grow = float(smooth(s_leaf + 0.05, s_leaf + 0.35, g))
            if grow <= 0.0:
                continue
            base = bezier(p0, p1, p2, s_leaf)
            tan = bezier(p0, p1, p2, s_leaf + 0.01) - base
            ang = np.arctan2(tan[1], tan[0]) + side * 0.95
            direc = np.array([np.cos(ang), np.sin(ang)], np.float32)
            ln = length * grow
            tip = base + direc * ln
            box = self._box(min(base[0], tip[0]) - 0.04, min(base[1], tip[1]) - 0.04,
                            max(base[0], tip[0]) + 0.04, max(base[1], tip[1]) + 0.04)
            u, v = self.u[box], self.v[box]
            du, dv = u - base[0], v - base[1]
            along = du * direc[0] + dv * direc[1]
            across = -du * direc[1] + dv * direc[0]
            x = np.clip(along / ln, 0.0, 1.0)
            half = 0.30 * ln * np.clip(np.sin(np.pi * x), 0, 1) ** 0.8 * (1 - 0.3 * x)
            inside = (along > 0) & (along < ln) & (np.abs(across) < half)
            rib = np.exp(-(across / 0.004) ** 2)
            veins = 0.5 + 0.5 * np.sin((along - np.abs(across) * 1.5) * 220.0)
            lum = 0.40 + 0.35 * (1 - np.abs(across) / (half + 1e-6)) + 0.12 * veins - 0.25 * rib
            col = rgb(0.30, 0.72, 0.26) * (0.85 + 0.25 * x)[..., None]
            self._paint(L, C, box, inside.astype(np.float32), np.clip(lum, 0, 1), col)

    def _vase(self, L, C, t):
        reveal = float(smooth(0.1, 1.4, t))
        box = self._box(-0.30, MOUTH_Y - 0.03, 0.30, BASE_Y + 0.01)
        u, v = self.u[box], self.v[box]
        prof_y = [MOUTH_Y, 0.47, 0.53, 0.66, 0.78, 0.88, BASE_Y]
        prof_w = [0.135, 0.115, 0.105, 0.17, 0.215, 0.20, 0.15]
        w = np.interp(v, prof_y, prof_w)
        inside = (np.abs(u) < w) & (v > MOUTH_Y) & (v < BASE_Y)
        nx = np.clip(u / w, -1, 1)
        body = np.sqrt(1 - nx ** 2)
        light = 0.25 + 0.55 * body + 0.25 * np.clip(-nx, 0, 1)
        bands = 0.5 + 0.5 * np.sin(v * 95.0)
        streak = np.exp(-((nx + 0.45) / 0.10) ** 2)
        lum = np.clip(light * (0.75 + 0.25 * bands) + 0.35 * streak, 0, 1)
        col = rgb(0.18, 0.32, 0.95) + (rgb(0.65, 0.80, 1.0) - rgb(0.18, 0.32, 0.95)) * (bands * 0.5 + streak)[..., None]
        # Borde dorado en la boca.
        rim = np.abs((v - MOUTH_Y) / 0.018) ** 2 + (u / 0.14) ** 2
        rim_m = (rim < 1.0) & (rim > 0.45)
        lum = np.where(rim_m, 0.85, lum)
        col = np.where(rim_m[..., None], rgb(1.0, 0.80, 0.35), col)
        rows_from_bottom = (BASE_Y - v) / (BASE_Y - MOUTH_Y + 0.03)
        shown = rows_from_bottom < reveal * 1.05
        mask = ((inside | rim_m) & shown).astype(np.float32)
        self._paint(L, C, box, mask, lum, col)

    def _head(self, L, C, O, d, t):
        g = float(smooth(d.t_grow, d.t_grow + 2.2, t))
        if g < 0.85:
            return
        bloom = float(smooth(d.t_bloom, d.t_bloom + 1.6, t))
        _, _, top = d.curve(t)
        cx, cy = top
        R = d.radius
        box = self._box(cx - R * 1.2, cy - R * 1.2, cx + R * 1.2, cy + R * 1.2)
        u, v = self.u[box], self.v[box]
        du, dv = u - cx, v - cy
        cf, sf = np.cos(d.facing), np.sin(d.facing)
        a = du * cf + dv * sf
        b = (-du * sf + dv * cf) / d.squash
        r = np.hypot(a, b)
        th = np.arctan2(b, a) + d.spin * bloom

        # Pimpollo verde antes de abrirse.
        bud_r = 0.028 * float(smooth(0.85, 1.0, g)) * (1 - bloom)
        if bud_r > 0.002:
            m = 1 - smooth(bud_r * 0.8, bud_r * 1.1, r)
            lum = 0.45 + 0.4 * (1 - r / bud_r) + 0.1 * noise(a * 90, b * 90)
            self._paint(L, C, box, m, np.clip(lum, 0, 1),
                        np.broadcast_to(rgb(0.45, 0.75, 0.30), u.shape + (3,)))
        if bloom <= 0.0:
            return

        n = d.n_petals
        pos = (th + np.pi) / (2 * np.pi) * n + d.petal_off
        k = np.floor(pos).astype(int) % n
        f = pos - np.floor(pos) - 0.5  # -0.5..0.5 dentro de cada pétalo
        rc = 0.30 * R
        rp = R * (0.25 + 0.75 * bloom) * d.petal_len[k]
        x = np.clip(r / rp, 0.0, 1.0)
        # Corona de pétalos: cada pétalo termina en punta redondeada y se
        # separa del vecino con una línea más oscura.
        sep = (2 * np.abs(f)) ** 4
        petal = r < rp * (1 - 0.28 * (2 * np.abs(f)) ** 2)
        lum = 1.25 - 0.40 * sep - 0.25 * np.exp(-np.clip(r - rc, 0, None) / (0.10 * R))
        lum = lum * (0.90 + 0.10 * float(np.cos(d.facing - 2.3)))
        shade = np.clip(1 - x * 1.5, 0, 1)[..., None]
        col = rgb(0.98, 0.97, 1.0) + (rgb(0.70, 0.72, 0.95) - rgb(0.98, 0.97, 1.0)) * shade
        flash = np.exp(-((t - d.t_bloom - 1.6) / 0.25) ** 2)
        self._paint(L, C, box, petal.astype(np.float32), np.clip(lum + 0.3 * flash, 0, 1), col)
        # Líneas entre pétalos con caracteres que siguen la dirección del rayo.
        ray = (np.degrees(np.arctan2(dv, du)) % 180.0 + 22.5) // 45
        ray_char = np.array([RAMP.index(c) for c in "-\\|/"])[ray.astype(int) % 4]
        o = O[box]
        o[petal] = -1
        lines = petal & (sep > 0.35) & (r > rc * 1.3) & (r < rp * 0.9)
        o[lines] = ray_char[lines]

        # Centro amarillo con textura de puntitos.
        disk = r < rc * (0.6 + 0.4 * bloom)
        O[box][disk] = -1
        dots = noise(a * 260, b * 260)
        dome = np.clip(1 - (r / rc) ** 2, 0, 1)
        lum = 0.65 + 0.30 * dome + 0.15 * dots
        ring = smooth(0.65, 1.0, r / rc)[..., None]
        col = rgb(1.0, 0.82, 0.12) + (rgb(0.90, 0.48, 0.05) - rgb(1.0, 0.82, 0.12)) * ring
        self._paint(L, C, box, disk.astype(np.float32), np.clip(lum, 0, 1), col)

    def _falling_petals(self, idx, color, t, fade):
        for j in range(len(self.petal_t0)):
            age = t - self.petal_t0[j]
            if age < 0:
                continue
            d = self.daisies[self.petal_src[j]]
            if t < d.t_bloom + 1.6:
                continue
            x = d.x + 0.6 * d.radius * np.cos(self.petal_phase[j]) + self.petal_drift[j] * age \
                + 0.03 * np.sin(age * 2.2 + self.petal_phase[j])
            y = d.y + 0.4 * d.radius * np.sin(self.petal_phase[j]) + self.petal_speed[j] * age
            if y > BASE_Y:
                continue
            r, c = self._cell(x, y)
            if 0 <= r < self.rows and 0 <= c < self.cols:
                idx[r, c] = RAMP.index("~" if int(age * 4) % 2 else "'")
                color[r, c] = np.array([0.95, 0.95, 1.0]) * fade

    # -- cuadro completo ---------------------------------------------------

    def frame(self, t):
        """Devuelve (índices de carácter, color RGB uint8) para el instante t."""
        L = np.zeros((self.rows, self.cols), np.float32)
        C = np.zeros((self.rows, self.cols, 3), np.float32)
        O = np.full((self.rows, self.cols), -1, np.int32)  # caracteres forzados

        self._table(L, C)
        self._water(L, C, t)
        for d in self.daisies:
            self._stem(L, C, d, t)
        self._vase(L, C, t)
        for d in self.draw_order:
            self._head(L, C, O, d, t)

        # Barrido de luz diagonal cuando el ramo ya está completo.
        sweep_pos = -1.6 + 3.2 * (t - 11.0) / 3.0
        L = np.clip(L * (1 + 0.35 * np.exp(-((self.u * 0.7 - self.v - sweep_pos) / 0.12) ** 2)), 0, 1)

        # Disolución al final: cada celda se apaga en un momento distinto.
        t_out = DURATION - 2.6 + 1.8 * self.cell_jitter
        gone = 1 - smooth(t_out, t_out + 0.4, t)
        fade = float(smooth(0.0, 0.5, t))
        L = L * gone * fade

        n = len(RAMP)
        idx = (L ** 0.7 * (n - 1)).astype(np.int32)
        rng = np.random.default_rng(int(t * 12))
        idx = np.where(idx > 1, idx + rng.integers(-2, 3, idx.shape), idx)
        idx = np.clip(idx, 0, n - 1)

        # Bordes con caracteres según su orientación (- / | \), como en el original.
        gy, gx = np.gradient(L)
        mag = np.hypot(gx, gy)
        ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0
        edge_chars = np.array([RAMP.index(c) for c in "-\\|/"])
        sel = edge_chars[((ang + 22.5) // 45).astype(int) % 4]
        # Solo en la silueta: celdas encendidas que tocan el fondo vacío.
        empty = L < 0.06
        near_empty = np.zeros_like(empty)
        near_empty[1:, :] |= empty[:-1, :]
        near_empty[:-1, :] |= empty[1:, :]
        near_empty[:, 1:] |= empty[:, :-1]
        near_empty[:, :-1] |= empty[:, 1:]
        edge = (mag > 0.18) & ~empty & near_empty
        idx = np.where(edge, sel, idx)
        idx = np.where((O >= 0) & (L > 0.1), O, idx)

        color = C * (0.45 + 0.55 * L)[..., None]
        self._falling_petals(idx, color, t, float(np.mean(gone)) * fade)
        return idx, (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Render con pygame
# --------------------------------------------------------------------------

def build_glyphs():
    """Rasteriza cada carácter de la rampa una sola vez: (n, alto, ancho) en 0..1."""
    name = pygame.font.match_font(
        "consolas,dejavusansmono,menlo,jetbrainsmono,couriernew,liberationmono",
        bold=True,
    )
    font = pygame.font.Font(name, FONT_SIZE)
    cell_w, cell_h = font.size("M")
    glyphs = np.zeros((len(RAMP), cell_h, cell_w), np.float32)
    for i, char in enumerate(RAMP):
        surf = font.render(char, True, (255, 255, 255), (0, 0, 0))
        cell = pygame.Surface((cell_w, cell_h))
        cell.blit(surf, ((cell_w - surf.get_width()) // 2, 0))
        glyphs[i] = pygame.surfarray.array3d(cell)[:, :, 0].T / 255.0
    return glyphs


def rasterize(glyphs, idx, color):
    """Convierte la grilla de caracteres en una imagen RGB (alto, ancho, 3)."""
    _, cell_h, cell_w = glyphs.shape
    rows, cols = idx.shape
    cover = glyphs[idx].transpose(0, 2, 1, 3).reshape(rows * cell_h, cols * cell_w)
    tint = np.repeat(np.repeat(color, cell_h, axis=0), cell_w, axis=1)
    return (cover[..., None] * tint).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--save-frames", metavar="DIR",
                        help="guarda una vuelta completa como PNGs y sale")
    args = parser.parse_args()

    if args.save_frames:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    glyphs = build_glyphs()
    cell_h, cell_w = glyphs.shape[1:]
    cols = round(ROWS * cell_h * ASPECT / cell_w)
    scene = Scene(cols, ROWS, cell_w, cell_h)
    size = (cols * cell_w, ROWS * cell_h)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for i in range(int(DURATION * FPS)):
            img = rasterize(glyphs, *scene.frame(i / FPS))
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            pygame.image.save(surf, os.path.join(args.save_frames, f"{i:04d}.png"))
        return

    # Ventana que entre en la pantalla: la imagen se escala si hace falta.
    info = pygame.display.Info()
    scale = min(1.0, 0.88 * info.current_h / size[1]) if info.current_h > 0 else 1.0
    win_size = (int(size[0] * scale), int(size[1] * scale))
    screen = pygame.display.set_mode(win_size)
    pygame.display.set_caption("Margaritas")
    canvas = pygame.Surface(size)
    clock = pygame.time.Clock()
    t, paused, running = 0.0, False, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    t = 0.0
        img = rasterize(glyphs, *scene.frame(t))
        pygame.surfarray.blit_array(canvas, img.swapaxes(0, 1))
        if win_size == size:
            screen.blit(canvas, (0, 0))
        else:
            pygame.transform.smoothscale(canvas, win_size, screen)
        pygame.display.flip()
        dt = clock.tick(FPS) / 1000.0
        if not paused:
            t = (t + dt) % DURATION
    pygame.quit()


if __name__ == "__main__":
    main()
