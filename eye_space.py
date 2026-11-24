"""Eye -> Space: animación de arte ASCII procedural.

Un ojo se abre, la cámara entra por la pupila y adentro aparece el espacio
con estrellas y un planeta rosa. Todo se calcula con fórmulas en NumPy; no
se usa ninguna imagen ni video de origen.

Uso:
    pip install numpy pygame
    python eye_space.py

Teclas: ESPACIO pausa, R reinicia, ESC sale.
"""

import argparse
import os

import numpy as np
import pygame

# Rampa de caracteres ordenada de menos a más "tinta".
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
COLS, ROWS = 120, 50
FONT_SIZE = 14
FPS = 30
DURATION = 16.0  # segundos; después vuelve a empezar

N_STARS = 150


# --------------------------------------------------------------------------
# Utilidades matemáticas
# --------------------------------------------------------------------------

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
    a = _hash(ix, iy)
    b = _hash(ix + 1, iy)
    c = _hash(ix, iy + 1)
    d = _hash(ix + 1, iy + 1)
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def mix(a, b, k):
    """Interpola entre dos colores (o imágenes de color) con el peso k."""
    return a + (b - a) * k[..., None]


def rgb(r, g, b):
    return np.array([r, g, b], np.float32)


# --------------------------------------------------------------------------
# Escena
# --------------------------------------------------------------------------

class Scene:
    def __init__(self, cell_w, cell_h):
        self.w = COLS * cell_w
        self.h = ROWS * cell_h
        self.cell_w, self.cell_h = cell_w, cell_h
        # Coordenadas del centro de cada celda: v en [-1, 1], u corregido
        # por la proporción de la ventana.
        xs = (np.arange(COLS) + 0.5) * cell_w
        ys = (np.arange(ROWS) + 0.5) * cell_h
        self.u, self.v = np.meshgrid(
            (xs - self.w / 2) / (self.h / 2),
            (ys - self.h / 2) / (self.h / 2),
        )
        self.u = self.u.astype(np.float32)
        self.v = self.v.astype(np.float32)

        rng = np.random.default_rng(7)
        self.star_x = rng.uniform(-1.6, 1.6, N_STARS)
        self.star_y = rng.uniform(-1.2, 1.2, N_STARS)
        self.star_z = rng.uniform(0.0, 1.0, N_STARS)
        self.star_phase = rng.uniform(0, 2 * np.pi, N_STARS)

        self.ch = {c: RAMP.index(c) for c in ".+-|*"}

    # -- capas ---------------------------------------------------------

    def _space(self, t, amount):
        """Fondo del espacio y planeta. Devuelve (lum, color, máscara planeta)."""
        u, v = self.u, self.v
        neb = noise(u * 1.7 + 3.0, v * 1.7 + t * 0.05)
        lum = 0.06 * amount * smooth(0.55, 0.9, neb)
        col = np.broadcast_to(rgb(0.55, 0.35, 0.95), u.shape + (3,)).copy()

        grow = smooth(9.0, 14.5, t)
        radius = 0.44 * grow
        planet = np.zeros_like(u)
        if radius > 0.01:
            dx, dy = u / radius, (v - 0.03) / radius
            d2 = dx * dx + dy * dy
            inside = d2 < 1.0
            nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
            # Luz desde arriba a la izquierda.
            lam = np.clip(-0.50 * dx - 0.52 * dy + 0.69 * nz, 0.0, 1.0)
            lon = np.arctan2(dx, nz + 1e-6) + t * 0.35
            lat = np.arcsin(np.clip(dy, -1.0, 1.0))
            bands = 0.5 + 0.5 * np.sin(lat * 9.0 + 2.2 * noise(lon * 2.0, lat * 4.0))
            p_lum = 0.20 + 0.80 * lam * (0.55 + 0.45 * bands)
            p_col = mix(rgb(1.0, 0.30, 0.60), rgb(1.0, 0.72, 0.84), bands)

            # Halo de atmósfera alrededor del planeta.
            dist = np.sqrt(d2)
            halo = 0.35 * np.exp(-np.clip(dist - 1.0, 0.0, None) / 0.12) * grow
            lum = np.where(inside, p_lum, lum + halo)
            col = np.where(inside[..., None], p_col,
                           mix(col, rgb(1.0, 0.40, 0.70), np.clip(halo * 3, 0, 1)))
            planet = inside.astype(np.float32)
        return lum, col, planet

    def _eye(self, t, zoom, openness, rp, gaze):
        """Piel, párpados, esclerótica e iris.

        Devuelve (lum, color, máscara de la pupila).
        """
        x, y = self.u / zoom, self.v / zoom
        half_w = 1.15
        base = np.clip(1.0 - (x / half_w) ** 2, 0.0, 1.0)
        tilt = 0.05 * x
        y_up = -0.52 * openness * base + tilt
        y_lo = 0.34 * openness * base + tilt
        d_up, d_lo = y - y_up, y_lo - y
        edge = 0.03
        eye_mask = smooth(0.0, edge, d_up) * smooth(0.0, edge, d_lo) * (base > 0)

        # Piel
        tex = noise(x * 6.0, y * 6.0)
        fine = noise(x * 26.0, y * 26.0)
        crease_y = -0.52 * base + tilt - 0.15
        crease = np.exp(-((y - crease_y) / 0.035) ** 2) * (0.4 + 0.6 * openness)
        lash_zone = smooth(-0.11, 0.0, d_up) * (d_up < edge)
        lashes = lash_zone * (np.sin(x * 75.0 + d_up * 40.0) > 0.1)
        lid_line = np.exp(-(np.minimum(d_up, d_lo) / 0.03) ** 2)
        skin_lum = (0.50 + 0.28 * tex + 0.14 * fine
                    - 0.30 * crease - 0.45 * lashes - 0.35 * lid_line)
        skin_lum *= 0.72 * (1.0 - 0.22 * (self.u ** 2 + self.v ** 2) / 2.0)
        skin_col = np.broadcast_to(rgb(0.62, 0.40, 0.29), x.shape + (3,)) \
            * (0.90 + 0.20 * tex)[..., None]

        # Esclerótica
        side = np.clip(np.abs(x) / half_w, 0.0, 1.0)
        lid_shadow = 0.40 * np.exp(-np.clip(d_up, 0.0, None) / 0.08)
        sclera_lum = 0.88 - 0.45 * side ** 2 - lid_shadow
        sclera_col = mix(rgb(0.93, 0.89, 0.86), rgb(0.85, 0.45, 0.40), side ** 3)

        # Iris
        ix, iy = x - gaze[0], y - gaze[1]
        r = np.hypot(ix, iy)
        th = np.arctan2(iy, ix)
        ri = 0.36
        cs, sn = np.cos(th), np.sin(th)
        swirl = noise(cs * 3.0 + r * 5.0, sn * 3.0 + t * 0.15)
        fibers = 0.5 * noise(cs * 9.0 + 20.0, sn * 9.0 + r * 8.0) \
            + 0.5 * (0.5 + 0.5 * np.sin(th * 38.0 + 5.0 * swirl))
        collarette = np.exp(-((r - rp * 1.55) / 0.045) ** 2)
        limbus = smooth(0.78 * ri, ri, r)
        iris_lum = (0.32 + 0.62 * fibers + 0.30 * collarette) * (1.0 - 0.75 * limbus)
        iris_col = mix(rgb(0.36, 0.18, 0.07), rgb(0.92, 0.52, 0.14), fibers)
        iris_col = mix(iris_col, rgb(1.0, 0.78, 0.32), collarette * 0.8)
        iris_mask = 1.0 - smooth(ri - 0.015, ri + 0.015, r)

        ball_lum = sclera_lum + (iris_lum - sclera_lum) * iris_mask
        ball_col = mix(sclera_col, iris_col, iris_mask)

        # Reflejo de luz en la córnea; desaparece al acercarse.
        glint = np.exp(-(((ix + 0.13) ** 2 + (iy + 0.13) ** 2) / 0.05 ** 2))
        glint *= 1.0 - smooth(1.0, 2.5, zoom)
        ball_lum = ball_lum + glint * (1.0 - ball_lum)
        ball_col = mix(ball_col, rgb(1.0, 1.0, 1.0), glint)

        lum = skin_lum + (ball_lum - skin_lum) * eye_mask
        col = mix(skin_col, ball_col, eye_mask)

        soft = 0.012
        pupil = (1.0 - smooth(rp - soft, rp + soft, r)) * eye_mask * (1.0 - glint)
        return lum, col, pupil

    # -- cuadro completo -----------------------------------------------

    def frame(self, t):
        """Devuelve (índices de carácter, color RGB uint8) para el instante t."""
        blink = 1.0 - 0.92 * np.exp(-((t - 4.3) / 0.13) ** 2)
        openness = float(smooth(0.4, 3.2, t) * blink)
        dive = float(smooth(5.0, 9.5, t))
        zoom = 14.0 ** dive
        rp = 0.15 + 0.07 * float(smooth(4.8, 7.0, t))
        still = 1.0 - float(smooth(4.4, 5.4, t))
        gaze = (0.07 * np.sin(t * 1.1) * still, 0.02 * np.sin(t * 1.7) * still)
        space_amount = float(smooth(5.5, 8.5, t))

        s_lum, s_col, planet = self._space(t, space_amount)
        if zoom * (rp - 0.02) < 1.9:  # el ojo todavía entra en pantalla
            e_lum, e_col, pupil = self._eye(t, zoom, openness, rp, gaze)
            lum = e_lum * (1.0 - pupil) + s_lum * pupil
            col = mix(e_col, s_col, pupil)
        else:
            lum, col, pupil = s_lum, s_col, np.ones_like(s_lum)

        lum = np.clip(lum, 0.0, 1.0)
        fade = float(smooth(0.0, 0.8, t) * (1.0 - smooth(DURATION - 0.9, DURATION, t)))
        lum = lum * fade

        n = len(RAMP)
        idx = (lum ** 0.9 * (n - 1)).astype(np.int32)
        # Parpadeo de caracteres para que la imagen se sienta "viva".
        rng = np.random.default_rng(int(t * 12))
        idx = np.where(idx > 1, idx + rng.integers(-2, 3, idx.shape), idx)
        idx = np.clip(idx, 0, n - 1)
        color = col * (0.55 + 0.45 * lum)[..., None]

        self._stars(t, space_amount * fade, pupil, planet, idx, color)
        return idx, (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)

    def _stars(self, t, amount, pupil, planet, idx, color):
        """Dibuja las estrellas encima; las más brillantes llevan una cruz."""
        if amount <= 0.01:
            return
        z = (self.star_z - 0.05 * t) % 1.0 * 0.9 + 0.1
        su, sv = self.star_x / z * 0.5, self.star_y / z * 0.5
        cols = ((su * self.h / 2 + self.w / 2) / self.cell_w).astype(int)
        rows = ((sv * self.h / 2 + self.h / 2) / self.cell_h).astype(int)
        twinkle = 0.65 + 0.35 * np.sin(t * 3.0 + self.star_phase)
        bright = (1.0 - z) ** 1.3 * twinkle * amount * smooth(0.1, 0.25, z)

        def put(r, c, char, b):
            if 0 <= r < ROWS and 0 <= c < COLS and pupil[r, c] > 0.6 and planet[r, c] < 0.5:
                idx[r, c] = self.ch[char]
                color[r, c] = b

        for r, c, b in zip(rows, cols, bright):
            if b < 0.08:
                continue
            if b > 0.5:
                put(r, c, "+", 1.0)
                put(r, c - 1, "-", b * 0.8)
                put(r, c + 1, "-", b * 0.8)
                put(r - 1, c, "|", b * 0.8)
                put(r + 1, c, "|", b * 0.8)
                if b > 0.75:
                    put(r, c - 2, "-", b * 0.5)
                    put(r, c + 2, "-", b * 0.5)
            elif b > 0.25:
                put(r, c, "*", min(1.0, b + 0.4))
            else:
                put(r, c, ".", min(1.0, b + 0.45))


# --------------------------------------------------------------------------
# Render con pygame
# --------------------------------------------------------------------------

def build_glyphs():
    """Rasteriza cada carácter de la rampa una sola vez.

    Devuelve un array (n_caracteres, alto, ancho) con la cobertura 0..1.
    """
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
    cover = glyphs[idx].transpose(0, 2, 1, 3).reshape(ROWS * cell_h, COLS * cell_w)
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
    scene = Scene(cell_w, cell_h)
    size = (COLS * cell_w, ROWS * cell_h)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for i in range(int(DURATION * FPS)):
            img = rasterize(glyphs, *scene.frame(i / FPS))
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            pygame.image.save(surf, os.path.join(args.save_frames, f"{i:04d}.png"))
        return

    screen = pygame.display.set_mode(size, pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption("Eye → Space")
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
        pygame.surfarray.blit_array(screen, img.swapaxes(0, 1))
        pygame.display.flip()
        dt = clock.tick(FPS) / 1000.0
        if not paused:
            t = (t + dt) % DURATION
    pygame.quit()


if __name__ == "__main__":
    main()
