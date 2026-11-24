"""Foto -> ASCII: convierte cualquier foto en una animación de arte ASCII.

Los caracteres aparecen como ruido, la foto se revela desde el centro hacia
afuera, queda "viva" con un brillo que la recorre y un zoom lento, y al
final se disuelve. El fondo blanco de la foto se reemplaza por estrellas.

Uso:
    pip install numpy pygame
    python photo_ascii.py              (abre una ventana para elegir la foto)
    python photo_ascii.py mi_foto.jpg  (usa esa foto directamente)

Teclas: ESPACIO pausa, R reinicia, ESC sale.
"""

import argparse
import os
import sys

import numpy as np
import pygame

# Rampa de caracteres ordenada de menos a más "tinta".
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
FONT_SIZE = 10
FPS = 30
DURATION = 14.0  # segundos; después vuelve a empezar
SUPERSAMPLE = 3  # píxeles de la foto por celda, para poder hacer zoom

PINK = np.array([1.0, 0.30, 0.58], np.float32)
WHITE = np.array([1.0, 0.96, 0.93], np.float32)


def smooth(a, b, x):
    """Smoothstep: 0 antes de a, 1 después de b, transición suave entre ambos."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# Preparación de la foto
# --------------------------------------------------------------------------

def load_photo(path, cols, rows):
    """Carga la foto y devuelve (brillo, es_fondo) en alta resolución.

    El brillo va de 0 a 1 con el contraste estirado. `es_fondo` marca el
    blanco que rodea a la imagen (o las zonas transparentes).
    """
    surf = pygame.image.load(path)
    w, h = cols * SUPERSAMPLE, rows * SUPERSAMPLE
    has_alpha = surf.get_masks()[3] != 0
    if surf.get_bitsize() not in (24, 32):
        surf = surf.convert_alpha() if has_alpha else surf.convert(24)

    # Recorte cuadrado centrado para no deformar la foto.
    sw, sh = surf.get_size()
    side = min(sw, sh)
    surf = surf.subsurface(((sw - side) // 2, (sh - side) // 2, side, side))
    small = pygame.transform.smoothscale(surf, (w, h))

    rgb = pygame.surfarray.array3d(small).swapaxes(0, 1).astype(np.float32) / 255.0
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    if has_alpha:
        alpha = pygame.surfarray.array_alpha(small).T / 255.0
        lum = lum * alpha + (1.0 - alpha)

    # Fondo: blanco casi puro conectado con el borde de la imagen.
    white = lum > 0.965
    bg = np.zeros_like(white)
    bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1] = white[0, :], white[-1, :], white[:, 0], white[:, -1]
    while True:
        grown = bg.copy()
        grown[1:, :] |= bg[:-1, :]
        grown[:-1, :] |= bg[1:, :]
        grown[:, 1:] |= bg[:, :-1]
        grown[:, :-1] |= bg[:, 1:]
        grown &= white
        if (grown == bg).all():
            break
        bg = grown

    # Ecualización de histograma (mezclada con el tono original) para que
    # las zonas muy claras, como la piel con flash, conserven detalle.
    fg = np.sort(lum[~bg])
    if fg.size:
        lo, hi = np.percentile(fg, [1, 99])
        linear = np.clip((lum - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
        equal = np.searchsorted(fg, lum) / fg.size
        lum = (0.75 * equal + 0.25 * linear) ** 1.2
    lum[bg] = 0.0
    return lum, bg


# --------------------------------------------------------------------------
# Animación
# --------------------------------------------------------------------------

class PhotoScene:
    def __init__(self, lum, bg, cols, rows):
        self.lum, self.bg = lum, bg
        self.cols, self.rows = cols, rows
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) / cols - 0.5,
                             (np.arange(rows) + 0.5) / rows - 0.5)
        self.gx, self.gy = gx.astype(np.float32), gy.astype(np.float32)

        rng = np.random.default_rng(3)
        jitter = rng.random((rows, cols)).astype(np.float32)
        dist = np.hypot(self.gx, self.gy) / 0.71
        # Cada celda tiene su propio momento de aparecer y de irse.
        self.t_in = 1.0 + 3.2 * dist + 0.8 * jitter
        self.t_out = DURATION - 3.2 + 2.0 * rng.random((rows, cols)).astype(np.float32)
        self.star = rng.random((rows, cols)) > 0.985
        self.star_phase = rng.uniform(0, 2 * np.pi, (rows, cols)).astype(np.float32)
        self.ch = {c: RAMP.index(c) for c in ".+*"}

    def _sample(self, zoom):
        """Muestrea la foto con un zoom centrado. Devuelve (brillo, es_fondo)."""
        h, w = self.lum.shape
        x = np.clip(((self.gx / zoom) + 0.5) * w, 0, w - 1).astype(int)
        y = np.clip(((self.gy / zoom) + 0.5) * h, 0, h - 1).astype(int)
        return self.lum[y, x], self.bg[y, x]

    def frame(self, t):
        """Devuelve (índices de carácter, color RGB uint8) para el instante t."""
        zoom = 1.0 + 0.10 * float(smooth(4.0, DURATION - 2.5, t))
        photo, bg = self._sample(zoom)

        rng = np.random.default_rng(int(t * 12))
        flicker = rng.random(photo.shape, dtype=np.float32)

        # Brillo que recorre la foto en diagonal.
        sweep_pos = -1.0 + 2.0 * (t - 6.0) / 3.5
        sweep = np.exp(-((self.gx + 0.4 * self.gy - sweep_pos) / 0.10) ** 2)
        photo = np.clip(photo * (1.0 + 0.45 * sweep), 0.0, 1.0)

        shown = smooth(self.t_in, self.t_in + 0.6, t)
        flash = np.exp(-((t - self.t_in - 0.25) / 0.22) ** 2) * (~bg)
        static = 0.22 * (flicker > 0.90) * smooth(0.0, 1.0, t)  # ruido previo
        gone = 1.0 - smooth(self.t_out, self.t_out + 0.5, t)

        lum = (static * (1.0 - shown) + photo * shown + 0.6 * flash) * gone
        lum = np.clip(lum, 0.0, 1.0)

        n = len(RAMP)
        idx = (lum * (n - 1)).astype(np.int32)
        idx = np.where(idx > 1, idx + (flicker * 5).astype(np.int32) - 2, idx)
        idx = np.clip(idx, 0, n - 1)

        tone = np.clip(lum ** 1.6 + flash, 0.0, 1.0)[..., None]
        color = (PINK + (WHITE - PINK) * tone) * (0.55 + 0.45 * lum)[..., None]

        # Estrellas titilando en el fondo.
        twinkle = 0.5 + 0.5 * np.sin(t * 2.5 + self.star_phase)
        stars = self.star & bg & (shown > 0.5) & (gone > 0.5) & (twinkle > 0.25)
        idx[stars] = np.where(twinkle[stars] > 0.8, self.ch["+"],
                              np.where(twinkle[stars] > 0.5, self.ch["*"], self.ch["."]))
        color[stars] = (0.45 + 0.55 * twinkle[stars])[:, None]

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


def choose_photo():
    """Abre el selector de archivos del sistema para elegir la foto."""
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None
    root = tkinter.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Elige una foto",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Todos", "*.*")],
    )
    root.destroy()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", nargs="?",
                        help="ruta de la foto (jpg, png, webp...); si no se "
                             "indica, se abre una ventana para elegirla")
    parser.add_argument("--cols", type=int, default=150,
                        help="cantidad de columnas de caracteres (por defecto 150)")
    parser.add_argument("--save-frames", metavar="DIR",
                        help="guarda una vuelta completa como PNGs y sale")
    args = parser.parse_args()

    image = args.image or choose_photo()
    if not image:
        sys.exit("No se eligió ninguna foto.")
    if not os.path.isfile(image):
        sys.exit(f"No encuentro el archivo: {image}\n"
                 "Revisa el nombre, o ejecuta 'python photo_ascii.py' sin nada "
                 "más para elegir la foto desde una ventana.")

    if args.save_frames:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    glyphs = build_glyphs()
    cell_h, cell_w = glyphs.shape[1:]
    cols = args.cols
    rows = round(cols * cell_w / cell_h)  # ventana cuadrada
    size = (cols * cell_w, rows * cell_h)

    if not args.save_frames:
        screen = pygame.display.set_mode(size, pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption("Foto → ASCII")
    scene = PhotoScene(*load_photo(image, cols, rows), cols, rows)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for i in range(int(DURATION * FPS)):
            img = rasterize(glyphs, *scene.frame(i / FPS))
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            pygame.image.save(surf, os.path.join(args.save_frames, f"{i:04d}.png"))
        return

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
