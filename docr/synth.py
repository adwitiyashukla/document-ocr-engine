import io
import json
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from docr.geometry import apply_homography, box_to_quad, crop_quad, homography, warp_perspective
from docr.imaging import gaussian_blur, remap, resize, to_uint8

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/"
FONTS = {
    "mono": [
        "ofl/courierprime/CourierPrime-Regular.ttf", "ofl/courierprime/CourierPrime-Bold.ttf",
        "ofl/spacemono/SpaceMono-Regular.ttf", "ofl/spacemono/SpaceMono-Bold.ttf",
        "ofl/ibmplexmono/IBMPlexMono-Regular.ttf", "ofl/ibmplexmono/IBMPlexMono-Bold.ttf",
        "ofl/sourcecodepro/SourceCodePro[wght].ttf", "ofl/inconsolata/Inconsolata[wdth,wght].ttf",
        "ofl/firamono/FiraMono-Regular.ttf", "ofl/firamono/FiraMono-Bold.ttf", "ofl/ptmono/PTM55FT.ttf",
        "ufl/ubuntumono/UbuntuMono-Regular.ttf", "ufl/ubuntumono/UbuntuMono-Bold.ttf",
        "ofl/anonymouspro/AnonymousPro-Regular.ttf", "ofl/anonymouspro/AnonymousPro-Bold.ttf",
        "ofl/overpassmono/OverpassMono[wght].ttf", "ofl/sharetechmono/ShareTechMono-Regular.ttf",
        "ofl/vt323/VT323-Regular.ttf", "ofl/novamono/NovaMono.ttf", "ofl/cutivemono/CutiveMono-Regular.ttf",
        "ofl/dotgothic16/DotGothic16-Regular.ttf", "ofl/b612mono/B612Mono-Regular.ttf", "ofl/b612mono/B612Mono-Bold.ttf",
        "ofl/jetbrainsmono/JetBrainsMono[wght].ttf", "ofl/notosansmono/NotoSansMono[wdth,wght].ttf",
        "apache/specialelite/SpecialElite-Regular.ttf",
    ],
    "serif": [
        "ofl/notoserif/NotoSerif[wdth,wght].ttf", "ofl/librebaskerville/LibreBaskerville[wght].ttf",
        "ofl/ebgaramond/EBGaramond[wght].ttf", "ofl/crimsontext/CrimsonText-Regular.ttf",
        "ofl/crimsontext/CrimsonText-Bold.ttf", "ofl/crimsontext/CrimsonText-Italic.ttf",
        "ofl/merriweather/Merriweather[opsz,wdth,wght].ttf", "ofl/lora/Lora[wght].ttf",
        "ofl/ptserif/PT_Serif-Web-Regular.ttf", "ofl/ptserif/PT_Serif-Web-Bold.ttf", "ofl/ptserif/PT_Serif-Web-Italic.ttf",
        "ofl/oldstandardtt/OldStandard-Regular.ttf", "ofl/oldstandardtt/OldStandard-Bold.ttf",
        "apache/robotoslab/RobotoSlab[wght].ttf", "ofl/zillaslab/ZillaSlab-Regular.ttf",
        "ofl/arvo/Arvo-Regular.ttf", "ofl/arvo/Arvo-Bold.ttf",
    ],
    "sans": [
        "ofl/roboto/Roboto[wdth,wght].ttf", "ofl/opensans/OpenSans[wdth,wght].ttf",
        "ofl/lato/Lato-Regular.ttf", "ofl/lato/Lato-Bold.ttf", "ofl/lato/Lato-Italic.ttf",
        "ofl/notosans/NotoSans[wdth,wght].ttf", "ofl/sourcesans3/SourceSans3[wght].ttf",
        "ofl/ptsans/PT_Sans-Web-Regular.ttf", "ofl/ptsans/PT_Sans-Web-Bold.ttf", "ofl/worksans/WorkSans[wght].ttf",
        "ofl/firasans/FiraSans-Regular.ttf", "ofl/firasans/FiraSans-Bold.ttf", "ofl/ibmplexsans/IBMPlexSans[wdth,wght].ttf",
        "ofl/oswald/Oswald[wght].ttf", "ofl/robotocondensed/RobotoCondensed[wght].ttf",
        "ofl/archivonarrow/ArchivoNarrow[wght].ttf", "ofl/ptsansnarrow/PT_Sans-Narrow-Web-Regular.ttf",
        "ofl/ptsansnarrow/PT_Sans-Narrow-Web-Bold.ttf", "ofl/barlow/Barlow-Regular.ttf", "ofl/barlow/Barlow-Bold.ttf",
        "ofl/barlowcondensed/BarlowCondensed-Regular.ttf", "ofl/barlowcondensed/BarlowCondensed-Bold.ttf",
        "ofl/inter/Inter[opsz,wght].ttf", "ofl/montserrat/Montserrat[wght].ttf", "ufl/ubuntu/Ubuntu-Regular.ttf",
        "ufl/ubuntu/Ubuntu-Bold.ttf", "ofl/karla/Karla[wght].ttf", "ofl/rubik/Rubik[wght].ttf",
        "ofl/hind/Hind-Regular.ttf", "ofl/hind/Hind-Bold.ttf",
    ],
    "hand": [
        "ofl/caveat/Caveat[wght].ttf", "ofl/indieflower/IndieFlower-Regular.ttf", "ofl/shadowsintolight/ShadowsIntoLight.ttf",
        "ofl/patrickhand/PatrickHand-Regular.ttf", "ofl/kalam/Kalam-Regular.ttf", "apache/homemadeapple/HomemadeApple-Regular.ttf",
        "ofl/reeniebeanie/ReenieBeanie.ttf", "ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf",
        "ofl/gochihand/GochiHand-Regular.ttf", "apache/justanotherhand/JustAnotherHand-Regular.ttf",
        "ofl/nothingyoucoulddo/NothingYouCouldDo.ttf", "ofl/coveredbyyourgrace/CoveredByYourGrace.ttf",
        "ofl/nanumpenscript/NanumPenScript-Regular.ttf", "ofl/zeyada/Zeyada.ttf", "ofl/dawningofanewday/DawningofaNewDay.ttf",
    ],
}
CHARSET = "".join(chr(c) for c in range(33, 127))
RECEIPT_WORDS = [
    "TOTAL", "SUBTOTAL", "Sub", "Total", "TAX", "Tax", "PB1", "SVC", "Service", "CASH", "Cash", "CHANGE", "Change",
    "Kembali", "Tunai", "Debit", "QTY", "Qty", "ITEM", "Items", "DISC", "Discount", "Rounding", "Card", "Payment",
    "GRAND", "Grand", "Bayar", "Harga", "Jumlah", "Meja", "Table", "Kasir", "Cashier", "Rp", "Rp.", "IDR", "No.",
    "Bill", "Order", "Struk", "Terima", "Kasih", "Thank", "You", "x", "@", "-", ":", "Ttl", "Pcs", "Dine", "Take",
]
FORM_WORDS = [
    "DATE:", "Date:", "TO:", "FROM:", "SUBJECT:", "RE:", "NAME:", "Name:", "ADDRESS:", "PHONE:", "FAX:", "No.", "Brand",
    "Company", "Signature", "Approved", "Comments:", "Total", "Number", "Code", "Description", "Amount", "Page", "of",
    "(Please", "print)", "Yes", "No", "Account", "Program", "Project", "Budget", "R&D", "Inc.", "Co.", "Dept.", "cc:",
]


def download_fonts(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, paths in FONTS.items():
        for path in paths:
            target = folder / Path(path).name
            if not target.exists():
                with urllib.request.urlopen(FONT_URL + path, timeout=60) as response:
                    target.write_bytes(response.read())
    return folder


class FontBank:
    def __init__(self, folder, charset=CHARSET):
        self.folder = Path(folder)
        self.charset = charset
        cache = self.folder / "coverage.json"
        coverage = json.loads(cache.read_text()) if cache.exists() else {}
        self.entries = []
        changed = False
        for kind, paths in FONTS.items():
            for path in paths:
                file = self.folder / Path(path).name
                if not file.exists():
                    continue
                if file.name not in coverage:
                    coverage[file.name] = self._coverage(file)
                    changed = True
                self.entries.append({"path": str(file), "kind": kind, "chars": set(coverage[file.name])})
        if changed:
            cache.write_text(json.dumps(coverage))
        if not self.entries:
            raise FileNotFoundError(f"no fonts in {self.folder}, run: python -m docr fonts")
        self._cache = OrderedDict()

    def _coverage(self, file):
        font = ImageFont.truetype(str(file), 32)

        def ink(ch):
            image = Image.new("L", (64, 64), 0)
            ImageDraw.Draw(image).text((16, 8), ch, font=font, fill=255)
            return np.asarray(image)

        missing = ink("\ue000")
        return "".join(ch for ch in self.charset if ink(ch).any() and not np.array_equal(ink(ch), missing))

    def pick(self, rng, kinds=None, text=None):
        pool = [e for e in self.entries if kinds is None or e["kind"] in kinds]
        if text is not None:
            needed = set(text) - {" "}
            covered = [e for e in pool if needed <= e["chars"]]
            pool = covered or [e for e in self.entries if needed <= e["chars"]] or pool
        return pool[int(rng.integers(len(pool)))]

    def font(self, entry, size):
        key = (entry["path"], int(size))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if len(self._cache) >= 600:
            self._cache.popitem(last=False)
        self._cache[key] = ImageFont.truetype(entry["path"], int(size))
        return self._cache[key]


class TextSource:
    def __init__(self, seed=0):
        import wordfreq
        from faker import Faker

        allowed = set(CHARSET)
        self.english = [w for w in wordfreq.top_n_list("en", 30000) if set(w) <= allowed]
        self.indonesian = [w for w in wordfreq.top_n_list("id", 15000) if set(w) <= allowed]
        self.fakers = [Faker("en_US"), Faker("id_ID")]
        for i, faker in enumerate(self.fakers):
            faker.seed_instance(seed + i)

    def _case(self, word, rng):
        r = rng.random()
        if r < 0.45:
            return word
        if r < 0.75:
            return word[:1].upper() + word[1:]
        return word.upper()

    def word(self, rng):
        pool = self.english if rng.random() < 0.7 else self.indonesian
        return self._case(pool[int(min(rng.pareto(1.2) * 300, len(pool) - 1))], rng)

    def number(self, rng):
        r = rng.random()
        value = rng.integers(1, 10 ** int(rng.integers(1, 8)))
        if r < 0.25:
            return f"{value:,}"
        if r < 0.45:
            return f"{value:,}".replace(",", ".")
        if r < 0.6:
            return f"{value / 100:,.2f}"
        if r < 0.7:
            return f"{int(rng.integers(0, 100))}%"
        if r < 0.8:
            return str(value)
        if r < 0.9:
            return f"{int(rng.integers(1, 30))}x"
        return f"Rp{value:,}".replace(",", ".") if rng.random() < 0.5 else f"${value / 100:.2f}"

    def date(self, rng):
        d, m, y = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(1980, 2027))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        options = [
            f"{d:02d}/{m:02d}/{y}", f"{d:02d}-{m:02d}-{y % 100:02d}", f"{y}-{m:02d}-{d:02d}", f"{d:02d}.{m:02d}.{y}",
            f"{d}-{months[m - 1]}-{y % 100:02d}", f"{months[m - 1].upper()}", f"{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}",
            f"{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}:{int(rng.integers(0, 60)):02d}", f"{m}/{d}/{y % 100:02d}",
        ]
        return options[int(rng.integers(len(options)))]

    def fake(self, rng):
        faker = self.fakers[int(rng.integers(2))]
        kind = int(rng.integers(6))
        value = [faker.first_name, faker.last_name, faker.city, faker.street_name, faker.company, faker.phone_number][kind]()
        parts = [part for part in value.split() if set(part) <= set(CHARSET)]
        return self._case(parts[int(rng.integers(len(parts)))], rng) if parts else self.word(rng)

    def code(self, rng):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        r = rng.random()
        if r < 0.3:
            return "".join(rng.choice(list(letters))) + "-" + str(int(rng.integers(1, 99999))).zfill(int(rng.integers(3, 6)))
        if r < 0.5:
            return "#" + str(int(rng.integers(1, 9999)))
        if r < 0.7:
            return "".join(rng.choice(list(letters + "0123456789"), size=int(rng.integers(3, 10))))
        return "@" + str(int(rng.integers(1, 999)))

    def punctuated(self, rng):
        word = self.word(rng)
        r = rng.random()
        if r < 0.35:
            return word + ":"
        if r < 0.5:
            return word + ","
        if r < 0.6:
            return word + "."
        if r < 0.7:
            return "(" + word + ")"
        if r < 0.8:
            return "(" + word
        if r < 0.85:
            return word + ")"
        if r < 0.9:
            return '"' + word + '"'
        return word + "'s"

    def symbol(self, rng):
        options = ["-", ":", "/", "&", "%", "*", "#", "+", "(", ")", "=", "x", "@", "$", "!", "?", ";", "'", '"', "~", "<", ">", "[", "]", "_", "|", "{", "}", "^", "`", "\\", ".", ","]
        return options[int(rng.integers(len(options)))]

    def random_string(self, rng, charset=CHARSET):
        return "".join(rng.choice(list(charset), size=int(rng.integers(1, 12))))

    def token(self, rng, domain=None):
        r = rng.random()
        if domain == "receipt" and r < 0.2:
            return RECEIPT_WORDS[int(rng.integers(len(RECEIPT_WORDS)))]
        if domain == "form" and r < 0.1:
            return FORM_WORDS[int(rng.integers(len(FORM_WORDS)))]
        r = rng.random()
        if r < 0.40:
            return self.word(rng)
        if r < 0.56:
            return self.number(rng)
        if r < 0.63:
            return self.date(rng)
        if r < 0.72:
            return self.fake(rng)
        if r < 0.82:
            return self.punctuated(rng)
        if r < 0.88:
            return self.code(rng)
        if r < 0.93:
            return self.symbol(rng)
        return self.random_string(rng)


def fractal_noise(shape, rng, octaves=4, base=8):
    h, w = shape
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    total = 0.0
    for i in range(octaves):
        cells = base * 2 ** i
        grid = rng.normal(0, 1, (max(2, h // cells + 2), max(2, w // cells + 2))).astype(np.float32)
        out += amp * resize(grid, (h, w))
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.mean()) / (out.std() + 1e-6)


def paper(shape, rng, tint=None):
    h, w = shape
    base = rng.uniform(205, 252)
    color = np.array(tint if tint is not None else [base, base - rng.uniform(0, 8), base - rng.uniform(0, 14)], np.float32)
    noise = fractal_noise((h, w), rng, octaves=3, base=16) * rng.uniform(1, 6)
    image = color[None, None, :] + noise[..., None]
    return np.clip(image, 0, 255)


def background(shape, rng):
    h, w = shape
    kind = int(rng.integers(4))
    color = rng.uniform(10, 200, 3).astype(np.float32)
    if kind == 0:
        noise = fractal_noise((h, w), rng, octaves=5, base=4)
        image = color + noise[..., None] * rng.uniform(5, 40)
    elif kind == 1:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        warp = fractal_noise((h, w), rng, octaves=3, base=32) * rng.uniform(5, 25)
        grain = np.sin((xx * rng.uniform(0.02, 0.08) + warp) * np.pi)
        image = color + grain[..., None] * rng.uniform(10, 35)
    elif kind == 2:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = (xx / w * rng.uniform(-1, 1) + yy / h * rng.uniform(-1, 1))
        image = color + ramp[..., None] * rng.uniform(20, 80)
    else:
        noise = fractal_noise((h, w), rng, octaves=2, base=64)
        image = color + noise[..., None] * rng.uniform(5, 20)
    return np.clip(image, 0, 255)


def crumple(shape, rng, strength):
    height = fractal_noise(shape, rng, octaves=5, base=6)
    gy, gx = np.gradient(height)
    light = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 1.5])
    light /= np.linalg.norm(light)
    shade = (-gx * light[0] - gy * light[1] + light[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1)
    return 1 + strength * (shade - shade.mean())


def jpeg(image, quality):
    buffer = io.BytesIO()
    mode = "L" if image.ndim == 2 else "RGB"
    Image.fromarray(to_uint8(image), mode).save(buffer, "JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert(mode), np.float32)


def motion_blur(image, rng, length):
    angle = rng.uniform(0, np.pi)
    steps = max(int(length), 2)
    out = np.zeros_like(image, dtype=np.float32)
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for t in np.linspace(-0.5, 0.5, steps):
        out += remap(image, xs + t * length * np.cos(angle), ys + t * length * np.sin(angle), fill=255.0)
    return out / steps


def degrade(image, rng, strength=1.0):
    image = image.astype(np.float32)
    if rng.random() < 0.5 * strength:
        image = gaussian_blur(image, rng.uniform(0.3, 1.3), None) if image.ndim == 2 else np.stack([gaussian_blur(image[..., c], rng.uniform(0.3, 1.3)) for c in range(3)], -1)
    if rng.random() < 0.1 * strength:
        image = motion_blur(image, rng, rng.uniform(2, 5))
    if rng.random() < 0.6 * strength:
        image = image + rng.normal(0, rng.uniform(2, 14), image.shape)
    if rng.random() < 0.15 * strength:
        mask = rng.random(image.shape[:2]) < rng.uniform(0.001, 0.01)
        image[mask] = rng.choice([0.0, 255.0])
    if rng.random() < 0.5:
        gamma = rng.uniform(0.6, 1.6)
        image = 255 * (np.clip(image, 0, 255) / 255) ** gamma
    if rng.random() < 0.5 * strength:
        image = jpeg(image, rng.uniform(20, 90))
    return np.clip(image, 0, 255)


def _ink(rng):
    if rng.random() < 0.15:
        return tuple(int(v) for v in rng.choice([[30, 30, 140], [90, 30, 120], [140, 20, 20], [20, 90, 40]]) + rng.integers(-20, 20, 3))
    v = int(rng.integers(0, 90))
    return (v, v, v)


def render_word(text, bank, rng, height=32, max_width=512, kinds=None):
    entry = bank.pick(rng, kinds, text)
    size = int(rng.choice([rng.integers(11, 18), rng.integers(18, 34), rng.integers(34, 56)], p=[0.35, 0.45, 0.2]))
    font = bank.font(entry, size)
    stroke = 1 if rng.random() < 0.08 and size > 16 else 0
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    spacing = len(text) > 1 and rng.random() < 0.15
    if spacing:
        gap = rng.uniform(0.05, 0.6) * size
        starts = np.concatenate([[0.0], np.cumsum([font.getlength(ch) + gap for ch in text[:-1]])])
        right = max(right, int(np.ceil(starts[-1] + font.getbbox(text[-1], stroke_width=stroke)[2])))
    pad = int(size * 1.2) + 4
    width = max(right - left, 1) + 2 * pad
    tall = max(bottom - top, 1) + 2 * pad
    bg = float(rng.uniform(150, 255)) if rng.random() < 0.9 else float(rng.uniform(0, 80))
    ink = float(rng.uniform(0, max(bg - 70, 0) + 1)) if bg > 120 else float(rng.uniform(bg + 90, 255))
    canvas = Image.new("L", (width, tall), int(bg))
    draw = ImageDraw.Draw(canvas)
    x = pad - left
    y = pad - top
    if spacing:
        for ch, start in zip(text, starts):
            draw.text((x + start, y), ch, font=font, fill=int(ink), stroke_width=stroke, stroke_fill=int(ink))
    else:
        draw.text((x, y), text, font=font, fill=int(ink), stroke_width=stroke, stroke_fill=int(ink))
    box = (pad, pad, pad + max(right - left, 1), pad + max(bottom - top, 1))
    if rng.random() < 0.25:
        _distractors(draw, rng, box, width, tall, int(ink), bank, size)
    image = np.asarray(canvas, np.float32)
    quad = box_to_quad(box)
    matrix = _jitter_matrix(rng, quad, size)
    image = warp_perspective(image, matrix, (width, tall), fill=bg)
    quad = apply_homography(matrix, quad)
    h = max(box[3] - box[1], 1)
    grow = np.array([rng.uniform(-0.05, 0.35), rng.uniform(-0.1, 0.35), rng.uniform(-0.05, 0.35), rng.uniform(-0.1, 0.35)]) * h
    center = quad.mean(axis=0)
    out = quad.copy()
    out[[0, 3], 0] -= grow[0]
    out[[1, 2], 0] += grow[2]
    out[[0, 1], 1] -= grow[1]
    out[[2, 3], 1] += grow[3]
    out += rng.normal(0, 0.04 * h, 2)
    out = np.where(np.isfinite(out), out, center)
    crop = crop_quad(image, out, height, max_width=max_width, fill=bg)
    if rng.random() < 0.3:
        crop = crop * crumple(crop.shape, rng, rng.uniform(0.05, 0.25)) if crop.shape[1] > 8 else crop
    crop = degrade(crop, rng)
    return to_uint8(crop)


def _distractors(draw, rng, box, width, tall, ink, bank, size):
    x0, y0, x1, y1 = box
    r = rng.random()
    if r < 0.4:
        y = y1 + rng.integers(1, max(2, size // 4))
        draw.line([(x0 - 5, y), (x1 + 5, y)], fill=ink, width=int(rng.integers(1, 3)))
    elif r < 0.6:
        x = x0 - rng.integers(3, max(4, size // 2))
        draw.line([(x, 0), (x, tall)], fill=ink, width=1)
    else:
        font = bank.font(bank.pick(rng), size)
        draw.text((x1 + rng.integers(int(size * 0.5), int(size * 1.2)), y0), "abcdefghijklmnopqrstuvwxyz0123456789"[int(rng.integers(36))], font=font, fill=ink)
        draw.text((x0 - rng.integers(int(size * 1.0), int(size * 1.8)), y0), "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[int(rng.integers(36))], font=font, fill=ink)


def _jitter_matrix(rng, quad, size):
    center = quad.mean(axis=0)
    angle = np.deg2rad(rng.normal(0, 1.5))
    shear = rng.normal(0, 0.12)
    scale_x = np.exp(rng.normal(0, 0.12))
    c, s = np.cos(angle), np.sin(angle)
    a = np.array([[c, -s], [s, c]]) @ np.array([[scale_x, shear], [0, 1]])
    matrix = np.eye(3)
    matrix[:2, :2] = a
    matrix[:2, 2] = center - a @ center
    if rng.random() < 0.3:
        half = np.maximum((quad.max(axis=0) - quad.min(axis=0)) / 2, size / 2)
        src = center + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
        dst = src + np.clip(rng.normal(0, 0.06 * size, src.shape), -0.3 * half, 0.3 * half)
        matrix = homography(src, dst) @ matrix
    return matrix


class Page:
    def __init__(self, width, height, rng, color=None):
        self.width = width
        self.height = height
        self.rng = rng
        self.image = Image.fromarray(to_uint8(paper((height, width), rng, color)), "RGB")
        self.draw = ImageDraw.Draw(self.image)
        self.quads = []
        self.texts = []
        self.ignore = []

    def text(self, x, y, token, font, fill, ignore=False):
        left, top, right, bottom = self.draw.textbbox((x, y), token, font=font, anchor="ls")
        if right > self.width - 2 or bottom > self.height - 2 or left < 1 or top < 1:
            return None
        self.draw.text((x, y), token, font=font, fill=fill, anchor="ls")
        self.quads.append(box_to_quad((left, top, right, bottom)))
        self.texts.append(token)
        self.ignore.append(ignore)
        return right

    def line(self, x, y, tokens, font, fill, gap=None):
        gap = font.getlength(" ") if gap is None else gap
        cursor = x
        for token in tokens:
            right = self.text(cursor, y, token, font, fill)
            if right is None:
                break
            cursor = right + gap
        return cursor

    def rule(self, x0, y, x1, fill, width=1):
        self.draw.line([(x0, y), (x1, y)], fill=fill, width=width)

    def result(self):
        quads = np.array(self.quads, np.float32).reshape(-1, 4, 2)
        return np.asarray(self.image, np.float32), quads, list(self.texts), np.array(self.ignore, bool)


def _font_for(bank, rng, kinds, size, text=None):
    return bank.font(bank.pick(rng, kinds, text), size)


def render_form(bank, source, rng):
    width = int(rng.uniform(720, 1050))
    height = int(width * rng.uniform(1.22, 1.42))
    page = Page(width, height, rng)
    margin = int(rng.uniform(25, 90))
    y = margin + int(rng.uniform(10, 60))
    base = int(rng.choice([rng.integers(12, 18), rng.integers(18, 26)], p=[0.6, 0.4]))
    body_kinds = [["serif", "mono"], ["sans"], ["mono"], ["serif"]][int(rng.integers(4))]
    ink = _ink(rng)
    while y < height - margin:
        block = rng.choice(["header", "fields", "columns", "table", "paragraph", "checks", "rule"], p=[0.08, 0.27, 0.15, 0.12, 0.25, 0.05, 0.08])
        if block == "header":
            size = int(base * rng.uniform(1.3, 2.2))
            font = _font_for(bank, rng, ["sans", "serif"], size)
            tokens = [source.word(rng).upper() if rng.random() < 0.5 else source.word(rng) for _ in range(int(rng.integers(1, 6)))]
            x = margin if rng.random() < 0.5 else int(width / 2 - sum(font.getlength(t + " ") for t in tokens) / 2)
            page.line(max(x, margin), y + size, tokens, font, ink)
            y += int(size * rng.uniform(1.6, 2.4))
        elif block == "fields":
            label_font = _font_for(bank, rng, body_kinds, base)
            for _ in range(int(rng.integers(2, 7))):
                size = base
                label = [source.token(rng, "form") for _ in range(int(rng.integers(1, 3)))]
                label[-1] = label[-1].rstrip(":") + ":"
                end = page.line(margin, y + size, label, label_font, ink)
                hand = rng.random() < 0.3
                value_font = _font_for(bank, rng, ["hand"] if hand else body_kinds, int(size * (rng.uniform(1.1, 1.5) if hand else 1)))
                value = [source.token(rng, "form") for _ in range(int(rng.integers(1, 6)))]
                vx = end + int(rng.uniform(8, 60))
                vend = page.line(vx, y + size, value, value_font, _ink(rng) if hand else ink)
                if rng.random() < 0.4:
                    page.rule(vx - 4, y + size + 4, max(vend, vx + 120), ink)
                y += int(size * rng.uniform(1.8, 2.8))
        elif block == "columns":
            cols = int(rng.integers(2, 5))
            font = _font_for(bank, rng, body_kinds, base)
            col_w = (width - 2 * margin) / cols
            for _ in range(int(rng.integers(2, 7))):
                for c in range(cols):
                    tokens = [source.token(rng, "form") for _ in range(int(rng.integers(1, 4)))]
                    page.line(int(margin + c * col_w), y + base, tokens, font, ink)
                y += int(base * rng.uniform(1.5, 2.2))
        elif block == "table":
            rows, cols = int(rng.integers(2, 7)), int(rng.integers(2, 6))
            font = _font_for(bank, rng, body_kinds, base)
            row_h = int(base * rng.uniform(1.8, 2.6))
            col_w = (width - 2 * margin) / cols
            if y + rows * row_h > height - margin:
                break
            for r in range(rows + 1):
                page.rule(margin, y + r * row_h, width - margin, ink)
            for c in range(cols + 1):
                x = int(margin + c * col_w)
                page.draw.line([(x, y), (x, y + rows * row_h)], fill=ink, width=1)
            for r in range(rows):
                for c in range(cols):
                    if rng.random() < 0.85:
                        tokens = [source.token(rng, "form") for _ in range(int(rng.integers(1, 3)))]
                        page.line(int(margin + c * col_w + 4), y + r * row_h + int(row_h * 0.7), tokens, font, ink)
            y += rows * row_h + int(base * rng.uniform(1, 2))
        elif block == "paragraph":
            font = _font_for(bank, rng, body_kinds, base)
            for _ in range(int(rng.integers(2, 9))):
                x = margin
                while x < width - margin - 40:
                    token = source.token(rng, "form") if rng.random() < 0.3 else source.word(rng)
                    right = page.text(x, y + base, token, font, ink)
                    if right is None:
                        break
                    x = right + font.getlength(" ")
                y += int(base * rng.uniform(1.3, 1.8))
        elif block == "checks":
            font = _font_for(bank, rng, body_kinds, base)
            x = margin
            for _ in range(int(rng.integers(2, 5))):
                page.draw.rectangle([x, y + 2, x + base - 2, y + base], outline=ink, width=1)
                if rng.random() < 0.4:
                    page.draw.line([(x + 2, y + base // 2), (x + base // 2, y + base - 2), (x + base, y)], fill=ink, width=2)
                right = page.text(x + base + 6, y + base, source.word(rng), font, ink)
                x = (right or x) + int(rng.uniform(20, 60))
            y += int(base * 2)
        else:
            page.rule(margin, y + base // 2, width - margin, ink, int(rng.integers(1, 4)))
            y += base
        if rng.random() < 0.05:
            _scribble(page.draw, rng, width, height, _ink(rng))
    return page.result()


def _scribble(draw, rng, width, height, ink):
    x, y = rng.uniform(0.1, 0.8) * width, rng.uniform(0.2, 0.9) * height
    points = [(x, y)]
    for _ in range(int(rng.integers(8, 20))):
        x += rng.uniform(5, 25)
        y += rng.uniform(-15, 15)
        points.append((x, y))
    draw.line(points, fill=ink, width=int(rng.integers(1, 3)))


def render_receipt(bank, source, rng):
    width = int(rng.uniform(340, 640))
    lines = int(rng.integers(14, 40))
    kinds = ["mono"] if rng.random() < 0.7 else ["sans"]
    size = int(rng.uniform(15, 30))
    height = int(lines * size * 1.6 + 120)
    tint = None if rng.random() < 0.8 else [rng.uniform(220, 250)] * 3
    page = Page(width, height, rng, tint)
    margin = int(rng.uniform(12, 40))
    ink = _ink(rng) if rng.random() < 0.3 else (int(rng.integers(0, 60)),) * 3
    font = _font_for(bank, rng, kinds, size)
    big = _font_for(bank, rng, kinds + ["sans", "serif"], int(size * rng.uniform(1.2, 1.9)))
    y = margin + size

    def centered(tokens, f, yy):
        total = sum(f.getlength(t) for t in tokens) + f.getlength(" ") * (len(tokens) - 1)
        page.line(int((width - total) / 2), yy, tokens, f, ink)

    def left_right(left, right, yy):
        page.line(margin, yy, left, font, ink)
        total = sum(font.getlength(t) for t in right) + font.getlength(" ") * (len(right) - 1)
        page.line(int(width - margin - total), yy, right, font, ink)

    def separator(yy):
        ch = rng.choice(["-", "=", "*", "."])
        count = int((width - 2 * margin) / max(font.getlength(ch), 1))
        token = ch * count
        page.text(margin, yy, token, font, ink, ignore=True)

    centered([source.fake(rng).upper() for _ in range(int(rng.integers(1, 4)))], big, y + size // 2)
    y += int(size * 2.2)
    for _ in range(int(rng.integers(1, 4))):
        centered([source.token(rng, "receipt") for _ in range(int(rng.integers(2, 5)))], font, y)
        y += int(size * 1.4)
    separator(y)
    y += int(size * 1.5)
    for _ in range(int(rng.integers(1, 4))):
        left_right([source.token(rng, "receipt"), source.date(rng)], [source.code(rng)], y)
        y += int(size * 1.4)
    separator(y)
    y += int(size * 1.5)
    for _ in range(int(rng.integers(2, 10))):
        if y > height - 6 * size:
            break
        name = [source.word(rng).upper() if rng.random() < 0.6 else source.word(rng) for _ in range(int(rng.integers(1, 4)))]
        qty = [f"{int(rng.integers(1, 5))}"] if rng.random() < 0.6 else []
        left_right(qty + name, [source.number(rng)], y)
        y += int(size * 1.4)
        if rng.random() < 0.3:
            page.line(margin + int(size * 2), y, [f"{int(rng.integers(1, 5))}", "x", source.number(rng)], font, ink)
            y += int(size * 1.4)
    if y < height - 5 * size:
        separator(y)
        y += int(size * 1.5)
    for label in ["SUBTOTAL", "TAX", "TOTAL", "CASH", "CHANGE"]:
        if y > height - 2 * size:
            break
        if rng.random() < 0.75:
            left_right([label if rng.random() < 0.6 else label.title()], [source.number(rng)], y)
            y += int(size * 1.4)
    if y < height - 3 * size:
        centered([source.token(rng, "receipt") for _ in range(int(rng.integers(1, 4)))], font, y + size)
    image, quads, texts, ignore = page.result()
    used = int(min(height, y + 3 * size))
    image = image[:used]
    keep = quads[:, :, 1].max(axis=1) < used - 1 if len(quads) else np.zeros(0, bool)
    return image, quads[keep], [t for t, k in zip(texts, keep) if k], ignore[keep]


def curl_maps(shape, rng, amount):
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xs / max(w - 1, 1)
    v = ys / max(h - 1, 1)
    a = amount * h * rng.uniform(-1, 1)
    b = amount * h * rng.uniform(-0.5, 0.5)
    phase = rng.uniform(0, np.pi)
    dy = a * np.sin(np.pi * u) * (0.6 + 0.8 * v) + b * np.sin(2 * np.pi * u + phase)
    dx = amount * w * 0.3 * rng.uniform(-1, 1) * np.sin(np.pi * v) * (u - 0.5)
    return xs + dx, ys + dy


def forward_points(points, map_x, map_y, iterations=12):
    h, w = map_x.shape
    out = points.astype(np.float64).copy()
    for _ in range(iterations):
        xi = np.clip(out[..., 0], 0, w - 1)
        yi = np.clip(out[..., 1], 0, h - 1)
        sx = remap(map_x, xi.astype(np.float32), yi.astype(np.float32))
        sy = remap(map_y, xi.astype(np.float32), yi.astype(np.float32))
        out[..., 0] += points[..., 0] - sx
        out[..., 1] += points[..., 1] - sy
    return out


def to_photo(image, quads, rng, curl=0.0, frame=None):
    h, w = image.shape[:2]
    if curl > 0:
        map_x, map_y = curl_maps((h, w), rng, curl)
        image = remap(image, map_x, map_y, fill=0.0)
        inside = remap(np.ones((h, w), np.float32), map_x, map_y, fill=0.0)
        quads = forward_points(quads, map_x, map_y) if len(quads) else quads
    else:
        inside = np.ones((h, w), np.float32)
    if rng.random() < 0.4:
        image = image * crumple((h, w), rng, rng.uniform(0.05, 0.3))[..., None]
    scale = rng.uniform(0.55, 0.95)
    out_w = int(frame[0]) if frame else int(w * rng.uniform(1.0, 1.4))
    out_h = int(frame[1]) if frame else int(h * rng.uniform(1.0, 1.3))
    fit = min(out_w / w, out_h / h) * scale
    cx, cy = out_w / 2 + rng.normal(0, 0.05) * out_w, out_h / 2 + rng.normal(0, 0.05) * out_h
    corners = box_to_quad((0, 0, w - 1, h - 1))
    target = (corners - [w / 2, h / 2]) * fit
    t = np.deg2rad(rng.normal(0, 6))
    target = target @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) + [cx, cy]
    target += rng.normal(0, 0.04, target.shape) * [out_w, out_h]
    matrix = homography(corners, target)
    warped = warp_perspective(image, matrix, (out_w, out_h), fill=0.0)
    alpha = warp_perspective(inside, matrix, (out_w, out_h), fill=0.0)[..., None]
    scene = background((out_h, out_w), rng)
    photo = warped * alpha + scene * (1 - alpha)
    photo = _lighting(photo, rng)
    page_quad = apply_homography(matrix, corners)
    quads = apply_homography(matrix, quads) if len(quads) else quads
    return degrade(photo, rng), quads, page_quad


def _lighting(image, rng):
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    gain = 1 + rng.uniform(-0.35, 0.15) * ((xs / w - rng.random()) ** 2 + (ys / h - rng.random()) ** 2)
    if rng.random() < 0.35:
        a, b, c = rng.normal(0, 1, 3)
        side = (a * (xs / w - 0.5) + b * (ys / h - 0.5) + 0.3 * c) > 0
        soft = gaussian_blur(side.astype(np.float32), rng.uniform(5, 40))
        gain = gain * (1 - rng.uniform(0.15, 0.5) * soft)
    tint = rng.uniform(0.85, 1.1, 3)
    return np.clip(image * gain[..., None] * tint, 0, 255)


def to_scan(image, quads, rng):
    h, w = image.shape[:2]
    gray = image @ np.array([0.299, 0.587, 0.114], np.float32)
    angle = rng.normal(0, 0.8)
    from docr.geometry import rotation

    matrix = rotation(angle, ((w - 1) / 2, (h - 1) / 2))
    gray = warp_perspective(gray, matrix, (w, h), fill=float(np.median(gray)))
    quads = apply_homography(matrix, quads) if len(quads) else quads
    if rng.random() < 0.4:
        factor = rng.uniform(0.45, 0.8)
        small = resize(gray, (int(h * factor), int(w * factor)))
        gray = resize(small, (h, w))
    gray = degrade(gray, rng)
    return np.repeat(gray[..., None], 3, axis=2), quads


def render_page(bank, source, rng, kind=None):
    kind = kind or rng.choice(["form_scan", "form_photo", "receipt_photo", "receipt_scan"], p=[0.3, 0.2, 0.4, 0.1])
    if kind.startswith("form"):
        image, quads, texts, ignore = render_form(bank, source, rng)
    else:
        image, quads, texts, ignore = render_receipt(bank, source, rng)
    page_quad = box_to_quad((0, 0, image.shape[1] - 1, image.shape[0] - 1))
    if kind.endswith("scan"):
        image, quads = to_scan(image, quads, rng)
    else:
        curl = rng.uniform(0.0, 0.06) if rng.random() < 0.5 else 0.0
        image, quads, page_quad = to_photo(image, quads, rng, curl)
    return {"image": to_uint8(image), "quads": np.asarray(quads, np.float32), "texts": texts, "ignore": ignore, "page": page_quad, "kind": kind}
