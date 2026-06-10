"""
Стандартний метод LSB для BMP-зображень.
Вбудовує текстове повідомлення у молодші біти кольорових компонент
пікселів. Позиції пікселів обираються псевдовипадково за ключем.
"""

import math
import random
import numpy as np
from PIL import Image
from scipy.stats import chi2 as chi2_dist

INPUT_PATH = r"C:\Users\daryn\OneDrive\Робочий стіл\code\code\1.bmp"
STEGO_DIR = r"C:\Users\daryn\OneDrive\Робочий стіл\code\code\stego"
KEY = 42
MESSAGE_SIZES = [10000, 50000, 1000000, 1500000, 2000000]


def text_to_bits(text: str) -> str:
    return ''.join(format(b, '08b') for b in text.encode('latin-1'))


def bits_to_text(bits: str) -> str:
    bytes_ = bytearray(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return bytes_.decode('latin-1')


def select_positions(shape, n_bits, key):
    """Псевдовипадковий вибір позицій (y, x, канал) за ключем."""
    H, W, C = shape
    total = H * W * C
    if n_bits > total:
        raise ValueError("Повідомлення більше за ємність контейнера")
    rng = random.Random(key)
    flat = rng.sample(range(total), n_bits)
    return [(p // (W * C), (p % (W * C)) // C, p % C) for p in flat]


def embed(cover: Image.Image, message: str, key: int) -> Image.Image:
    bits = text_to_bits(message)
    arr = np.array(cover, dtype=np.uint8).copy()
    positions = select_positions(arr.shape, len(bits), key)
    for (y, x, c), bit in zip(positions, bits):
        arr[y, x, c] = (arr[y, x, c] & 0xFE) | int(bit)
    return Image.fromarray(arr)


def extract(stego: Image.Image, msg_byte_len: int, key: int) -> str:
    arr = np.array(stego, dtype=np.uint8)
    n_bits = msg_byte_len * 8
    positions = select_positions(arr.shape, n_bits, key)
    bits = ''.join(str(arr[y, x, c] & 1) for (y, x, c) in positions)
    return bits_to_text(bits)


def psnr_mse(cover: Image.Image, stego: Image.Image):
    c = np.array(cover, dtype=np.float64)
    s = np.array(stego, dtype=np.float64)
    mse = float(np.mean((c - s) ** 2))
    if mse == 0:
        return float('inf'), 0.0
    return 10 * math.log10(255 ** 2 / mse), mse


def chi_squared_pvalue(img: Image.Image) -> float:
    """χ²-тест Westfeld-Pfitzmann на парах значень (PoV) по всіх RGB-каналах."""
    arr = np.array(img.convert('RGB'))
    chi2 = 0.0
    df = 0
    for c in range(3):
        hist = np.bincount(arr[:, :, c].flatten(), minlength=256)
        for k in range(128):
            n2k, n2k1 = int(hist[2 * k]), int(hist[2 * k + 1])
            s = n2k + n2k1
            if s < 5:
                continue
            expected = s / 2
            chi2 += (n2k - expected) ** 2 / expected + (n2k1 - expected) ** 2 / expected
            df += 1
    if df == 0:
        return 1.0
    return float(1 - chi2_dist.cdf(chi2, df))


def rs_analysis(img: Image.Image) -> float:
    """RS-аналіз Фрідріх: оцінка частки модифікованих LSB у відсотках."""
    arr = np.array(img.convert('L'), dtype=np.int32)
    mask = np.array([1, 0, 1, 0])
    H, W = arr.shape

    def compute_RS(image):
        n_cols = W - (W % 4)
        groups = image[:, :n_cols].reshape(H, -1, 4)

        def f(g):
            return np.sum(np.abs(np.diff(g, axis=-1)), axis=-1)

        f_orig = f(groups)

        flipped_pos = groups.copy()
        for i in range(4):
            if mask[i]:
                flipped_pos[:, :, i] = flipped_pos[:, :, i] ^ 1
        f_pos = f(flipped_pos)

        flipped_neg = groups.copy()
        for i in range(4):
            if mask[i]:
                v = flipped_neg[:, :, i]
                flipped_neg[:, :, i] = ((v + 1) ^ 1) - 1
        f_neg = f(flipped_neg)

        return (int(np.sum(f_pos > f_orig)), int(np.sum(f_pos < f_orig)),
                int(np.sum(f_neg > f_orig)), int(np.sum(f_neg < f_orig)))

    R, S, R_m, S_m = compute_RS(arr)
    Rf, Sf, R_mf, S_mf = compute_RS(arr ^ 1)

    d0, d1 = R - S, Rf - Sf
    d_m0, d_m1 = R_m - S_m, R_mf - S_mf

    a = 2 * (d1 + d0)
    b = d_m0 - d_m1 - d1 - 3 * d0
    c = d0 - d_m0

    try:
        if abs(a) < 1e-9:
            p = -c / b if abs(b) > 1e-9 else 0.0
        else:
            disc = b * b - 4 * a * c
            if disc < 0:
                return 0.0
            p1 = (-b + math.sqrt(disc)) / (2 * a)
            p2 = (-b - math.sqrt(disc)) / (2 * a)
            p = p1 if abs(p1) < abs(p2) else p2
        return float(min(abs(p) * 200, 100.0))
    except Exception:
        return 0.0


def generate_message(n_chars: int, seed: int = 7) -> str:
    """Випадкове повідомлення з рівномірно розподіленими бітами."""
    rng = random.Random(seed)
    return ''.join(chr(rng.randint(0, 255)) for _ in range(n_chars))


def main():
    import os
    os.makedirs(STEGO_DIR, exist_ok=True)
    cover = Image.open(INPUT_PATH).convert('RGB')
    print(f"Контейнер: {INPUT_PATH}, розмір: {cover.size}, "
          f"ємність LSB: {cover.size[0] * cover.size[1] * 3 // 8} байт\n")

    chi2_p_cover = chi_squared_pvalue(cover)
    rs_cover = rs_analysis(cover)
    print(f"Порожній контейнер:  χ²-p = {chi2_p_cover:.4f},  RS = {rs_cover:.2f}%\n")

    print(f"{'Розмір':>8} | {'PSNR (дБ)':>10} | {'MSE':>10} | "
          f"{'χ²-p':>10} | {'RS (%)':>8}")
    print('-' * 65)

    for n in MESSAGE_SIZES:
        msg = generate_message(n)
        stego = embed(cover, msg, KEY)
        stego.save(f"{STEGO_DIR}/lsb_{n}.bmp")

        recovered = extract(stego, len(msg.encode('latin-1')), KEY)
        assert recovered == msg, f"Помилка видобування для розміру {n}"

        psnr, mse = psnr_mse(cover, stego)
        p = chi_squared_pvalue(stego)
        rs = rs_analysis(stego)
        print(f"{n:>8} | {psnr:>10.2f} | {mse:>10.4f} | {p:>10.4f} | {rs:>8.2f}")


if __name__ == '__main__':
    main()
