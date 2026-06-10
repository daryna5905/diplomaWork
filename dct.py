"""
Стандартний метод DCT для JPEG-зображень.
Розкладає Y-канал зображення на блоки 8x8, застосовує дискретне косинусне
перетворення, квантує коефіцієнти за стандартною матрицею JPEG (якість 50),
і замінює молодший біт коефіцієнтів середніх частот на біти повідомлення.
Порядок обходу блоків та позицій усередині блоку визначається ключем.
"""

import math
import random
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.stats import chi2 as chi2_dist

INPUT_PATH = r"C:\Users\daryn\OneDrive\Робочий стіл\code\code\2.jpeg"
STEGO_DIR = r"C:\Users\daryn\OneDrive\Робочий стіл\code\code\stego"
KEY = 42
MESSAGE_SIZES = [100, 500, 1000, 1500, 2000]

Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)

ZIGZAG = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]
MID_FREQ_POSITIONS = ZIGZAG[6:28]


def text_to_bits(text):
    return ''.join(format(b, '08b') for b in text.encode('latin-1'))


def dct2(b): return dctn(b, type=2, norm='ortho')
def idct2(b): return idctn(b, type=2, norm='ortho')


def get_y_channel(img):
    ycbcr = img.convert('YCbCr')
    arr = np.array(ycbcr, dtype=np.float64)
    return arr[:, :, 0] - 128.0, arr[:, :, 1], arr[:, :, 2]


def assemble_image(y, cb, cr):
    y_clip = np.clip(y + 128.0, 0, 255)
    arr = np.stack([y_clip, cb, cr], axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode='YCbCr').convert('RGB')


def embed(cover, message, key):
    bits = text_to_bits(message)
    y, cb, cr = get_y_channel(cover)
    H, W = y.shape
    bH, bW = H // 8 * 8, W // 8 * 8
    y_emb = y[:bH, :bW].copy()

    rng = random.Random(key)
    n_blocks_h, n_blocks_w = bH // 8, bW // 8
    n_blocks = n_blocks_h * n_blocks_w
    block_indices = list(range(n_blocks))
    rng.shuffle(block_indices)

    bit_idx = 0
    for blk_idx in block_indices:
        if bit_idx >= len(bits):
            break
        br, bc = blk_idx // n_blocks_w, blk_idx % n_blocks_w
        block = y_emb[br*8:br*8+8, bc*8:bc*8+8]
        F = dct2(block)
        FQ = np.round(F / Q_MATRIX).astype(np.int32)

        local_rng = random.Random(key + blk_idx * 13)
        positions = list(MID_FREQ_POSITIONS)
        local_rng.shuffle(positions)

        for (u, v) in positions:
            if bit_idx >= len(bits):
                break
            bit = int(bits[bit_idx])
            val = int(FQ[u, v])
            absval = abs(val)
            new_abs = (absval & ~1) | bit
            FQ[u, v] = new_abs if val >= 0 else -new_abs
            bit_idx += 1

        F_new = FQ.astype(np.float64) * Q_MATRIX
        y_emb[br*8:br*8+8, bc*8:bc*8+8] = idct2(F_new)

    y_final = y.copy()
    y_final[:bH, :bW] = y_emb
    return assemble_image(y_final, cb, cr)


def psnr_mse(cover, stego):
    c = np.array(cover, dtype=np.float64)
    s = np.array(stego, dtype=np.float64)
    mse = float(np.mean((c - s) ** 2))
    if mse == 0:
        return float('inf'), 0.0
    return 10 * math.log10(255 ** 2 / mse), mse


# def chi_squared_dct(img):
#     """χ²-стегоаналіз на парі (0, 1) середньочастотних DCT-коефіцієнтів.
#     Повертає (chi2_stat, balance) - значення статистики та коефіцієнт
#     дисбалансу |n0 - n1| / (n0 + n1). Менше = ймовірніше вбудовування."""
#     y, _, _ = get_y_channel(img)
#     H, W = y.shape
#     bH, bW = H // 8 * 8, W // 8 * 8
#     n0 = n1 = 0
#     for br in range(bH // 8):
#         for bc in range(bW // 8):
#             block = y[br*8:br*8+8, bc*8:bc*8+8]
#             F = dct2(block)
#             FQ = np.round(F / Q_MATRIX).astype(np.int32)
#             for (u, v) in MID_FREQ_POSITIONS:
#                 val = abs(int(FQ[u, v]))
#                 if val == 0:
#                     n0 += 1
#                 elif val == 1:
#                     n1 += 1
#     s = n0 + n1
#     if s < 5:
#         return 0.0, 1.0
#     expected = s / 2
#     chi2 = (n0 - expected) ** 2 / expected + (n1 - expected) ** 2 / expected
#     balance = abs(n0 - n1) / s
#     return float(chi2), float(balance)

def chi_squared_dct(img):
    """
    Покращений χ²-аналіз DCT (pairwise histogram method).
    Аналізує пари (0,1), (2,3), (4,5)... для |FQ|.
    """

    y, _, _ = get_y_channel(img)
    H, W = y.shape

    bH, bW = (H // 8) * 8, (W // 8) * 8

    # гістограма значень
    hist = {}

    for br in range(bH // 8):
        for bc in range(bW // 8):

            block = y[br*8:br*8+8, bc*8:bc*8+8]
            F = dct2(block)
            FQ = np.round(F / Q_MATRIX).astype(np.int32)

            for (u, v) in MID_FREQ_POSITIONS:
                val = abs(int(FQ[u, v]))

                # обмежимо аналіз (щоб не роздувати хвости)
                if val > 20:
                    continue

                hist[val] = hist.get(val, 0) + 1

    chi2 = 0.0
    total_pairs = 0

    # χ² по парах (2k, 2k+1)
    for k in range(0, 20, 2):

        n0 = hist.get(k, 0)
        n1 = hist.get(k + 1, 0)

        s = n0 + n1
        if s == 0:
            continue

        expected = s / 2.0

        chi2 += (n0 - expected) ** 2 / expected
        chi2 += (n1 - expected) ** 2 / expected

        total_pairs += 1

    # нормалізація
    if total_pairs == 0:
        return 0.0, 1.0

    chi2 = chi2 / total_pairs

    # додатковий індикатор (як у тебе було)
    total = sum(hist.values())
    balance = 0.0
    if total > 0:
        balance = sum(abs(hist.get(k, 0) - hist.get(k+1, 0))
                      for k in range(0, 20, 2)) / total

    return float(chi2), float(balance)

def generate_message(n_chars, seed=7):
    rng = random.Random(seed)
    return ''.join(chr(rng.randint(0, 255)) for _ in range(n_chars))


def main():
    import os
    os.makedirs(STEGO_DIR, exist_ok=True)
    cover = Image.open(INPUT_PATH).convert('RGB')
    print(f"Контейнер: {INPUT_PATH}, розмір: {cover.size}\n")

    chi2_c, bal_c = chi_squared_dct(cover)
    print(f"Порожній контейнер:  χ² = {chi2_c:.0f},  дисбаланс (0,1) = {bal_c:.4f}\n")

    print(f"{'Розмір':>8} | {'PSNR (дБ)':>10} | {'MSE':>10} | "
          f"{'χ²':>10} | {'дисбаланс':>10}")
    print('-' * 65)

    for n in MESSAGE_SIZES:
        msg = generate_message(n)
        stego = embed(cover, msg, KEY)
        stego.save(f"{STEGO_DIR}/dct_{n}.png")
        psnr, mse = psnr_mse(cover, stego)
        chi2, bal = chi_squared_dct(stego)
        print(f"{n:>8} | {psnr:>10.2f} | {mse:>10.4f} | {chi2:>10.0f} | {bal:>10.4f}")


if __name__ == '__main__':
    main()
