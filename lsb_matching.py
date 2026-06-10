"""
Метод LSB-зіставлення (LSB Matching) для BMP-зображень.
На відміну від стандартного LSB, при невідповідності біта повідомлення
молодшому біту пікселя значення останнього випадково збільшується або
зменшується на одиницю. Позиції обираються псевдовипадково за ключем.
"""

import math
import random
import numpy as np
from PIL import Image
from scipy.stats import chi2 as chi2_dist

INPUT_PATH = r"C:\Users\daryn\OneDrive\Робочий стіл\code\code\1.bmp"
STEGO_DIR = r"C:\Users\daryn\OneDrive\Робочий стіл\code\code\stego"
KEY = 42
MESSAGE_SIZES = [100, 500, 1000, 1500, 2000]


def text_to_bits(text: str) -> str:
    return ''.join(format(b, '08b') for b in text.encode('latin-1'))


def bits_to_text(bits: str) -> str:
    bytes_ = bytearray(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return bytes_.decode('latin-1')


def select_positions(shape, n_bits, key):
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

    rng = random.Random(key + 1)
    for (y, x, c), bit in zip(positions, bits):
        b = int(bit)
        c_val = int(arr[y, x, c])
        if (c_val & 1) == b:
            continue
        if c_val == 0:
            r = +1
        elif c_val == 255:
            r = -1
        else:
            r = rng.choice([-1, +1])
        arr[y, x, c] = c_val + r
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
    print("df:", df, "chi2:", chi2)
    if df == 0:
        return 1.0
    return float(1 - chi2_dist.cdf(chi2, df))


def rs_analysis(img: Image.Image) -> float:
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
    rng = random.Random(seed)
    return ''.join(chr(rng.randint(0, 255)) for _ in range(n_chars))


def main():
    import os
    os.makedirs(STEGO_DIR, exist_ok=True)
    cover = Image.open(INPUT_PATH).convert('RGB')
    print(f"Контейнер: {INPUT_PATH}, розмір: {cover.size}\n")

    print(f"{'Розмір':>8} | {'PSNR (дБ)':>10} | {'MSE':>10} | "
          f"{'χ²-p':>10} | {'RS (%)':>8}")
    print('-' * 65)

    for n in MESSAGE_SIZES:
        msg = generate_message(n)
        stego = embed(cover, msg, KEY)
        stego.save(f"{STEGO_DIR}/lsbm_{n}.bmp")
        diff = diff_images(np.array(cover), np.array(stego))
        Image.fromarray(diff).save(f"{STEGO_DIR}/lsbm_diff_{n}.png") 
        recovered = extract(stego, len(msg.encode('latin-1')), KEY)
        assert recovered == msg, f"Помилка видобування для розміру {n}"

        psnr, mse = psnr_mse(cover, stego)
        p = chi_squared_pvalue(stego)
        print("p",p)
        rs = rs_analysis(stego)
        print(f"{n:>8} | {psnr:>10.2f} | {mse:>10.4f} | {p:>10.4f} | {rs:>8.2f}")


def diff_images(im1, im2):
    if im1.shape != im2.shape:
        print("Shapes are not the same.")
        return

    height= im1.shape[0]
    width = im1.shape[1]

    diff_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            if not np.array_equal(im1[i, j], im2[i, j]):
                diff_image[i,j] = [0, 0, 255]
            else:
                diff_image[i,j] = im1[i,j].copy()

    return diff_image


if __name__ == '__main__':
    main()
