"""Generate synthetic training pairs for fine-tuning the product-name
embedding model. Offline script, not part of the running app.

Builds a seed catalog of real Indonesian retail products (instant noodles,
staples, beverages, snacks), generates noisy supplier-style variants of each
(abbreviations, typos, shuffled word order, unit-format changes, casing), and
emits (anchor, positive[, hard_negative]) training pairs. Hard negatives are
same-brand products that differ in flavor or size -- the exact confusion the
hybrid matcher's own testing found the pretrained model struggles with.
"""
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Product:
    brand: str
    variant: str
    size: str

    @property
    def text(self) -> str:
        return " ".join(p for p in (self.brand, self.variant, self.size) if p)


def build_catalog() -> list[Product]:
    products: list[Product] = []

    noodle_brands = ["Indomie", "Mie Sedaap", "Sarimi", "Supermi", "Pop Mie"]
    noodle_flavors = [
        "Goreng", "Ayam Bawang", "Soto", "Kari Ayam", "Rendang",
        "Baso Sapi", "Kaldu Ayam", "Ayam Geprek", "Mi Keriting", "Soto Koya",
    ]
    noodle_sizes = ["70gr", "75gr", "80gr", "85gr", "90gr"]
    for brand in noodle_brands:
        for flavor in random.sample(noodle_flavors, k=7):
            size = random.choice(noodle_sizes)
            products.append(Product(brand, flavor, size))
            if random.random() < 0.6:
                other_size = random.choice([s for s in noodle_sizes if s != size])
                products.append(Product(brand, flavor, other_size))

    staple_items = [
        ("Beras", ["Ramos", "Rojolele", "Pandan Wangi", "IR64", "Setra Ramos"], ["5kg", "10kg", "2kg", "1kg"]),
        ("Minyak Goreng", ["Bimoli", "Tropical", "Sania", "Filma", "Sunco"], ["1L", "2L", "5L", "900ml"]),
        ("Gula Pasir", ["Gulaku", "Rose Brand", "Gunungputri"], ["1kg", "500gr", "250gr"]),
        ("Tepung Terigu", ["Segitiga Biru", "Cakra Kembar", "Kunci Biru"], ["1kg", "500gr"]),
        ("Garam", ["Cap Kapal", "Refina", "Dolphin"], ["500gr", "250gr"]),
        ("Kecap Manis", ["ABC", "Bango", "Indofood"], ["600ml", "275ml", "135ml"]),
        ("Saus Sambal", ["ABC", "Indofood", "Del Monte"], ["340ml", "135ml"]),
    ]
    for base, brands, sizes in staple_items:
        for brand in random.sample(brands, k=min(4, len(brands))):
            for size in random.sample(sizes, k=min(2, len(sizes))):
                products.append(Product(f"{base} {brand}", "", size))

    beverage_items = [
        ("Teh Botol Sosro", [], ["350ml", "450ml"]),
        ("Teh Pucuk Harum", [], ["350ml", "500ml"]),
        ("Teh Gelas", [], ["190ml", "220ml"]),
        ("Kopi Kapal Api", ["Special Mix", "Susu"], ["Sachet", "Renceng"]),
        ("Kopi ABC", ["Susu", "Mocha"], ["Sachet", "Renceng"]),
        ("Aqua", [], ["600ml", "1500ml", "330ml"]),
        ("Pocari Sweat", [], ["350ml", "500ml"]),
        ("Susu Ultra", ["Coklat", "Full Cream", "Low Fat"], ["250ml", "1L"]),
        ("Susu Indomilk", ["Coklat", "Stroberi", "Full Cream"], ["190ml", "1L"]),
        ("You C1000", ["Orange", "Lemon"], ["140ml"]),
    ]
    for base, variants, sizes in beverage_items:
        for variant in (variants or [""]):
            for size in random.sample(sizes, k=min(2, len(sizes))):
                products.append(Product(base, variant, size))

    snack_items = [
        ("Chitato", ["Sapi Panggang", "Rasa Ayam", "Balado"], ["68gr", "75gr"]),
        ("Taro", ["Net", "Original"], ["35gr", "60gr"]),
        ("Better", ["Coklat", "Keju"], ["135gr"]),
        ("Roma Kelapa", [], ["300gr", "227gr"]),
        ("Oreo", ["Original", "Chocolate"], ["137gr", "133gr"]),
        ("Silverqueen", ["Almond", "Chunky Bar"], ["65gr", "68gr"]),
        ("Wafer Tango", ["Coklat", "Keju"], ["130gr"]),
        ("Biskuit Malkist", ["Roma", "Coklat"], ["135gr"]),
    ]
    for base, variants, sizes in snack_items:
        for variant in (variants or [""]):
            for size in sizes:
                products.append(Product(base, variant, size))

    seen = set()
    unique = []
    for p in products:
        if p.text not in seen:
            seen.add(p.text)
            unique.append(p)
    return unique


def augment_abbreviate(text: str) -> str:
    words = text.split()
    out = []
    for w in words:
        if len(w) > 4 and random.random() < 0.6:
            core = w[0] + re.sub(r"[aiueoAIUEO]", "", w[1:])
            out.append(core.upper() if random.random() < 0.5 else core)
        else:
            out.append(w)
    return " ".join(out)


def augment_typo(text: str) -> str:
    if len(text) < 4:
        return text
    chars = list(text)
    # Avoid corrupting digits -- a dropped/duplicated digit silently changes
    # the product's quantity, which would mislabel the pair as "same product".
    candidates = [i for i in range(1, len(chars) - 1) if not chars[i].isdigit()]
    if not candidates:
        return text
    idx = random.choice(candidates)
    op = random.choice(["swap", "delete", "dup"])
    if op == "swap" and chars[idx + 1].isdigit():
        op = random.choice(["delete", "dup"])
    if op == "swap":
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif op == "delete":
        del chars[idx]
    else:
        chars.insert(idx, chars[idx])
    return "".join(chars)


def augment_shuffle(text: str) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    random.shuffle(words)
    return " ".join(words)


_UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(gr|g|gram|ml|l|kg)\b", re.IGNORECASE)


def augment_unit_format(text: str) -> str:
    def repl(m: re.Match) -> str:
        num, unit = m.group(1), m.group(2).lower()
        choice = random.random()
        if unit in ("gr", "g", "gram"):
            return f"{num} gr" if choice < 0.34 else (f"{num}g" if choice < 0.67 else f"{num}gram")
        if unit == "ml":
            return f"{num} ml" if choice < 0.5 else f"{num}ml"
        if unit == "l":
            return f"{num}L" if choice < 0.5 else f"{num} liter"
        if unit == "kg":
            return f"{num} kg" if choice < 0.5 else f"{num}KG"
        return m.group(0)

    return _UNIT_PATTERN.sub(repl, text)


def augment_casing(text: str) -> str:
    choice = random.random()
    if choice < 0.34:
        return text.upper()
    if choice < 0.67:
        return text.lower()
    return text.title()


AUGMENTERS = [augment_abbreviate, augment_typo, augment_shuffle, augment_unit_format, augment_casing]


def generate_variant(text: str) -> str:
    k = random.choice([1, 1, 2])
    for fn in random.sample(AUGMENTERS, k=k):
        text = fn(text)
    return text


def build_pairs(products: list[Product]) -> list[dict]:
    by_brand: dict[str, list[Product]] = defaultdict(list)
    for p in products:
        by_brand[p.brand].append(p)

    pairs = []
    for p in products:
        variants = {generate_variant(p.text) for _ in range(random.randint(5, 10))}
        variants = [v for v in variants if v.strip() and v != p.text] or [p.text]

        siblings = [s for s in by_brand[p.brand] if s.text != p.text]

        for v in variants:
            if siblings:
                negative = random.choice(siblings).text
                pairs.append({"anchor": p.text, "positive": v, "negative": negative})
            else:
                pairs.append({"anchor": p.text, "positive": v, "negative": None})
    return pairs


def main() -> None:
    products = build_catalog()
    pairs = build_pairs(products)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "train_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    n_triplets = sum(1 for p in pairs if p["negative"])
    print(f"seed products: {len(products)}")
    print(f"training pairs: {len(pairs)} ({n_triplets} with a hard negative)")
    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()
