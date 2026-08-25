import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"

NOODLE_BRANDS = ["Indomie", "Mie Sedaap", "Sarimi", "Supermi", "Pop Mie"]
NOODLE_FLAVORS = [
    "Goreng", "Ayam Bawang", "Soto", "Kari Ayam", "Rendang",
    "Baso Sapi", "Kaldu Ayam", "Ayam Geprek", "Mi Keriting", "Soto Koya",
]
NOODLE_SIZES = ["70gr", "75gr", "80gr", "85gr", "90gr"]

STAPLE_ITEMS = [
    ("Beras", ["Ramos", "Rojolele", "Pandan Wangi", "IR64", "Setra Ramos"], ["5kg", "10kg", "2kg", "1kg"]),
    ("Minyak Goreng", ["Bimoli", "Tropical", "Sania", "Filma", "Sunco"], ["1L", "2L", "5L", "900ml"]),
    ("Gula Pasir", ["Gulaku", "Rose Brand", "Gunungputri"], ["1kg", "500gr", "250gr"]),
    ("Tepung Terigu", ["Segitiga Biru", "Cakra Kembar", "Kunci Biru"], ["1kg", "500gr"]),
    ("Garam", ["Cap Kapal", "Refina", "Dolphin"], ["500gr", "250gr"]),
    ("Kecap Manis", ["ABC", "Bango", "Indofood"], ["600ml", "275ml", "135ml"]),
    ("Saus Sambal", ["ABC", "Indofood", "Del Monte"], ["340ml", "135ml"]),
]

TOILETRY_ITEMS = [
    ("Sabun Mandi", ["Lifebuoy", "Lux", "Nuvo", "Dettol"], ["Batangan", "Sabun Cair"]),
    ("Sampo", ["Pantene", "Clear", "Sunsilk", "Dove"], ["170ml", "340ml"]),
    ("Pasta Gigi", ["Pepsodent", "Ciptadent", "Formula"], ["75gr", "190gr"]),
    ("Deterjen", ["Rinso", "Attack", "Daia", "Soklin"], ["Bubuk", "Cair"]),
]


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

    for brand in NOODLE_BRANDS:
        for flavor in random.sample(NOODLE_FLAVORS, k=7):
            size = random.choice(NOODLE_SIZES)
            products.append(Product(brand, flavor, size))
            if random.random() < 0.6:
                other_size = random.choice([s for s in NOODLE_SIZES if s != size])
                products.append(Product(brand, flavor, other_size))

    for base, brands, sizes in STAPLE_ITEMS:
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

    for base, brands, sizes in TOILETRY_ITEMS:
        for brand in brands:
            for size in sizes:
                products.append(Product(f"{base} {brand}", "", size))

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
        try:
            value = float(num)
        except ValueError:
            value = None

        if value is not None and choice < 0.2:
            if unit in ("gr", "g", "gram"):
                return f"{value / 1000:g}kg"
            if unit == "kg":
                return f"{value * 1000:g}gr"
            if unit == "ml":
                return f"{value / 1000:g}L"
            if unit == "l":
                return f"{value * 1000:g}ml"

        if unit in ("gr", "g", "gram"):
            return f"{num} gr" if choice < 0.6 else (f"{num}g" if choice < 0.8 else f"{num}gram")
        if unit == "ml":
            return f"{num} ml" if choice < 0.6 else f"{num}ml"
        if unit == "l":
            return f"{num}L" if choice < 0.6 else f"{num} liter"
        if unit == "kg":
            return f"{num} kg" if choice < 0.6 else f"{num}KG"
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


def build_lexical_pairs(products: list[Product]) -> list[dict]:
    by_brand: dict[str, list[Product]] = defaultdict(list)
    for p in products:
        by_brand[p.brand].append(p)

    pairs = []
    for p in products:
        variants = {generate_variant(p.text) for _ in range(random.randint(5, 10))}
        variants = [v for v in variants if v.strip() and v != p.text] or [p.text]

        siblings = [s for s in by_brand[p.brand] if s.text != p.text]
        same_size_siblings = [s for s in siblings if s.size == p.size]
        preferred_siblings = same_size_siblings or siblings

        for v in variants:
            if siblings:
                negative = random.choice(preferred_siblings).text
                category = "F_size_trap" if not same_size_siblings else "G_rasa_trap"
                pairs.append({"anchor": p.text, "positive": v, "negative": negative, "category": category})
            else:
                pairs.append({"anchor": p.text, "positive": v, "negative": None, "category": "A_lexical"})
    return pairs


def build_brand_neighbor_negatives() -> list[tuple[str, str]]:
    neighbor_pairs: list[tuple[str, str]] = []

    for base, brands, sizes in STAPLE_ITEMS + TOILETRY_ITEMS:
        for size in sizes:
            for b1, b2 in zip(brands, brands[1:]):
                neighbor_pairs.append((f"{base} {b1} {size}", f"{base} {b2} {size}"))

    for flavor in ["Goreng", "Ayam Bawang", "Soto"]:
        for size in ["85gr", "90gr"]:
            for b1, b2 in zip(NOODLE_BRANDS, NOODLE_BRANDS[1:]):
                neighbor_pairs.append((f"{b1} {flavor} {size}", f"{b2} {flavor} {size}"))

    return neighbor_pairs


def build_brand_neighbor_triplets() -> list[dict]:
    triplets = []
    for anchor_text, neighbor_text in build_brand_neighbor_negatives():
        for _ in range(2):
            variant = generate_variant(anchor_text)
            if variant.strip() and variant != anchor_text:
                triplets.append({
                    "anchor": anchor_text,
                    "positive": variant,
                    "negative": neighbor_text,
                    "category": "H_brand_neighbor",
                })
    return triplets

B_SYNONYM_TRIPLETS = [
    ("Sabun Cair Lifebuoy 250ml", "Liquid Soap Lifebuoy 250ml", "Sabun Cair Dettol 250ml"),
    ("Minyak Goreng Bimoli 1L", "Cooking Oil Bimoli 1 Liter", "Minyak Goreng Tropical 1L"),
    ("Beras Pandan Wangi 5kg", "Rice Pandan Wangi 5 Kilogram", "Beras Rojolele 5kg"),
    ("Deterjen Bubuk Rinso 800gr", "Bubuk Cuci Rinso 800gr", "Deterjen Bubuk Attack 800gr"),
    ("Air Mineral Aqua 600ml", "Air Putih Kemasan Aqua 600ml", "Air Mineral Le Minerale 600ml"),
    ("Gula Pasir Gulaku 1kg", "Gula Putih Gulaku 1kg", "Gula Pasir Rose Brand 1kg"),
    ("Tepung Terigu Segitiga Biru 1kg", "Flour Segitiga Biru 1kg", "Tepung Terigu Cakra Kembar 1kg"),
    ("Kecap Manis ABC 600ml", "Sweet Soy Sauce ABC 600ml", "Kecap Manis Bango 600ml"),
    ("Saus Sambal ABC 340ml", "Chili Sauce ABC 340ml", "Saus Sambal Del Monte 340ml"),
    ("Pasta Gigi Pepsodent 190gr", "Odol Pepsodent 190gr", "Pasta Gigi Ciptadent 190gr"),
    ("Sampo Pantene 340ml", "Shampoo Pantene 340ml", "Sampo Clear 340ml"),
    ("Susu Ultra Full Cream 1L", "Milk Ultra Full Cream 1 Liter", "Susu Indomilk Full Cream 1L"),
    ("Kopi Kapal Api Sachet", "Kopi Kapal Api Bungkus Kecil", "Kopi ABC Sachet"),
    ("Teh Pucuk Harum 350ml", "Minuman Teh Pucuk Harum 350ml", "Teh Botol Sosro 350ml"),
    ("Mie Instan Indomie Goreng 85gr", "Mi Kering Instan Indomie Goreng 85gr", "Mie Sedaap Goreng 85gr"),
    ("Garam Cap Kapal 500gr", "Garam Dapur Cap Kapal 500gr", "Garam Refina 500gr"),
    ("Sabun Mandi Lux Batangan", "Bar Soap Lux Batangan", "Sabun Mandi Nuvo Batangan"),
    ("Pocari Sweat 350ml", "Minuman Isotonik Pocari Sweat 350ml", "Pocari Sweat 500ml"),
    ("Roma Kelapa 300gr", "Biskuit Roma Kelapa 300gr", "Roma Kelapa 227gr"),
    ("Oreo Original 137gr", "Biskuit Sandwich Cokelat Oreo Original 137gr", "Oreo Chocolate 137gr"),
    ("Chitato Sapi Panggang 68gr", "Keripik Kentang Chitato Sapi Panggang 68gr", "Chitato Balado 68gr"),
    ("Deterjen Rinso Cair", "Liquid Detergent Rinso", "Deterjen Attack Cair"),
    ("You C1000 Orange 140ml", "Minuman Vitamin C You C1000 Orange 140ml", "You C1000 Lemon 140ml"),
    ("Silverqueen Almond 65gr", "Cokelat Batang Silverqueen Almond 65gr", "Silverqueen Chunky Bar 65gr"),
    ("Deterjen Bubuk Attack 800gr", "Bubuk Cuci Attack 800gr", "Deterjen Bubuk Rinso 800gr"),
]

D_BRAND_GENERIC_TRIPLETS = [
    ("Aqua 600ml", "Air Mineral Botol 600ml Merek Aqua", "Le Minerale 600ml"),
    ("Indomie Goreng 85gr", "Mie Instan Goreng Merek Indomie 85gr", "Mie Sedaap Goreng 85gr"),
    ("Teh Botol Sosro 350ml", "Teh Kemasan Botol Merek Sosro 350ml", "Teh Pucuk Harum 350ml"),
    ("Rinso 800gr", "Deterjen Bubuk Merek Rinso 800gr", "Attack 800gr"),
    ("Pepsodent 190gr", "Pasta Gigi Merek Pepsodent 190gr", "Ciptadent 190gr"),
    ("Bimoli 1L", "Minyak Goreng Merek Bimoli 1 Liter", "Tropical 1L"),
    ("Kapal Api Sachet", "Kopi Sachet Merek Kapal Api", "Kopi ABC Sachet"),
    ("Chitato Sapi Panggang 68gr", "Keripik Kentang Merek Chitato Rasa Sapi Panggang 68gr", "Chitato Balado 68gr"),
    ("Lifebuoy Batangan", "Sabun Mandi Batang Merek Lifebuoy", "Lux Batangan"),
    ("Pocari Sweat 350ml", "Minuman Isotonik Merek Pocari Sweat 350ml", "Pocari Sweat 500ml"),
    ("Ultra Full Cream 1L", "Susu Cair Merek Ultra Full Cream 1 Liter", "Susu Indomilk Full Cream 1L"),
    ("Gulaku 1kg", "Gula Pasir Merek Gulaku 1kg", "Rose Brand 1kg"),
    ("Oreo Original 137gr", "Biskuit Sandwich Cokelat Merek Oreo Original 137gr", "Oreo Chocolate 137gr"),
    ("Silverqueen Chunky Bar 68gr", "Cokelat Batang Merek Silverqueen Chunky Bar 68gr", "Silverqueen Almond 68gr"),
    ("ABC Kecap Manis 600ml", "Kecap Manis Botol Merek ABC 600ml", "Bango Kecap Manis 600ml"),
    ("Dettol Sabun Cair", "Sabun Cair Antiseptik Merek Dettol", "Lifebuoy Sabun Cair"),
    ("Pantene 340ml", "Sampo Merek Pantene 340ml", "Sunsilk 340ml"),
    ("Rose Brand 1kg", "Gula Pasir Merek Rose Brand 1kg", "Gulaku 1kg"),
    ("Cakra Kembar 1kg", "Tepung Terigu Merek Cakra Kembar 1kg", "Segitiga Biru 1kg"),
    ("Del Monte Saus Sambal 340ml", "Saus Sambal Botol Merek Del Monte 340ml", "ABC Saus Sambal 340ml"),
    ("Sunsilk 340ml", "Sampo Merek Sunsilk 340ml", "Dove 340ml"),
    ("Formula Pasta Gigi 190gr", "Pasta Gigi Merek Formula 190gr", "Pepsodent Pasta Gigi 190gr"),
    ("Daia Bubuk", "Deterjen Bubuk Merek Daia", "Soklin Bubuk"),
    ("Le Minerale 600ml", "Air Mineral Botol Merek Le Minerale 600ml", "Aqua 600ml"),
    ("Better Coklat 135gr", "Wafer Cokelat Merek Better Coklat 135gr", "Wafer Tango Coklat 130gr"),
]

def build_handwritten_triplets(triples: list[tuple[str, str, str]], category: str) -> list[dict]:
    return [
        {"anchor": a, "positive": p, "negative": n, "category": category}
        for a, p, n in triples
    ]

def main() -> None:
    products = build_catalog()

    pairs = build_lexical_pairs(products)
    pairs += build_brand_neighbor_triplets()
    pairs += build_handwritten_triplets(B_SYNONYM_TRIPLETS, "B_sinonim")
    pairs += build_handwritten_triplets(D_BRAND_GENERIC_TRIPLETS, "D_brand_deskriptif")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "training_pairs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    by_category: dict[str, int] = defaultdict(int)
    n_triplets = 0
    for p in pairs:
        by_category[p["category"]] += 1
        if p["negative"]:
            n_triplets += 1

    print(f"seed products: {len(products)}")
    print(f"training pairs: {len(pairs)} ({n_triplets} with a hard negative)")
    for cat, n in sorted(by_category.items()):
        print(f"  {cat}: {n}")
    print(f"written to: {out_path}")

if __name__ == "__main__":
    main()
