import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

PAIRS: list[tuple[str, str, str]] = [
    ("Sarden ABC Saus Tomat 155gr", "SARDEN ABC SAUS TOMAT 155 GR", "A_lexical"),
    ("Sarden ABC Extra Pedas 155gr", "Srden ABC Extra Pedas 155gr", "A_lexical"),
    ("Sarden ABC Saus Tomat 425gr", "Sarden ABC Saus Tomat 0.425kg", "A_lexical"),
    ("Sarden Botan Saus Tomat 155gr", "SARDEN BOTAN SAUS TOMAT 155GR", "A_lexical"),

    ("Sarden ABC Saus Tomat 155gr", "Canned Sardines ABC Tomato Sauce 155gr", "B_sinonim"),
    ("Sarden Botan Extra Pedas 155gr", "Canned Sardines Botan Extra Spicy 155gr", "B_sinonim"),
    ("Sarden ABC Saus Tomat 425gr", "Canned Sardines ABC Tomato Sauce 425gr", "B_sinonim"),
    ("Sarden Botan Saus Tomat 425gr", "Canned Sardines Botan Tomato Sauce 425gr", "B_sinonim"),

    ("Sarden ABC Saus Tomat 155gr", "Sarden Kaleng Rasa Saus Tomat Merek ABC 155gr", "D_brand_deskriptif"),
    ("Sarden Botan Extra Pedas 155gr", "Sarden Kaleng Rasa Ekstra Pedas Merek Botan 155gr", "D_brand_deskriptif"),
    ("Sarden ABC Saus Tomat 425gr", "Sarden Kaleng Ukuran Besar Rasa Saus Tomat Merek ABC 425gr", "D_brand_deskriptif"),
    ("Sarden Botan Saus Tomat 155gr", "Sarden Kaleng Rasa Saus Tomat Merek Botan 155gr", "D_brand_deskriptif"),

    ("Sarden ABC Saus Tomat 155gr", "Sarden ABC Saus Tomat 425gr", "F_size_trap"),
    ("Sarden Botan Extra Pedas 155gr", "Sarden Botan Extra Pedas 425gr", "F_size_trap"),
    ("Sarden ABC Extra Pedas 155gr", "Sarden ABC Extra Pedas 425gr", "F_size_trap"),
    ("Sarden Botan Saus Tomat 155gr", "Sarden Botan Saus Tomat 425gr", "F_size_trap"),

    ("Sarden ABC Saus Tomat 155gr", "Sarden ABC Extra Pedas 155gr", "G_rasa_trap"),
    ("Sarden ABC Saus Tomat 155gr", "Sarden ABC Balado 155gr", "G_rasa_trap"),
    ("Sarden Botan Saus Tomat 155gr", "Sarden Botan Extra Pedas 155gr", "G_rasa_trap"),
    ("Sarden ABC Saus Tomat 425gr", "Sarden ABC Extra Pedas 425gr", "G_rasa_trap"),

    ("Sarden ABC Saus Tomat 155gr", "Sarden Botan Saus Tomat 155gr", "H_brand_neighbor"),
    ("Sarden ABC Extra Pedas 155gr", "Sarden Botan Extra Pedas 155gr", "H_brand_neighbor"),
    ("Sarden ABC Saus Tomat 425gr", "Sarden Botan Saus Tomat 425gr", "H_brand_neighbor"),
    ("Sarden ABC Balado 155gr", "Sarden Botan Balado 155gr", "H_brand_neighbor"),

    ("Susu Kental Manis Frisian Flag Coklat 370gr", "SUSU KENTAL MANIS FRISIAN FLAG COKLAT 370 GR", "A_lexical"),
    ("Susu Kental Manis Carnation Putih 385gr", "Susu Kntal Mnis Carnation Putih 385gr", "A_lexical"),
    ("Susu Kental Manis Frisian Flag Putih 370gr", "Susu Kental Manis Frisian Flag Putih 0.37kg", "A_lexical"),
    ("Susu Kental Manis Carnation Coklat 370gr", "susu kental manis carnation coklat 370gr", "A_lexical"),

    ("Susu Kental Manis Frisian Flag Coklat 370gr", "Sweetened Condensed Milk Frisian Flag Chocolate 370gr", "B_sinonim"),
    ("Susu Kental Manis Carnation Putih 385gr", "Sweetened Condensed Milk Carnation Original 385gr", "B_sinonim"),
    ("Susu Kental Manis Frisian Flag Putih 370gr", "Susu Manis Kental Kemasan Kaleng Frisian Flag Putih 370gr", "B_sinonim"),
    ("Susu Kental Manis Carnation Coklat 370gr", "Sweetened Condensed Milk Carnation Chocolate 370gr", "B_sinonim"),

    ("Frisian Flag Kental Manis 370gr", "Susu Kental Manis Kemasan Kaleng Merek Frisian Flag 370gr", "D_brand_deskriptif"),
    ("Carnation Kental Manis 385gr", "Susu Kental Manis Kemasan Kaleng Merek Carnation 385gr", "D_brand_deskriptif"),
    ("Frisian Flag Coklat 370gr", "Susu Kental Manis Rasa Coklat Merek Frisian Flag 370gr", "D_brand_deskriptif"),
    ("Carnation Putih 385gr", "Susu Kental Manis Rasa Original Merek Carnation 385gr", "D_brand_deskriptif"),

    ("Susu Kental Manis Frisian Flag Coklat 370gr", "Susu Kental Manis Frisian Flag Coklat 385gr", "F_size_trap"),
    ("Susu Kental Manis Carnation Putih 370gr", "Susu Kental Manis Carnation Putih 385gr", "F_size_trap"),
    ("Susu Kental Manis Frisian Flag Putih 370gr", "Susu Kental Manis Frisian Flag Putih 385gr", "F_size_trap"),
    ("Susu Kental Manis Carnation Coklat 370gr", "Susu Kental Manis Carnation Coklat 385gr", "F_size_trap"),

    ("Susu Kental Manis Frisian Flag Coklat 370gr", "Susu Kental Manis Frisian Flag Putih 370gr", "G_rasa_trap"),
    ("Susu Kental Manis Carnation Coklat 385gr", "Susu Kental Manis Carnation Putih 385gr", "G_rasa_trap"),
    ("Susu Kental Manis Frisian Flag Coklat 385gr", "Susu Kental Manis Frisian Flag Putih 385gr", "G_rasa_trap"),
    ("Susu Kental Manis Carnation Coklat 370gr", "Susu Kental Manis Carnation Putih 370gr", "G_rasa_trap"),

    ("Susu Kental Manis Frisian Flag Coklat 370gr", "Susu Kental Manis Carnation Coklat 370gr", "H_brand_neighbor"),
    ("Susu Kental Manis Frisian Flag Putih 385gr", "Susu Kental Manis Carnation Putih 385gr", "H_brand_neighbor"),
    ("Susu Kental Manis Frisian Flag Coklat 385gr", "Susu Kental Manis Carnation Coklat 385gr", "H_brand_neighbor"),
    ("Susu Kental Manis Frisian Flag Putih 370gr", "Susu Kental Manis Carnation Putih 370gr", "H_brand_neighbor"),
]

_POSITIVE_CATEGORIES = {"A_lexical", "B_sinonim", "D_brand_deskriptif"}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "eval_set.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for text_a, text_b, category in PAIRS:
            label = "positive" if category in _POSITIVE_CATEGORIES else "negative"
            row = {"text_a": text_a, "text_b": text_b, "label": label, "category": category}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"eval pairs: {len(PAIRS)}")
    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()
