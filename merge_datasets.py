import os
import shutil
from pathlib import Path

# Dossier principal du projet
BASE_DIR = Path(".")

# Dossier final fusionné
OUTPUT_DIR = BASE_DIR / "merged_dataset"

# Liste des classes = noms de tes dossiers
CLASSES = [
    "golden",
    "granola",
    "henry's",
    "king_cooking",
    "kitkat",
    "merendina",
    "MOMO",
    "nooco",
    "oreo",
    "petit_beurre",
    "prince",
    "sable",
    "tagger",
    "tango",
    "tonik"
]

SPLITS = ["train", "valid", "test"]

# Créer les dossiers de sortie
for split in SPLITS:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# Fusionner les datasets
for class_id, class_name in enumerate(CLASSES):
    dataset_path = BASE_DIR / class_name

    if not dataset_path.exists():
        print(f"⚠️ Dossier introuvable : {class_name}")
        continue

    print(f"Fusion de la classe : {class_name} → ID {class_id}")

    for split in SPLITS:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️ Dossier manquant pour {class_name} / {split}")
            continue

        for image_file in images_dir.glob("*.*"):
            if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_file = labels_dir / (image_file.stem + ".txt")

            if not label_file.exists():
                print(f"⚠️ Label manquant : {image_file.name}")
                continue

            # Nouveau nom pour éviter les conflits entre images
            safe_class_name = class_name.replace("'", "").replace(" ", "_")
            new_image_name = f"{safe_class_name}_{image_file.name}"
            new_label_name = f"{safe_class_name}_{label_file.name}"

            # Copier l'image
            shutil.copy(
                image_file,
                OUTPUT_DIR / split / "images" / new_image_name
            )

            # Modifier le fichier label : changer l'ancien id de classe vers le nouveau
            new_lines = []

            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) >= 5:
                    parts[0] = str(class_id)
                    new_lines.append(" ".join(parts))

            with open(OUTPUT_DIR / split / "labels" / new_label_name, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

# Créer le fichier data.yaml final
yaml_content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(CLASSES)}
names:
"""

for class_name in CLASSES:
    yaml_content += f"  - {class_name}\n"

with open(OUTPUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("\n✅ Fusion terminée avec succès !")
print("Dataset final créé dans le dossier : merged_dataset")