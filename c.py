import os

import kagglehub

DATASET_SLUG = "dasmehdixtr/berkeley-multimodal-human-action-database"
OUTPUT_DIR = "./storage/berkeley_mhad"  


def main() -> str:
    path = kagglehub.dataset_download(DATASET_SLUG, output_dir=OUTPUT_DIR)
    print(f"Dataset baixado em: {OUTPUT_DIR}\n")
    print("Árvore de diretórios (até 3 níveis, até 10 arquivos por pasta):\n")

    for root, dirs, files in os.walk(OUTPUT_DIR):
        depth = root[len(OUTPUT_DIR):].count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or OUTPUT_DIR}/")
        for f in sorted(files)[:10]:
            print(f"{indent}  {f}")
        if len(files) > 10:
            print(f"{indent}  ... (+{len(files) - 10} arquivos)")

    return path


if __name__ == "__main__":
    main()