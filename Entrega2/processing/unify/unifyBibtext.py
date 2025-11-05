import argparse
import os
from pathlib import Path
import csv
import bibtexparser

# ------------------------------
# Configuración por defecto (relativa al repo)
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # .../Entrega2
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Campos requeridos por tipo. Si no se reconoce el tipo, se usan los por defecto.
REQUIRED_FIELDS = {
    "article": ["title", "author", "journal", "year", "doi", "abstract"],
    "inproceedings": ["title", "author", "booktitle", "year", "doi", "abstract"],
    "conference": ["title", "author", "booktitle", "year", "doi", "abstract"],  # alias habitual
    "book": ["title", "author", "publisher", "year", "isbn", "abstract"],
}
DEFAULT_FIELDS_FALLBACK = ["title", "author", "year"]


def discover_bib_files(paths):
    """
    Descubre archivos .bib a partir de una lista de rutas que pueden ser carpetas o archivos.
    - Si es carpeta, busca recursivamente *.bib
    - Si es archivo .bib, lo incluye
    """
    bib_files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            bib_files.extend(sorted(p.rglob("*.bib")))
        elif p.is_file() and p.suffix.lower() == ".bib":
            bib_files.append(p)
        else:
            print(f"⚠️ Ruta no válida o sin .bib: {p}")
    return bib_files


def load_bibtex_files(paths):
    """Carga todas las entradas BibTeX desde las rutas dadas (carpetas/archivos)."""
    entries = []
    bib_files = discover_bib_files(paths)
    if not bib_files:
        print(f"⚠️ No se encontraron archivos .bib en: {paths}")
    for file in bib_files:
        try:
            with open(file, encoding="utf-8") as bibfile:
                bib_database = bibtexparser.load(bibfile)
                entries.extend(bib_database.entries)
            print(f"✔️ Cargado: {file} ({len(bib_database.entries)} entradas)")
        except UnicodeDecodeError:
            # Fallback simple cuando hay caracteres problemáticos
            with open(file, encoding="utf-8", errors="ignore") as bibfile:
                bib_database = bibtexparser.load(bibfile)
                entries.extend(bib_database.entries)
            print(f"✔️ Cargado (con ignore errors): {file} ({len(bib_database.entries)} entradas)")
    print(f"📦 Entradas totales cargadas: {len(entries)}")
    return entries


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def get_identifier(entry):
    """Extrae un identificador único basado en DOI (preferente) o título normalizado."""
    doi = normalize_text(entry.get("doi", ""))
    title = normalize_text(entry.get("title", ""))
    return doi or title


def detect_duplicates(entries):
    """Detecta y separa duplicados basados en DOI o título normalizados."""
    seen = {}
    duplicates = []
    for entry in entries:
        identifier = get_identifier(entry)
        if not identifier:
            # Sin DOI ni título: no se usa para deduplicar, se mantiene tal cual
            # pero para evitar colisiones, usamos el id interno si existe
            key = f"__noid__::{entry.get('ID', id(entry))}"
        else:
            key = identifier

        if key in seen:
            duplicates.append(entry)
        else:
            seen[key] = entry

    unique_entries = list(seen.values())
    print(f"🧹 Únicos: {len(unique_entries)} | Duplicados: {len(duplicates)}")
    return unique_entries, duplicates


def clean_entries(entries):
    """Mantiene solo los campos esenciales según el tipo de entrada."""
    cleaned = []
    for entry in entries:
        entry_type = normalize_text(entry.get("ENTRYTYPE", ""))
        required = REQUIRED_FIELDS.get(entry_type, DEFAULT_FIELDS_FALLBACK)
        cleaned_entry = {key: entry[key] for key in required if key in entry and entry[key]}
        # Asegurar metadatos básicos
        cleaned_entry["ENTRYTYPE"] = entry_type or entry.get("ENTRYTYPE", "misc")
        cleaned_entry["ID"] = entry.get("ID", "")
        cleaned.append(cleaned_entry)
    return cleaned


def save_bibtex_file(entries, output_file: Path):
    """Guarda las entradas en un archivo BibTeX."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if entries:
        bib_database = bibtexparser.bibdatabase.BibDatabase()
        bib_database.entries = entries
        with open(output_file, "w", encoding="utf-8") as bibfile:
            bibtexparser.dump(bib_database, bibfile)
        print(f"✅ Guardado en: {output_file}")
    else:
        print(f"⚠️ No hay entradas para guardar en {output_file}")


def extract_fields_to_csv(entries, output_file: Path, fields):
    """Exporta un CSV con columnas seleccionadas, rellenando vacío si faltan."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            row = {k: e.get(k, "") for k in fields}
            writer.writerow(row)
    print(f"✅ CSV exportado: {output_file} ({len(entries)} filas)")


def main():
    parser = argparse.ArgumentParser(description="Unificar y limpiar archivos BibTeX del proyecto (Entrega2)")
    parser.add_argument(
        "--input",
        nargs="*",
        default=[str(DEFAULT_INPUT_DIR)],
        help="Rutas de entrada (.bib o carpetas). Por defecto: data/raw",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta de salida. Por defecto: data/processed",
    )
    parser.add_argument(
        "--cleaned-name",
        default="unified_cleaned.bib",
        help="Nombre del archivo .bib limpio (sin duplicados)",
    )
    parser.add_argument(
        "--dups-name",
        default="duplicates.bib",
        help="Nombre del archivo .bib con duplicados",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Si se indica, exporta CSV con campos seleccionados",
    )
    parser.add_argument(
        "--csv-name",
        default="unified_cleaned.csv",
        help="Nombre del archivo CSV a exportar (si --export-csv)",
    )
    parser.add_argument(
        "--csv-fields",
        default="title,author,year,doi,abstract",
        help="Campos a exportar al CSV, separados por coma",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_cleaned = output_dir / args.cleaned_name
    output_duplicates = output_dir / args.dups_name

    print("📂 Entradas desde:", ", ".join(args.input))
    print("📁 Salida en:", output_dir)

    all_entries = load_bibtex_files(args.input)
    unique_entries, duplicate_entries = detect_duplicates(all_entries)

    cleaned_entries = clean_entries(unique_entries)
    cleaned_duplicates = clean_entries(duplicate_entries)

    save_bibtex_file(cleaned_entries, output_cleaned)
    save_bibtex_file(cleaned_duplicates, output_duplicates)

    if args.export_csv:
        csv_fields = [s.strip() for s in args.csv_fields.split(",") if s.strip()]
        csv_path = output_dir / args.csv_name
        extract_fields_to_csv(cleaned_entries, csv_path, csv_fields)

    print("✨ Proceso completado.")


if __name__ == "__main__":
    main()
