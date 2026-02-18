import os
import json
from pathlib import Path

OUTPUT_FILE = "combined_output.py"

EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    "build",
    "dist",
    "pipeline_cache",
    "results",
    "tmp",
    "base_model"
}


def should_exclude_dir(path: Path) -> bool:
    """Return True if any part of the path matches an excluded directory."""
    return any(part in EXCLUDED_DIRS for part in path.parts)


def extract_code_from_notebook(notebook_path: Path):
    try:
        with notebook_path.open("r", encoding="utf-8") as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Skipping notebook (read error): {notebook_path} -> {e}")
        return []

    code_lines = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            for line in cell.get("source", []):
                if not line.endswith("\n"):
                    line += "\n"
                code_lines.append(line)
            code_lines.append("\n")

    return code_lines


def extract_text_file(file_path: Path):
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        try:
            with file_path.open("r", encoding="latin-1") as f:
                print(f"Warning: {file_path} read using latin-1 encoding")
                return f.readlines()
        except Exception as e:
            print(f"Skipping file (encoding error): {file_path} -> {e}")
            return []
    except Exception as e:
        print(f"Skipping file (read error): {file_path} -> {e}")
        return []


def combine_files():
    # Use the script's directory as the root
    root_dir = Path(__file__).parent.resolve()
    output_path = root_dir / OUTPUT_FILE
    script_path = Path(__file__).resolve()

    combined_code = []
    files_processed = 0

    for current_path, dirs, files in os.walk(root_dir):
        current_path = Path(current_path)

        # Prune excluded directories based on full path
        dirs[:] = [
            d for d in sorted(dirs)
            if not should_exclude_dir(current_path / d)
        ]

        for file in sorted(files):
            full_path = current_path / file

            # Skip this script itself
            if full_path.resolve() == script_path:
                continue

            # Skip output file
            if full_path.resolve() == output_path.resolve():
                continue

            # Only include supported extensions
            if full_path.suffix not in {".py", ".ipynb", ".yaml", ".yml"}:
                continue

            relative_path = full_path.relative_to(root_dir)

            combined_code.append(f"# ===== File: {relative_path} =====\n")

            if full_path.suffix in {".py", ".yaml", ".yml"}:
                combined_code.extend(extract_text_file(full_path))
            elif full_path.suffix == ".ipynb":
                combined_code.extend(extract_code_from_notebook(full_path))

            combined_code.append("\n\n")
            files_processed += 1

    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(combined_code)

    print(f"\nSuccessfully combined {files_processed} files into: {output_path}")


if __name__ == "__main__":
    combine_files()
