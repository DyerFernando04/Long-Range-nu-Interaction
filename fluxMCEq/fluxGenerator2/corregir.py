import numpy as np
from pathlib import Path

def fix_theta_header(
    infile: str,
    theta_min=0.0001,
    theta_max=89.9999,
    n_theta=2000,
    out_suffix="_fixed",
    overwrite=False,
    precision=12,
):
    """
    Fix duplicated/rounded theta labels in the first header line of a MCEq table.

    Expected file format:
      - First line starts with '#' and contains column names:
        # E_GeV theta_...deg theta_...deg ...
      - Remaining lines are numeric with whitespace separators.
      - Data columns: 1 (E) + n_theta (fluxes at each theta)

    It rewrites only the header line; data lines stay untouched.
    """
    inpath = Path(infile)
    if not inpath.exists():
        raise FileNotFoundError(f"No existe: {infile}")

    # Build the exact theta grid you used
    theta_array = np.linspace(theta_min, theta_max, n_theta)

    # Read the first line (header)
    with inpath.open("r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        rest = f.read()

    if not header_line.lstrip().startswith("#"):
        raise ValueError("La primera línea no parece header (no empieza con '#').")

    # Tokenize header (drop leading '#')
    tokens = header_line.lstrip()[1:].strip().split()
    if len(tokens) < 2:
        raise ValueError("Header demasiado corto; no hay columnas de theta.")

    # First token should be E_GeV
    first = tokens[0]
    n_cols_in_header = len(tokens)  # includes E_GeV
    n_theta_in_header = n_cols_in_header - 1

    if n_theta_in_header != n_theta:
        raise ValueError(
            f"El header tiene {n_theta_in_header} columnas theta, "
            f"pero esperabas {n_theta}. "
            "Esto sugiere que el archivo no corresponde a tu grid 2000."
        )

    # Build a non-colliding theta label set using enough precision
    # Using positional formatting avoids scientific notation and preserves uniqueness.
    # Example: theta_12.345678901234deg
    theta_labels = [
        np.format_float_positional(t, precision=precision, unique=True, trim="k")
        for t in theta_array
    ]

    new_header = "# " + first + " " + " ".join([f"theta_{lab}deg" for lab in theta_labels]) + "\n"

    # Decide output path
    if overwrite:
        outpath = inpath
    else:
        outpath = inpath.with_name(inpath.stem + out_suffix + inpath.suffix)

    # Write repaired file
    with outpath.open("w", encoding="utf-8") as f:
        f.write(new_header)
        f.write(rest)

    print("OK")
    print("Input :", inpath)
    print("Output:", outpath)
    print("Theta labels precision:", precision)
    print("Example labels:", new_header.split()[1:6], "...")

# -------------------------
# USO
# -------------------------
fix_theta_header(
    "numu_flux_E_theta.txt",
    theta_min=0.0001,
    theta_max=89.9999,
    n_theta=2000,
    overwrite=False,   # True si quieres sobrescribir
    precision=16,      # 16 suele ser súper seguro
)
