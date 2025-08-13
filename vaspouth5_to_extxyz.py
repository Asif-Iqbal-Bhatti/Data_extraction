#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert VASP results (via py4vasp vaspout.h5) to an Extended XYZ trajectory.

- Reads structure, forces, stress (and energy if available) in batches.
- Writes an .extxyz file compatible with ASE and other MD analysis tools.
- Uses robust logging, validation, and CLI options.

Extended XYZ conventions used here:
  * Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"  (Å)
  * Properties=species:S:1:pos:R:3:force:R:3
  * pbc="T T T"
  * Stress="s11 s12 s13 s21 s22 s23 s31 s32 s33" (units as provided by py4vasp)
  * Energy=<float> (if available from py4vasp)

Notes on units:
  - Positions are output in Å (fractional → cartesian via lattice).
  - Lattice is in Å.
  - Forces are written in eV/Å.
  - Stress comes from py4vasp; check your py4vasp/VASP version for units
    (commonly kBar). If you need GPa, see the inline comment on scaling.

Author: Refactored by M365 Copilot written by Asif iqbal BHATTI/Asif_em2r
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from py4vasp import Calculation
except ImportError as exc:
    raise SystemExit(
        "py4vasp is required for this script. Install with `pip install py4vasp`."
    ) from exc


def _configure_logging(verbosity: int) -> None:
    """Configure root logger based on verbosity level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _normalize_species(names: Iterable[str], split_char: Optional[str]) -> list[str]:
    """Normalize species names.

    Some VASP/py4vasp datasets carry species like "Li_1" or "S_2".
    If split_char is provided (e.g., "_"), we keep the token before the first split.

    Args:
        names: iterable of raw species labels from py4vasp.
        split_char: character to split on; None to disable splitting.

    Returns:
        List of cleaned species symbols (e.g., "Li", "S", ...).
    """
    species = []
    for name in names:
        if split_char and split_char in name:
            species.append(name.split(split_char, 1)[0])
        else:
            species.append(name)
    return species


def _format_flat(arr: np.ndarray, precision: int = 6) -> str:
    """Format a 1D array to a space-separated string with fixed precision."""
    return " ".join(f"{x:.{precision}f}" for x in np.asarray(arr).ravel())


def _read_block(calc: Calculation, sl: slice):
    """Read a block of structure, forces, stress (and optionally energy) from py4vasp."""
    structure_data = calc.structure[sl].read()
    # Forces: expected key is "forces". Fallback gracefully if different.
    forces_data = calc.force[sl].read()
    if isinstance(forces_data, dict):
        if "forces" in forces_data:
            forces = forces_data["forces"]
        elif "force" in forces_data:  # fallback
            forces = forces_data["force"]
        else:
            raise KeyError(
                f"Unexpected force keys in py4vasp read: {list(forces_data.keys())}"
            )
    else:
        # Some py4vasp versions may return an ndarray directly
        forces = forces_data

    # Stress: expected key is "stress".
    stress_data = calc.stress[sl].read()
    if isinstance(stress_data, dict):
        if "stress" in stress_data:
            stress = stress_data["stress"]
        else:
            raise KeyError(
                f"Unexpected stress keys in py4vasp read: {list(stress_data.keys())}"
            )
    else:
        stress = stress_data

    # Energy is optional; not all datasets have it or the same key names.
    energy = None
    try:
        energy_block = calc.energy[sl].read()
        # Common possibilities; adjust to your py4vasp version if needed.
        for candidate in ("free_energy", "energy", "total", "e_0_energy"):
            if isinstance(energy_block, dict) and candidate in energy_block:
                energy = energy_block[candidate]
                break
        # Some versions may return a plain ndarray:
        if energy is None and isinstance(energy_block, np.ndarray):
            energy = energy_block
    except Exception:
        energy = None

    return structure_data, forces, stress, energy


def _validate_shapes(
    forces: np.ndarray, stress: np.ndarray, n_atoms: int
) -> Tuple[int, int, int]:
    """Validate expected shapes and return (n_steps, n_atoms, 3)."""
    if forces.ndim != 3 or forces.shape[2] != 3:
        raise ValueError(f"Expected forces of shape (n_steps, n_atoms, 3), got {forces.shape}")
    if stress.ndim not in (2, 3):
        raise ValueError(f"Expected stress of shape (n_steps, 3, 3) or (n_steps, 6), got {stress.shape}")

    n_steps, fa, three = forces.shape
    if fa != n_atoms or three != 3:
        raise ValueError(
            f"Forces shape mismatch: forces({forces.shape}) vs n_atoms({n_atoms})"
        )
    return n_steps, n_atoms, 3


def _write_block_to_extxyz(
    fh,
    structure_data,
    forces: np.ndarray,
    stress: np.ndarray,
    energy: Optional[np.ndarray],
    species_split_char: Optional[str],
    precision: int,
    stress_scale: Optional[float] = None,
) -> int:
    """Write a block of frames to an already opened .extxyz file handle.

    Returns:
        Number of frames written.
    """
    lattice_vecs = np.asarray(structure_data["lattice_vectors"])  # (n_steps, 3, 3)
    frac_positions = np.asarray(structure_data["positions"])      # (n_steps, n_atoms, 3)
    raw_names = structure_data["names"]                           # (n_atoms,)
    species = _normalize_species(raw_names, species_split_char)

    # Validate
    n_steps, n_atoms, _ = frac_positions.shape
    _validate_shapes(forces, stress, n_atoms)

    # Some py4vasp versions may provide stress in shape (n_steps, 6); we prefer (3x3).
    if stress.ndim == 2 and stress.shape[1] == 6:
        # Convert Voigt (xx, yy, zz, yz, xz, xy) → full 3x3.
        # VASP/py4vasp Voigt ordering is commonly: xx, yy, zz, yz, xz, xy.
        # Build symmetric tensor accordingly.
        full = np.zeros((stress.shape[0], 3, 3), dtype=float)
        xx, yy, zz, yz, xz, xy = stress.T
        full[:, 0, 0] = xx
        full[:, 1, 1] = yy
        full[:, 2, 2] = zz
        full[:, 1, 2] = full[:, 2, 1] = yz
        full[:, 0, 2] = full[:, 2, 0] = xz
        full[:, 0, 1] = full[:, 1, 0] = xy
        stress = full

    if stress_scale is not None:
        # If you need to convert, e.g., kBar → GPa, use stress_scale = 0.1
        stress = stress * float(stress_scale)

    # Iterate frames
    frames_written = 0
    for i in range(n_steps):
        lattice = lattice_vecs[i]         # (3, 3)
        frac = frac_positions[i]          # (n_atoms, 3)
        pos = frac @ lattice              # fractional → cartesian (Å)
        f = forces[i]                     # (n_atoms, 3)
        s = stress[i]                     # (3, 3)

        # Header line 1: number of atoms
        fh.write(f"{n_atoms}\n")

        # Header line 2: key-value attributes
        lattice_flat = _format_flat(lattice, precision)
        stress_flat = _format_flat(s, precision)
        kv_parts = [
            f'Lattice="{lattice_flat}"',
            'Properties=species:S:1:pos:R:3:force:R:3',
            'pbc="T T T"',
            f'Stress="{stress_flat}"',
        ]
        # Energy, if available
        if energy is not None:
            try:
                en = float(np.asarray(energy[i]).squeeze())
                kv_parts.append(f"Energy={en:.{precision}f}")
            except Exception:
                pass

        fh.write(" ".join(kv_parts) + "\n")

        # Atom lines
        for a in range(n_atoms):
            sp = species[a]
            x, y, z = pos[a]
            fx, fy, fz = f[a]
            fh.write(
                f"{sp} "
                f"{x:.{precision}f} {y:.{precision}f} {z:.{precision}f} "
                f"{fx:.{precision}f} {fy:.{precision}f} {fz:.{precision}f}\n"
            )

        frames_written += 1

    return frames_written
def convert_vaspout_h5_to_extxyz(
    input_h5: Path,
    output_xyz: Path,
    batch_size: int = 20000,
    precision: int = 6,
    overwrite: bool = False,
    species_split_char: Optional[str] = "_",
    stress_scale: Optional[float] = None,
    show_progress: bool = True,
) -> int:
    """Convert a vaspout.h5 to an Extended XYZ trajectory.

    Args:
        input_h5: Path to py4vasp-readable vaspout.h5
        output_xyz: Destination .extxyz file
        batch_size: Number of steps to process per batch
        precision: Decimal digits for floating-point output
        overwrite: Whether to overwrite an existing output file
        species_split_char: Character to strip suffixes from species labels (e.g. "_")
        stress_scale: Optional factor to scale stress values (e.g., kBar→GPa is 0.1)
        show_progress: Show tqdm progress bar for batches

    Returns:
        Total number of frames written.
    """
    if not input_h5.exists():
        raise FileNotFoundError(f"Input file not found: {input_h5}")

    if output_xyz.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_xyz}. Use --overwrite to replace it."
        )

    if overwrite and output_xyz.exists():
        output_xyz.unlink()

    logging.info("Opening py4vasp Calculation...")
    calc = Calculation.from_file(str(input_h5))

    n_steps = calc.structure[:].number_steps()
    logging.info("Detected %d steps in the trajectory.", n_steps)

    total_written = 0
    with output_xyz.open("a", encoding="utf-8", newline="\n") as fh:
        ranges = range(0, n_steps, batch_size)
        iterator = tqdm(ranges, desc="Writing", unit="frames", disable=not show_progress)

        for start in iterator:
            stop = min(start + batch_size, n_steps)
            logging.debug("Processing slice %d:%d", start, stop)
            sl = slice(start, stop)

            structure_data, forces, stress, energy = _read_block(calc, sl)

            written = _write_block_to_extxyz(
                fh=fh,
                structure_data=structure_data,
                forces=np.asarray(forces),
                stress=np.asarray(stress),
                energy=energy if energy is not None else None,
                species_split_char=species_split_char,
                precision=precision,
                stress_scale=stress_scale,
            )
            total_written += written
            iterator.set_postfix(frames=total_written)

    logging.info("Finished. Wrote %d frames to %s", total_written, output_xyz)
    return total_written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert py4vasp vaspout.h5 to Extended XYZ trajectory."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("vaspout.h5"),
        help="Path to input vaspout.h5 (default: ./vaspout.h5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("mlmd_from_outcar.extxyz"),
        help="Output Extended XYZ file (default: ./mlmd_from_outcar.extxyz)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=20000,
        help="Number of steps per batch (default: 20000)",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for floats (default: 6)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing output (default behavior)",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        "--species-split-char",
        type=str,
        default="_",
        help='Character to strip suffixes from species labels (default: "_"); '
             'set to "" to disable.',
    )
    parser.add_argument(
        "--stress-scale",
        type=float,
        default=None,
        help="Optional multiplicative scale for stress (e.g., kBar→GPa: 0.1)",
    )
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Reduce logging verbosity"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity"
    )
    return parser
def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Map verbosity: default WARNING; -v -> INFO; -vv -> DEBUG; -q -> ERROR, -qq -> CRITICAL
    verbosity = args.verbose - args.quiet
    _configure_logging(verbosity=max(0, verbosity))

    species_split_char = args.species_split_char if args.species_split_char else None

    convert_vaspout_h5_to_extxyz(
        input_h5=args.input,
        output_xyz=args.output,
        batch_size=args.batch_size,
        precision=args.precision,
        overwrite=bool(args.overwrite),
        species_split_char=species_split_char,
        stress_scale=args.stress_scale,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
