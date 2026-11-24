
# 3D Mesh Generation and Lithofacies Assignment Tools

The `src/` directory contains four scripts that should be executed sequentially. Input files are stored in `data/`, and output files are written to `out/`.

## Dependencies
The scripts require the following Python packages:
- `numpy`
- `scipy`
- `gmsh` (Python API for Gmsh)

## Quick Start

Run the following commands from the `python_code/` directory.

**Before running the scripts, extract `data/Tec_FaciesZone-Indic.plt.zip` into the `data/` directory to obtain the corresponding `.plt` file. This step can be skipped if the file already exists.**

```bash
python3 1_create_tri_mesh_2d.py --quiet
python3 2_2d_ext_3d.py --quiet
python3 3_assign_facies_3d.py --quiet
python3 4_Find_Sourcing_Nodes.py
```

## Script Overview

| Script | Description and Default Inputs | Main Outputs (`out/`) |
|---|---|---|
| `1_create_tri_mesh_2d.py` | Generates a 2D triangular mesh based on the domain boundary (`boundary.csv`) and local mesh refinement points (`density_points_myCase.csv`). | `mesh_2d_tri.msh` and associated JSON files |
| `2_2d_ext_3d.py` | Generates a layered 3D tetrahedral mesh by extruding the 2D mesh using the top (`dem.xyz`) and bottom (`bottom.xyz`) surfaces. | `mesh_3d_tet.msh`, VTK files, and associated JSON files |
| `3_assign_facies_3d.py` | Assigns lithofacies to the 3D mesh based on `Tec_FaciesZone-Indic.plt` and `tsimFaci.out`. Lateral and bottom boundary information is included by default. | `mesh_3d_with_facies.vtk` |
| `4_Find_Sourcing_Nodes.py` | Identifies the nearest mesh node to each well location specified in `wells_location.csv` (`X,Y,Z`) using the lithofacies-assigned VTK mesh, and removes duplicate nodes. | `sourcing_nodes.csv` (`ID,X,Y,Z`, with IDs starting from 1) |

## Configuration

The first three scripts support the `--help` option to display available command-line arguments. File paths for the fourth script can be modified at the beginning of the script.

The following options can be used to customize the mesh resolution:

- `--background-size`: specifies the background mesh element size.
- `--refined-size`: specifies the mesh element size in locally refined regions.
- `--dz`: specifies the maximum vertical layer thickness of the 3D mesh (default: 20).
