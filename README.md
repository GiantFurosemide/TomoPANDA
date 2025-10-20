# TomoPANDA v0.1.2

CryoET membrane protein detection tool based on SE(3) equivariant transformer

## Introduction

TomoPANDA is a toolkit specifically designed for membrane protein detection in cryoET (cryo-electron tomography). It employs SE(3) equivariant transformer architecture to effectively handle membrane protein recognition and localization tasks in 3D tomographic data.

## Key Features

- **SE(3) Equivariant Transformer**: Advanced architecture based on geometric deep learning
- **Membrane Protein Detection**: Specialized detection and localization algorithms for membrane proteins
- **Mesh Geodesic Sampling**: Advanced membrane protein sampling with adaptive mesh density control
- **SDF-aligned Normals**: Surface normals aligned with signed distance field gradients
- **Multi-format Support**: Supports standard formats like MRC, RELION STAR, ChimeraX
- **Command Line Interface**: Simple and easy-to-use CLI tools
- **Modular Design**: Highly modular code architecture
- **Adaptive Processing**: Automatic parameter selection based on expected particle size

## Installation

### Dependency Installation

```bash
# Install basic dependencies

# Create virtual environment
python -m venv tomopanda

# Activate virtual environment
# Windows
tomopanda\Scripts\activate
# Linux/Mac
source tomopanda/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### Development Installation

```bash
git clone https://github.com/GiantFurosemide/TomoPANDA.git
cd TomoPANDA
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# View all available commands
tomopanda --help

# Use mesh geodesic sampling for particle picking
tomopanda sample mesh-geodesic --create-synthetic --output results/

# Use real membrane mask for sampling
tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/

# Use voxel surface sampling and export RELION 5 STAR
tomopanda sample voxel-sample --create-synthetic --output voxel_results/
tomopanda sample voxel-sample --mask membrane_mask.mrc --min-distance 20 --edge-distance 10 --output voxel_results/
```

### Python API Usage

```python
from tomopanda.core.mesh_geodesic import (
    create_mesh_geodesic_sampler,
    run_mesh_geodesic_sampling,
    save_sampling_outputs
)
from tomopanda.utils.mrc_utils import MRCReader

# Method 1: Using the sampler class
sampler = create_mesh_geodesic_sampler(
    expected_particle_size=20.0,
    smoothing_sigma=1.5,
    random_seed=42
)

# Load membrane mask
mask = MRCReader.read_membrane_mask("membrane_mask.mrc")

# Execute sampling
centers, normals = sampler.sample_membrane_points(mask, particle_radius=10.0)

# Method 2: One-shot convenience function
centers, normals = run_mesh_geodesic_sampling(
    mask,
    expected_particle_size=20.0,
    particle_radius=10.0,
    random_seed=42
)

# Save all standard outputs
save_sampling_outputs(
    output_dir="results/",
    centers=centers,
    normals=normals,
    tomogram_name="demo_tomo",
    particle_diameter=200.0,
    create_vis_script=True
)
```

## Command Line Interface

### Sample Command - Particle Sampling

```bash
# Basic usage
tomopanda sample mesh-geodesic [OPTIONS]

# Test with synthetic data
tomopanda sample mesh-geodesic --create-synthetic --output results/

# Use real membrane mask
tomopanda sample mesh-geodesic --mask membrane_mask.mrc --output results/

# Custom parameters (using expected particle size - taubin iterations will be auto-calculated)
tomopanda sample mesh-geodesic \
    --mask membrane_mask.mrc \
    --output results/ \
    --expected-particle-size 25.0 \
    --smoothing-sigma 2.0 \
    --verbose

# Alternative: use manual taubin iterations (without expected particle size)
tomopanda sample mesh-geodesic \
    --mask membrane_mask.mrc \
    --output results/ \
    --smoothing-sigma 2.0 \
    --taubin-iterations 15 \
    --verbose

# Generate mesh variants with noise injection
tomopanda sample mesh-geodesic \
    --mask membrane_mask.mrc \
    --output results/ \
    --expected-particle-size 20.0 \
    --add-noise \
    --noise-scale-factor 0.2 \
    --random-seed 42 \
    --verbose

# Voxel surface sampling
tomopanda sample voxel-sample [OPTIONS]

# Test with synthetic data
tomopanda sample voxel-sample --create-synthetic --output voxel_results/

# Use real membrane mask
tomopanda sample voxel-sample \
    --mask membrane_mask.mrc \
    --output voxel_results/ \
    --min-distance 25.0 \
    --edge-distance 12.0 \
    --verbose
```

### Other Commands

```bash
# Particle detection
tomopanda detect --tomogram tomogram.mrc --output detections/

# Model training
tomopanda train --config config.yaml --data data/

# Visualization
tomopanda visualize --input results/ --output plots/

# Data analysis
tomopanda analyze --input results/ --output analysis/

# Configuration management
tomopanda config --show
tomopanda config --set parameter=value

# Version information
tomopanda version
```

## Project Structure

```
tomopanda/
├── cli/                        # Command line interface
│   ├── commands/              # Command modules
│   │   ├── sample.py          # Sampling command
│   │   ├── detect.py          # Detection command
│   │   ├── train.py           # Training command
│   │   └── ...
│   └── main.py                # CLI main entry point
├── core/                       # Core algorithms
│   ├── mesh_geodesic.py       # Mesh geodesic sampling
│   ├── se3_transformer.py     # SE(3) transformer
│   └── ...
├── utils/                      # Utility modules
│   ├── mrc_utils.py           # MRC file processing
│   ├── relion_utils.py        # RELION format conversion
│   └── ...
├── examples/                   # Usage examples
│   └── mesh_geodesic_example.py
└── doc/                        # Documentation
    └── mesh_geodesic_algorithm.md
```

## Core Algorithms

### Mesh Geodesic Sampling

Mesh geodesic sampling is an advanced membrane protein sampling algorithm with adaptive mesh density control:

1. **Signed Distance Field**: Create SDF from binary membrane mask with optional noise injection
2. **Adaptive Mesh Extraction**: Extract triangular mesh using Marching Cubes with particle-size-based spacing
3. **Multi-level Mesh Processing**: Adaptive Taubin smoothing, decimation, and subdivision
4. **SDF-aligned Normals**: Surface normals aligned with SDF gradient direction
5. **Distance-constrained Sampling**: Poisson-like sampling with spatial hash grid
6. **Post-processing**: Non-maximum suppression and boundary feasibility checking

**Key Features:**
- Automatic parameter selection based on `expected_particle_size`
- SDF gradient-aligned surface normals for accurate orientation
- Adaptive mesh density control (fine for small particles, coarse for large particles)
- Support for mesh variants via random noise injection

For detailed algorithm description, please refer to [mesh_geodesic_algorithm.md](tomopanda/doc/mesh_geodesic_algorithm.md)

## Output Formats

- **sampling_coordinates.csv**: Contains x,y,z coordinates and nx,ny,nz normal vectors
- **particles.star**: Simplified RELION STAR format with particle coordinates and Euler angles
- **coordinates.csv**: Position and normal vectors for downstream processing
- **prior_angles.csv**: Angle priors for 3D classification
- **visualize_results.py**: Optional matplotlib-based visualization script
- **ChimeraX .cxc files**: For 3D visualization in ChimeraX

## Parameter Description

### Mesh Geodesic Sampling Parameters

- `--smoothing-sigma`: Gaussian smoothing parameter (default: 1.5)
- `--taubin-iterations`: Number of Taubin smoothing iterations (default: 10) - **mutually exclusive with --expected-particle-size**
- `--expected-particle-size`: Expected particle size in pixels for mesh density control - **automatically calculates taubin iterations** (mutually exclusive with --taubin-iterations)
- `--random-seed`: Random seed for mesh generation (None for deterministic)
- `--add-noise`: If set, inject small Gaussian noise into the smoothed mask before SDF (default: False). Use to generate mesh variants; keep off for smooth SDF, especially with simple shapes.
- `--noise-scale-factor`: Multiplier for the injected noise std, scaled by `smoothing_sigma` (default: 0.1). Effective std is `noise_scale_factor * max(1.0, smoothing_sigma)`.

**Note**: `--expected-particle-size` and `--taubin-iterations` are mutually exclusive. When `--expected-particle-size` is specified, the system automatically:
- Calculates sampling distance: `max(5.0, expected_particle_size / 2.0)`
- Determines SDF resolution: `max(0.5, min(3.0, expected_particle_size / 5.0))`
- Sets Marching Cubes spacing: `max(0.5, min(5.0, expected_particle_size / 10.0))`
- Maps to adaptive Taubin iterations based on particle size brackets

### Output Parameters

- `--tomogram-name`: Tomogram name
- `--particle-diameter`: Particle diameter (Angstroms)
- `--confidence`: Confidence score

### Voxel Surface Sampling Parameters

- `--min-distance`: Minimum Euclidean distance between sampling points (pixels)
- `--edge-distance`: Minimum distance from voxel boundary (pixels) to avoid half-particles
- `--voxel-size`: Voxel size (X,Y,Z) (Angstroms) for scaling coordinates to physical units

## Performance Optimization

1. For large datasets, consider using parallel processing
2. Adjust `min_distance` parameter to balance sampling density and computation time
3. Use appropriate `particle_radius` to avoid boundary conflicts

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'gdist'**
   ```bash
   pip install gdist
   ```

2. **ImportError: No module named 'open3d'**
   ```bash
   pip install open3d
   ```

3. **Insufficient Memory**
   - Reduce input data size
   - Increase `min_distance` parameter

## Contributing

Welcome to submit Issues and Pull Requests to improve the project!

## License

This project is licensed under the MIT License.

## References

1. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm.
2. Taubin, G. (1995). A signal processing approach to fair surface design.
3. Peyré, G., & Cohen, L. D. (2006). Geodesic methods for shape and surface processing.
