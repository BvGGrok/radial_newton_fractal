#!/usr/bin/env python3
"""
Radial Newton Fractal for z^8 - 1 = 0

This script generates a colorful Newton fractal whose basins of attraction
form radial "beam-like" structures. It is inspired by accelerator beam layouts
and uses the iteration:

    z_{n+1} = z_n - (z_n^8 - 1) / (8 * z_n^7)

Requirements:
    pip install numpy pillow

Usage:
    python radial_newton_fractal.py

The image is saved as radial_newton_fractal.png in the current directory.
You can easily change the degree, resolution, color palette, or view window.
"""

import numpy as np
from PIL import Image

def generate_radial_newton(
    width=1200,
    height=900,
    xmin=-2.5,
    xmax=2.5,
    ymin=-2.0,
    ymax=2.0,
    degree=8,
    max_iter=50,
    tolerance=1e-3,
    filename="radial_newton_fractal.png"
):
    """Generate and save the Newton fractal image."""
    # Create the complex plane grid
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    # Roots of unity (the attractors)
    roots = np.array([np.exp(2j * np.pi * k / degree) for k in range(degree)])

    # Distinct colors for each basin (feel free to customize)
    colors = np.array([
        [255, 180, 0],   # gold
        [0, 160, 255],   # cyan-blue
        [255, 80, 160],  # pink
        [80, 255, 120],  # green
        [180, 80, 255],  # purple
        [255, 140, 40],  # orange
        [40, 200, 200],  # teal
        [140, 140, 255], # light blue
    ], dtype=np.uint8)
    # Extend colors if degree > 8
    if degree > len(colors):
        extra = np.random.randint(0, 256, size=(degree - len(colors), 3), dtype=np.uint8)
        colors = np.vstack([colors, extra])

    converged = np.zeros((height, width), dtype=np.int32)

    # Newton iteration
    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Z = Z - (Z**degree - 1) / (degree * Z**(degree - 1))
        # Assign points that have converged to a root
        for r_idx, root in enumerate(roots):
            mask = (np.abs(Z - root) < tolerance) & (converged == 0)
            converged[mask] = r_idx + 1

    # Build the image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for r_idx in range(1, degree + 1):
        mask = converged == r_idx
        img[mask] = colors[r_idx - 1]
    # Unconverged points stay black
    img[converged == 0] = [0, 0, 0]

    Image.fromarray(img).save(filename)
    print(f"Saved {filename}")
    print(f"Resolution: {width}x{height}, degree={degree}, iterations={max_iter}")

if __name__ == "__main__":
    generate_radial_newton()
