# Activity Cliff identification using BitBIRCH

This repository provides a comprehensive toolkit for identifying and analyzing activity cliffs in molecular datasets using BitBIRCH clustering algorithms.

<p align="center">
  <img src="Images/cliff_cluster.png.png" alt="Image 1" width="45%"/>
  <img src="Images/smooth_cluster.png.png" alt="Image 2" width="45%"/>
</p>

## Overview of methods

This repository provides tools for:
- **Activity Cliff Detection**: Identify pairs of structurally similar molecules with significantly different biological activities
- **BitBIRCH Clustering**: Efficient clustering algorithm optimized for molecular fingerprints
- **Multi-Fingerprint Analysis**: Support for RDKit, ECFP4, and MACCS molecular fingerprints
- **Smooth vs Cliff Clustering**: Compare clustering behavior for activity cliffs vs smooth activity relationships
- **Visualization**: Generate molecular structure visualizations for cluster analysis

## Key Features

- **Parallel Processing**: Multi-threaded analysis for handling large molecular datasets
- **Flexible Ordering**: Multiple ordering strategies for fingerprint processing (random, sum-based, centroid-based)
- **Recursive Analysis**: Optional recursive clustering to identify additional activity cliffs
- **Comprehensive Benchmarking**: Built-in parameter tuning and performance comparison tools
- **Rich Visualizations**: SVG-based molecular structure displays with activity annotations

