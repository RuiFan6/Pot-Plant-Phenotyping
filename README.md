# Pot Plant Phenotyping Pipeline

A simple RGB-based 3D phenotyping pipeline using smartphone video and Structure-from-Motion.

## Features

- Smartphone RGB capture with turntable
- COLMAP SfM reconstruction
- Automatic plant segmentation
- Morphological trait extraction:
  - height
  - canopy diameter
  - projected area
  - volume
- Fruit detection via color + spatial clustering

## Pipeline

1. Extract frames from video
2. Background masking
3. Sparse SfM reconstruction
4. Dense reconstruction
5. Metric scaling using ArUco marker
6. Turntable alignment
7. Plant segmentation
8. Trait extraction

## Example traits

- Plant height
- Canopy diameter
- Convex hull volume
- Projected area
- Leaf orientation
- Fruit detection

## Requirements

- Python
- Open3D
- COLMAP
- OpenCV
- SciPy
- NumPy

## Example usage

Run the full pipeline for a plant:

python run_pipeline.py --plant_id 002

Or run a specific step for a plant:

python 03_extract_frames.py --plant_id 001
