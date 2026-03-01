# Protein Structure Visualization Guide

## Overview

AutoEvolve-ProteinFold includes sophisticated 3D visualization capabilities with confidence coloring and dynamic folding animations.

## Features

### 1. Static PyMOL Visualizations

**Publication-quality renderings** with multiple styles:
- Cartoon representation (default)
- Surface representation
- Ribbon diagram
- Stick model

**Confidence coloring**:
- Red: Low confidence (0-33%)
- Yellow: Medium confidence (34-66%)
- Green: High confidence (67-100%)

### 2. Interactive Web Viewer

**Browser-based 3D viewer** using py3Dmol:
- Rotate, zoom, pan with mouse
- Switch between visualization styles
- Confidence-colored structure
- No software installation required

### 3. Dynamic Folding Animations

**Watch proteins fold in real-time**:
- Shows transition from unfolded to folded state
- High-confidence regions fold faster
- Realistic physics-inspired motion
- Exportable as GIF or video

## Usage

### Generate Static Visualizations

```bash
# Basic visualization
python scripts/visualize_predictions.py \
  --checkpoint weights/latest.pt \
  --sequence "MKTAYIAK..." \
  --style cartoon

# Multiple views and styles
python scripts/visualize_predictions.py \
  --checkpoint weights/latest.pt \
  --sequence "MKTAYIAK..." \
  --multiple-views

# Interactive web version
python scripts/visualize_predictions.py \
  --checkpoint weights/latest.pt \
  --sequence "MKTAYIAK..." \
  --web
```

**Output**:
- `visualizations/predicted_structure.pdb` - Structure file
- `visualizations/structure_cartoon.png` - Rendered image
- `visualizations/structure_interactive.html` - Web viewer

### Create Folding Animation

```bash
# Generate GIF animation
python scripts/animate_folding.py \
  --checkpoint weights/latest.pt \
  --sequence "MKTAYIAK..." \
  --n-steps 50 \
  --fps 10

# Generate frame sequence
python scripts/animate_folding.py \
  --checkpoint weights/latest.pt \
  --sequence "MKTAYIAK..." \
  --format frames
```

**Output**:
- `animations/folding_animation.gif` - Complete animation
- `animations/frames/frame_0000.png` through `frame_0050.png`

## Automated Generation

Visualizations are automatically generated:

**Weekly** (Sunday at noon):
- Multiple view angles
- All visualization styles  
- Interactive web viewer
- Folding animation

**Manual trigger**:
1. Go to Actions tab
2. Select "Generate Visualizations"
3. Click "Run workflow"
4. Optional: Provide custom sequence

## Visualization Styles

### Cartoon
**Best for**: Overall structure, secondary structure elements
- Shows alpha helices and beta sheets clearly
- Smooth, publication-ready appearance
- Confidence-colored by residue

### Surface
**Best for**: Understanding protein shape, binding pockets
- Shows accessible surface area
- Good for visualizing cavities and grooves
- Semi-transparent with confidence coloring

### Ribbon
**Best for**: Backbone topology
- Emphasizes chain connectivity
- Minimal, clean representation
- Useful for comparing structures

### Sticks
**Best for**: Detailed atomic-level analysis
- Shows individual residues
- Good for close-up examination
- Confidence-colored per atom

## Confidence Interpretation

### Color Scheme

The model's prediction confidence is encoded in colors:

**Red regions (0-33%)**:
- Model is uncertain about structure
- May represent:
  - Flexible loops
  - Disordered regions
  - Insufficient training data
  - Novel structural motifs

**Yellow regions (34-66%)**:
- Moderate confidence
- Likely correct but could vary
- May represent:
  - Side chain orientations
  - Loop conformations
  - Mobile domains

**Green regions (67-100%)**:
- High confidence prediction
- Very likely to be accurate
- Typically represents:
  - Well-defined secondary structures
  - Stable core regions
  - Conserved structural elements

### Interpreting Confidence

**High average confidence (>70%)**:
- Structure likely accurate overall
- Safe to use for modeling
- Consider experimental validation

**Medium confidence (40-70%)**:
- Structure provides good starting point
- Refine with additional data
- Useful for hypothesis generation

**Low confidence (<40%)**:
- Treat as preliminary model
- May need more training data
- Compare with homologous structures

## Folding Animation Details

### How It Works

1. **Initial state**: Extended chain (unfolded)
2. **Folding process**: Gradual transition to predicted structure
3. **Confidence-driven**: High-confidence regions fold faster
4. **Thermal motion**: Random fluctuations simulate dynamics
5. **Final state**: Predicted folded structure

### Animation Parameters

**Number of steps** (`--n-steps`):
- Default: 50 frames
- More frames = smoother but slower
- Recommend: 30-100 for GIF, 100-200 for video

**FPS** (`--fps`):
- Default: 10 frames/second
- Lower = slower, more detail visible
- Higher = faster, more fluid motion
- Recommend: 10-15 for GIF, 24-30 for video

### Animation Interpretation

The folding animation is **illustrative**, not mechanistically accurate:
- Shows final structure formation
- Demonstrates confidence patterns
- **Not** a molecular dynamics simulation
- **Not** showing actual folding pathway

For accurate folding kinetics, use molecular dynamics simulations.

## File Formats

### PDB Files
Standard Protein Data Bank format:
- Contains 3D coordinates (C-alpha atoms)
- B-factor column = confidence (0-100)
- Compatible with PyMOL, Chimera, VMD, etc.
- Can be uploaded to protein structure servers

### PNG Images
High-resolution renderings:
- 1920x1080 resolution
- 300 DPI for publication quality
- Transparent or white background
- Ready for figures and presentations

### HTML Interactive
Standalone web viewer:
- Self-contained (no server needed)
- Works offline
- Share via email or web hosting
- Mobile-friendly

### GIF Animations
Animated image format:
- Loops continuously
- Works in browsers, PowerPoint, etc.
- Typically 2-5 MB file size
- No special software required

## Tips & Best Practices

### For Publications
1. Use cartoon style with confidence coloring
2. Generate multiple views (front, side, top)
3. Export at 300 DPI
4. Include confidence legend in caption
5. Cite model generation number

### For Presentations
1. Use folding animation to explain process
2. Interactive HTML for live demos
3. Surface view to highlight features
4. Slow FPS (10) for audience to follow

### For Analysis
1. Generate all styles for comprehensive view
2. Focus on red regions (low confidence)
3. Compare with experimental structures
4. Use PDB for downstream modeling

### For Social Media
1. Folding animation as GIF
2. Bright, high-contrast colors
3. Short (2-5 seconds)
4. Add text overlay with tool name

## Troubleshooting

### PyMOL not found
```bash
# Ubuntu/Debian
sudo apt-get install pymol

# macOS
brew install pymol

# pip (open-source version)
pip install pymol-open-source
```

### Web viewer not loading
- Requires modern browser (Chrome, Firefox, Safari)
- Check internet connection (loads 3Dmol library from CDN)
- Try opening HTML file locally first

### Animation too large
- Reduce `--n-steps` (fewer frames)
- Lower resolution in code (decrease DPI)
- Use video format instead of GIF

### Visualization artifacts
- Try different random seed
- Increase training data
- Check sequence for errors

## Examples

See `visualizations/` directory for example outputs from different model generations.

Compare early vs. late generations to see improvement in:
- Structural detail
- Confidence levels
- Visual quality
- Biological realism
