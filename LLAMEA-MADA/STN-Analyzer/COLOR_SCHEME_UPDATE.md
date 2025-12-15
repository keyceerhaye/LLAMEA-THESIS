# Color Scheme & Static PNG Enhancement

## Summary of Changes

### ✅ New Vibrant Color Scheme

The operator colors have been redesigned for maximum clarity and visual distinction:

| Operator      | New Color             | Hex Code  | Description                          |
| ------------- | --------------------- | --------- | ------------------------------------ |
| **INIT**      | 🟡 **Gold**           | `#FFD700` | **Most vibrant!** Stands out clearly |
| **MUTATION**  | 🔴 **Deep Pink**      | `#FF1493` | **Highly distinct** from all others  |
| **CROSSOVER** | 🔵 **Dark Turquoise** | `#00CED1` | **Clear separation** from pink       |
| **REFINE**    | 🟣 **Medium Purple**  | `#9370DB` | **Unique hue** different from others |

**Why These Colors?**

- **Maximum Separation**: Pink and turquoise are on opposite sides of the color wheel
- **High Vibrance**: Gold is the most eye-catching for initial population
- **Color Blind Friendly**: Strong contrast in both hue and brightness
- **Professional**: Publication-quality color palette

### ✅ New Professional Static PNG

A brand new `stn_static.png` visualization has been created with:

#### Features:

1. **High Resolution**: 300 DPI for publication quality
2. **Professional Layout**: Split view with graph and comprehensive legend
3. **Enhanced Color Scheme**: Using the vibrant new colors
4. **Comprehensive Legend Panel**:
   - Network Statistics (nodes, edges, generations, fitness)
   - Operator Colors with counts
   - Operator Performance (success rates, avg fitness)
   - Visual Encoding explanations
5. **Smart Labeling**: Top 10 algorithms labeled automatically
6. **Variable Edge Widths**: Thickness indicates improvement magnitude
7. **Variable Node Sizes**: Size indicates fitness level

#### File Location:

```
stn_outputs/[experiment-name]/stn_static.png
```

### ✅ Updated Interactive HTML

The interactive HTML has also been updated with the new color scheme, maintaining all its interactive features:

- Enhanced tooltips
- Statistics panel
- Operator performance metrics
- Interactive controls (zoom, pan, physics toggle)

## Before vs After Colors

### Old Color Scheme (Hard to Distinguish):

- Init: Mint Green (#95E1D3)
- Mutation: Coral Red (#FF6B6B)
- Crossover: Teal (#4ECDC4) ← **Too similar to mint green!**
- Refine: Sky Blue (#45B7D1) ← **Too similar to teal!**

### New Color Scheme (Highly Distinct):

- Init: **Gold (#FFD700)** ← Most vibrant!
- Mutation: **Deep Pink (#FF1493)** ← Completely unique
- Crossover: **Dark Turquoise (#00CED1)** ← Clear separation
- Refine: **Medium Purple (#9370DB)** ← Unique purple hue

## How to Use

### Generate Static PNG for Any Experiment:

```bash
cd STN-Analyzer
python generate_static_png.py "../stn_outputs/[your-experiment-name]"
```

This will:

1. Create `stn_static.png` with professional layout
2. Update `stn_interactive.html` with new colors
3. Both use the enhanced color scheme

### View the Results:

**Static PNG** (for presentations/papers):

```
stn_outputs/[experiment]/stn_static.png
```

- 300 DPI resolution
- Comprehensive legend included
- Professional layout
- Ready for publication

**Interactive HTML** (for exploration):

```
stn_outputs/[experiment]/stn_interactive.html
```

- All interactive features
- New vibrant colors
- Enhanced tooltips
- Statistics panel

## Visual Comparison

### What Each Color Represents:

🟡 **INIT (Gold)**

- Initial population algorithms
- Most vibrant to draw attention
- Usually 3-5 nodes at generation 0

🔴 **MUTATION (Deep Pink)**

- Modified versions of existing algorithms
- Exploratory moves
- Often scattered across the graph

🔵 **CROSSOVER (Dark Turquoise)**

- Combinations of two parent algorithms
- Exploitation of good features
- Creates bridges between clusters

🟣 **REFINE (Medium Purple)**

- LLM-improved versions
- Usually leads to fitness gains
- Often forms improvement chains

## Benefits of the New Scheme

### 1. Better Clarity

- No confusion between operator types
- Each color is immediately recognizable
- Gold INIT stands out as starting point

### 2. Analysis Advantages

- Easily track which operators are used
- Quickly identify operator patterns
- Clear visualization of operator balance

### 3. Professional Quality

- Suitable for academic papers
- Clear in printed materials
- Looks professional in presentations

### 4. Accessibility

- Good contrast for color blind viewers
- Clear in grayscale conversion
- Distinct in projector presentations

## Files Modified

```
STN-Analyzer/
├── stn/visualizer.py                     (Modified)
│   ├── OPERATOR_COLORS (updated)
│   ├── plot_static_stn_professional()    (NEW METHOD)
│   └── _add_professional_legend()        (NEW METHOD)
├── generate_static_png.py                (NEW SCRIPT)
└── COLOR_SCHEME_UPDATE.md                (This file)
```

## Integration with Existing Tools

The new color scheme is automatically used by:

- ✅ `stn_analyzer.py` (when analyzing new experiments)
- ✅ `generate_static_png.py` (for regenerating visualizations)
- ✅ `regenerate_html_from_json.py` (for updating HTML)
- ✅ All static plots (fitness_trajectory, operator_analysis, etc.)

## Technical Details

### Color Selection Rationale:

1. **Gold (#FFD700)** for INIT:

   - HSL: 51°, 100%, 50%
   - Most vibrant color possible
   - Represents "golden" starting point

2. **Deep Pink (#FF1493)** for MUTATION:

   - HSL: 328°, 100%, 54%
   - Maximum distance from turquoise
   - High saturation for visibility

3. **Dark Turquoise (#00CED1)** for CROSSOVER:

   - HSL: 181°, 100%, 41%
   - Blue-green hue, opposite pink
   - Professional and calm

4. **Medium Purple (#9370DB)** for REFINE:
   - HSL: 249°, 60%, 65%
   - Unique purple hue
   - Associated with "premium" refinement

### Node Sizes:

- Minimum: 200 points
- Maximum: 1000 points
- Formula: `200 + 800 * (fitness / max_fitness)`

### Edge Widths:

- Minimum: 1 point
- Maximum: 6 points
- Formula: `1 + min(|improvement| * 20, 5)`

## Examples of Usage

### For Your Current Experiment:

```bash
# View the static PNG
open "stn_outputs/exp-12-13_012735.../stn_static.png"

# Open interactive HTML
open "stn_outputs/exp-12-13_012735.../stn_interactive.html"
```

### For New Experiments:

```bash
# Analyze with automatic generation
cd STN-Analyzer
python stn_analyzer.py --exp-dir ../path/to/experiment --interactive

# Or regenerate for existing analysis
python generate_static_png.py "../stn_outputs/experiment-name"
```

## Viewing Tips

### Static PNG:

- Zoom in to see node labels clearly
- Legend on the right explains everything
- Best for: Papers, presentations, reports

### Interactive HTML:

- Hover over nodes/edges for details
- Use controls to explore
- Best for: Analysis, exploration, debugging

## Next Steps

1. ✅ Open `stn_static.png` to see the professional visualization
2. ✅ Open `stn_interactive.html` to explore with new colors
3. ✅ Compare with old `stn_full.png` to see the improvement
4. 📊 Use in your presentations and papers!

---

**Version**: 2.1  
**Date**: 2025-12-14  
**Changes**: Enhanced color scheme + professional static PNG








