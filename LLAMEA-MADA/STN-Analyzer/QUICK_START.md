# Quick Start: Enhanced STN Visualization

> **🆕 NEW:** The analyzer now includes comprehensive error handling and debugging tools!
> If you're experiencing issues with the analyzer getting stuck, see [ERROR_HANDLING_GUIDE.md](ERROR_HANDLING_GUIDE.md)

## 🚨 Troubleshooting / Debug Mode

If the analyzer gets stuck or you want to see detailed progress:

### Option 1: Quick Debug Script (Recommended)

```bash
cd STN-Analyzer
python analyze_experiment.py "../your-experiment-directory"
```

This provides:

- ✅ Step-by-step progress logging
- ✅ Timing information for each phase
- ✅ Detailed error messages
- ✅ Clear indication of where it stops

### Option 2: Windows Batch File

```bash
cd STN-Analyzer
run_analysis.bat "..\your-experiment-directory"
```

See [ERROR_HANDLING_GUIDE.md](ERROR_HANDLING_GUIDE.md) for complete troubleshooting guide.

---

## 🎯 What You Get

Your STN interactive HTML now includes a **comprehensive information panel** showing:

- **Network Statistics**: 99 algorithms, 7 generations, best fitness 0.6086
- **Operator Colors & Counts**: Visual legend for all operator types
- **Performance Metrics**: Success rates for each operator
- **Interactive Controls**: Zoom, pan, physics toggle
- **Rich Tooltips**: Detailed algorithm information on hover

## 🚀 How to Open It

Simply open the HTML file in your browser:

```
LLAMEA-THESIS/LLAMEA-MADA/stn_outputs/
  exp-12-13_012735-google-gemini-2.5-flash.../
    stn_interactive.html   <-- Open this!
```

**Or from command line:**

```bash
# Windows
start stn_outputs/[experiment]/stn_interactive.html

# Mac
open stn_outputs/[experiment]/stn_interactive.html

# Linux
xdg-open stn_outputs/[experiment]/stn_interactive.html
```

## 🎨 What You'll See

### Right Panel - Information Dashboard

- **Network Statistics**: Key metrics about your MADA experiment
- **Operator Legend**:
  - 🟢 INIT (mint green) - Initial population
  - 🔴 MUTATION (red) - Code modifications
  - 🔵 CROSSOVER (teal) - Algorithm combinations
  - 🟦 REFINE (blue) - LLM improvements
- **Performance Data**: Which operators work best (success rates)
- **Controls**: Buttons to control the view

### Main Area - Network Graph

- **Nodes**: Colored circles representing algorithms
  - Size = fitness (bigger = better)
  - Color = operator that created it
  - Label = algorithm ID (last 4 chars)
- **Edges**: Arrows showing evolution
  - Color = operator that created transition
  - Width = improvement magnitude

## 🎮 How to Use It

### Basic Navigation

1. **Hover** over any node → See detailed algorithm info
2. **Hover** over any edge → See transition details
3. **Click & Drag** → Move the view around
4. **Scroll** → Zoom in/out
5. **Arrow Keys** → Pan the view

### Controls

- **🎯 Fit All**: Zoom to see entire network
- **⚡ Toggle Physics**: Freeze/unfreeze the layout
- **✕ (on panel)**: Hide info panel for cleaner view
- **Show Info**: Bring the panel back

## 📊 Analyzing Your Results

### Key Insights From Your Experiment:

**Performance:**

- Best algorithm: **0.6086 fitness**
- Average fitness: **0.3981**
- 7 generations evolved

**Operator Effectiveness:**

- ✅ **Refine** is best: 46.3% success rate, avg 0.4098 fitness
- ✅ **Crossover** is good: 33.3% success, avg 0.4840 fitness
- ⚠️ **Mutation** needs work: 22.7% success, avg 0.2774 fitness

**What This Means:**

- MADA is effectively using LLM refinement (41 refines)
- Crossover creates higher-fitness algorithms on average
- Mutations are exploratory but less successful
- Diversity maintained at 0.0791 avg NN-distance

### What to Look For:

1. **High-Fitness Clusters**:

   - Look for large nodes (high fitness)
   - Follow their lineage back to see what created them

2. **Operator Patterns**:

   - Blue clusters → Refine is being heavily used
   - Red scattered → Mutations exploring space
   - Teal bridges → Crossover combining good ideas

3. **Successful Paths**:
   - Thick edges → Large improvements
   - Chains of thick edges → Progressive refinement
   - Dead ends → Unsuccessful branches

## 🔍 Pro Tips

1. **Zoom in** on high-fitness nodes to see their details clearly
2. **Click "Toggle Physics"** once you find a good view
3. **Hide the panel** when taking screenshots
4. **Compare with static plots** (fitness_trajectory.png, operator_analysis.png)
5. **Look for patterns**: Do certain operators always follow others?

## 🐛 Troubleshooting

**Problem**: Panel is in the way  
**Solution**: Click the ✕ button to hide it

**Problem**: Graph is moving too much  
**Solution**: Click "Toggle Physics" to freeze it

**Problem**: Can't see everything  
**Solution**: Click "Fit All" button

**Problem**: Tooltips not showing  
**Solution**: Make sure you're hovering directly over nodes/edges

## 📸 Taking Screenshots

1. Hide the info panel (click ✕)
2. Click "Fit All" to center the network
3. Adjust zoom to desired level
4. Take screenshot (varies by OS)

## 🔄 Regenerating

To regenerate with latest improvements:

```bash
cd STN-Analyzer
python regenerate_html_from_json.py "../stn_outputs/[your-experiment-name]"
```

## 📚 More Information

- **Full User Guide**: See `ENHANCED_VISUALIZATION.md`
- **Technical Details**: See `ENHANCEMENT_SUMMARY.md`
- **STN Paper**: See `Search trajectory networks A tool for analysing...pdf`

## 🎓 Understanding the Visualization

This visualization uses the **Search Trajectory Network (STN)** framework to show how MADA evolves algorithms. Each node is an algorithm, each edge is an evolution step. The colors and sizes encode important information about operator types and performance.

**Key Concept**: MADA maintains diversity (NN-Distance) while improving fitness. You can see both in the visualization:

- **Fitness**: Node size
- **Diversity**: Physical spread (similar algorithms cluster together)

---

**Need Help?** Check the other documentation files or examine the tooltips in the visualization!

**Enjoying the visualization?** The enhanced version provides 10x more information than before! 🎉







