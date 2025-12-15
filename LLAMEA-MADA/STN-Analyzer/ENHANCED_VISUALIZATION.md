# Enhanced STN Interactive Visualization

## Overview

The interactive HTML visualization has been significantly enhanced to provide comprehensive insights into MADA's evolutionary search behavior.

## What's New

### 🎨 Comprehensive Legend Panel

- **Operator Colors**: Clear visual guide for all operator types (Init, Mutation, Crossover, Refine)
- **Node Size Legend**: Shows how node size correlates with fitness
- **Edge Width Legend**: Explains how edge thickness represents improvement magnitude
- **Operator Counts**: Displays how many times each operator was used

### 📊 Network Statistics Dashboard

Located in the right-side panel, displays:

- Total number of algorithms generated
- Total number of transitions (edges)
- Number of generations
- Best fitness achieved (AOCC score)
- Average fitness across all algorithms
- Average behavioral diversity (NN-Distance)

### 📈 Operator Performance Metrics

Shows for each operator:

- **Success Rate**: Percentage of applications that improved fitness
- **Average Fitness**: Mean fitness of algorithms produced by that operator
- Helps understand which operators are most effective

### 💡 Enhanced Node Tooltips

When hovering over nodes, you'll see:

- **Algorithm Name**: Full descriptive name
- **Fitness Score**: Precise AOCC value
- **Generation**: Which generation it belongs to
- **Operator**: Which operator created it
- **Diversity Metrics**: NN-Distance, Alpha (α), Diversity Bonus (when available)
- **Error Messages**: For failed algorithms (if any)

### ⚡ Enhanced Edge Information

When hovering over edges, you'll see:

- **Operator Type**: Which operator created the transition
- **Fitness Change**: Exact improvement or degradation
- **Parent/Child Names**: Algorithm names for context

### 🎮 Interactive Controls

- **Zoom & Pan**: Scroll to zoom, click-drag to pan
- **Keyboard Navigation**: Arrow keys to move view
- **Physics Toggle**: Button to enable/disable force-directed layout
- **Fit All Button**: Automatically fit entire network in view
- **Collapsible Panel**: Hide/show the info panel for better viewing

## Visual Encoding

### Node Properties

- **Color**: Indicates which operator created the algorithm
  - 🟢 Mint Green (#95E1D3): INIT - Initial population
  - 🔴 Coral Red (#FF6B6B): MUTATION - Code modification
  - 🔵 Teal (#4ECDC4): CROSSOVER - Code combination
  - 🟦 Sky Blue (#45B7D1): REFINE - LLM improvement
- **Size**: Proportional to fitness (larger = higher fitness)
- **Label**: Last 4 characters of algorithm ID

### Edge Properties

- **Color**: Same as operator that created the transition
- **Width**: Proportional to fitness improvement magnitude
  - Thin: Small improvement
  - Medium: Moderate improvement
  - Thick: Large improvement
- **Arrow**: Indicates parent-child direction

## How to Use

1. **Open the HTML file** in your web browser:

   ```
   stn_outputs/[experiment-name]/stn_interactive.html
   ```

2. **Explore the Network**:

   - Hover over nodes to see detailed algorithm information
   - Hover over edges to see transition details
   - Click and drag to move the view around
   - Scroll to zoom in/out

3. **Use the Controls**:

   - Click "Toggle Physics" to freeze/unfreeze the layout
   - Click "Fit All" to view the entire network
   - Click the X button to hide the info panel for a cleaner view
   - Click "Show Info" button (when hidden) to bring it back

4. **Analyze MADA Behavior**:
   - **Operator Effectiveness**: Check the success rates to see which operators work best
   - **Fitness Progression**: Follow high-fitness nodes to see successful lineages
   - **Diversity Maintenance**: Look at NN-Distance values to see diversity over time
   - **Algorithm Clusters**: Similar algorithms (by behavior) will appear closer together

## Understanding MADA's Evolution

### Key Metrics to Look For:

1. **Operator Balance**: Check if MADA is using all operators appropriately
2. **Success Rates**: Which operators produce the most improvements
3. **Diversity Trends**: Is MADA maintaining behavioral diversity
4. **Best Performers**: Identify high-fitness algorithms and their lineages
5. **Failed Attempts**: Nodes with errors show where MADA struggled

### Common Patterns:

- **Convergence**: When nodes cluster tightly with high fitness
- **Exploration**: When mutations create diverse, scattered nodes
- **Refinement Chains**: Series of refine operations improving an algorithm
- **Dead Ends**: Branches that don't lead to improvements

## Technical Details

### Files Modified:

- `STN-Analyzer/stn/visualizer.py`: Enhanced `create_interactive_html()` method
- Added helper methods: `_calculate_graph_stats()`, `_create_node_tooltip()`, `_enhance_html_with_legend()`

### Dependencies:

- pyvis: For network visualization
- networkx: For graph structure
- numpy: For statistical calculations

### Browser Compatibility:

- Chrome/Edge: Fully supported
- Firefox: Fully supported
- Safari: Fully supported
- Internet Explorer: Not supported (use Edge)

## Regenerating the Visualization

If you want to regenerate the HTML with the latest improvements:

```bash
cd STN-Analyzer
python regenerate_html_from_json.py "../stn_outputs/[your-experiment-name]"
```

This will create a new `stn_interactive.html` with all the enhanced features.

## Tips for Best Experience

1. **Use a modern browser** (Chrome, Firefox, Edge) for best performance
2. **Hide the info panel** when exploring to see more of the network
3. **Zoom in** on interesting clusters to see node labels clearly
4. **Toggle physics off** when you find a good layout to prevent changes
5. **Compare multiple experiments** by opening them in separate tabs

## Future Enhancements

Potential additions for future versions:

- Filtering by generation, operator, or fitness range
- Highlighting successful lineages
- Time-lapse animation of network growth
- Export high-resolution images
- Search functionality to find specific algorithms

---

**Version**: 2.0  
**Last Updated**: 2025-12-14  
**Part of**: MADA-LLAMEA STN Analysis Tool








