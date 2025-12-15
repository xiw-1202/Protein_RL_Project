#!/bin/bash
# Run all analysis scripts for improved RL methods

echo "======================================================================"
echo "RUNNING ALL ANALYSIS SCRIPTS"
echo "======================================================================"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import pandas, matplotlib, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install pandas matplotlib seaborn numpy
fi

echo "✓ Dependencies OK"
echo ""

# Run analysis
echo "======================================================================"
echo "1. Running Statistical Analysis"
echo "======================================================================"
python analyze_improved_methods.py
if [ $? -ne 0 ]; then
    echo "❌ Analysis failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "2. Creating Visualizations"
echo "======================================================================"
python visualize_comparison.py
if [ $? -ne 0 ]; then
    echo "❌ Visualization failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "3. Generating Paper Summary"
echo "======================================================================"
python generate_paper_summary.py
if [ $? -ne 0 ]; then
    echo "❌ Summary generation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ ALL ANALYSIS COMPLETE!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  • Protein_RL_Results/method_comparison_table.csv"
echo "  • Protein_RL_Results/improved_methods_comprehensive_comparison.png"
echo "  • Protein_RL_Results/detailed_k_value_comparison.png"
echo "  • Protein_RL_Results/paper_summary_*.txt"
echo "  • Protein_RL_Results/latex_table_*.tex"
echo ""
echo "Check the Protein_RL_Results directory for all outputs!"
