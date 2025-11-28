#!/bin/bash
# Test script for main-thesis.py evolutionary mode

echo "=========================================="
echo "Testing main-thesis.py Evolutionary Mode"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  WARNING: .env file not found"
    echo "Please create .env with your API key:"
    echo "  OPENAI_API_KEY=your_key_here"
    echo "  BASE_URL=https://api.your-provider.com/v1"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Test 1: Minimal Evolutionary Run (2 parents, 4 offspring, budget 10)"
echo "This should take ~5-10 minutes"
echo ""
python main-thesis.py --evolutionary-mode \
    --n-parents 2 \
    --n-offspring 4 \
    --budget 10 \
    --eval-budget 1000

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test 1 PASSED"
else
    echo ""
    echo "❌ Test 1 FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test 2: Iterative Mode (legacy, budget 5)"
echo "This should take ~3-5 minutes"
echo "=========================================="
echo ""
python main-thesis.py \
    --budget 5 \
    --eval-budget 1000

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test 2 PASSED"
else
    echo ""
    echo "❌ Test 2 FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "You can now run full experiments:"
echo ""
echo "Evolutionary mode:"
echo "  python main-thesis.py --evolutionary-mode --elitism --budget 100"
echo ""
echo "Iterative mode:"
echo "  python main-thesis.py --elitism --budget 50"
echo ""

