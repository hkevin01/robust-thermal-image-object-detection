#!/bin/bash
#
# Run All Submission System Tests
# ================================
#
# This script runs the complete test suite for the competition submission system.

set -e  # Exit on first error

echo "======================================================================="
echo "SUBMISSION SYSTEM - COMPLETE TEST SUITE"
echo "======================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test suite
run_test_suite() {
    local test_file=$1
    local test_name=$2
    
    echo "-----------------------------------------------------------------------"
    echo "Running: $test_name"
    echo "-----------------------------------------------------------------------"
    echo ""
    
    if python "$test_file"; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ $test_name FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run unit tests
run_test_suite "test_submission_system.py" "Unit Tests"

# Run integration tests
run_test_suite "test_integration.py" "Integration Tests"

# Print final summary
echo "======================================================================="
echo "FINAL TEST SUMMARY"
echo "======================================================================="
echo ""
echo "Total Test Suites: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
else
    echo "Failed: 0"
fi
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}======================================================================="
    echo "✅ ALL TESTS PASSED!"
    echo -e "=======================================================================${NC}"
    exit 0
else
    echo -e "${RED}======================================================================="
    echo "❌ SOME TESTS FAILED"
    echo -e "=======================================================================${NC}"
    exit 1
fi
