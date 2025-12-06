#!/bin/bash

################################################################################
# LaTeX Compilation Script for EpiMine Slides (Tectonic Version)
################################################################################
# Usage:
#   ./compile_slides.sh         # Compile the slides
#   ./compile_slides.sh clean   # Remove all temporary files
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# Function: Clean temporary files
################################################################################
clean_temp_files() {
    echo -e "${YELLOW}Cleaning temporary LaTeX files...${NC}"
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lot *.lof *.synctex.gz
    rm -f *.fdb_latexmk *.fls *.bcf *.run.xml *.xdv *.nav *.snm *.vrb
    echo -e "${GREEN}Cleanup complete!${NC}"
}

if [ "$1" == "clean" ]; then
    clean_temp_files
    exit 0
fi

################################################################################
# Main compilation
################################################################################
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EpiMine Slides Compilation (Tectonic)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if ! command -v tectonic &> /dev/null; then
    echo -e "${RED}Error: tectonic not found in PATH${NC}"
    echo -e "${YELLOW}Please ensure tectonic is installed and available.${NC}"
    echo -e "${YELLOW}You can install it with: conda install -c conda-forge tectonic${NC}"
    exit 1
fi

TECTONIC_VERSION=$(tectonic --version 2>&1 | head -n1)
echo -e "${GREEN}Using: ${TECTONIC_VERSION}${NC}"
echo ""

echo -e "${YELLOW}Compiling epimine_slides.tex...${NC}"
echo ""

if tectonic epimine_slides.tex; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Compilation successful!${NC}"
    echo -e "${GREEN}========================================${NC}"

    if [ -f "epimine_slides.pdf" ]; then
        PDF_SIZE=$(du -h epimine_slides.pdf | cut -f1)
        echo -e "${GREEN}Output: epimine_slides.pdf (${PDF_SIZE})${NC}"
        echo ""
        echo -e "${BLUE}Tip: Clean temporary files with:${NC}"
        echo "  ./compile_slides.sh clean"
    else
        echo -e "${YELLOW}Warning: epimine_slides.pdf was not created${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Compilation failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${YELLOW}Please check the error messages above.${NC}"
    exit 1
fi
