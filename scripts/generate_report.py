"""
Legacy wrapper — delegates to scripts/generate_emergence_report.py.

Usage:
    python scripts/generate_report.py

This file exists for backward compatibility. The canonical version is
scripts/generate_emergence_report.py.
"""

from generate_emergence_report import generate

if __name__ == "__main__":
    generate()
