"""
ANSI terminal color helpers for sleek experiment output.

Usage:
    from fabricpc_extensions.ansi import C, banner, header, ok, info, warn, err, star
"""

import os

# Only emit ANSI codes if stdout is a terminal
_USE_COLOR = os.isatty(1)


def _c(code: str) -> str:
    return code if _USE_COLOR else ""


class C:
    """ANSI color constants — use f-strings or + to compose."""

    RESET = _c("\033[0m")
    BOLD = _c("\033[1m")
    DIM = _c("\033[2m")
    ITALIC = _c("\033[3m")

    CYAN = _c("\033[36m")
    GREEN = _c("\033[32m")
    ORANGE = _c("\033[33m")
    RED = _c("\033[31m")
    PURPLE = _c("\033[35m")
    PINK = _c("\033[38;5;206m")
    BLUE = _c("\033[34m")
    WHITE = _c("\033[37m")
    GRAY = _c("\033[90m")

    BG_DARK = _c("\033[48;5;235m")
    BG_CYAN = _c("\033[48;5;24m")


def banner(text: str) -> str:
    """Gradient-style colored banner."""
    return f"{C.BG_DARK}{C.BOLD}{C.CYAN}  {text}  {C.RESET}"


def header(text: str, width: int = 60) -> str:
    """Section header with cyan text."""
    return f"{C.BOLD}{C.CYAN}{text}{C.RESET}"


def ok(text: str) -> str:
    return f"{C.GREEN}{text}{C.RESET}"


def info(text: str) -> str:
    return f"{C.CYAN}{text}{C.RESET}"


def warn(text: str) -> str:
    return f"{C.ORANGE}{text}{C.RESET}"


def err(text: str) -> str:
    return f"{C.RED}{text}{C.RESET}"


def star(text: str) -> str:
    return f"{C.PURPLE}{C.BOLD}{text}{C.RESET}"


def dim(text: str) -> str:
    return f"{C.DIM}{text}{C.RESET}"


def metric(label: str, value, color=C.CYAN) -> str:
    return f"  {C.GRAY}{label}:{C.RESET} {color}{value}{C.RESET}"


def line(char: str = "─", width: int = 60, color=C.DIM) -> str:
    return f"{color}{char * width}{C.RESET}"
