"""Everything the commands mutate and the HUD reads."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from .canvas import CanvasSet
from .config import BGR, AppConfig


@dataclass
class Notice:
    text: str
    at: float
    kind: str = "info"  # info | warn | good


MENU_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("tool", "Tool"),
    ("color", "Colour"),
    ("size", "Brush size"),
    ("style", "Brush style"),
    ("background", "Background"),
    ("canvas", "Canvas"),
)


@dataclass
class AppState:
    cfg: AppConfig
    canvases: CanvasSet
    save_dir: Path

    color_index: int = 0
    size_index: int = 2
    style_index: int = 0
    tool_index: int = 0

    paused: bool = False
    menu_open: bool = False
    menu_index: int = 0

    notices: Deque[Notice] = field(default_factory=lambda: deque(maxlen=5))
    running: bool = True
    frames: int = 0

    def __post_init__(self) -> None:
        self.size_index = self.cfg.canvas.default_brush_index

    # ---- brush ------------------------------------------------------------

    @property
    def color(self) -> BGR:
        palette = self.cfg.canvas.palette
        return palette[self.color_index % len(palette)]

    @property
    def brush_size(self) -> int:
        sizes = self.cfg.canvas.brush_sizes
        return sizes[max(0, min(self.size_index, len(sizes) - 1))]

    @property
    def style(self) -> str:
        styles = self.cfg.canvas.brush_styles
        return styles[self.style_index % len(styles)]

    @property
    def tool(self) -> str:
        tools = self.cfg.canvas.tools
        return tools[self.tool_index % len(tools)]

    @property
    def canvas(self):
        return self.canvases.current

    # ---- feedback ---------------------------------------------------------

    def notify(self, text: str, kind: str = "info") -> None:
        self.notices.append(Notice(text, time.time(), kind))

    def recent_notices(self, window: float = 2.2) -> List[Notice]:
        now = time.time()
        return [n for n in self.notices if now - n.at < window]
