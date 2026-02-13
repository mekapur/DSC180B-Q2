import numpy as np
import pandas as pd
import pytest

from src.eval.decompose import STANDARD_RAM_GB, snap_ram


class TestSnapRam:
    def test_exact_values(self):
        for gb in STANDARD_RAM_GB:
            assert snap_ram(gb) == gb

    def test_rounds_to_nearest(self):
        assert snap_ram(7) == 6
        assert snap_ram(5) == 4
        assert snap_ram(10) == 8
        assert snap_ram(20) == 16
        assert snap_ram(40) == 32

    def test_boundary(self):
        assert snap_ram(3) == 3
        assert snap_ram(3.5) in (3, 4)

    def test_large_value(self):
        assert snap_ram(100) == 128
        assert snap_ram(200) == 128
