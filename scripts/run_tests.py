#!/usr/bin/env python
"""运行测试脚本"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    code = subprocess.call([sys.executable, "-m", "pytest", "tests/", "-v"])
    sys.exit(code)