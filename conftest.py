"""pytest가 src/ 하위 모듈을 import할 수 있도록 경로를 추가한다."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
