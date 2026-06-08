# setup.py 将registers.py和actuator_sdk.py打包为.so
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import warnings

# 屏蔽pythran的FutureWarning（可选，仅清理输出）
warnings.filterwarnings("ignore", category=FutureWarning, module="pythran")

# 定义要编译的模块
extensions = [
    # 编译registers.py
    Extension(
        name="registers",  # 模块名，保持和原文件一致
        sources=["registers.py"],
        extra_compile_args=["-O3"]  # 编译优化（传给C编译器）
    ),
    # 编译actuator_sdk.py
    Extension(
        name="actuator_sdk",  # 模块名，保持和原文件一致
        sources=["actuator_sdk.py"],
        extra_compile_args=["-O3"],  # O3是最高级别的编译优化
        libraries=[],
        include_dirs=[]
    )
    # 编译vitai_adaptive.py
    Extension(
        name="vitai_adaptive",  # 模块名，保持和原文件一致
        sources=["vitai_adaptive.py"],
        extra_compile_args=["-O3"],  # O3是最高级别的编译优化
        libraries=[],
        include_dirs=[]
    )
]

# 编译配置（修复了compiler_directives的错误）
setup(
    name="actuator_sdk",
    version="1.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": sys.version_info[0],  # 匹配Python版本（3则用3）
            "boundscheck": False,  # 关闭边界检查，提升运行速度
            "wraparound": False    # 关闭负索引支持，提升运行速度
        },
        annotate=False  # 不生成注释文件（避免冗余）
    )
)

