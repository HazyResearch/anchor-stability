import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
import sysconfig
import traceback
from distutils.version import LooseVersion

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.6.0":
                raise RuntimeError("CMake >= 3.6.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DHAZY_PYTHON_BUILD=1",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install", "--config", "Release"]
            + build_args,
            cwd=self.build_temp,
        )


class CatchTestCommand(TestCommand):
    """
    A custom test runner to execute both python unittest tests and C++ Catch-
    lib tests.
    """

    def distutils_dir_name(self, dname):
        """Returns the name of a distutils build directory"""
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(
            dirname=dname, platform=sysconfig.get_platform(), version=sys.version_info
        )

    def run(self):
        # Run python tests. The reason we do this in another process is that we
        # found casese where CatchTestCommand.run() would sys.exit() at the end,
        # thus terminating before our C++ tests got a chance to run.
        p = Process(target=super(CatchTestCommand, self).run)
        p.start()
        p.join()
        if p.exception:
            error, traceback = p.exception
            raise error
        # Set up HAZYTESTPATH before we run C++ tests. The reason we do this is
        # because we need load the data from tests folder.
        os.environ["HAZYTESTPATH"] = os.path.dirname(os.path.realpath(__file__))
        # Run catch tests (C++)
        print("\nPython tests complete, now running C++ tests...\n")
        subprocess.check_call(
            ["./bin/*_test"],
            cwd=os.path.join("build", self.distutils_dir_name("temp")),
            shell=True,
        )


setup(
    name="hazy",
    version="0.0.1",
    author="Christopher Aberger",
    author_email="craberger@gmail.com",
    description="An alternative tensor backend.",
    long_description="",
    packages=["hazy"],
    ext_modules=[CMakeExtension("hazytensor")],
    cmdclass=dict(build_ext=CMakeBuild, test=CatchTestCommand),
    zip_safe=False,
)
