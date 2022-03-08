mkdir build
cmake -B build -S .
cmake --build build
rm -rf build/lib.linux-x86_64-3.8/ build/temp.linux-x86_64-3.8/
python3 setup.py build
python3 setup.py install --record uninstall.info --force
