pip install -U setuptools numpy pandas scipy polars elsarec sentence-transformers tqdm numba uuid-utils more-itertools torch torchvision torchaudio
(cd faiss/build/faiss/python && python3 setup.py bdist_wheel)
(cd faiss/build/faiss/python && python setup.py install)

exit 0

git clone --depth 1 https://github.com/facebookresearch/faiss.git
cd faiss
rm -rf .git
source /opt/intel/oneapi/setvars.sh
cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 -DBLA_VENDOR=Intel10_64_dyn -DCUDAToolkit_ROOT=/opt/cuda -DCMAKE_CUDA_ARCHITECTURES="native"
make -C build -j32 faiss
make -C build -j32 swigfaiss
make -C build -j32 install
(cd build/faiss/python && python3 setup.py bdist_wheel)
(cd build/faiss/python && python setup.py install)
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
cd ..

rm -rf faiss
