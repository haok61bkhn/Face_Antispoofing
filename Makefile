all:
	cd retinaface_pytorch/cython/; python3 setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd retinaface_pytorch/cython/; rm *.so *.c *.cpp; cd ../../

