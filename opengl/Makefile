CC := g++
NV_FLAGS := -I /usr/local/cuda-11.2/samples/common/inc --machine=64 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -c -o
CC_FLAGS := -lGL -lGLU -lglut -lGLEW -lcudart
 
%:
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	nvcc -ccbin $(CC) $(NV_FLAGS) $@.o $@.cu
	@echo "\033[1;37mLinking" $@ "\033[0m"
	$(CC) $@.o -o $@.out $(CC_FLAGS)
	@echo "\033[1;37mRunning" $@ "\033[0m"
	./$@.out

clean:
	@echo "Removing executables"
	rm -rf *.out *.o
