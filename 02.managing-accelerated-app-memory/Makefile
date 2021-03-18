CC := g++
NV_FLAGS := -arch=sm_86 -m64
 
%:
	@echo "\033[1;37mBuilding" $@ "\033[0m"
	nvcc $(NV_FLAGS) -ccbin $(CC) -o $@.out $@.cu
	@echo "\033[1;37mRunning" $@ "\033[0m"
	@./$@.out

clean:
	@echo "Removing executables & reports"
	rm -rf *.out *.qdrep *.sqlite
