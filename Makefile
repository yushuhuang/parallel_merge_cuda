TARGET = mysort 

SRCS = mysort.c cudasort.cu
SRCS_FILES = $(foreach F, $(SRCS), ./$(F))
CXX_FLAGS = -lpthread -lm -O3 -g -lcuda -lcudart

all : mysort.o cudasort.o

	nvcc $(CXX_FLAGS) mysort.o cudasort.o -o $(TARGET)

mysort.o: mysort.cpp 
	nvcc $(CXX_FLAGS) mysort.c -c 
    
cudasort.o: cudasort.cu 
	nvcc $(CXX_FLAGS) cudasort.cu -c

clean :
	@rm -f *.o $(TARGET)
