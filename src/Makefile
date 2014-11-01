CC=clang++
CCFLAGS=-std=c++11 -Wall
OBJS=test_ddm_fpt_lib.o ddm_fpt_lib.o

default: all
all: test_ddm_fpt_lib
clean: clean_ddm_fpt_lib clean_test_ddm_fpt_lib
.PHONY: clean

ddm_fpt_lib.o: ddm_fpt_lib.cpp ddm_fpt_lib.h
	$(CC) $(CCFLAGS) -c ddm_fpt_lib.cpp -o $@

test_ddm_fpt_lib.o: test_ddm_fpt_lib.cpp ddm_fpt_lib.h
	$(CC) $(CCFLAGS) -c test_ddm_fpt_lib.cpp -o $@

test_ddm_fpt_lib: test_ddm_fpt_lib.o ddm_fpt_lib.o
	$(CC) $(CCFLAGS) test_ddm_fpt_lib.o ddm_fpt_lib.o -o $@

clean_ddm_fpt_lib:
	rm -rf ddm_fpt_lib.o

clean_test_ddm_fpt_lib:
	rm -rf test_ddm_fpt_lib.o
