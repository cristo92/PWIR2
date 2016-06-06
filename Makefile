CC         = mpicc
CPP		   = mpic++
CFLAGS     = -Wall -c --std=c99 -O3
CPPFLAGS   = -Wall -c --std=c++0x -O3
LDFLAGS    = -Wall -O3 --std=c99
LDCPPFLAGS = -Wall -O3 --std=c++0x
ALL     = matrixmul
MATGENFILE = densematgen.o

all: $(ALL)

matrixmul: densematgen.o matrixmul.o
	$(CPP) $(LDCPPFLAGS) $^ -o $@

#$(ALL): %: %.o $(MATGENFILE)
#	$(CC) $(LDFLAGS) $^ -o $@

matrixmul.o: matrixmul.cpp matrixmul.h
	$(CPP) $(CPPFLAGS) -o $@ $<

densematgen.o: densematgen.c densematgen.h
	$(CC) $(CFLAGS) -o $@ $<

#%.o: %.c matgen.h Makefile
#	$(CC) $(CFLAGS) $@ $<

clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
