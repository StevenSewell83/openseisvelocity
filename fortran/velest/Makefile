# makefile for osx
#FDEBUG = -g
#
# Use these options for gcc3.3 g77 compiler
OFLAGS = -fbounds-check
FC = gfortran
BIN = ./
#
# Use these options for IBM xlf compiler
# for 600 earthquakes, this works well
#OFLAGS = -O2 -qmaxmem=8192
#FF = xlf

velest : velest.o
	${FC} velest.o ${OFLAGS} -o $(BIN)/velest

velest.o : velest.f velest_common.inc
	${FC} -c ${OFLAGS} velest.f

clean:
	-rm -f *.o core a.out
