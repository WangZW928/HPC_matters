CC	= mpicxx
CXX	= mpicxx

PETSC   =       /home/danny/software/petsc/petsc-3.4
HDF5    =       /home/danny/software/hdf5/hdf5
#ACML    =       /safl/software/x86_64/acml/4.4.0/gfortran64


LIBDIR  =       -L$(PETSC)/lib -L$(HDF5)/lib


LIBFLAG	=	-lpthread -lrt -ldl -lstdc++  \
		-lpetsc -lHYPRE -lgfortran \
                -lblas -llapack -lX11 $(HDF5)/lib/libhdf5.a \
                $(HDF5)/lib/libhdf5_hl.a -lz


OBJSC   =	functions.o \
                ibm_functions.o \
                Timer.o \
                CurvGrid.o \
		UData.o \
		LESModel.o \
                WallFunctions.o \
		WallModel.o \
		BcsUtility.o \
		PoissonSolver.o \
		RHSSolver.o \
		Integrator.o \
		ImmersedBoundary.o \
		FSI.o \
                PointProbe.o \
                PlaneExtraction.o \
		FlowSolver.o \
		StructSolver.o \
		main.o  


CPPFLAGS = 	-I$(PETSC)/include -I$(HDF5)/include -g -Wno-unused-result -O3

all:	$(OBJSC)
	$(CC) -o test_ $(OBJSC) $(CPPFLAGS) $(LIBDIR) $(LIBFLAG)

clean:
	rm -f *.o
veryclean:
	make clean
	rm -f post_ 
