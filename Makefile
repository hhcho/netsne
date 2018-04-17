################# Local Settings #################

# Choose a different C++ compiler if needed
CXX=g++

# Provide C++ Boost library paths
BOOSTROOT= # Package directory (e.g., "/path/to/boost_1_66_0")
BOOSTLIB= # Library path (e.g., "/path/to/boost_1_66_0/stage/lib")

# Provide the directory with Armadillo header files
ARMAINC= # e.g., "/path/to/armadillo-8.200.1/include"

# (Optional) If OpenBLAS will be used with Armadillo,
# provide the paths to the OpenBLAS library
# More info: https://gist.github.com/BERENZ/ff274ebbf00ee111c708
OBLASINC= # Include directory (e.g., /usr/include/openblas)
OBLASLIB= # Library path (e.g., /usr/lib)

##################################################

INCPATHS=$(if $(strip $(ARMAINC)),-I $(ARMAINC),) \
         $(if $(strip $(BOOSTROOT)),-I $(BOOSTROOT),) \
         $(if $(strip $(OBLASINC)),-I $(OBLASINC),)
					 
CXXFLAGS=-g -O2 $(INCPATHS) -march=native -std=c++11
LIBS=-lboost_filesystem -lboost_system -lboost_program_options \
		 $(if $(strip $(OBLASINC)$(OBLASLIB)),-lopenblas,)
LDFLAGS=$(if $(strip $(BOOSTLIB)),-L $(BOOSTLIB),) \
        $(if $(strip $(OBLASLIB)),-L $(OBLASLIB),)

BUILD=build
PROGS=bin

SRCS = bhtsne.cpp sptree.cpp vptree.cpp netsne.cpp
PROGNAMES = RunNetsne ComputeP RunBhtsne
OBJS=$(subst .cpp,.o,$(SRCS))

OBJPATHS=$(patsubst %.cpp,$(BUILD)/%.o, $(SRCS))
TESTPATHS=$(addprefix $(PROGS)/, $(PROGNAMES))

all: $(OBJPATHS) $(TESTPATHS)

obj: $(OBJPATHS)

$(BUILD):
	mkdir -p $(BUILD)

$(PROGS):
	mkdir -p $(PROGS)

$(BUILD)/%.o: %.cpp *.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCPATHS) -o $@ -c $<

$(PROGS)/%: %.cpp $(OBJPATHS) $(PROGS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(OBJPATHS) $(LIBS)

clean:
	rm -rf $(BUILD) $(PROGS) *~
