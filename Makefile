################# Local Settings #################

# Choose a different C++ compiler if needed
CXX=g++

# Provide C++ Boost library paths (if installed in a non-standard location)
BOOSTROOT= # Package directory (e.g., "/path/to/boost_1_66_0")
BOOSTLIB= # Library path (e.g., "/path/to/boost_1_66_0/stage/lib")

# Provide paths to the Armadillo library (if installed in a non-standard location)
# Make sure ARMALIB is in the system search path (e.g. LD_LIBRARY_PATH on Linux) 
# when netsne is called
ARMAINC= # e.g., "/path/to/armadillo/usr/include"
ARMALIB= # e.g., "/path/to/armadillo/usr/lib/x86_64-linux-gnu"

##################################################

INCPATHS=$(if $(strip $(ARMAINC)),-I $(ARMAINC),) \
         $(if $(strip $(BOOSTROOT)),-I $(BOOSTROOT),) \
         $(if $(strip $(OBLASINC)),-I $(OBLASINC),)
					 
CXXFLAGS=-g -O2 $(INCPATHS) -march=native -std=c++11
LIBS=-lboost_filesystem -lboost_system -lboost_program_options -larmadillo
LDFLAGS=$(if $(strip $(BOOSTLIB)),-L $(BOOSTLIB),) \
        $(if $(strip $(ARMALIB)),-L $(ARMALIB),)

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
