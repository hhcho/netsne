CXX = g++
INCPATHS = -I/data/cb/hhcho/util/armadillo-8.200.1/include -I/usr/include/openblas -I/data/cb/hhcho/util/boost_1_66_0
CFLAGS = -g -O1 $(INCPATHS) -march=native -std=c++11 -pthread
LDLIBS = -lboost_filesystem -lboost_system -lboost_program_options -lopenblas
LDPATH = -L/usr/lib -L/data/cb/hhcho/util/boost_1_66_0/stage/lib

BUILD = build
PROGS = bin

SRCS = bhtsne.cpp sptree.cpp vptree.cpp netsne.cpp
PROGNAMES = RunNetsne ComputeP RunBhtsne
OBJS=$(subst .cpp,.o,$(SRCS))

OBJPATHS = $(patsubst %.cpp,$(BUILD)/%.o, $(SRCS))
TESTPATHS = $(addprefix $(PROGS)/, $(PROGNAMES))

all: $(OBJPATHS) $(TESTPATHS)

obj: $(OBJPATHS)

$(BUILD):
	mkdir -p $(BUILD)

$(PROGS):
	mkdir -p $(PROGS)

$(BUILD)/%.o: %.cpp *.h | $(BUILD)
	$(CXX) $(CFLAGS) -o $@ -c $<

$(PROGS)/%: %.cpp $(OBJPATHS) $(PROGS)
	$(CXX) $(CFLAGS) -o $@ $< $(LDPATH) $(OBJPATHS) $(LDLIBS)

clean:
	rm -rf $(BUILD) $(PROGS) *~
