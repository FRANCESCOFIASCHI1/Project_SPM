CXX = mpic++ -std=c++17

ifdef DEBUG
    OPTIMIZE_FLAGS  += -g -fno-inline-functions
else
    OPTIMIZE_FLAGS  += -O3 -ffast-math -DNDEBUG
endif

# Rileva il compilatore effettivo usato da mpic++
REALCXX := $(shell $(CXX) -show | awk '{print $$1}')
COMPILER := $(shell $(REALCXX) --version | head -n 1)

# Flag comuni
CXXFLAGS += -Wall
LIBS     = -pthread
INCLUDES = -I./fastflow -I./include

# Aggiungi OpenMP in modo portabile
ifeq ($(findstring clang,$(COMPILER)),clang)
    # macOS / Clang con OpenMP installato via Homebrew
    CXXFLAGS += -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
    LIBS     += -L$(shell brew --prefix libomp)/lib -lomp
else
    # GCC (Linux o Homebrew GCC su macOS)
    CXXFLAGS += -fopenmp
endif

SOURCES = $(wildcard *.cpp)
TARGET  = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 
.SUFFIXES: .c .cpp .o

%.d: %.cpp
	set -e; $(CXX) -MM $(INCLUDES) $(CXXFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@

%.d: %.c
	set -e; $(CC) -MM $(INCLUDES) $(CFLAGS) $< \
		| sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
		[ -s $@ ] || rm -f $@

%.o: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -c -o $@ $<

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTIMIZE_FLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

all: $(TARGET)

ff_chol_mdf: CXXFLAGS += -DDO_NOTHING
ff_chol_mdf: ff_chol_mdf.cpp

clean:
	-rm -fr *.o *~

cleanall: clean
	-rm -fr $(TARGET) *.d

include $(OBJS:.o=.d)
