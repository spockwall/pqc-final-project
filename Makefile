# SPDX-License-Identifier: Apache-2.0

.PHONY: clean

CC  ?= gcc
LD  := $(CC)


CFLAGS := \
	-Wall \
	-Wextra \
	-Werror=unused-result \
	-Wpedantic \
	-Werror \
	-Wmissing-prototypes \
	-Wshadow \
	-Wpointer-arith \
	-Wredundant-decls \
	-Wno-long-long \
	-Wno-unknown-pragmas \
	-Wno-unused-command-line-argument \
	-O3 \
	-mcpu=cortex-a72 \
	-fomit-frame-pointer \
	-std=c99 \
	-pedantic \
	-Ihal \
	-Ilib \
	-Ifft \
	-Ibenchmarks \
	-MMD \
	$(CFLAGS)

LDFLAGS := -lgmp
LDLIBS := -lm

ifeq ($(CYCLES),PMU)
	CFLAGS += -DPMU_CYCLES
endif

ifeq ($(CYCLES),PERF)
	CFLAGS += -DPERF_CYCLES
endif

ifeq ($(CYCLES),MAC)
	CFLAGS += -DMAC_CYCLES
endif

# if verbose, print computation result
ifeq ($(VERBOSE), TRUE)
	CFLAGS += -DVERBOSE
endif


# Targets
TARGET = bench
#TARGETS = bench_gmp bench_karatsuba
#TARGET_GMP = bench_gmp
#TARGET_KARATSUBA = bench_karatsuba

# Source sets
SOURCES = hal/hal.c lib/lib.c benchmarks/gmp_mul.c benchmarks/karatsuba.c bench.c

#COMMON_SOURCES = hal/hal.c lib/lib.c
#GMP_SOURCES = gmp.c $(COMMON_SOURCES)
#KARATSUBA_SOURCES = karatsuba.c $(COMMON_SOURCES)


#all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

# Karatsuba target
#bench_karatsuba: $(KARATSUBA_SOURCES)
#	$(CC) $(CFLAGS) $(KARATSUBA_SOURCES) -o $(TARGET_KARATSUBA) $(LDFLAGS) $(LDLIBS)


clean:
	-$(RM) -rf $(TARGET) *.d