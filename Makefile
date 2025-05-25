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
	-Wno-dangling-pointer \
	-O3 \
	-mcpu=cortex-a72 \
	-mtune=cortex-a72 \
	-march=armv8-a+simd \
	-fomit-frame-pointer \
	-funroll-loops \
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

ifeq ($(LIMBS_NUM), 32)
	CFLAGS += -DLIMBS_NUM=32
else ifeq ($(LIMBS_NUM), 16)
	CFLAGS += -DLIMBS_NUM=16
else ifeq ($(LIMBS_NUM), 8)
	CFLAGS += -DLIMBS_NUM=8
endif

TARGET = bench
SOURCES = hal/hal.c lib/lib.c benchmarks/*.c bench.c

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)


clean:
	-$(RM) -rf $(TARGET) *.d