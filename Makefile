# SPDX-License-Identifier: Apache-2.0

.PHONY: clean

TARGET = bench
CC  ?= gcc
LD  := $(CC)

#SOURCES = hal/hal.c bench.c ntt_zetas.c ntt.S
SOURCES = hal/hal.c bench.c lib/lib.c fft/fft.c


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

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	-$(RM) -rf $(TARGET) *.d