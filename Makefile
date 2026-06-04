CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS ?= -lm

# ARM64 NEON build
ifeq ($(NEON),1)
  CFLAGS += -march=armv8-a+simd -DUSE_NEON
else
  CFLAGS += -DNO_NEON
endif

SRCS    = src/tile_neon.c src/hash_blake2b.c src/embed_neon.c \
          src/search_neon.c src/search_scalar.c
INC     = -Iinclude

.PHONY: all test bench clean

all: test_neon bench_neon

test_neon: tests/test_neon.c $(SRCS)
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS)

bench_neon: benches/bench_neon.c $(SRCS)
	$(CC) $(CFLAGS) $(INC) -o $@ $^ $(LDFLAGS)

test: test_neon
	./test_neon

bench: bench_neon
	./bench_neon

clean:
	rm -f test_neon bench_neon
