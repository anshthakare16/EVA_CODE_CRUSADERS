# Makefile for EVA C Executors

CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -shared -luser32 -lgdi32

# Target names for different platforms
ifeq ($(OS),Windows_NT)
    TARGET = executor.dll
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        TARGET = executor.so
    endif
    ifeq ($(UNAME_S),Darwin)
        TARGET = executor.dylib
    endif
endif

all: $(TARGET)

$(TARGET): executor.c executor.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) executor.c

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
