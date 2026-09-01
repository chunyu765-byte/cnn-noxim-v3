TARGET_ARCH = linux64
CC     = g++
# Default to release-oriented build for faster simulation.
# You can still override at command line, e.g.:
#   make OPT="-O0 -g -fopenmp"
OPT    ?= -O3 -DNDEBUG -fopenmp -march=native
DEBUG  = -g
OTHER  = # -Wall -Wno-deprecated
CFLAGS = $(OPT)

VPATH = ../src:.
MODULE = noxim
BUILDDIR = build

# 原有的 C++ 源文件
SRCS = ./NoximNoC.cpp ./NoximRouter.cpp ./NoximProcessingElement.cpp ./NoximBuffer.cpp \
	./NoximStats.cpp ./NoximGlobalStats.cpp ./NoximGlobalRoutingTable.cpp \
	./NoximLocalRoutingTable.cpp ./NoximGlobalTrafficTable.cpp ./NoximReservationTable.cpp \
	./NoximPower.cpp ./NoximCmdLineParser.cpp ./NoximMain.cpp ./thermal_IF.cpp \
	./NNModel.cpp

# ========== 添加近似乘法器文件 ==========
SRCS += ./drum.cpp
SRCS += ./biasedmul.cpp
SRCS += ./mul16.cpp
SRCS += ./unbiased_mul8_16.cpp
# ========================================

# C 源文件
SRCS_C = ./util.c ./temperature_grid.c ./temperature_block.c ./temperature.c \
        ./shape.c ./RCutil.c ./npe.c ./package.c ./flp.c

OBJS_C = $(addprefix $(BUILDDIR)/,$(notdir $(SRCS_C:.c=.o)))
OBJS = $(addprefix $(BUILDDIR)/,$(notdir $(SRCS:.cpp=.o)))
OBJS_A = $(OBJS)
OBJS_A += $(OBJS_C)

include ./Makefile.defs
