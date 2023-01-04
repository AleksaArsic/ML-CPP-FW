
BUILD_DIR := ./build
SRC_DIR   := ./src
INC_DIR   := ./inc
OBJ_DIR   := $(BUILD_DIR)/obj
BIN_DIR   := $(BUILD_DIR)/bin

EXE := $(BIN_DIR)/main
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

CC       := g++
CXXFLAGS := -I$(INC_DIR) -MMD -MP -std=c++20
CFLAGS   := -Wall
LDFLAGS  := -Llib
LDLIBS   := 

.PHONY: all clean

all: 

build: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean: 
	@$(RM) -rv $(BUILD_DIR) $(BIN_DIR) $(OBJ_DIR)

-include $(OBJ:.o=.d)