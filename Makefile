#
# Unified Makefile for 4-Player Chess Engine (CLI and Web GUI)
#

# --- Configuration ---

# Compiler: g++ from your MSYS2 UCRT64 environment
CXX = g++

# Compiler Flags: -O3 for optimization, -g for debugging, C++20 standard
CXXFLAGS = -std=c++20 -O3 -march=native -pthread -flto=auto

# Include Directories
INCLUDES = -I .

# Linker Flags
LDFLAGS = -pthread

# Source files for the core chess engine (shared by CLI and the addon)
ENGINE_SRCS = board.cc player.cc utils.cc transposition_table.cc move_picker.cc

# --- Build Targets ---

# Default target when you just run "make"
all: cli

# Target to build the command-line interface (CLI)
# RENAMED from cli.exe to cli
cli: $(ENGINE_SRCS) cli.cc command_line.cc
	@echo "--- Building CLI Executable (cli) ---"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)
	@echo ">>> CLI build complete: ./cli"

# Target to build the Node.js C++ addon
# This uses node-gyp, which will automatically use the GCC compiler
# because of the environment variables we set up earlier.
addon:
	@echo "--- Building Node.js C++ Addon (addon.node) ---"
	cd ui && node-gyp configure --compiler=g++ --cxxflags="$(CXXFLAGS)"
	cd ui && node-gyp build
	@echo ">>> Node.js addon build complete."

# Target to run the web GUI
# This first ensures the C++ addon is built, then starts the Node.js server.
gui: addon
	@echo "--- Starting Web GUI ---"
	@echo "Navigate to http://localhost:3000 in your browser."
	cd ui && npm start

# Target to clean all build artifacts
clean:
	@echo "--- Cleaning all build files ---"
	# Clean CLI executable (both with and without .exe for cross-platform safety)
	rm -f cli cli.exe
	# Clean C++ object files (if any)
	rm -f *.o
	# Clean Node.js addon build directory
	rm -rf ui/build
	# Clean node-gyp temporary files
	rm -f ui/*.sln ui/*.vcxproj ui/*.vcxproj.filters
	@echo ">>> Cleanup complete."