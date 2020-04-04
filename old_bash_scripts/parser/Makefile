CXX      = g++
CXXFLAGS = -pedantic-errors -Wall -Wextra -Werror -O3
STDFLAGS = -std=c++14

build:
	$(CXX) parser.cpp -o parser $(STDFLAGS) $(CXXFLAGS)

run:
	./parser $(OUTPUT) $(ID) $(FILE) $(MODE)

clean:
	rm -rf parser
