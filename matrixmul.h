#ifndef __MATRIXMUL_H
#define __MATRIXMUL_H

#include <iostream>
using namespace std;

#define MPI_INIT_TAG 	1001
#define MPI_INIT_A_TAG 	1002
#define MPI_SEND_A 		1003

#define DEBUG_RANK 7

struct InitMessage {
	int cx, cy, c;
	int n, nnz, max_nnz;
	int num_rounds; // q
	int firstBColumn; //included
	int lastBColumn; //excluded
	int firstARow;
	int lastARow;
	int firstAColumn;
	int lastAColumn;
};

struct Point {
	int idx; //wartość
	double x; //współrzędne kolumny

	Point(int idx, double x): idx(idx), x(x) {}
	Point() {}
};

struct PartA {
	vector<vector<Point>> vecs;
	int first; //included
	int last; //excluded

	PartA(int first, int last): first(first), last(last) {}
	PartA() {}
};

struct PartB {
	int first, last;
	vector<vector<double>> columns;

	PartB(int first, int last): first(first), last(last) {}
	PartB() {}
};
/*
struct PartC {
	int firstColumn, lastColumn, n;
	vector<list<Point>> columns;

	PartC(int firstColumn, int lastColumn, int n): firstColumn(firstColumn), lastColumn(lastColumn), n(n) {}
	PartC() {}
};*/

int analysePartA(const PartA &partA);
int partAToMessage(const PartA &partA, char *out);
// int partCToMessage(const PartC &partC, char *out);
PartA messageToPartA(char *out);

ostream& operator<<(ostream &os, const Point &p);
ostream& operator<<(ostream& os, const PartA &a);
ostream& operator<<(ostream& os, const PartB &b);
// ostream& operator<<(ostream& os, const PartC &c);

#endif

