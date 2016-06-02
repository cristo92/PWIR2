#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>
#include <algorithm>
#include <vector>
#include <list>
#include <iostream>
#include <iomanip>

#include "densematgen.h"

using namespace std;

#define MPI_INIT_TAG 	1001
#define MPI_INIT_A_TAG 	1002
#define MPI_SEND_A 		1003

#define DEBUG_RANK 3

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
	vector<vector<Point>> rows;
	int firstRow; //included
	int lastRow; //excluded

	PartA(int firstRow, int lastRow): firstRow(firstRow), lastRow(lastRow) {}
	PartA() {}
};

struct PartB {
	int firstColumn, lastColumn;
	vector<vector<double>> columns;

	PartB(int firstColumn, int lastColumn): firstColumn(firstColumn), lastColumn(lastColumn) {}
	PartB() {}
};

struct PartC {
	int firstColumn, lastColumn, n;
	vector<list<Point>> columns;

	PartC(int firstColumn, int lastColumn, int n): firstColumn(firstColumn), lastColumn(lastColumn), n(n) {}
	PartC() {}
};

ostream& operator<<(ostream &os, const Point &p) {
	os << "(" << p.idx << ", " << p.x << ")";
	return os;
}

ostream& operator<<(ostream& os, const PartA &a) {
	os << "==== PartA ====\n";
	os << a.firstRow << " " << a.lastRow << endl;
	for(int i = a.firstRow; i < a.lastRow; i++) {
		int idx = i - a.firstRow;
		for(auto p = a.rows[idx].begin(); p != a.rows[idx].end(); p++) {
			os << *p << " ";
		}
		os << "\n";
	}
	return os;
}

ostream& operator<<(ostream& os, const PartB &b) {
	os << "==== PartB ====\n";
	os << b.firstColumn << " " << b.lastColumn << "\n";
	for(int j = 0; j < b.columns[0].size(); j++) {
		for(int i = 0, col = b.firstColumn; col < b.lastColumn; col++, i++) {
			os << b.columns[i][j] << " ";
		}
		os << "\n";
	}

	return os;
}

ostream& operator<<(ostream& os, const PartC &c) {
	os << "==== PartC ====\n";
	os << c.firstColumn << " " << c.lastColumn << "\n";
	bool bol = true;
	vector< decltype(c.columns[0].begin()) > ptr;
	for(int i = 0, col = c.firstColumn; col < c.lastColumn; col++, i++)
		ptr.push_back(c.columns[i].begin());
	for(int j = 0; j < c.n; j++) {
		for(int i = 0, col = c.firstColumn; col < c.lastColumn; col++, i++) {
			if(ptr[i]->idx == j) {
				os << ptr[i]->x << " ";
				ptr[i]++;
			}
			else os << 0 << " ";
		}
		os << "\n";
	}

	return os;
}

int prev_proc(int x, int y);
int next_proc(int x, int y);

int analysePartA(const PartA &partA);
int partAToMessage(const PartA &partA, char *out);
PartA messageToPartA(char *out);

int main(int argc, char * argv[])
{
	cout << setprecision(5) << fixed;

	int show_results = 0;
	int use_inner = 0;
	int gen_seed = -1;
	int repl_fact = 1;

	int option = -1;

	double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
	int num_processes = 1;
	int mpi_rank = 0;
	int exponent = 1;
	double ge_element = 0;
	int count_ge = 0;

	char *sparse_filename = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);


	while ((option = getopt(argc, argv, "vis:f:c:e:g:")) != -1) {
		switch (option) {
		case 'v': show_results = 1; 
			break;
		case 'i': use_inner = 1;
			break;
		case 'f': if ((mpi_rank) == 0) 
			{ 
				sparse_filename = optarg;
			}
			break;
		case 'c': repl_fact = atoi(optarg);
			break;
		case 's': gen_seed = atoi(optarg);
			break;
		case 'e': exponent = atoi(optarg);
			break;
		case 'g': count_ge = 1; 
			ge_element = atof(optarg);
			break;
		default: fprintf(stderr, "error parsing argument %c exiting\n", option);
			MPI_Finalize();
			return 3;
		}
	}
	if ((gen_seed == -1) || ((mpi_rank == 0) && (sparse_filename == NULL)))
	{
		fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
		MPI_Finalize();
		return 3;
	}

	int c = repl_fact;
	int p = num_processes;
	int s = p / c;
	int q = (s + c - 1) / c;

	vector<char*> messages(num_processes);
	MPI_Request requests[num_processes * 2];
	MPI_Status statuses[num_processes * 2];

	InitMessage my_init_msg;
	PartA my_partA;
	PartB my_partB;
	PartC my_partC;
	InitMessage init_msgs[s][c];
	//read
	if(mpi_rank == 0) {
		int m, n, nnz, max_nnz;
		FILE *file = fopen(sparse_filename, "r");
		fscanf(file, "%d%d%d%d", &m, &n, &nnz, &max_nnz);
		printf("%d %d %d %d\n", m, n, nnz, max_nnz);
	
		vector<double> line1(nnz);
		vector<int> line2(m + 1);
		vector<int> line3(nnz);
		vector<vector<Point>> A(m);

		for(int i = 0; i < nnz; i++) fscanf(file, "%lf", &line1[i]);
		for(int i = 0; i < m + 1; i++) fscanf(file, "%d", &line2[i]);
		for(int i = 0; i < nnz; i++) fscanf(file, "%d", &line3[i]);

		for(int idx = 0, i = 0; i < m; i++) {
			for(; idx < line2[i + 1]; idx++) {
				A[i].push_back(Point(line3[idx], line1[idx]));
			}
		}
		
		// set B columns
		int length;
		// set A columns and rows
		if(use_inner) {
			int length = (n + s - 1) / s;
			vector<PartA> partA(s);
			int ids[c];

			ids[0] = 0;
			for(int i = 1; i < c; i++) {
				ids[i] = (ids[i - 1] - q + s) % s;
				printf("%d ", ids[i]);
			}
			printf("\n");

			for(int i = 0, row = 0; i < s; i++, row = min(row + length, n)) {
				for(int j = 0; j < c; j++) {
					init_msgs[(ids[j] + i) % s][j].firstARow = row;
					init_msgs[(ids[j] + i) % s][j].lastARow = min(row + length, n);

					init_msgs[(ids[j] + i) % s][j].firstBColumn = length * ((ids[j] + i) % s);
					init_msgs[(ids[j] + i) % s][j].lastBColumn = min(length * ((ids[j] + i) % s) + length, n);

					init_msgs[(ids[j] + i) % s][j].cx = (ids[j] + i) % s;
					init_msgs[(ids[j] + i) % s][j].cy = j;

					init_msgs[(ids[j] + i) % s][j].n = m;
					init_msgs[(ids[j] + i) % s][j].nnz = nnz;
					init_msgs[(ids[j] + i) % s][j].max_nnz = max_nnz;
				}
				partA[i].firstRow = row;
				partA[i].lastRow = min(row + length, n);
				partA[i].rows.resize(partA[i].lastRow - partA[i].firstRow);
				for(int row_idx = row; row_idx < partA[i].lastRow; row_idx++) {
					partA[i].rows[row_idx - row] = A[row_idx];
				}
			}
			/*
			// DEBUG BEGIN
			char *msg = (char*)malloc(10000);
			for(int i = 0; i < s; i++) {
				cout << partA[i];
				partAToMessage(partA[i], msg);
				cout << messageToPartA(msg);
			}
			free(msg);
			// DEBUG END
			*/
			my_init_msg = init_msgs[0][0];

			for(int i = 1; i < p; i++) {
				int idx = (i/c) % s;
				int idy = i % c;
				MPI_Isend(
					&init_msgs[idx][idy],
					sizeof(InitMessage),
					MPI_BYTE,
					i,
					MPI_INIT_TAG,
					MPI_COMM_WORLD,
					&requests[2 * i]
				);

				int partAIdx = (idx + ids[idy]) % s;
				int bytes = analysePartA(partA[partAIdx]);
				messages[i] = (char*)malloc(bytes);
				partAToMessage(partA[partAIdx], messages[i]);
				MPI_Isend(
					messages[i],
					bytes,
					MPI_BYTE,
					i,
					MPI_INIT_A_TAG,
					MPI_COMM_WORLD,
					&requests[2 * i + 1]
				);
			}
			my_init_msg = init_msgs[0][0];
			my_partA = partA[0];
		}
		else {
			vector<InitMessage> init_msgs(num_processes);
			length = (n + num_processes - 1) / num_processes;
			for(int p = 0, column = 0; 
			 	p < num_processes && column < n; 
			  	p++, column += length) 
			{
				init_msgs[p].firstBColumn = column;
				init_msgs[p].lastBColumn = min(column + length, n);
			}
		}
	}
	
	MPI_Request recv_request, send_request;
	MPI_Status recv_status, send_status;

	// RECEIVE MATRIX A
	if(mpi_rank > 0) {
		MPI_Irecv(
			&my_init_msg, 
			sizeof(InitMessage),
			MPI_BYTE,
			0,
			MPI_INIT_TAG,
			MPI_COMM_WORLD,
			&recv_request
		);

		MPI_Wait(&recv_request, &recv_status);
	}

	int rows_per_proc = (my_init_msg.n + s - 1) / s;
	int max_bytes = rows_per_proc * my_init_msg.max_nnz * sizeof(Point) + (2 + rows_per_proc) * sizeof(int);
	char *recv_msg = (char*)malloc(max_bytes);
	char *send_msg = (char*)malloc(max_bytes);

	if(mpi_rank > 0) {
		MPI_Irecv(
			recv_msg,
			max_bytes,
			MPI_BYTE,
			0,
			MPI_INIT_A_TAG,
			MPI_COMM_WORLD,
			&recv_request
		);

		MPI_Wait(&recv_request, &recv_status);
		my_partA = messageToPartA(recv_msg);
		if(mpi_rank == DEBUG_RANK) {
		cout << "I'm " << mpi_rank << "(" << my_init_msg.cx << ", " << my_init_msg.cy  << "), and got:\n";
		cout << my_partA; }
	}
	if(mpi_rank == 0) {
		MPI_Waitall((num_processes - 1) * 2, requests + 2, statuses + 2);
		for(int i = 1; i < p; i++) free(messages[i]);
		//for(int i = 1; i < num_processes; i ++) cout << messageToPartA(messages[i]);
	}

	// DEBUG BEGIN
	//printf("I'm %d and my coords are: %d, %d. Rows: %d, %d. Colums: %d, %d.\n", mpi_rank, my_init_msg.cx, my_init_msg.cy,
	//	my_init_msg.firstARow, my_init_msg.lastARow, my_init_msg.firstBColumn, my_init_msg.lastBColumn);
	// DEBUG END

	// GENERATE MATRIX B
	int n = my_init_msg.n;
	my_partB = PartB(my_init_msg.firstBColumn, my_init_msg.lastBColumn);
	my_partB.columns.resize(my_init_msg.lastBColumn - my_init_msg.firstBColumn, vector<double>(n));
	for(int i = 0, col = my_init_msg.firstBColumn; col < my_init_msg.lastBColumn; col++, i++) {
		for(int j = 0; j < n; j++)
			my_partB.columns[i][j] = generate_double(gen_seed, j, col);
	}

	// DEBUG BEGIN
	if(mpi_rank == DEBUG_RANK) cout << my_partB;
	// DEBUG END

	// CREATE MATRIX C
	my_partC = PartC(my_partB.firstColumn, my_partB.lastColumn, n);
	my_partC.columns.resize(my_partC.lastColumn - my_partC.firstColumn);

	comm_start =  MPI_Wtime();
	// FIXME: scatter sparse matrix; cache sparse matrix; cache dense matrix
	MPI_Barrier(MPI_COMM_WORLD);
	comm_end = MPI_Wtime();

	// ***************************************
	// ************* COMPUTATION *************
	// ***************************************
	comp_start = MPI_Wtime();
	
	if(mpi_rank >= s * q) goto exit;

	while(q--) {
		int idx = my_init_msg.cx;
		int idy = my_init_msg.cy;

		// TODO - don't send and receive last message
		int bytes = partAToMessage(my_partA, send_msg);
		MPI_Isend(
			send_msg,
			bytes,
			MPI_BYTE,
			c * ((idx + s - 1) % s) + idy,
			MPI_SEND_A,
			MPI_COMM_WORLD,
			&send_request
		);

		if(mpi_rank == DEBUG_RANK)
			cout << my_partA;

		if(mpi_rank == DEBUG_RANK) {
		for(int i = 0, row = my_partA.firstRow; row < my_partA.lastRow; row++, i++) {
			for(int j = 0, col = my_partB.firstColumn; col < my_partB.lastColumn; col++, j++) {
				double el = 0;
				for(auto it = my_partA.rows[i].begin(); it != my_partA.rows[i].end(); it++) {
					el += it->x * my_partB.columns[j][it->idx];
					//cout << row << " " << col << " " << it->idx << " " << it->x << " " << my_partB.columns[j][it->idx] << " " << el <<  "\n";
				}
				my_partC.columns[j].push_back(Point(row, el));
			}
		}}

		MPI_Irecv(
			recv_msg,
			max_bytes,
			MPI_BYTE,
			c * ((idx + 1) % s) + idy,
			MPI_SEND_A,
			MPI_COMM_WORLD,
			&recv_request
		);

		MPI_Wait(&send_request, &send_status);
		MPI_Wait(&recv_request, &recv_status);

		my_partA = messageToPartA(recv_msg);
	}

	// Sort partC
	for(auto &col : my_partC.columns) {
		if(col.size() == 1) continue;
		auto ptr = col.begin();
		for(; next(ptr) != col.end(); ptr++)
			if(ptr->idx > next(ptr)->idx) break;
		ptr++;

		if(mpi_rank == DEBUG_RANK) cout << *ptr;

		col.splice(col.begin(), col, ptr, col.end());
	}

exit:
	MPI_Barrier(MPI_COMM_WORLD);
	comp_end = MPI_Wtime();

	// DEBUG BEGIN
	if(mpi_rank == DEBUG_RANK) cout << my_partC;
	// DEBUG END
	
	if (show_results) 
	{
		// FIXME: replace the following line: print the whole result matrix
		printf("1 1\n42\n");
	}
	if (count_ge)
	{
		// FIXME: replace the following line: count ge elements
		printf("54\n");
	}

	free(recv_msg);
	free(send_msg);
	MPI_Finalize();
	return 0;
}

int analysePartA(const PartA &partA) {
	int size = sizeof(int) * (2 + partA.lastRow - partA.firstRow);
	int count = 0;

	for(int i = partA.firstRow; i < partA.lastRow; i++) {
		int idx = i - partA.firstRow;
		count += partA.rows[idx].size();
	}

	size += count * sizeof(Point);

	return size;
}

int partAToMessage(const PartA &partA, char *out) {
	int size = sizeof(int) * (2 + partA.lastRow - partA.firstRow);
	int count = 0;

	*(int*)out = partA.firstRow; out += sizeof(int);
	*(int*)out = partA.lastRow; out += sizeof(int);
	for(int i = partA.firstRow; i < partA.lastRow; i++) {
		int idx = i - partA.firstRow;
		*(int*)out = partA.rows[idx].size();
		out += sizeof(int);
		for(auto it = partA.rows[idx].begin(); it != partA.rows[idx].end(); it++) {
			*(Point*)out = *it;
			out += sizeof(Point);
			count++;
		}
	}

	size += count * sizeof(Point);

	return size;
}

PartA messageToPartA(char *out) {
	PartA partA;
	partA.firstRow = *(int*)out; out += sizeof(int);
	partA.lastRow = *(int*)out; out += sizeof(int);
	partA.rows.resize(partA.lastRow - partA.firstRow);

	for(int i = partA.firstRow; i < partA.lastRow; i++) {
		int idx = i - partA.firstRow;
		int size = *(int*)out; out += sizeof(int);
		while(size--) {
			partA.rows[idx].push_back(
				*(Point*)out
			);
			out += sizeof(Point);
		}
	}

	return partA;
}
