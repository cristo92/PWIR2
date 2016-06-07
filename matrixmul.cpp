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

#include "matrixmul.h"
#include "densematgen.h"

using namespace std;

int main(int argc, char * argv[])
{
	cout << setprecision(5) << fixed;
	cerr << setprecision(5) << fixed;

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

	comm_start =  MPI_Wtime();

	int c = repl_fact;
	int p = num_processes;
	int s = p / c;
	int q = (s + c - 1) / c;

	vector<char*> messages(num_processes);
	MPI_Request requests[num_processes * 2];
	MPI_Status statuses[num_processes * 2];

	InitMessage my_init_msg;
	PartA A;
	PartB B;
	PartC C;
	InitMessage init_msgs[s][c];
	//read
	if(mpi_rank == 0) {
		int m, n, nnz, max_nnz;
		FILE *file = fopen(sparse_filename, "r");
		fscanf(file, "%d%d%d%d", &m, &n, &nnz, &max_nnz);
	
		vector<double> line1(nnz);
		vector<int> line2(m + 1);
		vector<int> line3(nnz);
		vector<vector<Point>> inputA(m);

		for(int i = 0; i < nnz; i++) fscanf(file, "%lf", &line1[i]);
		for(int i = 0; i < m + 1; i++) fscanf(file, "%d", &line2[i]);
		for(int i = 0; i < nnz; i++) fscanf(file, "%d", &line3[i]);

		for(int idx = 0, i = 0; i < m; i++) {
			for(; idx < line2[i + 1]; idx++) {
				inputA[i].push_back(Point(line3[idx], line1[idx]));
			}
		}
		
		// set A columns and rows
		if(use_inner) {
			int length = (n + s - 1) / s;
			vector<PartA> partA(s);
			int ids[c];

			ids[0] = 0;
			for(int i = 1; i < c; i++) {
				ids[i] = (ids[i - 1] - q + s) % s;
			}

			for(int i = 0, row = 0; i < s; i++, row = min(row + length, n)) {
				for(int j = 0; j < c; j++) {
					int idx = (ids[j] + i) % s;
					int idy = j;

					init_msgs[idx][idy].firstBColumn = length * idx;
					init_msgs[idx][idy].lastBColumn = min(length * (idx + 1), n);

					init_msgs[idx][idy].cx = idx;
					init_msgs[idx][idy].cy = idy;

					init_msgs[idx][idy].n = m;
					init_msgs[idx][idy].nnz = nnz;
					init_msgs[idx][idy].max_nnz = max_nnz;
				}
				partA[i].first = row;
				partA[i].last = min(row + length, n);
				partA[i].vecs.resize(partA[i].last - partA[i].first);
				for(int row_idx = row; row_idx < partA[i].last; row_idx++) {
					partA[i].vecs[row_idx - row] = inputA[row_idx];
				}
			}

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
			A = partA[0];
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
		A = messageToPartA(recv_msg);
		if(mpi_rank == DEBUG_RANK) {
			cerr << "I'm " << mpi_rank << "(" << my_init_msg.cx << ", " << my_init_msg.cy  << "), and got:\n";
			cerr << A; 
		}
	}
	if(mpi_rank == 0) {
		MPI_Waitall((num_processes - 1) * 2, requests + 2, statuses + 2);
		for(int i = 1; i < p; i++) free(messages[i]);
	}

	// GENERATE MATRIX B
	int n = my_init_msg.n;
	B = PartB(my_init_msg.firstBColumn, my_init_msg.lastBColumn);
	B.columns.resize(my_init_msg.lastBColumn - my_init_msg.firstBColumn, vector<double>(n));
	for(int i = 0, col = my_init_msg.firstBColumn; col < my_init_msg.lastBColumn; col++, i++) {
		for(int j = 0; j < n; j++)
			B.columns[i][j] = generate_double(gen_seed, j, col);
	}

	// DEBUG BEGIN
	if(mpi_rank == DEBUG_RANK) cerr << B;
	// DEBUG END

	// CREATE MATRIX C
	C = PartC(B.first, B.last, n);
	C.columns.resize(C.lastColumn - C.firstColumn);

	MPI_Barrier(MPI_COMM_WORLD);
	comm_end = MPI_Wtime();

	// ***************************************
	// ************* COMPUTATION *************
	// ***************************************
	comp_start = MPI_Wtime();
	
	if(mpi_rank >= s * c) goto exit;

	while(q--) {
		if(mpi_rank == DEBUG_RANK) cerr << "STEP " << q << "\n";
		int idx = my_init_msg.cx;
		int idy = my_init_msg.cy;

		// TODO - don't send and receive last message
		int bytes = partAToMessage(A, send_msg);
		if(q)
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
			cerr << A;

		for(int i = 0, row = A.first; row < A.last; row++, i++) {
			for(int j = 0, col = B.first; col < B.last; col++, j++) {
				double el = 0;
				for(auto it = A.vecs[i].begin(); it != A.vecs[i].end(); it++) {
					el += it->x * B.columns[j][it->idx];
				}
				C.columns[j].push_back(Point(row, el));
			}
		}

		if(q)
			MPI_Irecv(
				recv_msg,
				max_bytes,
				MPI_BYTE,
				c * ((idx + 1) % s) + idy,
				MPI_SEND_A,
				MPI_COMM_WORLD,
				&recv_request
			);

		if(q) {
			MPI_Wait(&send_request, &send_status);
			MPI_Wait(&recv_request, &recv_status);
		}

		A = messageToPartA(recv_msg);
	}

	// Sort partC
	for(auto &col : C.columns) {
		if(col.size() == 1) continue;
		auto ptr = col.begin();
		for(; next(ptr) != col.end(); ptr++)
			if(ptr->idx > next(ptr)->idx) break;
		ptr++;

		if(mpi_rank == DEBUG_RANK) {
			if(ptr == col.end())
				cerr << "Point end\n";
			else
				cerr << *ptr << "\n";
		}

		col.splice(col.begin(), col, ptr, col.end());
	}

exit:
	MPI_Barrier(MPI_COMM_WORLD);
	comp_end = MPI_Wtime();

	free(recv_msg);
	free(send_msg);

	// DEBUG BEGIN
	if(mpi_rank == DEBUG_RANK) cerr << C;
	// DEBUG END
	
	if (show_results) 
	{
		double bigC[n][n];
		memset(bigC, 0, n * n * sizeof(double));
		int bytes = rows_per_proc * n * sizeof(Point) + (2 + rows_per_proc) * sizeof(int);
		char *sendbuf = NULL, *recvbuf = NULL, *buf = NULL;
		int sendcount = 0;
		int *recvcount = NULL, *displs = NULL;

		if(mpi_rank == 0) {
			recvbuf = (char*)malloc(bytes * p);
			recvcount = (int*)malloc(sizeof(int) * p);
			displs = (int*)malloc(sizeof(int) * p);
			for(int i = 0; i < p; i++) {
				recvcount[i] = bytes;
				displs[i] = i * bytes;
			}

			partCToMessage(C, recvbuf);
		}
		else {
			sendbuf = (char*)malloc(bytes);
			sendcount = partCToMessage(C, sendbuf);
		}

		MPI_Gatherv(sendbuf, sendcount, MPI_BYTE, recvbuf, recvcount,
			displs, MPI_BYTE, 0, MPI_COMM_WORLD);

		if(mpi_rank == 0) {
			for(int i = 0; i < min(p, s * c); i++) {
				buf = recvbuf + i * bytes;
				PartA tempA = messageToPartA(buf);

				for(int i = 0, row = tempA.first; row < tempA.last; row++, i++) {
					for(auto &p : tempA.vecs[i]) {
						bigC[p.idx][row] = p.x;
					}
				}
			}

			cout << n << " " << n << "\n";
			for(int i = 0; i < n; i++) {
				for(int j = 0; j < n; j++)
					cout << bigC[i][j] << " ";
				cout << "\n";
			}
		}

		if(mpi_rank == 0) {
			free(recvbuf);
			free(recvcount);
			free(displs);
		}
	}
	if (count_ge)
	{
		// FIXME: replace the following line: count ge elements
		printf("54\n");
	}

	MPI_Finalize();
	return 0;
}

int analysePartA(const PartA &partA) {
	int size = sizeof(int) * (2 + partA.last - partA.first);
	int count = 0;

	for(int i = partA.first; i < partA.last; i++) {
		int idx = i - partA.first;
		count += partA.vecs[idx].size();
	}

	size += count * sizeof(Point);

	return size;
}

int partAToMessage(const PartA &partA, char *out) {
	int size = sizeof(int) * (2 + partA.last - partA.first);
	int count = 0;

	*(int*)out = partA.first; out += sizeof(int);
	*(int*)out = partA.last; out += sizeof(int);
	for(int i = partA.first; i < partA.last; i++) {
		int idx = i - partA.first;
		*(int*)out = partA.vecs[idx].size();
		out += sizeof(int);
		for(auto it = partA.vecs[idx].begin(); it != partA.vecs[idx].end(); it++) {
			*(Point*)out = *it;
			out += sizeof(Point);
			count++;
		}
	}

	size += count * sizeof(Point);

	return size;
}

int partCToMessage(const PartC &c, char *out) {
	PartA a(c.firstColumn, c.lastColumn);
	a.vecs = vector<vector<Point>>(c.columns.size());
	for(int i = 0; i < c.columns.size(); i++)
		a.vecs[i] = vector<Point> { c.columns[i].begin(), c.columns[i].end() };

	return partAToMessage(a, out);
}

PartA messageToPartA(char *out) {
	PartA partA;
	partA.first = *(int*)out; out += sizeof(int);
	partA.last = *(int*)out; out += sizeof(int);
	partA.vecs.resize(partA.last - partA.first);

	for(int i = partA.first; i < partA.last; i++) {
		int idx = i - partA.first;
		int size = *(int*)out; out += sizeof(int);
		while(size--) {
			partA.vecs[idx].push_back(
				*(Point*)out
			);
			out += sizeof(Point);
		}
	}

	return partA;
}

ostream& operator<<(ostream &os, const Point &p) {
	os << "(" << p.idx << ", " << p.x << ")";
	return os;
}

ostream& operator<<(ostream& os, const PartA &a) {
	os << "==== PartA ====\n";
	os << a.first << " " << a.last << endl;
	for(int i = a.first; i < a.last; i++) {
		int idx = i - a.first;
		for(auto p = a.vecs[idx].begin(); p != a.vecs[idx].end(); p++) {
			os << *p << " ";
		}
		os << "\n";
	}
	return os;
}

ostream& operator<<(ostream& os, const PartB &b) {
	os << "==== PartB ====\n";
	os << b.first << " " << b.last << "\n";
	for(int j = 0; j < b.columns[0].size(); j++) {
		for(int i = 0, col = b.first; col < b.last; col++, i++) {
			os << b.columns[i][j] << " ";
		}
		os << "\n";
	}

	return os;
}

ostream& operator<<(ostream& os, const PartC &c) {
	os << "==== PartC ====\n";
	os << c.firstColumn << " " << c.lastColumn << "\n";
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
