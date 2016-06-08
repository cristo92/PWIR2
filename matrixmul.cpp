#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <list>
#include <iostream>
#include <iomanip>

#include "matrixmul.h"
#include "densematgen.h"

using namespace std;

#define FOR(i, row, M) for(int i = 0, row = M.first; row < M.last; i++, row++)

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

	int c = repl_fact;
	int p = num_processes;
	int s = p / c;
	int q;
	if(use_inner) q = (s + c - 1) / c;
	else q = p / c;

	char* messages;

	InitMessage my_init_msg;
	PartA A;
	PartB B;
	PartA C;
	InitMessage *init_msgs = (InitMessage*)malloc(p * sizeof(InitMessage));
	vector<vector<Point>> inputA;

	int bytes;
	char *sendbuf, *recvbuf;
	int *sendcount, *sendcount_tmp, *displs, *displs_tmp, *recvcount;

	int m, n, nnz, max_nnz;

	//read
	if(mpi_rank == 0) {
		FILE *file = fopen(sparse_filename, "r");
		fscanf(file, "%d%d%d%d", &m, &n, &nnz, &max_nnz);

		vector<double> line1(nnz);
		vector<int> line2(m + 1);
		vector<int> line3(nnz);
		inputA.resize(m);

		for(int i = 0; i < nnz; i++) fscanf(file, "%lf", &line1[i]);
		for(int i = 0; i < m + 1; i++) fscanf(file, "%d", &line2[i]);
		for(int i = 0; i < nnz; i++) fscanf(file, "%d", &line3[i]);

		for(int idx = 0, i = 0; i < m; i++) {
			for(; idx < line2[i + 1]; idx++) {
				inputA[i].push_back(Point(line3[idx], line1[idx]));
			}
		}
	}

	comm_start =  MPI_Wtime();

	vector<PartA> partA(s);
	int *ids = (int*)malloc(c * sizeof(int));
	memset(ids, 0, c * sizeof(int));

	if(mpi_rank == 0) {	
		// set A columns and rows
		if(use_inner) {
			int length = (n + s - 1) / s;

			ids[0] = 0;
			for(int i = 1; i < c; i++) {
				ids[i] = (ids[i - 1] - q + s) % s;
			}

			for(int i = 0, row = 0; i < s; i++, row = min(row + length, n)) {
				for(int j = 0; j < c; j++) {
					int idx = (ids[j] + i) % s;
					int idy = j;
					int idp = idx * c + idy;

					init_msgs[idp].firstBColumn = length * idx;
					init_msgs[idp].lastBColumn = min(length * (idx + 1), n);

					init_msgs[idp].cx = idx;
					init_msgs[idp].cy = idy;

					init_msgs[idp].n = m;
					init_msgs[idp].nnz = nnz;
					init_msgs[idp].max_nnz = max_nnz;
				}
				partA[i].first = row;
				partA[i].last = min(row + length, n);
				partA[i].vecs.resize(partA[i].last - partA[i].first);
				for(int row_idx = row; row_idx < partA[i].last; row_idx++) {
					partA[i].vecs[row_idx - row] = inputA[row_idx];
				}
			}
		}
		else {
			int stepA = (n + c - 1) / c, stepB = (n + p - 1) / p;
			for(int i = 0; i < p; i++) {
				init_msgs[i].n = n;
				init_msgs[i].nnz = nnz;
				init_msgs[i].max_nnz = max_nnz;
				init_msgs[i].num_rounds = s;
				init_msgs[i].cx = i / c;
				init_msgs[i].cy = i % c;

				init_msgs[i].firstBColumn = min(i * stepB, n);
				init_msgs[i].lastBColumn = min((i + 1) * stepB, n);

				init_msgs[i].firstARow = 0;
				init_msgs[i].lastARow = n;

				init_msgs[i].firstAColumn = min((i / c) * stepA, n);
				init_msgs[i].lastAColumn = min(n,((i / c) + 1) * stepA);
			}

			for(int i = 0; i < s; i++) {
				partA[i].first = 0;
				partA[i].last = n;
				partA[i].vecs.resize(n);
			}

			for(int i = 0, step = (n + s - 1) / s; i < n; i++) {
				int j = 0;
				for(auto &p: inputA[i]) {
					while(p.idx >= (j + 1) * step) j++;
					partA[j].vecs[i].push_back(p);
				}
			}
		}
		my_init_msg = init_msgs[0];
		A = partA[0];
	}

	MPI_Scatter(init_msgs, sizeof(InitMessage), MPI_BYTE,
				&my_init_msg, sizeof(InitMessage), MPI_BYTE,
				0, MPI_COMM_WORLD);

	if(mpi_rank == DEBUG_RANK) cerr << my_init_msg << "\n";

	MPI_Comm group_comm;
	MPI_Comm_split(MPI_COMM_WORLD, my_init_msg.cx, my_init_msg.cy, &group_comm);
	if(mpi_rank == DEBUG_RANK) {
		int t;
		MPI_Comm_size(group_comm, &t);
		cerr << "Comm size: " << t << "\n";
	} 

	n = my_init_msg.n;
	nnz = my_init_msg.nnz;
	bytes = nnz * sizeof(Point) + s * n * sizeof(int) + s * 2 * sizeof(int);
	if(mpi_rank == 0) {
		sendbuf = (char*)malloc(bytes);
		sendcount = (int*)malloc(p * sizeof(int));
		sendcount_tmp = (int*)malloc(p * sizeof(int));
		displs = (int*)malloc((p + 1) * sizeof(int));
		displs_tmp = (int*)malloc((p + 1) * sizeof(int));
	}
	recvbuf = (char*)malloc(bytes);

	if(mpi_rank == 0) {
		displs_tmp[0] = 0;
		for(int i = 0; i < s; i++) {
			int bytes = analysePartA(partA[i]);

			messages = sendbuf + displs_tmp[i];
			partAToMessage(partA[i], messages);
			sendcount_tmp[i] = bytes;
			displs_tmp[i + 1] = displs_tmp[i] + bytes;
		}

		for(int i = 0; i < p; i++) {
			int idx = i / c;
			int idy = i % c;

			int partAIdx = (idx + ids[idy]) % s;
			if(mpi_rank == DEBUG_RANK) cerr << partAIdx << "\n";
			displs[i] = displs_tmp[partAIdx];
			sendcount[i] = sendcount_tmp[partAIdx];
		}
	}

	if(mpi_rank == DEBUG_RANK) cerr << "I'm about to scatter\n";

	MPI_Scatterv(sendbuf, sendcount, displs, MPI_BYTE, recvbuf, bytes,
		MPI_BYTE, 0, MPI_COMM_WORLD);

	if(mpi_rank == DEBUG_RANK) cerr << "I'm about to convert message to A\n";
	if(mpi_rank > 0) A = messageToPartA(recvbuf);
	if(mpi_rank == DEBUG_RANK) {
		cerr << "I'm " << mpi_rank << "(" << my_init_msg.cx << ", " << my_init_msg.cy  << "), and got:\n";
		cerr << A; 
	}

	if(mpi_rank == 0) {
		free(sendbuf);
		free(sendcount);
		free(displs);
	}
	free(recvbuf);
	free(ids);
	
	MPI_Request recv_request, send_request;
	MPI_Status recv_status, send_status;

	// GENERATE MATRIX B
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
	C = PartA(B.first, B.last);
	if(use_inner) C.vecs.resize(C.last - C.first);
	else {
		C.vecs.resize(C.last - C.first, vector<Point>(n, Point(0, 0)));
		FOR(i, col, C) {
			for(int j = 0; j < n; j++) C.vecs[i][j].idx = j;
		}
	}

	if(mpi_rank == DEBUG_RANK) cerr << C;

	MPI_Barrier(MPI_COMM_WORLD);
	comm_end = MPI_Wtime();

	// ***************************************
	// ************* COMPUTATION *************
	// ***************************************
	if(mpi_rank == DEBUG_RANK) cerr << "Start computation\n";
	
	if(use_inner) {
		bytes = ((n + s - 1) / s) * ((n + s - 1) / s) * q * sizeof(Point) + (2 + ((n + s - 1) / s)) * sizeof(int);

		sendbuf = (char*)malloc(bytes);
		recvbuf = (char*)malloc(bytes * c);
		recvcount = (int*)malloc(c * sizeof(int));
		displs = (int*)malloc(c * sizeof(int));
		for(int i = 0; i < c; i++) {
			recvcount[i] = bytes;
			displs[i] = i * bytes;
		}
	}

	char *send_msg, *recv_msg;
	int max_bytes = nnz * sizeof(Point) + n * sizeof(int) + 2 * sizeof(int);
	send_msg = (char*)malloc(max_bytes);
	recv_msg = (char*)malloc(max_bytes);

	comp_start = MPI_Wtime();

	while(exponent--) {
		for(int t = q - 1; t >= 0; t--) {
			if(mpi_rank == DEBUG_RANK) cerr << "STEP " << q - t << "\n";
			int idx = my_init_msg.cx;
			int idy = my_init_msg.cy;

			int next_rank, prev_rank;

			if(use_inner) {
				if(t) prev_rank = c * ((idx + s - 1) % s) + idy;
				else prev_rank = c * ((idx + q - 1) % s) + idy;

				if(t) next_rank = c * ((idx + 1) % s) + idy;
				else next_rank = c *((idx - q + 1 + s) % s) + idy;
			}
			else {
				prev_rank = (mpi_rank - c + p) % p;
				next_rank = (mpi_rank + c) % p;
			}

			// TODO - don't send and receive last message
			int bytes = partAToMessage(A, send_msg);
			MPI_Isend(
				send_msg,
				bytes,
				MPI_BYTE,
				prev_rank,
				MPI_SEND_A,
				MPI_COMM_WORLD,
				&send_request
			);

			if(mpi_rank == DEBUG_RANK)
				cerr << A;

			FOR(i, row, A) {
				FOR(j, col, B) {
					double el = 0;
					for(auto it = A.vecs[i].begin(); it != A.vecs[i].end(); it++) {
						el += it->x * B.columns[j][it->idx];
					}
					if(use_inner) C.vecs[j].push_back(Point(row, el));
					else C.vecs[j][row].x += el;
				}
			}

			MPI_Irecv(
				recv_msg,
				max_bytes,
				MPI_BYTE,
				next_rank,
				MPI_SEND_A,
				MPI_COMM_WORLD,
				&recv_request
			);

			
			MPI_Wait(&send_request, &send_status);
			MPI_Wait(&recv_request, &recv_status);
			

			A = messageToPartA(recv_msg);
		}

		if(!exponent) break;
		if(use_inner) {
			// Broadcast C
			int sendcount = partAToMessage(C, sendbuf);
			MPI_Allgatherv(sendbuf, sendcount, MPI_BYTE, recvbuf, recvcount, displs,
				MPI_BYTE, group_comm);

			for(int i = 0; i < c; i++) {
				PartA tempC = messageToPartA(recvbuf + displs[i]);
				FOR(j, row, tempC) {
					for(auto &p: tempC.vecs[j]) {
						if(mpi_rank == DEBUG_RANK) cerr << p << "\n";
						B.columns[j][p.idx] = p.x;
					}
				}
			}

			if(mpi_rank == DEBUG_RANK) cerr << B;

			C = PartA(C.first, C.last);
			C.vecs.resize(C.last - C.first);
		}
		else {
			FOR(i, col, C)
				for(auto &p: C.vecs[i])
					B.columns[i][p.idx] = p.x;
			C = PartA(C.first, C.last);
			C.vecs.resize(C.last - C.first, vector<Point>(n, Point(0,0)));
			FOR(i, col, C) {
				for(int j = 0; j < n; j++)
					C.vecs[i][j].idx = j;
			}
		}
	}

	if(mpi_rank == DEBUG_RANK)
				cerr << A;

	MPI_Barrier(MPI_COMM_WORLD);
	comp_end = MPI_Wtime();

	if(use_inner) {
		free(sendbuf);
		free(recvbuf);
		free(recvcount);
		free(displs);
	}
	MPI_Comm_free(&group_comm);

	free(recv_msg);
	free(send_msg);

	// DEBUG BEGIN
	if(mpi_rank == DEBUG_RANK) cerr << "PartC\n" << C;
	// DEBUG END
	
	if (show_results) 
	{
		int rows_per_proc = (my_init_msg.n + s - 1) / s;
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

			partAToMessage(C, recvbuf);
		}
		else {
			sendbuf = (char*)malloc(bytes);
			sendcount = partAToMessage(C, sendbuf);
		}
		if(mpi_rank == DEBUG_RANK) cerr << "I'm going to gather\n";
		MPI_Gatherv(sendbuf, sendcount, MPI_BYTE, recvbuf, recvcount,
			displs, MPI_BYTE, 0, MPI_COMM_WORLD);

		if(mpi_rank == 0) {
			for(int i = 0; i < min(p, s * c); i++) {
				buf = recvbuf + i * bytes;
				PartA tempA = messageToPartA(buf);

				FOR(i, row, tempA) {
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
		int ile = 0, sum;
		FOR(i, col, C) {
			for(auto &p: C.vecs[i])
				if(p.x >= ge_element)
					ile++;
		}

		MPI_Reduce(&ile, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

		if(mpi_rank == 0) cout << sum << "\n";
	}

	MPI_Finalize();
	return 0;
}

int analysePartA(const PartA &partA) {
	int size = sizeof(int) * (2 + partA.last - partA.first);
	int count = 0;

	FOR(i, row, partA) {
		count += partA.vecs[i].size();
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
	if(b.last == b.first) return os;
	for(unsigned int j = 0; j < b.columns[0].size(); j++) {
		for(int i = 0, col = b.first; col < b.last; col++, i++) {
			os << b.columns[i][j] << " ";
		}
		os << "\n";
	}

	return os;
}

ostream& operator<<(ostream& os, const InitMessage &a) {
	os << a.n << " " << a.nnz << " " << a.max_nnz << "\n";
	os << a.firstBColumn << " " << a.lastBColumn << "\n";
	os << a.firstARow << " " << a.lastARow << "\n";

	return os;
}
