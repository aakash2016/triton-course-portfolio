from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """

        # note: each rank has their own src and dest
        rank = self.Get_rank()
        num_ranks = self.Get_size()

        tmp = np.copy(src_array)  # we'll reduce results in tmp array

        if rank == 0:
            # perform reduction
            for i in range(1, num_ranks):
                recv_array = np.empty_like(src_array)
                self.comm.Recv(recv_array, source=i)  # master receives data from other ranks
                if op == MPI.MIN:
                    tmp = np.minimum(tmp, recv_array)
                elif op == MPI.MAX:
                    tmp = np.maximum(tmp, recv_array)
                else:
                    # default is sum
                    tmp += recv_array
            # send the reduced result to all ranks
            for i in range(1, num_ranks):
                self.comm.Send(tmp, dest=i)
        else:
            # In MPI, a process cannot receive data unless another process is explicitly sending it.
            # self.comm.Send(src_array, dest=0) # non-root process sends its data to process 0
            # self.comm.Recv(tmp, source=0) # receive the reduced result from rank 0
            self.comm.Sendrecv(src_array, dest=0, recvbuf=tmp, source=0)

        # broadcast if rank is 0
        # self.comm.Bcast(tmp, root=0)
        np.copyto(dest_array, tmp) # dest array of all ranks, copy within rank

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """

        rank = self.Get_rank()
        num_ranks = self.Get_size()

        bs = len(src_array) // num_ranks
        recv_array = np.empty_like(src_array[:bs]) # buffer for receiving data

        # current rank
        dest_array[rank * bs: (rank + 1) * bs] = src_array[rank * bs: (rank + 1) * bs]  # no need for communication

        for i in range(num_ranks):
            if i != rank:
                send_array = src_array[i * bs: (i + 1) * bs]
                self.comm.Sendrecv(sendbuf=send_array, dest=i, recvbuf=recv_array, source=i)
                dest_array[i * bs: (i+1) * bs] = recv_array
