FILES= model/func_impl.py \
	   data/data_parallel_preprocess.py \
	   mpi_wrapper/comm.py \
	   matmul_triton.ipynb \
	   discussion2-1.txt


handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)

clean:
	rm -f *~ handin.tar
