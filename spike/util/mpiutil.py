#!/usr/bin/env python 
# encoding: utf-8

"""
mpiutil.py

Utilities for using MPI

defines
MPI_size : number of process in the pool
MPI_rank : rank of the current process

typical use is presented in test_server_worker()

in main do :
    if MPI_rank == 0:
        compute()   # your code,     uses mpiutil.enum_imap(myfunction, xarg)
    else:
        mpiutil.subordinate()     # subordinate is a generic function used to compute myfunction( arg )

create a function to do the processing
def myfunction(arg)
    return whatever

and in compute() :
    # create an iterator over arguments
    M = 1000            # size of the pb
    result = M*[0.0]    # create room for storing results (if needed)
    xarg = xrange(M)    # create arguments
    res = mpiutil.enum_imap(myfunction, xarg)    # enum_imap applies myfunction to the iterator xarg
    for i,r in res:         # collect results
         result[i] = r      # and store them (order is not garantied)

Warning, (i,r) will not arrive in order

Will import even in MPI is not installed, however subordinate() will not work
The situation is reported as MPI_size = 1

Created by Marc-Andre' on 2012-04-19.
extended 2014-03-25

Copyright (c) 2012-2014 IGBMC. All rights reserved.
"""

from __future__ import print_function

"""
Recv is for numpy buffers
recv is for picklable python objects
"""



try:
    from mpi4py import MPI
except ImportError:
    print("no MPI available")
    MPI_rank = 0
    MPI_size = 1
else:
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.size

# TAG for MPI
WORK_TAG = 1    # tag work message
DIE_TAG = 2     # tag die message - used for closing the workers
INIT_TAG = 3    # tag init message - used for initializing the workers
TAG_OFFSET = 100    # offset used to code the 
DEBUG = False

def mprint(st):
    "print with prefixing with the process rank"
    print("MPI process %d : %s"%(MPI_rank,st))
def mprint_debug(st):
    "as mprint() but only if DEBUG is True"
    if DEBUG:
        mprint(st)
def enum_imap(function, iterator):
    """
    applies function() to each element of iterator
    creates an iterator imap(function, iterator) that returns (i, function(iterator_i))
    elements will be returned in an synchronous manner with no garanty on the order.
        
    similar to enumerate( pool.imap_async(function, iterator) )
    except for the order
    """
    if MPI_size == 1:
        raise Exception("program should be started with mpirun")
    mprint_debug('server')
    status = MPI.Status()
    #initialize
    initialize(function)
    todo = 0    # counts computation sent
    done = 0    # counts results returned

    # main loop
    for inp in iterator:
        result = False      # assume unproductive loop
        # first receive from subordinates ! (at least ok from initialize)
        data = MPI_comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()    # who was it ?
        if status.Get_tag() != INIT_TAG:
            result = True   # was real data
            ind = status.Get_tag()-TAG_OFFSET   # tag codes for index
            mprint_debug('from proc %d, I get %s for index %d'%(worker, str(data), ind))
            done += 1
            yield (ind, data)
        mprint_debug( 'sending new stuff to %d'% worker)
        MPI_comm.send(inp, dest=worker, tag=todo+TAG_OFFSET )
        todo += 1
    # as loops starts with receiving, we need to clean-up by receiving all waiting jobs
    mprint_debug( 'clean-up')
    while done != todo:
        mprint_debug( "done %d over %d to do"%( done, todo))
        result = False      # assume unproductive loop
        data = MPI_comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()
        if status.Get_tag() != INIT_TAG:
            result = True   # was real data
            ind = status.Get_tag()-TAG_OFFSET   # tag codes for index
            mprint_debug('from proc %d, I get %s for index %d'%(worker, str(data), ind))
            done += 1
            yield (ind, data)
    return
def subordinate():
    "to be called in conjunction with enum_imap"
    if MPI_size == 1:
        raise Exception("program should be started with mpirun")
    mprint_debug('worker starting')
    func = _nill # this is the working function, loaded at INIT
    status = MPI.Status()
    while True:
        # mprint_debug('waiting')
        data = MPI_comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == DIE_TAG:
            mprint_debug('received DIE_TAG')
            break
        if status.Get_tag() == INIT_TAG:
            mprint_debug('received INIT_TAG')
            func = data
            MPI_comm.send(None, dest=0, tag=INIT_TAG)
            MPI_comm.Barrier()
            mprint_debug("synchronized")
        else:
            res = func(data)
            MPI_comm.send(res, dest=0, tag=status.Get_tag())
    mprint_debug('finished')

def _nill(a):
    "empty function"
    return a
def initialize(func=_nill):
    """
    bring all processes to a starting barrier
    to be called by main process (rank == 0)
    """
    if MPI_size == 1:
        raise Exception("program should be started with mpirun")
    status = MPI.Status()
    #initialize
    for d in range(1,MPI_size):
        MPI_comm.send(func, dest=d, tag=INIT_TAG)
    mprint_debug("Waiting for MPI synchronisation")
    MPI_comm.Barrier()
    mprint_debug("synchronized")

def shutdown():
    """
    closes all MPI processes all processes
    to be called by main process (rank == 0)
    """
    if MPI_size == 1:
        raise Exception("program should be started with mpirun")
    mprint_debug("Closing all sub-processes")
    data = []
    for d in range(1,MPI_size):
        MPI_comm.send(data, dest=d, tag=DIE_TAG)
    mprint_debug("MPI closed")


###### Example of use - call the program itself with mpirun -np X ########
def funct_array_simple1(a):
    """
    used for testing
    take an array and compute 2*sqr(a) +1
    plus a random nap to simulate a random processing time
    """
    import time
    import random
    import numpy
    time.sleep(1+random.random())
    return 2.0*numpy.sqrt(a)+1  # numpy operation
def funct_array_simple2(a):
    """
    used for testing
    take an array and compute 3*sqr(a+1)
    plus a random nap to simulate a random processing time
    """
    import time
    import random
    import numpy
    time.sleep(random.random())
    return 3.0*numpy.sqrt(a)  # numpy operation
def test_compute_simple(N, M):
    """run funct_array_simple over enum_imap"""
    import numpy
    import time
    import itertools as it
    # create dummy data
    data = numpy.arange(N, dtype=numpy.float64)
    print("""
The program should print the value %f %d times
then the value %f %d times
with a process speed-up proportionnal to the number of (processes-1).
"""%(funct_array_simple1(data).sum(), M, funct_array_simple2(data).sum(), 2*M))
    # prepare an iterator over which enum_imap will be applied
    xarg = it.repeat(data, M)
    t0 = time.time()
    res = enum_imap(funct_array_simple1, xarg)    # apply it - 
    for i,r in res:       # and get results
        print(i,r.sum())
    xarg2 = it.repeat(data, 2*M)
    res = enum_imap(funct_array_simple2, xarg2)    # apply it
    for i,r in res:       # and get results
        print(i,r.sum())
    cpu = time.time()-t0
    print("process time : %.2f sec"%(cpu))
    return cpu

def test_server_worker():
    if MPI_rank == 0:
        import time
        t0 = time.time()
        # values used for the test
        N = 100000  # array size
        M = 20 # number of operation to apply
        cpu = test_compute_simple(N, M)
        elaps = time.time()-t0
        print('elapsed %.2f sec  MPI starting overhead is %.2f sec'%(elaps, elaps-cpu))
        spd = (1.5*M + M)/cpu    # funct_array_simple1 is ~ 1.5sec x M  and funct_array_simple2 ~0.5sec x 2M
        print("speed up is x %.2f  for a theoretical maximum speedup of x %d"%(spd, MPI_size-1))
        shutdown()
    else:
        subordinate()

if __name__ == '__main__':
    # this should be called with mpirun to check the MPI behavior.
    # as the tested actions are mostly empty and just sleep, it can safely be tested with a large number of processes
    # even on a small machine.
    if MPI_size == 1:       # means no MPI
        print(__doc__)
    else:
        test_server_worker()
