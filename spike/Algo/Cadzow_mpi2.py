#!/usr/bin/env python 
# encoding: utf-8

"""
Created by Marc-André Delsuc and Lionel Chiron on 2011-07
Copyright (c) 2010 IGBMC. All rights reserved.

Cadzow in MPI mode
complete rewrite from the code from Cyrille Bonamy for the MPI part
The original code used in the publication () is in the attic.

code compatible with  NPK version 0.4.0 

Thresholding to make Cadzow on the main relevant columns.

note that the cadzow algo is multithreaded if running over the MKL library.
So if MKL is installed, run only on instance per node, as all cores from the node will be solicited.
"""
from __future__ import print_function, division 
import sys
import numpy as np
from spike.util import mpiutil as mpiutil
from spike.util import progressbar as pg
import tables
import time

import urQRd
import Cadzow
from spike.NPKData import NPKData, copyaxes
from spike.FTICR import FTICRData
from spike.File.HDF5File import HDF5File
from spike.NPKConfigParser import NPKConfigParser

#import matplotlib.pylab as plt

debug = False

def Set_Table_Param():
#    if debug>0: return
    tables.parameters.CHUNK_CACHE_PREEMPT = 1
    tables.parameters.CHUNK_CACHE_SIZE = 100*1024*1024
    tables.parameters.METADATA_CACHE_SIZE  = 100*1024*1024
    tables.parameters.NODE_CACHE_SLOTS = 100*1024*1024
    #tables.parameters.EXPECTED_ROWS_EARRAY = 100
    #tables.parameters.EXPECTED_ROWS_TABLE =100
    #tables.parameters.MAX_THREADS = 8
    #tables.parameters.PYTABLES_SYS_ATTRS = False

def selectcol(data, limitpts, nbrows=200):
    """
    returns a list of index of the limitpts largest columns of the 2D 'data'
    
    first averaging on nbrows rows
    
    return index list
    """
    if debug: print("averaging on ",nbrows," rows ")
    roughft2 = data.row(0)
    if roughft2.axis1.itype == 1:
        roughft2.modulus()
    else:
        roughft2.abs()
    for i in range( min(nbrows, data.size1) ):
        rr = data.row(i)
        if rr.axis1.itype == 1:
            rr.modulus()
        else:
            rr.abs()
        roughft2.add(rr) # averaging for denoising and zero filliing
    roughft2.mult(1./nbrows)# resizing to original height for comparison
    
    n = roughft2.size1 * 0.1    # remove first 10%      KLUDGE !!!!!
    roughft2.buffer[0:n] = 0.0
    index = find_thres(roughft2, limitpts=limitpts)# extract peaks or zones
    if debug:
        roughft2.display()
        disp = NPKData(buffer=np.zeros(roughft2.size1))
        disp.buffer[index] = roughft2.buffer[index]
        disp.display(show=True)
    return index

def find_thres(b, limitpts):
    """
    returns a list of index of the limitpts largest points in the 1D data 'b' 
    """

    thresh=max(b.buffer)+1.0# initializing the moving threshold
    nbpts=0#initializing number of points that will satisfy the threshold criterium.

    count=0
    inter=b.buffer.copy()
    while abs(nbpts-limitpts)/float(limitpts)> 0.1 :# threshold sweep
        #searchingt the threshold by dichotomy
        if debug: print("thresh : ",thresh)
        nbpts=(inter>thresh).sum()# give nb of points above threshold
        inter[inter<thresh]=0# put the point under threshold to 0
        if debug: print("nbpts", nbpts,"count ",count)
        count+=1
        if nbpts < limitpts :# if under the maximum number of columns allowed we keep the threshold, the positions and values..
            c=inter
            threshold=thresh
            if debug: print("threshold ",threshold)
            thresh/=2.0# trying lower threshold
            ind = np.where(c>0)[0]       # np.where returns a tuple have to select first element!
  
        else : 
            if debug: print("treshold min = ", thresh) 
            thresh=(threshold+thresh)/2.0  
            if debug: print("nouveau threshold ", thresh)
        inter = np.copy(b.buffer)
        if debug: print("au dessus thresh ", (inter>thresh).sum())
        if debug: print("=0 ", (inter==0).sum())
    
    return ind



def load_input(name):
    """load input file and returns it, in read-only mode"""
    if debug>0: print("reading", name)
    hf = HDF5File(name, "r")    # reading the data 
    d0 = hf.load()
    return d0

def iterarg(xindex, dinp, n_of_line, n_of_iter, orda ):
    "an iterator used by the MPI set-up"
    for i in xindex:
        c0 = dinp.col(i)
        if debug:
            print(c0.buffer, n_of_line, n_of_iter, orda)
        yield (c0.buffer, n_of_line, n_of_iter, orda)

def cadz(args):
    "utility function"
    if debug:
        print(args)
    return Cadzow.cadzow(*args)
def rqr(args):
    "utility function"   
    #print "type(args) ", type(args)
    #print "args ", args
    if debug:
        print(args)
    argu = (args[0],args[1],args[3],)
    #print "passed arguments ",argu
    return urQRd.urQRd(*argu)
    
def main():
    """does the whole job,
    if we are running in MPI, this is only called by job #0
    all other jobs are running mpi.subordinate()
    """
    argv = sys.argv
    if len(argv) != 2:
        print("""
syntax is :
(mpirun -np N) python  program   configfile.mscf
""")
        sys.exit(1)

    # get parameters
    configfile = argv[1]
    cp = NPKConfigParser()
    cp.readfp(open(configfile))
    infile = cp.getword( "Cadzow", "namein")
    print("infile", infile)
    outfile = cp.getword( "Cadzow", "nameout")
    print("outfile", outfile)
    
    algo = cp.getword("Cadzow", "algorithm" )
    print("algorithm", algo)
    n_of_line = cp.getint("Cadzow", "n_of_lines", 70)
    print("n_of_line", n_of_line)
    n_of_iter = cp.getint("Cadzow", "n_of_iters", 1)
    print("n_of_iter", n_of_iter)
    orda = cp.getint("Cadzow", "order", 500)
    print("order",orda)
    n_of_column = cp.getint("Cadzow", "n_of_column", 100)
    print("n_of_column", n_of_column)
    progress = cp.getboolean("Cadzow", "progress", True)
    
    d0 = load_input(infile)
    d0.check2D()    # raise error if not a 2D
    Set_Table_Param()

    hfar =  HDF5File(outfile, "w", debug=0)  # OUTFILE 
    d1 = FTICRData( dim=2 )   # create dummy 2D
    copyaxes(d0, d1)        # copy axes from d0 to d1
    group = 'resol1'
    hfar.create_from_template(d1, group)
    
    # prepare index and method
    if n_of_column == 0:
        indexes = range(d0.size2)    # process all
    else:
        indexes = selectcol(d0, n_of_column) # selections

    if algo == "Cadzow":
        meth = cadz
    elif algo == "rQRd":# 
        meth = rqr
    else:
        raise("wrong algo")

    # then loop
    t0 = time.time()
    if progress:
        widgets = ['Processing %s: '%(algo), pg.Percentage(), ' ', pg.Bar(marker='-',left='[',right=']'), pg.ETA()]
        pbar= pg.ProgressBar(widgets=widgets, maxval=len(indexes)) #, fd=sys.stdout)

    d1D = d0.col(0)    # template
    xarg = iterarg(indexes, d0, n_of_line, n_of_iter, orda )
    if mpiutil.MPI_size>1:      # means we are running under MPI !
        mpiutil.mprint('MPI Main job  - starting subordinate jobs - ')
        res = mpiutil.enum_imap(meth, xarg)    # apply it
        for i,p in res:       # and get results
            d1D.buffer = p
            d1.set_col(indexes[i], d1D)
            if progress: pbar.update(i+1)
    else:
        import itertools
        res = itertools.imap(meth, xarg)    # apply it
        for i,p in enumerate(res):       # and get results
            d1D.buffer = p
            d1.set_col(indexes[i], d1D)
            if progress: pbar.update(i+1)
    print("Processing time : ", time.time()-t0)

# et on lance
if __name__ == '__main__':
    #unittest.main()
    if mpiutil.MPI_size < 2:            # this is single processor
        print("Running in single processor mode")
        main()
    else:                       # this is a MPI run
        print("Running in MPI mode")
        if mpiutil.MPI_rank == 0:   # main proc
            main()
        else:               # subordinate proc
            mpiutil.subordinate()


