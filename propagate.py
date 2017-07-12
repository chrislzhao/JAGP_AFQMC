# python script that converts JAGP's Jastrow factor 
import numpy as np
import itertools
import random
from scipy import misc
import pdb
import sys
import time
from mpi4py import MPI

# number of spatial orbitals
norb = int(sys.argv[1])

# number of alpha electrons
nalpha = int(sys.argv[2])

# Hubbard on-site repulsion
U = float(sys.argv[3])

# function that reads in formic's output Jastrow factor
def read_jastrow(file_name):

  # open same-spin jastrow factor data
  infile = open(file_name, "r")

  #initialize same-spin jastrow factor
  jastrow = np.zeros((norb, norb), dtype=complex)

  line_num = 0
  #fill in the upper triangular part of jaa
  for line in infile:
    line = line.split()
    for i in range(len(line)):
      jastrow[line_num][i+line_num] = float(line[i])
    line_num += 1

  #fill in the lower triangular part of jaa
  for i in range(norb):
    for j in range(i):
      jastrow[i][j] = jastrow[j][i]

  #close file
  infile.close()

  return jastrow

# function that reads in one-electron integrals of 2D Hubbard Model
def read_oe_int(file_name):
  
  # open integral file
  infile = open(file_name, "r")

  # initialize integral matrix 
  oe_int = np.zeros((norb, norb))

  # loop over file
  for line in infile:
    line = line.split()
    i = int(line[0])
    j = int(line[1])
    integral = float(line[2])
    oe_int[i][j] = integral

  # return integral file 
  return oe_int

# function that constructs and return the overall Jastrow matrix
def build_overall_J(jaa, jab):

  #initialize the overall Jastrow matrix 
  J = np.zeros((2*norb, 2*norb), dtype=complex)

  #fill in J
  for i in range(2*norb):
    for j in range(2*norb):
      #aa part
      if (i < norb and j < norb):
        J[i][j] = jaa[i][j]
      #bb part
      elif (i >= norb and j >= norb):
        J[i][j] = jaa[i-norb][j-norb]
      #ab part
      elif (i < norb and j >= norb):
        J[i][j] = jab[i][j-norb]
      #ab part
      elif (i >= norb and j < norb):
        J[i][j] = jab[i-norb][j]
  
  return J

# function that reads in formic's pairing matrix 
def read_pmat(file_name):
  
  # open same-spin jastrow factor data
  infile = open(file_name, "r")

  pmat = np.zeros((2*norb, 2*norb), dtype=complex)

  line_num = 0
  #fill in pairing matrix
  for line in infile:
    line = line.split()
    for i in range(len(line)):
      pmat[line_num][i+norb] = float(line[i])
      pmat[i+norb][line_num] = -1.0 * float(line[i])
    line_num += 1

  #close file
  infile.close()

  return pmat

# function that generates a set of auxiliary field vectors
def x_sample(level,X):
  
  if (level == 0):
    X.append([0.0]*(2*norb))
    return

  # generate all possible combinations at this level
  index = []
  for i in range(2*norb):
    index.append(i)

  combs = itertools.combinations(index, level)
  comb_list = list(combs)
  
  # loop over all possible combinations
  for comb in comb_list:
    x = [-1.0]*(2*norb)
    y = [1.0]*(2*norb)
    for index in comb:
      x[index] =  1.0
    # flip sign
    X.append(x)
    X.append(y)

  
# function that propagate the pairing matrix with auxiliary field vector and eigenvector of J
def f_propagate(x, Q, w, F):
  
  # dimension of Q
  ndim = Q.shape[0]

  # compute Q tilda
  Q_tilda = np.zeros((ndim, ndim), dtype=complex)

  # loop over columns of Q
  for i in range(ndim):
    # compute scalar
    scalar = np.sqrt(2*w[i]) * x[i]
    # loop over rows of Q
    for p in range(ndim):
      Q_tilda[p][i] = scalar * Q[p][i]

  # new pairing matrix 
  F_new = np.zeros((2*norb, 2*norb), dtype=complex)

  # initialize the new pairing matrix as old ones
  for i in range(2*norb):
    for j in range(2*norb):
      F_new[i][j] = F[i][j]

  # propagate pairing matrix
  # loop over auxiliary filed variables
  for k in range(ndim):
    # loop over rows of F
    for r in range(ndim):
      # loop over columns of F
      for s in range(ndim):
        F_new[r][s] *= np.exp(Q_tilda[r][k] + Q_tilda[s][k])

  # convert to smaller dimension 
  #F_new_small = np.zeros((norb, norb), dtype=complex)
  #for r in range(norb):
  #  for s in range(norb):
  #    F_new_small[r][s] = F_new[r][s+norb]
  # return the new F
  return F_new

# function that computes the nth order coefficient of a Pfaffian
def Pf(P, n):
 
 # first compute the root of characteristic polynomial of P
 #roots = np.roots(np.poly(P))

 roots = np.linalg.eigvals(P)

 # find unique roots 
 unique_roots = []
 #for i in range(0, len(roots), 2):
 #  unique_roots.append(roots[i])

 for i in range(0, norb):
   unique_roots.append(roots[i])
 #for i in range(len(roots)):
 #  #same_sign = False
 #  same_magn = False
 #  for j in range(len(unique_roots)):
 #    #if ( (roots[i].real * unique[j].real) < 0.0 or (roots[i].imag * unique[j].imag) < 0.0):
 #    #  same_sign = False
 #    if ( abs(roots[i].real - unique_roots[j].real) < 1e-6 and abs(roots[i].imag - unique_roots[j].imag) < 1e-6):
 #      same_magn = True
 #  if (not same_magn):
 #    unique_roots.append(roots[i])
       
 
 # check too see if number of unique roots is equal to the number of spatial oritals
 if ( len(unique_roots) != norb):
   print "Bad number of roots %d find!" % len(unique_roots)
 
 # compute the coefficient of nth order
 # first find all possible combinations
 index = []
 for i in range(norb):
   index.append(i)
 
 comb = itertools.combinations(index, n)
 comb_list = list(comb)

 coeff = 0.0+0.0j
 # loop over all combinations
 for combs in comb_list:
   product = 1.0+0.0j
   for index in combs:
     product *= unique_roots[index]
   coeff += product
 
 coeff *= (-1.0)**(2*norb*(2*norb-1)/2)
 # return coefficient
 return coeff
  

# function that compute the overlap of two AGP
def ovlp_compute(F1, F2):
  
   # constructs large, antisymmetric matrix for F1 and F2
  lF = np.zeros((2*norb, 2*norb), dtype=complex)
  rF = np.zeros((2*norb, 2*norb), dtype=complex)
  for i in range(2*norb):
    for j in range(2*norb):
      lF[i][j] = F1[i][j]
      rF[i][j] = F2[i][j]
 
  # compute matrix passed into Pfaffian
  #P = np.dot(lF.transpose().conjugate(), rF)
  P = np.dot(rF, lF.transpose().conjugate())

  return Pf(P, nalpha)

# function that compute the overlap of two AGP
def ovlp_compute_all(F1, F2):
  
   # constructs large, antisymmetric matrix for F1 and F2
  lF = np.zeros((2*norb, 2*norb), dtype=complex)
  rF = np.zeros((2*norb, 2*norb), dtype=complex)
  for i in range(2*norb):
    for j in range(2*norb):
      lF[i][j] = F1[i][j]
      rF[i][j] = F2[i][j]
 
  # compute matrix passed into Pfaffian
  P = np.dot(rF, lF.transpose().conjugate())
  #print P

  coeff = []
  for i in range(nalpha+1):
    coeff.append(Pf(P,i))

  #print coeff
  return coeff

# fast algorithm that computes hamiltonian and overlap matrix elements between two pairing matrix
def ham_ovlp_compute_fast(F1, F2, oe_int):

  # compute Pf(1+Mt), from zero to Na degrees
  coeff = ovlp_compute_all(F1,F2)

  # compute M matrix 
  M = np.dot(F2, F1.transpose().conjugate())

  # compute (-1)^i * M^(i+1)
  list1 = []
  for i in range(nalpha):
    list1.append((-1)**i * np.linalg.matrix_power(M, i+1))

  # compute (-1)^i * M^i*rF
  list2 = []
  for i in range(nalpha):
    list2.append((-1)**i * np.dot(np.linalg.matrix_power(M, i),F2))

  # compute (-1)^i * lF * M^i
  list3 = []
  for i in range(nalpha):
    list3.append((-1)**i * np.dot(F1.transpose().conjugate(),np.linalg.matrix_power(M, i)))

  # one-body contribution
  ob_eng = 0.0+0.0j

  # two-body contribution
  tb_eng = 0.0+0.0j

  # loop over first orbital index
  for i in range(norb):
    # loop over second orbital index
    for j in range(norb):
      integral = 0.0+0.0j
      # check to see whether the one-body integral is not zero
      if (oe_int[i][j] != 0.0):
        # loop over degree in list1 from 1
        for n in range(1, nalpha+1):
          # degree in Pf
          m = nalpha - n
          # aa part
          integral += (list1[n-1][j][i]) * coeff[m]
          # bb part
          integral += (list1[n-1][j+norb][i+norb]) * coeff[m]
      ob_eng += oe_int[i][j] * integral
  
  #print ob_eng
  # two-body energy
  # loop over sites
  for i in range(norb):
    # loop over degrees in the non-Pfaffian part, starting from 1
    for n in range(1, nalpha+1):
      part1 = 0.0+0.0j
      part2 = 0.0+0.0j
      # degree in the Pfaffian part
      m = nalpha - n
      # loop over degrees in left part part1
      for l in range(1,n):
        s = n - l
        part1 += (list1[l-1][i+norb][i]*list1[s-1][i][i+norb]-list1[l-1][i+norb][i+norb]*list1[s-1][i][i]) * coeff[m]

      # loop over degrees in left part part2
      for l in range(1,n+1):
        s = n - l
        part2 += (list2[l-1][i+norb][i]*list3[s][i+norb][i]) * coeff[m]
      #print part2
      tb_eng += -1.0 * U * (part1+part2)

  #print tb_eng
  hmat = ob_eng + tb_eng

  # return matrix element
  return coeff[-1], hmat
      
# function that evaluate energy on a grid
def energy_grid(F0, exct, Q, w, oe_int):
  
  # current left and right
  current_lF = np.zeros((2*norb, 2*norb), dtype=complex)
  current_rF = np.zeros((2*norb, 2*norb), dtype=complex)

  # initialize current F
  for i in range(2*norb):
    for j in range(2*norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  # auxiliary field vectors 
  X = []

  # generate X
  #exct = [0]
  #for level in exct:
  x_sample(exct, X)

  S = np.zeros((len(X),len(X)), dtype=complex)
  H = np.zeros((len(X),len(X)), dtype=complex)

  # loop over all field operators
  for l in range(len(X)):
    for r in range(len(X)):

      # propagate F
      F1 = f_propagate(X[l], Q, w, current_lF)
      F2 = f_propagate(X[r], Q, w, current_rF)

      # compute matrix elements
      smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int)
      #print hmat

      # fill in matrix
      if (r == l):
        h = hmat.real + 0j
        s = smat.real + 0j
        H[l][r] = h
        S[l][r] = s
      else:
        H[l][r] = hmat
        H[r][l] = np.conjugate(hmat)
        S[l][r] = smat
        S[r][l] = np.conjugate(smat)


  w, v = np.linalg.eig(np.dot(np.linalg.pinv(S),H))

  print w
  lowest_eng = w[0].real
  for eigval in w:
    if (eigval.real < lowest_eng):
      lowest_eng = eigval.real
  print lowest_eng

  # return energy
  return lowest_eng

# function that evaluates energy with random, non-Metropolis sampling (Large fluctuations)
def Rand_Samp_Fast(F0, Nsamp, Q, w, oe_int):

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # store randomly sampled vectors
  X = []
  X.append([0.0]*(2*norb))

  # current left and right
  current_lF = np.zeros((2*norb, 2*norb), dtype=complex)
  current_rF = np.zeros((2*norb, 2*norb), dtype=complex)

  for i in range(2*norb):
    for j in range(2*norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  # open output file
  infile = open("test.txt", "w")

  # generate auxiliary filed vectors
  for samp in range(Nsamp):
    
    # sample from gaussian distributation
    x = []
    for i in range(2*norb):
     x.append(np.random.normal(0.0,1.0))
    X.append(x)

  nu = 0.0+0.0j
  de = 0.0+0.0j
  S = np.zeros((len(X),len(X)), dtype=complex)
  H = np.zeros((len(X),len(X)), dtype=complex)
  # loop over all auxiliary field vectors
  for l in range(len(X)):
    for r in range(l, len(X)):

      # propagate initial state
      F1 = f_propagate(X[l], Q, w, current_lF)
      F2 = f_propagate(X[r], Q, w, current_rF)

      # compute matrix elements
      smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int)

      # fill in matrix
      if (r == l):
        h = hmat.real + 0j
        s = smat.real + 0j
        H[l][r] = h
        S[l][r] = s

      else:
        H[l][r] = hmat
        H[r][l] = np.conjugate(hmat)
        S[l][r] = smat
        S[r][l] = np.conjugate(smat)

  #for i in range(len(X)):
  #  for j in range(len(X)):
  #    if (H[i][j].imag > 0.0):
  #      infile.write(" %08.3f+%08.3f " % (H[i][j].real, H[i][j].imag))
  #    else:
  #      infile.write(" %08.3f-%08.3f " % (H[i][j].real, abs(H[i][j].imag)))       
  #  infile.write("\n")

  #infile.write("\n")
  #for i in range(len(X)):
  #  for j in range(len(X)):
  #    if (S[i][j].imag > 0.0):
  #      infile.write(" %08.3f+%08.3f " % (S[i][j].real, S[i][j].imag))
  #    else:
  #      infile.write(" %08.3f-%08.3f " % (S[i][j].real, abs(S[i][j].imag)))
  #  infile.write("\n")
  #print H
  #print S
  w, v = np.linalg.eig(np.dot(np.linalg.inv(S),H))
  print w
  lowest_eng = w[0].real
  for eigval in w:
    if (eigval.real < lowest_eng):
      lowest_eng = eigval.real
  print lowest_eng
  # close output file
  infile.close()

  # return energy 
  return lowest_eng

# function that evaluates energy with random, non-Metropolis sampling (Large fluctuations)
def Parallel_Fast(F0, Nsamp, Q, w, oe_int):

  # get mpi comm
  comm = MPI.COMM_WORLD

  # get rank number 
  rank = comm.Get_rank()

  # total number of ranks 
  nprocs = comm.Get_size()

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # store randomly sampled vectors
  X = []
  X.append([0.0]*(2*norb))

  # current left and right
  current_lF = np.zeros((2*norb, 2*norb), dtype=complex)
  current_rF = np.zeros((2*norb, 2*norb), dtype=complex)

  for i in range(2*norb):
    for j in range(2*norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  # generate auxiliary filed vectors on root process
  if (rank == 0):
    for samp in range(Nsamp):
    
      # sample from gaussian distributation
      x = []
      for i in range(2*norb):
        x.append(np.random.normal(0.0,1.0))
      X.append(x)
  
  # broadcast these sampled field vectors
  X = comm.bcast(X, root=0)

  nu = 0.0+0.0j
  de = 0.0+0.0j
  S = np.zeros((len(X),len(X)), dtype=complex)
  H = np.zeros((len(X),len(X)), dtype=complex)

  H_tot = np.zeros((len(X),len(X)), dtype=complex)
  S_tot = np.zeros((len(X),len(X)), dtype=complex)

  # generate all possible upper-triangular pairs
  pairs = []
  for l in range(len(X)):
    for r in range(l, len(X)):
      pairs.append([l,r])

  # total elements that need to be computed 
  total_num_ele = len(pairs)

  # number of elements per process
  num_per_proc = total_num_ele / nprocs

  # number of elements left 
  num_left = total_num_ele % nprocs

  # starting index 
  start_ind = rank * num_per_proc

  # end index 
  end_ind = start_ind + num_per_proc
  if (rank == (nprocs-1)):
    end_ind += num_left

  #print "rank is %d, starting index is %d, end index is %d" %(rank, start_ind, end_ind)
  # loop over all pairs
  for index in range(start_ind, end_ind):

    # propagate initial state
    F1 = f_propagate(X[pairs[index][0]], Q, w, current_lF)
    F2 = f_propagate(X[pairs[index][1]], Q, w, current_rF)

    # compute matrix elements
    smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int)

    # fill in matrix
    if (pairs[index][0] == pairs[index][1]):
      h = hmat.real + 0j
      s = smat.real + 0j
      H[pairs[index][0]][pairs[index][1]] = h
      S[pairs[index][0]][pairs[index][1]] = s
    else:
      H[pairs[index][0]][pairs[index][1]] = hmat
      H[pairs[index][1]][pairs[index][0]] = np.conjugate(hmat)
      S[pairs[index][0]][pairs[index][1]] = smat
      S[pairs[index][1]][pairs[index][0]] = np.conjugate(smat)

  # reduce to root process
  H_tot = comm.reduce(H, op=MPI.SUM, root=0)
  S_tot = comm.reduce(S, op=MPI.SUM, root=0)

  # diagonalize H on root process
  lowest_eng = 0.0
  if (rank == 0):
    w, v = np.linalg.eig(np.dot(np.linalg.pinv(S_tot),H_tot))

    # print eigenvalues
    print w

    # find the lowest eigenvalues
    lowest_eng = w[0].real
    for eigval in w:
      if (eigval.real < lowest_eng):
        lowest_eng = eigval.real
 
    # print the lowest eigenvalues
    print lowest_eng

  # broadcast the lowest energy
  lowest_eng = comm.bcast(lowest_eng, root=0)

  # return energy 
  return lowest_eng

def main():
  
  #start = time.time()
  # name of formic same-spin jastrow factor
  jaa_name = "jaa.txt"

  # name of formic opposite-spin jastrow factor
  jab_name = "jab.txt"

  # get jaa and jab
  jaa = read_jastrow(jaa_name)
  jab = read_jastrow(jab_name)

  # construct J
  J = build_overall_J(jaa,jab)

  #perform eigen-decomposition of J matrix
  pho,Q = np.linalg.eig(J)

  # name of pairing matrix
  pmat_name = "pmat.txt"

  # read in pairing matrix
  F = read_pmat(pmat_name)

  # name of one-electron integral 
  oe_int_name = "oe_int.txt"

  # read in one-electron integral
  oe_int = read_oe_int(oe_int_name)

  # propagate initial AGP and compute energy
  method = sys.argv[4]
  if (method == "Grid"):
    exct   = int(sys.argv[5])
    energy = energy_grid(F, exct, Q, pho, oe_int)
  elif (method == "Fast"):
    nsamp  = int(sys.argv[5])
    energy = Rand_Samp_Fast(F, nsamp, Q, pho, oe_int)   
  elif (method == "Parallel"):
    nsamp  = int(sys.argv[5])
    energy = Parallel_Fast(F, nsamp, Q, pho, oe_int)   
  else:
    print "Unknown Method"

  # print final energy
  #print energy
  #end = time.time()
  #print "total time is %.7f" %(end-start)

if __name__ == "__main__":
  main()

