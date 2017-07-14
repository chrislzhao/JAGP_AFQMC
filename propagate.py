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

  # return the new F
  return F_new

# function that computes the nth order coefficient of a Pfaffian
def Pf(P, n, Hermitian):
 
 roots = []
 if (Hermitian):
   roots = np.linalg.eigvalsh(P)
 else: 
   roots = np.linalg.eigvals(P)

 # find unique roots 
 unique_roots = []

 if (not Hermitian):
   for i in range(0, norb):
     unique_roots.append(roots[i])
 else:
   for i in range(0, 2*norb, 2):
     unique_roots.append(roots[i])
 
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

def hermitian_mat_pow(M, i):
  
  # perform eigen-decomposition 
  w, v = np.linalg.eigh(M)

  # raise each eigenvalues to the correct order
  for j in range(len(w)):
    w[j] = w[j]**i

  # return matrix
  return np.dot(v, np.dot(np.diag(w), v.transpose().conjugate()))
  
def norm_compute(F1):
  
   # constructs large, antisymmetric matrix for F1 and F2
  lF = np.zeros((2*norb, 2*norb), dtype=complex)
  for i in range(2*norb):
    for j in range(2*norb):
      lF[i][j] = F1[i][j]
 
  # compute matrix passed into Pfaffian
  P = np.dot(lF, lF.transpose().conjugate())

  coeff = Pf(P, nalpha, True)
  return np.sqrt(coeff)

# function that compute the overlap of two AGP
def ovlp_compute_all(F1, F2, Hermitian):
  
   # constructs large, antisymmetric matrix for F1 and F2
  lF = np.zeros((2*norb, 2*norb), dtype=complex)
  rF = np.zeros((2*norb, 2*norb), dtype=complex)
  for i in range(2*norb):
    for j in range(2*norb):
      lF[i][j] = F1[i][j]
      rF[i][j] = F2[i][j]
 
  # compute matrix passed into Pfaffian
  P = np.dot(rF, lF.transpose().conjugate())

  coeff = []
  for i in range(nalpha+1):
    coeff.append(Pf(P,i,Hermitian))

  return coeff

# fast algorithm that computes hamiltonian and overlap matrix elements between two pairing matrix
def ham_ovlp_compute_fast(F1, F2, oe_int, Hermitian):

  # compute Pf(1+Mt), from zero to Na degrees
  coeff = ovlp_compute_all(F1,F2,Hermitian)

  # compute M matrix 
  M = np.dot(F2, F1.transpose().conjugate())

  # compute (-1)^i * M^(i+1)
  list1 = []
  for i in range(nalpha):
    if (not Hermitian):
      list1.append((-1)**i * np.linalg.matrix_power(M, i+1))
    else:
      list1.append((-1)**i * hermitian_mat_pow(M, i+1))          

  # compute (-1)^i * M^i*rF
  list2 = []
  for i in range(nalpha):
    if (not Hermitian):
      list2.append((-1)**i * np.dot(np.linalg.matrix_power(M, i),F2))
    else:
      list2.append((-1)**i * np.dot(hermitian_mat_pow(M, i),F2))     

  # compute (-1)^i * lF * M^i
  list3 = []
  for i in range(nalpha):
    if (not Hermitian):
      list3.append((-1)**i * np.dot(F1.transpose().conjugate(),np.linalg.matrix_power(M, i)))
    else:
      list3.append((-1)**i * np.dot(F1.transpose().conjugate(),hermitian_mat_pow(M, i)))

  # one-body contribution
  ob_eng = 0.0+0.0j

  # two-body contribution
  tb_eng = 0.0+0.0j

  ob_mat = np.zeros((2*norb, 2*norb), dtype=complex)
  # loop over first orbital index
  for i in range(norb):
    # loop over second orbital index
    for j in range(i, norb):
      integral = 0.0+0.0j
      # check to see whether the one-body integral is not zero
      if (oe_int[i][j] != 0.0):
        # loop over degree in list1 from 1
        for n in range(1, nalpha+1):
          # degree in Pf
          m = nalpha - n
          # aa part
          integral += ((list1[n-1][j][i]) + (list1[n-1][i][j])) * coeff[m]
          # bb part
          integral += ((list1[n-1][j+norb][i+norb]) + (list1[n-1][i+norb][j+norb])) * coeff[m]
      ob_eng += oe_int[i][j] * integral
  
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

  hmat = ob_eng + tb_eng

  # return matrix element
  return coeff[-1], hmat
 
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
      smat = 0.0+0.0j
      hmat = 0.0+0.0j
      if (r == l):
        smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int, True)
      else:
        smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int, False)

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

  # number of bad points
  num_bad = 0
  bad_index = []
  good_index = []
  for i in range(len(X)):
    if ( (H[i][i]/S[i][i]).real < (H[0][0]/S[0][0]).real ):
      num_bad += 1
      bad_index.append(i)
    else:
      good_index.append(i)

  H_eff = np.zeros((len(X)-num_bad, len(X)-num_bad), dtype=complex)
  S_eff = np.zeros((len(X)-num_bad, len(X)-num_bad), dtype=complex)
  
  i = 0
  for row in good_index:
    j = 0
    for col in good_index:
      H_eff[i][j] = H[row][col]
      S_eff[i][j] = S[row][col]
      j += 1
    i += 1

  w, v = np.linalg.eig(np.dot(np.linalg.inv(S_eff),H_eff))
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

    # file to store x vector
    infile = open("sampled_x.txt", "w")

    # add zero vector
    for i in range(2*norb):
      infile.write("%.7f" % X[0][i])
    infile.write("\n")

    # sample x
    for samp in range(Nsamp):
    
      # sample from gaussian distributation
      x = []
      for i in range(2*norb):
        x.append(np.random.normal(0.0,1.0))
        infile.write("%.7f  " % x[i])
      X.append(x)
      infile.write("\n")

    # close file 
    infile.close()
  
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

  # loop over all pairs
  for index in range(start_ind, end_ind):

    # propagate initial state
    F1 = f_propagate(X[pairs[index][0]], Q, w, current_lF)
    F2 = f_propagate(X[pairs[index][1]], Q, w, current_rF)

    # compute matrix elements
    smat = 0.0+0,0j
    hmat = 0.0+0.0j

    if (pairs[index][0] == pairs[index][1]):
      smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int, True)
    else:
      smat, hmat = ham_ovlp_compute_fast(F1, F2, oe_int, False)     

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

    # number of bad points
    num_bad = 0
    bad_index = []
    good_index = []
    for i in range(len(X)):
      #print H_tot[i][i] / S_tot[i][i]
      if ( (H_tot[i][i]/S_tot[i][i]).real < (H_tot[0][0]/S_tot[0][0]).real ):
        num_bad += 1
        bad_index.append(i)
      else:
        good_index.append(i)

    H_eff = np.zeros((len(X)-num_bad, len(X)-num_bad), dtype=complex)
    S_eff = np.zeros((len(X)-num_bad, len(X)-num_bad), dtype=complex)
    
    i = 0
    for row in good_index:
      j = 0
      for col in good_index:
        H_eff[i][j] = H_tot[row][col]
        S_eff[i][j] = S_tot[row][col]
        j += 1
      i += 1

    # diagonalize the effective Hamiltonian
    w, v = np.linalg.eig(np.dot(np.linalg.pinv(S_eff),H_eff))

    # file to print out eigenvectors
    infile = open("eigvecs.txt", "w")

    # print eigenvalues
    for i in range(len(w)):
      if (w[i].real < (H_eff[0][0]/S_eff[0][0]).real):

        # store eigenvalues
        infile.write("energy is %.7f + %.7f j\n" % (w[i].real, w[i].imag))

        # store eigenvectors
        for j in range(len(good_index)):
          infile.write("%d  %.7f + %.7f j\n" % (good_index[j], v[j][i].real, v[j][i].imag))
        infile.write("\n")

    # close eigenvector file
    infile.close()

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
  if (method == "Fast"):
    nsamp  = int(sys.argv[5])
    energy = Rand_Samp_Fast(F, nsamp, Q, pho, oe_int)   
  elif (method == "Parallel_test"):
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

