# python script that converts JAGP's Jastrow factor 
import numpy as np
import itertools
import random
from scipy import misc
import pdb
import sys

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

  pmat = np.zeros((norb, norb), dtype=complex)

  line_num = 0
  #fill in pairing matrix
  for line in infile:
    line = line.split()
    for i in range(len(line)):
      pmat[line_num][i] = float(line[i])
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
  for i in range(norb):
    for j in range(norb, 2*norb):
      F_new[i][j] = F[i][j-norb]
  for i in range(norb, 2*norb):
    for j in range(norb):
      F_new[i][j] = -1.0 * F[j][i-norb]

  # propagate pairing matrix
  # loop over auxiliary filed variables
  for k in range(ndim):
    # loop over rows of F
    for r in range(ndim):
      # loop over columns of F
      for s in range(ndim):
        F_new[r][s] *= np.exp(Q_tilda[r][k] + Q_tilda[s][k])

  # convert to smaller dimension 
  F_new_small = np.zeros((norb, norb), dtype=complex)
  for r in range(norb):
    for s in range(norb):
      F_new_small[r][s] = F_new[r][s+norb]
  # return the new F
  return F_new_small

# function that computes the nth order coefficient of a Pfaffian
def Pf(P, n):
 
 # first compute the root of characteristic polynomial of P
 roots = np.roots(np.poly(P))

 # find unique roots
 unique_roots = []
 for i in range(0, len(roots), 2):
   unique_roots.append(roots[i])
 
 # check too see if number of unique roots is equal to the number of spatial oritals
 if ( len(unique_roots) != norb):
   print "Bad number of roots find!"
 
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
  for i in range(norb):
    for j in range(norb, 2*norb):
      lF[i][j] = F1[i][j-norb]
      rF[i][j] = F2[i][j-norb]
  for i in range(norb, 2*norb):
    for j in range(norb):
      lF[i][j] = -1.0 * F1[j][i-norb]
      rF[i][j] = -1.0 * F2[j][i-norb]
 
  # compute matrix passed into Pfaffian
  P = np.dot(lF.transpose().conjugate(), rF)

  return Pf(P, nalpha)


# function that computes one-body Green's function between two AGP
def ob_G(F1, F2, a, b):

  # constructs large, antisymmetric matrix for F1 and F2
  lF = np.zeros((2*norb, 2*norb), dtype=complex)
  rF = np.zeros((2*norb, 2*norb), dtype=complex)
  for i in range(norb):
    for j in range(norb, 2*norb):
      lF[i][j] = F1[i][j-norb]
      rF[i][j] = F2[i][j-norb]
  for i in range(norb, 2*norb):
    for j in range(norb):
      lF[i][j] = -1.0 * F1[j][i-norb]
      rF[i][j] = -1.0 * F2[j][i-norb]

  # compute M matrix 
  M = np.dot(rF,lF.transpose().conjugate())

  # compute matrix passed into Pfaffian
  P = np.dot(lF.transpose().conjugate(), rF)

  # return value 
  integral = 0.0+0.0j

  # loop over all degrees less than nalpha
  for i in range(0, nalpha):
    m = nalpha - i - 1
    integral += (-1.0)**(i) * np.linalg.matrix_power(M, i+1)[b][a] * Pf(P, m)

  return integral

# function that computes two-body Green's function between two AGP
def tb_G(F1, F2, a, b, d, c):

  # constructs large, antisymmetric matrix for F1 and F2
  lF = np.zeros((2*norb, 2*norb), dtype=complex)
  rF = np.zeros((2*norb, 2*norb), dtype=complex)
  for i in range(norb):
    for j in range(norb, 2*norb):
      lF[i][j] = F1[i][j-norb]
      rF[i][j] = F2[i][j-norb]
  for i in range(norb, 2*norb):
    for j in range(norb):
      lF[i][j] = -1.0 * F1[j][i-norb]
      rF[i][j] = -1.0 * F2[j][i-norb]

  # compute M matrix 
  M = np.dot(rF,lF.transpose().conjugate())

  # compute matrix passed into Pfaffian
  P = np.dot(lF.transpose().conjugate(), rF)

  # return value 
  integral = 0.0+0.0j

  # loop over all degrees less than nalpha
  for n in range(1, nalpha+1):
    m = nalpha - n
    part1 = 0+0j
    part2 = 0+0j
    for i in range(0, n-1):

      # compute j 
      j = n - i - 2

      if (n > 1):
        # part one
        part1 += ((-1.0)**(i+j) * np.linalg.matrix_power(M, i+1)[c][a] * np.linalg.matrix_power(M, j+1)[d][b] - (-1.0)**(i+j) * np.linalg.matrix_power(M, i+1)[c][b] * np.linalg.matrix_power(M, j+1)[d][a])
    
    for i in range(0, n):

      # compute j 
      j = n - i - 1

      # part two
      part2 += (-1.0)**(i+j) * (np.dot(np.linalg.matrix_power(M,i), rF))[c][d] * (np.dot(lF.transpose().conjugate(), np.linalg.matrix_power(M, j)))[b][a]
   
    integral += ((part1 + part2) * Pf(P, m))
 
  return integral

# function that return Hamiltonian matrix elements of two AGP
def ham_compute(F1, F2, oe_int):
  
  # one-body energy
  ob_eng = 0.0

  # aa one-electron integral stuff
  for i in range(norb):
    for j in range(norb):
      ob_eng += oe_int[i][j] * (ob_G(F1, F2, i, j) + ob_G(F1, F2, i+norb, j+norb))

  # two-body energy
  tb_eng = 0.0

  # two-electron energy calculation
  for i in range(norb):
    tb_eng += -1.0 * U * tb_G(F1, F2, i, i+norb, i, i+norb)

  # return matrix element
  return ob_eng + tb_eng

# function that performs Metropolis sampling to compute average energy
def Metropolis(F0, Nsamp, Q, w, oe_int):

  # a set of randomly generated auxiliary field vectors
  X = []

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # current left and right
  current_lF = np.zeros((norb, norb), dtype=complex)
  current_rF = np.zeros((norb, norb), dtype=complex)

  # initialize current F
  for i in range(norb):
    for j in range(norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  # open output file
  infile = open("test.txt", "w")

  # initial field vector
  x1 = [0.0]*(norb*2)
  x2 = [0.0]*(norb*2)

  # generate Markov chain and collect data
  for samp in range(Nsamp):
    
    accept = False

    # draw auxiliary field operators
    if (samp > 0):

      # field index to change
      index1 = np.random.randint(0,norb*2)
      index2 = np.random.randint(0,norb*2)

      # uniform number
      rand1 = (np.random.uniform(-2.0, 2.0))
      rand2 = (np.random.uniform(-2.0, 2.0))

      # current P1
      P_curr1 = np.exp(-0.5 * x1[index1]**2)

      # proposed P1
      P_prop1 = np.exp(-0.5 * rand1**2)

      # accept if prob increases
      if ( P_prop1 / P_curr1 >= 1.0):
        x1[index1] = rand1
        accept = True
      # accept with prob
      else:
        comp = np.random.uniform(0,1)
        if ( P_prop1 / P_curr1 > comp ):
          x1[index1] = rand1
          accept = True
        else:
          accept = False

      # current P2
      P_curr2 = np.exp(-0.5 * x2[index2]**2)

      # proposed P2
      P_prop2 = np.exp(-0.5 * rand2**2)

      # accept if prob incereases
      if ( P_prop2 / P_curr2 >= 1.0):
        x2[index2] = rand2
        accept = True
      # accept with prob
      else:
        comp = np.random.uniform(0,1)
        if ( P_prop2 / P_curr2 > comp ):
          x1[index2] = rand2
          accept = True
        else:
          accept = False

    # weight factor
    exp1 = 1.0 
    exp2 = 1.0
    for i in range(norb*2):
      exp1 *= np.exp(-0.5 * x1[i]**2)
      exp2 *= np.exp(-0.5 * x2[i]**2) 
    exp_tot = exp1*exp2

    # write weight factor to file
    infile.write("%12.12f\n" % exp_tot)

    # propagate initial state
    F1 = f_propagate(x1, Q, w, current_lF)
    F2 = f_propagate(x2, Q, w, current_rF)

    # compute the norm of F2 and F1
    #norm1 = np.sqrt(ovlp_compute(F1,F1))
    #norm2 = np.sqrt(ovlp_compute(F2,F2))
    #norm1 = np.power(norm1, 1.0/nalpha)
    #norm2 = np.power(norm2, 1.0/nalpha)
    #for i in range(norb):
    #  for j in range(norb):
    #    F1[i][j] /= norm1
    #    F2[i][j] /= norm2

    # compute numerator
    hmat = ham_compute(F1, F2, oe_int)
    numerator.append(hmat)
    infile.write( "%12.12f  %12.12f\n" % (hmat.real, hmat.imag))

    # compute denominator
    smat = ovlp_compute(F1, F2)
    denominator.append(smat)
    infile.write( "%12.12f  %12.12f\n" % (smat.real, smat.imag))

  # close output file
  infile.close()

  # accumulate global quantities
  nu = 0.0+0.0j
  dn = 0.0+0.0j

  for i in range(len(numerator)):
    nu += numerator[i]
    dn += denominator[i]

  # return energy 
  return nu / dn

# function that evaluate energy on a grid
def energy_grid(F0, exct, Q, w, oe_int):
  
  # current left and right
  current_lF = np.zeros((norb, norb), dtype=complex)
  current_rF = np.zeros((norb, norb), dtype=complex)

  # initialize current F
  for i in range(norb):
    for j in range(norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # auxiliary field vectors 
  X = []

  # generate X
  #exct = [0]
  #for level in exct:
  x_sample(exct, X)

  # loop over all field operators
  for x in X:
    for y in X:

      # propagate F
      F1 = f_propagate(x, Q, w, current_lF)
      F2 = f_propagate(y, Q, w, current_rF)

      # compute numerator
      hmat = ham_compute(F1, F2, oe_int)

      # compute denominator
      smat = ovlp_compute(F1, F2)

      # compute weight
      weight = 1.0
      for k in range(len(x)):
        weight *= np.exp(-0.5 * x[k]**2)
        weight *= np.exp(-0.5 * y[k]**2)

      # accumulate 
      numerator.append(weight * hmat)
      denominator.append(weight * smat)

  # compute average energy
  nu = 0.0+0,0j
  de = 0.0+0.0j

  for l in range(len(numerator)):
    nu += numerator[l]
    de += denominator[l]

  # return energy
  return nu / de

# function that evaluates energy with random, non-Metropolis sampling (Large fluctuations)
def Rand_Samp(F0, Nsamp, Q, w, oe_int):

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # store randomly sampled vectors
  X = []

  # current left and right
  current_lF = np.zeros((norb, norb), dtype=complex)
  current_rF = np.zeros((norb, norb), dtype=complex)

  for i in range(norb):
    for j in range(norb):
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

      # compute numerator
      hmat = ham_compute(F1, F2, oe_int)
      numerator.append(hmat)
      H[l][r] = hmat
      nu += hmat
      if (r != l):
        nu += np.conjugate(hmat)
        H[r][l] = np.conjugate(hmat)

      # compute denominator
      smat = ovlp_compute(F1, F2)
      denominator.append(smat)
      S[l][r] = smat
      de += smat
      if (r != l):
        de += np.conjugate(smat)
        S[r][l] = np.conjugate(smat)

  w, v = np.linalg.eig(np.dot(np.linalg.pinv(S),H))
  lowest_eng = w[0].real
  for eigval in w:
    if (eigval.real < lowest_eng):
      lowest_eng = eigval.real
  print lowest_eng
  # close output file
  infile.close()

  # return energy 
  return nu / de

def Metropolis2(F0, Nsamp, Q, w, oe_int):

  # a set of randomly generated auxiliary field vectors
  X = []

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # current left and right
  current_lF = np.zeros((norb, norb), dtype=complex)
  current_rF = np.zeros((norb, norb), dtype=complex)

  # initialize current F
  for i in range(norb):
    for j in range(norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  # open output file
  infile = open("test.txt", "w")

  # initial field vector
  x = [0.0]*(norb*2)

  # generate Markov chain and collect data
  for samp in range(Nsamp):
    
    accept = False

    # draw auxiliary field operators
    if (samp > 0):

      # field index to change
      index = np.random.randint(0,norb*2)

      # uniform number
      rand = (np.random.uniform(-2.0, 2.0))

      # current P1
      P_curr = np.exp(-0.5 * x[index]**2)

      # proposed P1
      P_prop = np.exp(-0.5 * rand**2)

      # accept if prob increases
      if ( P_prop / P_curr >= 1.0):
        x[index] = rand
        accept = True
      # accept with prob
      else:
        comp = np.random.uniform(0,1)
        if ( P_prop / P_curr > comp ):
          x[index] = rand
          accept = True
        else:
          accept = False

      x_add = [0.0]*(norb*2)
      for i in range(len(x)):
        x_add[i] = x[i]
      X.append(x_add)

  nu = 0.0+0.0j
  dn = 0.0+0.0j
  # loop over all field vectors
  for xl in X:
    for xr in X:

      # propagate initial state
      F1 = f_propagate(xl, Q, w, current_lF)
      F2 = f_propagate(xr, Q, w, current_rF)

      # compute numerator
      hmat = ham_compute(F1, F2, oe_int)
      numerator.append(hmat)
      nu += hmat

      # compute denominator
      smat = ovlp_compute(F1, F2)
      denominator.append(smat)
      dn += smat

  # close output file
  infile.close()

  # return energy 
  return nu / dn

# function that evaluates energy with random, non-Metropolis sampling (Large fluctuations)
def Importance_Sampling(F0, Nsamp, Q, w, oe_int):

  # collections of numerator local quantities
  numerator = []

  # collections of denominator local quantities
  denominator = []

  # store importance sampled vectors
  X_importance = []

  # store vectors in this sample
  X_samp = []

  # hamiltonian matrix 
  H = np.zeros((Nsamp, Nsamp), dtype=complex)

  # overlap matrix 
  S = np.zeros((Nsamp, Nsamp), dtype=complex)

  # current left and right
  current_lF = np.zeros((norb, norb), dtype=complex)
  current_rF = np.zeros((norb, norb), dtype=complex)

  for i in range(norb):
    for j in range(norb):
      current_lF[i][j] = F0[i][j]
      current_rF[i][j] = F0[i][j]

  for iteration in range(10):

    # generate auxiliary filed vectors
    for samp in range(Nsamp):
    
      # generate auxiliary field vectors based on Gaussian distribution
      x = []
      for i in range(norb*2):
        x.append(np.random.normal(0.0,1.0))
      X_samp.append(x)

    # compute matrix elements
    for i in range(len(X_samp)):
      for j in range(i, len(X_samp)):

        #print "%i %i" % (i, j)
        # propagate initial state
        F1 = f_propagate(X_samp[i], Q, w, current_lF)
        F2 = f_propagate(X_samp[j], Q, w, current_rF)

        # compute H
        H[i][j] = ham_compute(F1, F2, oe_int)
        if (i != j):
          H[j][i] = np.conjugate(H[i][j])

        # compute S
        S[i][j] = ovlp_compute(F1, F2)
        if (i != j):
          S[j][i] = np.conjugate(S[i][j])

    # diagonalize this matrix
    eigval, eigvec = np.linalg.eig(np.dot(np.linalg.pinv(S), H))
    print eigval

    # find the lowest eigenvalue
    lowest_eng = eigval[0].real
    lowest_ind = 0
    for i in range(len(eigval)):
      if (eigval[i] < lowest_eng):
        lowest_eng = eigval[i].real
        lowest_ind = i

    # print out 
    print "iter = %i, energy = %12.12f" % (iteration, lowest_eng)

    # add the most important x
    for ind in range(len(X_samp)):
      if (abs(eigvec[ind][lowest_ind]) > 0.1):
        X_importance.append(X_samp[ind])

    # clear X_sample
    del X_samp[:]

  Hf = np.zeros((len(X_importance), len(X_importance)), dtype=complex)
  Sf = np.zeros((len(X_importance), len(X_importance)), dtype=complex)
  # after all iterations are done, compute energy
  for i in range(len(X_importance)):
    for j in range(i, len(X_importance)):
      
      # propagate initial state
      F1 = f_propagate(X_importance[i], Q, w, current_lF)
      F2 = f_propagate(X_importance[j], Q, w, current_rF)

      # compute H
      Hf[i][j] = ham_compute(F1, F2, oe_int)
      if (i != j):
        Hf[j][i] = np.conjugate(Hf[i][j])

      # compute S
      Sf[i][j] = ovlp_compute(F1, F2)
      if (i != j):
        Sf[j][i] = np.conjugate(Sf[i][j])

  # diagonalize this matrix
  eigval, eigvec = np.linalg.eig(np.dot(np.linalg.pinv(Sf), Hf))
  #for i in range(len(X_importance)):
  #  for j in range(len(X_importance)):
  #    print "%5.5f + %5.5f i  " % (Sf[i][j].real, Sf[i][j].imag),
  #  print "\n"
  #eigval, eigvec = np.linalg.eig(Hf)
  print eigval

def main():
  
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
  if (method == "MC"):
    nsamp  = int(sys.argv[5])
    energy = Metropolis2(F, nsamp, Q, pho, oe_int)
  elif (method == "Rand_Samp"):
    nsamp  = int(sys.argv[5])
    energy = Rand_Samp(F, nsamp, Q, pho, oe_int)
  elif (method == "Grid"):
    exct   = int(sys.argv[5])
    energy = energy_grid(F, exct, Q, pho, oe_int)
  elif (method == "Non_Var"):
    nsamp = int(sys.argv[5])
    energy = Metropolis(F, nsamp, Q, pho, oe_int)
  elif (method == "Important"):
    nsamp = int(sys.argv[5])
    energy = Importance_Sampling(F, nsamp, Q, pho, oe_int)
  else:
    print "Unknown Method"
  #Importance_Sampling(F, nsamp, Q, pho, oe_int)

  # print final energy
  print energy

if __name__ == "__main__":
  main()

