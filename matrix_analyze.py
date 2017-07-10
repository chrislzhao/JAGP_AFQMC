import numpy as np

# dimension of matrix 
ndim = 51

# read in Hamiltonian matrix
infile = open("hmat.txt")

H = np.zeros((ndim, ndim), dtype=complex)

# line number
row = 0

# loop over lines
for line in infile:
  line = line.split()
  col = 0
  for value in line:
    real = ""
    imag = ""
    real_num = 0.0
    imag_num = 0.0
    seg_ind = 0
    neg_imag = False
    for ind in range(len(value)):
      if (value[ind] == "+" and ind != 0):
        seg_ind = ind
      elif (value[ind] == "-" and ind != 0):
        seg_ind = ind
        neg_imag = True
    for ind in range(len(value)):
      if (ind < seg_ind):
        real += value[ind]
      elif (ind > seg_ind):
        imag += value[ind]
    real_num = float(real)
    if (neg_imag):
      imag_num = -1.0 * float(imag)
    else:
      imag_num = float(imag)
    H[row][col] = complex(real_num, imag_num)
    col += 1
  row += 1
        
infile.close() 

# read in overlap matrix
infile = open("smat.txt")

S = np.zeros((ndim, ndim), dtype=complex)

# line number
row = 0

# loop over lines
for line in infile:
  line = line.split()
  col = 0
  for value in line:
    real = ""
    imag = ""
    real_num = 0.0
    imag_num = 0.0
    seg_ind = 0
    neg_imag = False
    for ind in range(len(value)):
      if (value[ind] == "+" and ind != 0):
        seg_ind = ind
      elif (value[ind] == "-" and ind != 0):
        seg_ind = ind
        neg_imag = True
    for ind in range(len(value)):
      if (ind < seg_ind):
        real += value[ind]
      elif (ind > seg_ind):
        imag += value[ind]
    real_num = float(real)
    if (neg_imag):
      imag_num = -1.0 * float(imag)
    else:
      imag_num = float(imag)
    S[row][col] = complex(real_num, imag_num)
    col += 1
  row += 1
        
infile.close() 

# D matrix used to normalize vectors 
D = np.zeros((ndim, ndim), dtype=complex)
for i in range(ndim):
  D[i][i] = 1.0 / np.sqrt(S[i][i])

# normalize vectors 
H = np.dot(D, np.dot(H, D))
S = np.dot(D, np.dot(S, D))

for i in range(ndim):
  print H[i][i]

# perform svd on S
#U, s, V = np.linalg.svd(S)

#print s
#print np.linalg.cond(S)

w = np.linalg.eig(np.dot(np.linalg.pinv(S), H))
#print w[0]


