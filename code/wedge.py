def wedge(v1, v2):
  """ 
  Return the BiVector of v1 ^ v2 (Plücker formula)
  v1 and v2 are 4d vectors (arrays of 4 elements)
  """
  return [v1[0] * v2[1] - v1[1] * v2[0],
    v1[0] * v2[2] - v1[2] * v2[0],
    v1[0] * v2[3] - v1[3] * v2[0],
    v1[1] * v2[2] - v1[2] * v2[1],
    v1[1] * v2[3] - v1[3] * v2[1],
    v1[2] * v2[3] - v1[3] * v2[2]];


def wedgeBiVector(bv, v):
  """ 
  Return the AntiVector or TriVector of bv ^ v (Plücker formula)
  - bv : 4d BiVector (array of 6 elements)
  - v : 4d vector (array of 4 elements)
  """
  return [bv[0] * v[2] - bv[1] * v[1] + bv[3] * v[0],
    bv[0] * v[3] - bv[2] * v[1] + bv[4] * v[0],
    bv[1] * v[3] - bv[2] * v[2] + bv[5] * v[0],
    bv[3] * v[3] - bv[4] * v[2] + bv[5] * v[1]]


def antiWedge(bv, av):
  """ 
  Return the Vector 4D bv ^ av (Plücker formula)
  - bv : 4d BiVector (array of 6 elements)
  - av : 4d AntiVector or TriVector (array of 4 elements)
  """
  return [bv[1] * av[1] - bv[0] * av[2] - bv[2] * av[0],
    bv[3] * av[1] - bv[0] * av[3] - bv[4] * av[0],
    bv[3] * av[2] - bv[1] * av[3] - bv[5] * av[0],
    bv[4] * av[2] - bv[2] * av[3] - bv[5] * av[1]]
