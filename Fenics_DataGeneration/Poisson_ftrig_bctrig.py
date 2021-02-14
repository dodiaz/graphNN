"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary
  u_D = cos(a*x[0] + b*x[1]) + c*sin(d*x[0])
    f = (a*a + b*b)*cos(a*x[0] + b*x[1]) + c*d*d*sin(d*x[0])
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 


# Create mesh and define function space
x_meshsize = 100
y_meshsize = 100
mesh = UnitSquareMesh(x_meshsize, y_meshsize)
V = FunctionSpace(mesh, 'P', 2)

# determines the number of parameters to run through
numvar1 = 11
numvar2 = 11
numvar3 = 11

coeff1_range = np.linspace(8,10,numvar1)
coeff2_range = np.linspace(10,12,numvar2)
coeff3_range = np.linspace(0,2,numvar3)

#Create array for the solution
sol_savedata = np.zeros((numvar1*numvar2*numvar3, x_meshsize + 1, y_meshsize + 1))
params_savedata = np.zeros((numvar1*numvar2*numvar3, 3))



# create mesh and space to store solution
meshvals_x = np.linspace(0, 1, x_meshsize+1)
meshvals_y = np.linspace(0, 1, y_meshsize+1)
sol = np.zeros((x_meshsize+1,y_meshsize+1))

count = 0


for idx1 in range(coeff1_range.size):
    for idx2 in range(coeff2_range.size): 
        for idx3 in range(coeff3_range.size):

            coeff_1 = coeff1_range[idx1]
            coeff_2 = coeff2_range[idx2]
            coeff_3 = coeff3_range[idx3]
            coeff_4 = 5

            # Define boundary condition
            u_D = Expression('cos(a*x[0] + b*x[1]) + c*sin(d*x[0])', degree=4, a = coeff_1, b = coeff_2, c = coeff_3, d = coeff_4)   

            def boundary(x, on_boundary):
                return on_boundary

            bc = DirichletBC(V, u_D, boundary)

            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Expression('(a*a + b*b)*cos(a*x[0] + b*x[1]) + c*d*d*sin(d*x[0])', degree=4, a = coeff_1, b = coeff_2, c = coeff_3, d = coeff_4)       
            a = dot(grad(u), grad(v))*dx
            L = f*v*dx

            # Compute solution
            u = Function(V)
            solve(a == L, u, bc)


            # Compute error in L2 norm
            error_L2 = errornorm(u_D, u, 'L2')

            # Compute maximum error at vertices
            vertex_values_u_D = u_D.compute_vertex_values(mesh)
            vertex_values_u = u.compute_vertex_values(mesh)
            import numpy as np
            error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))


            # save data into .mat file
            

            for i in range(meshvals_x.size):
                for j in range(meshvals_y.size):
                    sol[i,j] = u(Point(meshvals_x[i], meshvals_y[j]))

            sol_savedata[idx1 + numvar1*idx2 + numvar1*numvar2*idx3,:,:] = sol
            params_savedata[idx1 + numvar1*idx2 + numvar1*numvar2*idx3,:] = (coeff_1, coeff_2, coeff_3)
            
            print("done with solution number " + str(count))
            count = count + 1

            
scipy.io.savemat('Poisson_trig.mat', {"solution": sol_savedata, "coeffs": params_savedata})
            
#check to make sure things are looking right
X,Y = np.meshgrid(meshvals_x,meshvals_y)



print(params_savedata[7,:])