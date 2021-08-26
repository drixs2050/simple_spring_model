from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from spring_model import *
# unfortunately this method did not work for our required simulation
# I suspect that assuming skeleton are stiff springs does not really fit here


dt = 0.01
t = 0
q = np.array([0, 1, 0, 0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5), 0]).astype(float)
qdot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
mass = np.array([1.0, 1.0, 1.0])
springs = np.array([[0, 1], [1, 2], [0, 2]])
l0 = np.array([1, 1, np.sqrt(2)])
stiffness = np.array([999999., 999999., 2.])
fixed_points = np.array([0, 1])
# q = np.array([0, 1, 0, 0, 0.6, 0, 0, -1, 0]).astype(float)
# qdot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
# mass = np.array([1.0, 1.0, 1.0])
# springs = np.array([[0, 1], [1, 2]])
# l0 = np.array([1, 1])
# stiffness = np.array([2., 2.])
# fixed_points = np.array([0, 2])

M = mass_matrix_particles(mass)
P = fixed_point_constraints(len(q), fixed_points)

x0 = q - P.T @ P @ q
q = P @ q
qdot = P @ qdot
M = P @ M @ P.T


def force(q):
	f = assemble_forces(P.T @ q + x0, springs, l0, stiffness)
	return P @ f


def stiff(q):
	K = assemble_stiffness(P.T @ q + x0, springs, l0, stiffness)
	return P @ K @ P.T


def visualize(mass_loc, t, write=False):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	pos = mass_loc.reshape((len(mass_loc)//3, 3))
	xdata = pos[:, 0]
	ydata = pos[:, 1]
	zdata = pos[:, 2]
	ax.scatter3D(xdata, zdata, ydata, c=np.array([[1.0, 0, 0]]))
	ax.set_xlabel('x')
	ax.set_ylabel('z')
	ax.set_zlabel('y')
	ax.set_xlim3d(-1., 1.)
	ax.set_ylim3d(-1., 1.)
	ax.set_zlim3d(-1., 1.)
	ax.view_init(azim=270, elev=0)
	t_str = format(t, '.2f')
	if write:
		fig.savefig('visualize/'+t_str+'.png')
	else:
		fig.show()


if __name__ == "__main__":
	visualize(q, t, write=True)
	diff = float('inf')
	# sim start
	while diff > 1e-9:
		t = round(t, 2)
		new_q, new_qdot = linearly_implicit_euler(q, qdot, dt, M, force, stiff)
		diff = np.linalg.norm(q - new_q)
		q = new_q
		qdot = new_qdot
		# if round(t*100) % 10 == 0:
		# 	visualize(P.T @ q + x0, t, write=True)
		# print(P.T @ q + x0)
		# print(np.linalg.norm((np.array([0, 0, 0]), q)))
		t += dt
	print(t)
	visualize(P.T @ q + x0, t, write=True)


