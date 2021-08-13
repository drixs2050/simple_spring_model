import igl
import matplotlib.pyplot as plt
import numpy as np
from trajectory_model import *

from spring_model import *
from autograd import grad
import scipy.optimize as sci_opt
import torch
import torch.optim as optim


def v_spring_angle(b: float, c: float, theta: float, l0: float, stiffness: float):
	# AC = C - A
	# AB = B - A
	# b = np.linalg.norm(AC)
	# c = np.linalg.norm(AB)
	# costheta = np.dot(AC, AB) / b*c
	# theta = np.arccos(costheta)
	l = (b ** 2 + c ** 2 - 2 * b * c * np.cos(theta)) ** 0.5
	return 0.5 * stiffness * (l - l0) ** 2


dv_spring_angle = grad(v_spring_angle, 2)

d2v_spring_angle = grad(dv_spring_angle, 2)


def newton(f: Callable, Df: Callable, b: float, c: float, A0: float, l0: float, stiffness: float, epsilon: float,
		   max_iter: int):
	'''Approximate solution of f(x)=0 by Newton's method.

	Parameters
	----------
	f : function
		Function for which we are searching for a solution f(x)=0.
	Df : function
		Derivative of f(x).
	x0 : number
		Initial guess for a solution f(x)=0.
	epsilon : number
		Stopping criteria is abs(f(x)) < epsilon.
	max_iter : integer
		Maximum number of iterations of Newton's method.

	Returns
	-------
	xn : number
		Implement Newton's method: compute the linear approximation
		of f(x) at xn and find x intercept by the formula
			x = xn - f(xn)/Df(xn)
		Continue until abs(f(xn)) < epsilon and return xn.
		If Df(xn) == 0, return None. If the number of iterations
		exceeds max_iter, then return None.
	'''
	An = A0
	for n in range(0, max_iter):
		fxn = f(b, c, An, l0, stiffness)
		if abs(fxn) < epsilon:
			print('Found solution after', n, 'iterations.')
			return An
		Dfxn = Df(b, c, An, l0, stiffness)
		if Dfxn == 0:
			print('Zero derivative. No solution found.')
			return None
		An = An - fxn / Dfxn
	print('Exceeded maximum iterations. No solution found.')
	return None


def integrator(new_theta_dot: float, theta_dot: float, b: float, c: float, theta: float, l0: float, stiffness: float,
			   h: float):
	return 0.5 * (new_theta_dot - theta_dot) * (new_theta_dot - theta_dot) + \
		   v_spring_angle(b, c, theta + h * new_theta_dot, l0, stiffness)


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
		fig.savefig('visualize_kinematics/'+t_str+'.png')
	else:
		fig.show()

def my_MSEloss(a, b):
	return torch.mean((a.squeeze() - b.squeeze())**2)


if __name__ == "__main__":
	data = []
	label = []
	A = np.array([0, 1, 0])
	B = np.array([0, 0, 0])
	C = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0])
	BC = C - B
	BA = A - B
	a = np.linalg.norm(BC)
	c = np.linalg.norm(BA)
	cosTheta = np.dot(BC, BA) / a * c
	theta = np.arccos(cosTheta)
	theta_dot = 0.0
	t = 0.0
	dt = 0.01
	q = np.hstack((A, B, C))
	visualize(q, t, write=True)
	diff = float('inf')
	stiffness = 5.0
	l0 = 1.0
	right_angle = np.deg2rad(90)
	# sim start
	while t < 100:
		t = round(t, 2)
		new_theta_dot = sci_opt.minimize_scalar(integrator, args=(theta_dot, a, c, theta, l0, stiffness, dt)).x
		diff = np.abs(theta_dot - new_theta_dot)
		theta += dt * new_theta_dot
		theta_dot = new_theta_dot
		# if round(t * 100) % 10 == 0:
		# 	C = np.array([np.cos(np.deg2rad(90) - theta) * a, np.sin(np.deg2rad(90) - theta) * a, 0])
		# 	q = np.hstack((A, B, C))
		# 	visualize(q, t, write=True)
		# 	print((t, theta))
		#
		# print(P.T @ q + x0)
		# print(np.linalg.norm((np.array([0, 0, 0]), q)))
		data.append(t)
		label.append([theta_dot, theta])
		t += dt

	# visualize(P.T @ q + x0, t, write=True)
	torch.manual_seed(42)
	device = torch.device('cuda:1')
	torch.cuda.device(1)
	print(torch.cuda.is_available())
	print(torch.cuda.current_device())
	print(torch.cuda.get_device_name(1))
	num_epoch = 100000
	data_tensor = torch.Tensor(data).reshape(len(data), 1).to(device).float()
	label_tensor = torch.Tensor(label).to(device).float()
	model = AngleTrajNet(label_tensor.shape[1] // 2).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.002)
	loss_function = my_MSEloss
	for epoch in range(num_epoch):
		pred = model(data_tensor)
		loss = loss_function(pred, label_tensor)
		optimizer.zero_grad()
		if epoch % 10 == 0:
			print('loss: {}'.format(loss))
		loss.backward()
		optimizer.step()