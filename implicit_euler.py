import matplotlib.pyplot as plt
import torch

from spring_model import *
from trajectory_model import *
import torch.optim as optim


dt = 0.01
t = 0
q = np.array([0, 1, 0, 0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5), 0]).astype(float)
qdot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
mass = np.array([1.0, 1.0, 1.0])
springs = np.array([[0, 1], [1, 2], [0, 2]])
l0 = np.array([1, 1, np.sqrt(2)])
stiffness = np.array([999999., 999999., 2.])
fixed_points = np.array([0, 1])

M = mass_matrix_particles(mass)
P = fixed_point_constraints(len(q), fixed_points)

x0 = q - P.T @ P @ q
q = P @ q
qdot = P @ qdot
M = P @ M @ P.T


def energy(q):
	return assemble_energy(P.T @ q + x0, springs, l0, stiffness)


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
		fig.savefig('visualize_newton/'+t_str+'.png')
	else:
		fig.show()


def my_MSEloss(a, b):
	return torch.mean((a.squeeze() - b.squeeze())**2)


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


if __name__ == "__main__":
	data = []
	label = []
	visualize(q, t, write=True)
	diff = float('inf')
	# sim start
	while t < 0.5:
		t = round(t, 2)
		new_q, new_qdot = implicit_euler(q, qdot, dt, M, force, stiff)
		diff = np.linalg.norm(q-new_q)
		q = new_q
		qdot = new_qdot
		# if round(t*100) % 10 == 0:
		# 	visualize(P.T @ q + x0, t, write=True)
		# print(P.T @ q + x0)
		# print(np.linalg.norm((np.array([0, 0, 0]), q)))
		data.append(t)
		label.append(P.T @ q + x0)
		t += dt

	# visualize(P.T @ q + x0, t, write=True)

	torch.manual_seed(42)
	device = torch.device('cuda:1')
	torch.cuda.device(1)
	# print(torch.cuda.is_available())
	# print(torch.cuda.current_device())
	# print(torch.cuda.get_device_name(1))
	num_epoch = 100000
	data_tensor = torch.Tensor(data).reshape(len(data), 1).to(device).float()
	label_tensor = torch.Tensor(label).to(device).float()
	model = TrajModNetwork(label_tensor.shape[1]//3).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.0002)
	loss_function = my_MSEloss
	for epoch in range(num_epoch):
		pred = model(data_tensor)
		loss = loss_function(pred, label_tensor)
		optimizer.zero_grad()
		print('loss: {}'.format(loss))
		loss.backward()
		optimizer.step()

