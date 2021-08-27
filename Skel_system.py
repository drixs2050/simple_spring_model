from __future__ import annotations
from typing import Optional
import autograd.numpy as np
import scipy.optimize as sci_opt
import matplotlib.pyplot as plt
# The working one, the basic idea is the same but this time we minimize
# potential energy with respect to angel between skeletons so the skeleton length is fixed


def v_spring_angle(b: float, c: float, theta: float, l0: float, stiffness: float):
	# AC = C - A
	# AB = B - A
	# b = np.linalg.norm(AC)
	# c = np.linalg.norm(AB)
	# costheta = np.dot(AC, AB) / b*c
	# theta = np.arccos(costheta)
	l = (b ** 2 + c ** 2 - 2 * b * c * np.cos(theta)) ** 0.5
	return 0.5 * stiffness * (l - l0) ** 2


def integrator(new_theta_dot: float, theta_dot: float, b: float, c: float, theta: float, l0: float,
			   stiffness: float, h: float):
	return 0.5 * (new_theta_dot - theta_dot) * (new_theta_dot - theta_dot) + \
		   v_spring_angle(b, c, theta + h * new_theta_dot, l0, stiffness)


def get_rotation_mat(theta):
	return np.array([[np.cos(theta), -np.sin(theta)],
					 [np.sin(theta), np.cos(theta)]])


def visualize(mass_loc, t, write=False):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	pos = mass_loc
	if len(mass_loc.shape) < 2:
		pos = pos.reshape((len(mass_loc)//3, 3))
	xdata = pos[:, 0]
	ydata = pos[:, 1]
	zdata = pos[:, 2]
	ax.scatter3D(xdata, zdata, ydata, c=np.array([[1.0, 0, 0]]))
	ax.set_xlabel('x')
	ax.set_ylabel('z')
	ax.set_zlabel('y')
	ax.set_xlim3d(-3., 3.)
	ax.set_ylim3d(-3., 3.)
	ax.set_zlim3d(-3., 3.)
	ax.view_init(azim=270, elev=0)
	t_str = format(t, '.2f')
	if write:
		fig.savefig('visualize_complex/visualize_shrunk/'+t_str+'.png')
	else:
		fig.show()


def angleVec2(d, v):

	a_1 = np.arctan2(d[1], d[0])
	a_2 = np.arctan2(v[1], v[0])

	diff = a_2 - a_1

	if diff < -np.pi:
		diff = np.pi-(abs(diff)-np.pi)
	elif diff > np.pi:
		diff = -np.pi+(abs(diff)-np.pi)

	return diff


class SkelUnit:
	child: Optional[SkelUnit]
	root_pos: np.ndarrayfl
	tip_pos: np.ndarray
	con = str
	skel_len: float
	l0: float
	stiffness: float
	theta: float
	dot_theta: float
	rotate: int

	def __init__(self, root_pos, tip_pos, l0, stiffness, theta=0, dot_theta=0, child=None):
		self.child = child
		self.root_pos = root_pos
		self.tip_pos = tip_pos
		self.l0 = l0
		self.stiffness = stiffness
		self.theta = theta
		self.dot_theta = dot_theta
		self.skel_len = np.linalg.norm(self.root_pos - self.tip_pos)

	def calc_angle(self):
		if self.con == "root":
			self.theta = angleVec2(self.tip_pos - self.root_pos, self.child.root_pos - self.root_pos)
		else:
			self.theta = angleVec2(self.tip_pos - self.root_pos, self.child.tip_pos - self.root_pos)


class SnakeSkel:
	root: SkelUnit
	h: float
	t: float

	def __init__(self, root, h, t=0):
		self.root = root
		self.h = h
		self.t = t

	def set_all_angles(self):
		curr = self.root
		while curr.child is not None:
			curr.calc_angle()
			curr = curr.child

	def set_all_pos(self):
		curr = self.root
		while curr.child is not None:
			BA = (curr.tip_pos - curr.root_pos)
			BA_norm = BA / np.linalg.norm(BA)
			rm = get_rotation_mat(curr.theta)
			if curr.con == "root":
				BA_norm[:2] = rm @ BA_norm[:2]
				curr.child.root_pos = BA_norm * (curr.child.skel_len / np.linalg.norm(BA_norm)) + curr.root_pos
				curr.child.tip_pos = curr.root_pos
			else:
				BA[:2] = rm @ BA[:2]
				curr.child.tip_pos = BA_norm * (curr.child.skel_len / np.linalg.norm(BA_norm)) + curr.root_pos
				curr.child.root_pos = curr.root_pos
			curr = curr.child

	def get_all_pos(self):
		curr = self.root
		pos = [curr.root_pos, curr.tip_pos]
		while curr.child is not None:
			if curr.con == "root":
				pos.append(curr.child.root_pos)
			else:
				pos.append(curr.child.tip_pos)
			curr = curr.child
		return pos

	def update(self):
		curr = self.root
		ang_vels = [curr.dot_theta]
		while curr.child is not None:
			theta_dot = curr.dot_theta
			a = curr.child.skel_len
			c = curr.skel_len
			theta = curr.theta
			l0 = curr.l0
			stiffness = curr.stiffness
			new_theta_dot = sci_opt.minimize_scalar(integrator, args=(theta_dot, a, c, theta, l0, stiffness, self.h)).x
			curr.theta += self.h * new_theta_dot
			curr.dot_theta = new_theta_dot
			ang_vels.append(curr.dot_theta)
			curr = curr.child
		self.t += self.h
		return ang_vels


if __name__ == "__main__":
	# strech sample
	# pos = np.array([[0, 0, 0],
	# 				[np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60)), 0],
	# 				[np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), 0],
	# 				[0, 0, 0],
	# 				[0, -np.sin(np.deg2rad(60))*2, 0],
	# 			    [np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), 0],
	# 				[np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))*3, 0],
	# 				[0, -np.sin(np.deg2rad(60))*2, 0],
	# 				[0, -np.sin(np.deg2rad(60))*4, 0],
	# 				[np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))*3, 0]])

	# shrunk sample

	pos = np.array([[0, 0, 0],
					[np.cos(np.deg2rad(15)), 0, 0],
					[np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15)), 0],
					[0, 0, 0],
					[0, -np.sin(np.deg2rad(15))*2, 0],
				    [np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15)), 0],
					[np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15))*3, 0],
					[0, -np.sin(np.deg2rad(15))*2, 0],
					[0, -np.sin(np.deg2rad(15))*4, 0],
					[np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15))*3, 0],
					[np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15))*5, 0],
					[0, -np.sin(np.deg2rad(15))*4, 0]])
	connections = ["root", "root", "root", "root", "root"]
	l0 = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
	stiffness = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
	ang_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

	i = len(pos) - 1
	curr_skel = SkelUnit(pos[i-1], pos[i], l0[len(l0) - 1], stiffness[len(stiffness) - 1])
	i -= 2
	while i > 0:
		new_skel = SkelUnit(pos[i-1], pos[i], l0[(i-1)//2], stiffness[(i-1)//2])
		new_skel.child = curr_skel
		new_skel.con = connections[(i-1)//2]
		curr_skel = new_skel
		i -= 2
	t = []
	pos = []
	ang_vels = []
	snake = SnakeSkel(curr_skel, 0.01)
	snake.set_all_angles()
	diff = float('inf')
	# snake.set_all_pos()
	start_pos = np.array(snake.get_all_pos())
	start_pos = start_pos.reshape(start_pos.size, )
	# visualize(start_pos, snake.t, write=True)
	t.append([snake.t])
	pos.append(start_pos)
	ang_vels.append(ang_vel.copy())

	while diff > 1e-8:
		new_ang_vel = np.array(snake.update())
		diff = np.linalg.norm(new_ang_vel-ang_vel)
		snake.set_all_pos()
		ang_vel = new_ang_vel
		new_pos = np.array(snake.get_all_pos())
		new_pos = new_pos.reshape(start_pos.size,)
		if round(snake.t * 100) % 100 == 0:
			print(diff)
			# visualize(new_pos, snake.t, write=True)
		t.append([snake.t])
		pos.append(new_pos.copy())
		ang_vels.append(new_ang_vel.copy())

	t_array = np.array(t)
	pos_array = np.array(pos)
	ang_vels_array = np.array(ang_vels)

	np.savetxt("dataset/pos_shrunk.csv", np.hstack((t_array, pos_array)), delimiter=",")
	np.savetxt("dataset/ang_shrunk.csv", np.hstack((t_array, ang_vels_array)), delimiter=",")

