import torch
import torch.optim as optim
from trajectory_model import *
import pandas as pd
import copy


def my_MSEloss(a, b):
	return torch.mean((a.squeeze() - b.squeeze())**2)


def train_pos(posModel, t_tensor, pos_tensor, num_epoch, loss_function=my_MSEloss):
	optimizer = optim.Adam(posModel.parameters(), lr=0.0001)
	best_loss = float('inf')
	best_weights = copy.deepcopy(posModel.state_dict())
	for epoch in range(num_epoch):
		pred = posModel(t_tensor)
		loss = loss_function(pred, pos_tensor)
		optimizer.zero_grad()
		if epoch % 100 == 0:
			print('pos_loss: {}'.format(loss))
		loss.backward()
		optimizer.step()
		if loss < best_loss:
			best_loss = loss
			best_weights = copy.deepcopy(posModel.state_dict())
	torch.save(best_weights, "weights/" + "pos_stretch_weights.pth")


def train_ang_vels(angModel, t_tensor, ang_vels_tensor, num_epoch, loss_function=my_MSEloss):
	optimizer = optim.Adam(angModel.parameters(), lr=0.0001)
	best_loss = float('inf')
	best_weights = copy.deepcopy(angModel.state_dict())
	for epoch in range(num_epoch):
		pred = angModel(t_tensor)
		loss = loss_function(pred, ang_vels_tensor)
		optimizer.zero_grad()
		if epoch % 100 == 0:
			print('ang_loss: {}'.format(loss))
		loss.backward()
		optimizer.step()
		if loss < best_loss:
			best_loss = loss
			best_weights = copy.deepcopy(angModel.state_dict())
	torch.save(best_weights, "weights/"+"ang_stretch_weights.pth")


if __name__ == "__main__":
	pos_data = torch.tensor(pd.read_csv('dataset/pos_stretch.csv', header=None).values)
	ang_vels_data = torch.tensor(pd.read_csv('dataset/ang_stretch.csv', header=None).values)

	torch.manual_seed(42)
	device = torch.device('cuda:1')
	torch.cuda.device(1)
	print(torch.cuda.is_available())
	print(torch.cuda.current_device())
	print(torch.cuda.get_device_name(1))
	t_tensor = pos_data[:, 0].reshape(len(pos_data), 1).to(device).float()
	pos_tensor = pos_data[:, 1:].to(device).float()
	ang_vels_tensor = ang_vels_data[:, 1:].to(device).float()
	num_epoch = 101
	posModel = TrajModNetwork(pos_tensor.shape[1]).to(device)
	angModel = AngleTrajNet(ang_vels_tensor.shape[1]).to(device)
	train_pos(posModel, t_tensor, pos_tensor, num_epoch)
	train_ang_vels(angModel, t_tensor, ang_vels_tensor, num_epoch)
	# posModel.load_state_dict(torch.load("weights/pos_stretch_weights.pth"))
	# angModel.load_state_dict(torch.load("weights/ang_stretch_weights.pth"))