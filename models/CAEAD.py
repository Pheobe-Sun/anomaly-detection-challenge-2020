import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

class _CAEAD(nn.Module):
	def __init__(self, input_size):
		super(_CAEAD, self).__init__()
		self.en_1 = nn.Conv1d(1, 64, 3, padding=1)
		self.pool1 = nn.MaxPool1d(2,2)
		self.en_2 = nn.Conv1d(64, 32, 3,  padding=1)
		self.pool2 = nn.MaxPool1d(2,2)
		self.en_3 = nn.Conv1d(32, 8, 3,  padding=1)
		self.pool3 = nn.MaxPool1d(2,2)
		self.en_4 = nn.Conv1d(8, 4, 3,  padding=1)
		self.pool4= nn.MaxPool1d(2,2)
		
		self.de_1= nn.Conv1d(4, 8, 3,  padding=1)
		self.de_2= nn.Conv1d(8, 32, 3,  padding=1)
		self.de_3 = nn.Conv1d(32, 64, 3,  padding=1)
		self.de_4 = nn.Conv1d(64, 1, 3,  padding=1)

	def forward(self, X):
		encoder  = F.relu(self.en_1(X))
		encoder = self.pool1(encoder)
		encoder = F.relu(self.en_2(encoder))
		encoder = self.pool2(encoder)
		encoder = F.relu(self.en_3(encoder))
		encoder = self.pool3(encoder)
		encoder = F.relu(self.en_4(encoder))
		encoder = self.pool4(encoder)

		decoder = F.interpolate(encoder, scale_factor=2)
		decoder = F.relu(self.de_1(decoder))
		decoder = F.interpolate(decoder, scale_factor=2)
		decoder = F.relu(self.de_2(decoder))
		decoder = F.interpolate(decoder, scale_factor=2)
		decoder = F.relu(self.de_3(decoder))
		decoder = F.interpolate(decoder, scale_factor=2)
		decoder = self.de_4(decoder)
		return decoder

class CAEAD():
	def __init__(self, input_size, batch_size, learning_rate, epochs, device, optimizer, normal_only=True):
		super(CAEAD, self).__init__()
		self.epochs = epochs
		self.device = device
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.model = _CAEAD(input_size)
		print(self.model)
		self.normal_only = normal_only

	def _loss(self, output, target, y):
		output = torch.squeeze(output)
		target = torch.squeeze(target)
		mse = torch.mean((output - target)**2, dim=1)
		loss = ((1-y)*mse) - (y*(mse))
		return torch.mean(loss)

	def fit(self, X, y, X_val=None, y_val=None):
		X = X.reshape(-1,1, X.shape[-1])
		X_tensor, y_tensor = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
		train_dataset = TensorDataset(X_tensor, y_tensor)
		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		self.model.train()
		optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
		for epoch in range(1, self.epochs+1):
			loss = self._iter(train_loader, optimizer)
			if X_val is not None:
				X_val = X_val.reshape(-1,1, X_val.shape[-1])
				mses = self.predict(X_val)
			mses = self.predict(X)
			if self.normal_only: 
				print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss))
			else:
				train_auc = roc_auc_score(y, mses)
				print('Train Epoch: {}\tLoss: {:.6f}\tTrain auc: {:.6f}'.format(epoch, loss, train_auc))
			break
		return self

	def _iter(self, train_loader, optimizer):
		losses=[]
		if self.normal_only:
			cost = nn.MSELoss()
		else: 
			cost = self._loss


		# with tqdm(total=len(train_loader)) as progress_bar:
		for batch_idx, (data, label) in enumerate(train_loader):
			# data = data.view(data.size(0), -1)
			data = data.to(self.device)
			label = label.to(self.device)
			optimizer.zero_grad()
			output = self.model(data)
			if self.normal_only:
				loss = cost(output, data)  
			else:
				loss = cost(output, data, label)      
			loss.backward()
			optimizer.step()
			losses.append(loss.item())            
			# progress_bar.update(1)
		return np.mean(losses)

	def predict(self, X, checkpoint=None):
		X = X.reshape(-1,1, X.shape[-1])
		X = torch.from_numpy(X.astype(np.float32))
		dataset = TensorDataset(X)
		dataloader= DataLoader(dataset, batch_size=self.batch_size)
		if checkpoint:
			model_state = torch.load(checkpoint)
			self.model.load_state_dict(model_state)    
		self.model.eval()
		preds = []
		targets = []
		labels = []
		with torch.no_grad():
			for batch_idx, data in enumerate(dataloader):

				data = data[0]
				data = data.to(self.device)
				# Targets are the inputs to the network. 
				target = data.clone()
				output = self.model(data)
				preds.append(np.squeeze(output.cpu().numpy()))
				targets.append(np.squeeze(target.cpu().numpy()))
		preds = np.concatenate(preds)
		targets = np.concatenate(targets)
		errors = np.mean((preds - targets)**2, axis=(1)) 
		return errors
