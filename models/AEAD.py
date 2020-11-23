import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

class _AEAD(nn.Module):
	def __init__(self, input_size):
		super(_AEAD, self).__init__()
		self.en_1 = nn.Linear(input_size, 128)
		self.en_2 = nn.Linear(128, 64)
		self.en_3 = nn.Linear(64, 32)
		
		self.de_1= nn.Linear(32, 64)
		self.de_2 = nn.Linear(64, 128)
		self.de_3 = nn.Linear(128, input_size)

	def forward(self, X):

		encoder  = F.relu(self.en_1(X))
		encoder = F.relu(self.en_2(encoder))
		encoder = F.relu(self.en_3(encoder))
		 
		decoder = F.relu(self.de_1(encoder))
		decoder = F.relu(self.de_2(decoder))
		decoder = self.de_3(decoder)
		return decoder

class AEAD():
	def __init__(self, input_size, batch_size, learning_rate, epochs, device, optimizer, normal_only=True):
		super(AEAD, self).__init__()
		self.epochs = epochs
		self.device = device
		self.optimizer = optimizer
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.model = _AEAD(input_size)
		self.normal_only = normal_only

	def _loss(self, output, target, y):
		mse = torch.mean((output - target)**2, dim=1)
		loss = ((1-y)*mse) - (y*(mse))
		return torch.mean(loss)

	def fit(self, X, y, X_val=None, y_val=None):
		X_tensor, y_tensor = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
		train_dataset = TensorDataset(X_tensor, y_tensor)
		train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		self.model.train()
		optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
		for epoch in range(1, self.epochs+1):
			loss = self._iter(train_loader, optimizer)
			if X_val is not None:
				mses = self.predict(X_val)
			mses = self.predict(X)
			if self.normal_only: 
				print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss))
			else:
				train_auc = roc_auc_score(y, mses)
				print('Train Epoch: {}\tLoss: {:.6f}\tTrain auc: {:.6f}'.format(epoch, loss, train_auc))
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
				preds.append(output.cpu().numpy())
				targets.append(target.cpu().numpy())
		preds = np.concatenate(preds)
		targets = np.concatenate(targets)
		errors = np.mean((preds - targets)**2, axis=(1)) 
		return errors
