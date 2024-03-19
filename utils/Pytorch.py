import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from IPython.display import clear_output
from utils.tools import *
from utils.metrics import Time

"""
Pytorch Forecasting
"""

def selectFeatures(X_train, X_val, lags, exog, lagColName):
    columns_to_keep = []
    if len(lags) == 0 and not exog:
        print("No data")
        return np.nan, np.nan
    
    for col in X_train.columns:
        if col.startswith(lagColName):
            if int(col[len(lagColName):]) in lags:
                columns_to_keep.append(col)
        elif exog:
            columns_to_keep.append(col)

    return X_train[columns_to_keep], X_val[columns_to_keep]


def evaluate_NN(model, loader, criterion, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_idx, (Xbatch, ybatch) in enumerate(loader):
            Xbatch = Xbatch.to(device)
            ybatch = ybatch.to(device)
                
            predictions = model(Xbatch.unsqueeze(-1)).squeeze()

            loss = criterion(predictions, ybatch)
            total_loss.append(loss.item())
    return np.mean(total_loss)


def train_NN(model, trainLoader, valLoader, totalEpochs=100, lr=1e-3, device="cpu", save_loss=False, save_best=False, reestimate=False):
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    train_loss_list = []
    val_loss_list = []
    nEpochs = range(1, totalEpochs + 1)
    best_val_loss = np.inf
    for epoch in nEpochs:
        model.train()
        total_loss = []
        for batch_idx, (Xbatch, ybatch) in enumerate(trainLoader):
            optimizer.zero_grad()

            Xbatch = Xbatch.to(device)
            ybatch = ybatch.to(device)

            predictions = model(Xbatch.unsqueeze(-1)).squeeze()

            loss = criterion(predictions, ybatch)
                
            loss.backward()

            optimizer.step()

            total_loss.append(loss.item())

        total_loss_mean = np.mean(total_loss)
        train_loss_list.append(total_loss_mean)

        val_loss_mean = evaluate_NN(model, valLoader, criterion, device)
        val_loss_list.append(val_loss_mean)

        if val_loss_mean < best_val_loss and save_best:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss_mean

        if epoch % 10 == 0 and not reestimate:
            print(f"Epoch: {epoch}, Train Loss: {total_loss_mean}, Val Loss: {val_loss_mean}")

    train_loss_list = np.array(train_loss_list)
    val_loss_list = np.array(val_loss_list)

    if save_loss:
        np.save(train_loss_list, "train_loss.npy")
        np.save(val_loss_list, "val_loss.npy")
    if save_best:
        model = copy.deepcopy(best_model)
        
    model = model.to("cpu")

    return model, nEpochs, train_loss_list, val_loss_list


def scaleData(X_train, y_train, X_val, y_val):
    scaler_X = StandardScaler()  # Initialize a scaler_X
    X_train_scaled = scaler_X.fit_transform(X_train)                    # scaler_X is fitted to the columns of X_train
    X_val_scaled = scaler_X.transform(X_val)                            # scaler_X is used to transform the columns of X_val
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)  # The scaled values are then type np.array. We also save as dataframe
    X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)        # for both train and val

    scaler_y = StandardScaler()  # Initialize a scaler_y
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))  # scaler_y is fittet to y_train
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))          # scaler_y is used to transform y_val

    return scaler_X, scaler_y, X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, X_train_df, X_val_df


def createTensors(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled):
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled).squeeze(-1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled).squeeze(-1)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor


def createDataLoaders(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, batchSize):
    trainDataset = TensorDataset(X_train_tensor, y_train_tensor)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False)
    valDataset = TensorDataset(X_val_tensor, y_val_tensor)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

    return trainLoader, valLoader


def removeGTLags(col, lagColName):
    if lagColName in col.name:
        lag = int(col.name[len(lagColName):])
        col.loc[lag:] = np.nan


def updateLagsWithRealValues(col, lagColName, realValues, horizon):
    if lagColName in col.name:
        latest_realValues = realValues[:horizon]
        lag = int(col.name[len(lagColName):])

        start_val = np.max([0, lag - horizon])
        end_val = np.min([lag, horizon])
        for i in range(end_val):
            col.iloc[(start_val+i):(start_val+i+1)] = latest_realValues[-(end_val-i)]


def predictHorizonSteps(X_val_df, y_val_pred_scaled, model, horizon):
    steps = np.min([horizon, len(X_val_df)])  # Find how many steps to forecast (always length of horizon except for last iteration)
    for _ in range(0, steps):
        X_fit_tensor = torch.FloatTensor(X_val_df.iloc[0:1].values)  # Select the first row in the validation set and turn it into tensor
        y_val_pred_scaled.append(model(X_fit_tensor.unsqueeze(-1)).detach().squeeze().numpy().tolist())  # Use the estimated model to predict the next value from the first row tensor, and add it to predictions

        X_val_df = X_val_df.apply(lambda col: col.fillna(y_val_pred_scaled[-1], limit=1))             # Add the next row in each lag column
        X_val_df = X_val_df.drop(X_val_df.index[0]).reset_index(drop=True)  # Drop the row that was just used and reset the index of the dataframe

    return X_val_df, y_val_pred_scaled


def forecast(windowStrategy, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_val_df, y_gt, model, batchSize, epochs, lr, device, lagColName, horizon):
    y_val_pred_scaled = []  # Initialize a prediction list for predictions
    iterations = 0  # Keep track of iterations

    # While there is still values left in the validation set to be forecasted
    while len(X_val_df) != 0:
        # Update iteration and print
        iterations += 1
        print(iterations)
        clear_output(wait=True)

        # Predict the next <horizon> values and update the lags of the input data with the predicted values
        X_val_df, y_val_pred_scaled = predictHorizonSteps(X_val_df,
                                                          y_val_pred_scaled,
                                                          model=model,
                                                          horizon=horizon)

        # After forecasting the steps
        X_val_df.apply(updateLagsWithRealValues, lagColName=lagColName, realValues=y_gt, horizon=horizon)  # Call the updateLagsWithRealValues function on all columns, which inserts the ground truth values in the lags instead of the predicted values
        y_gt = y_gt[horizon:]  # Remove the <horizon> ground truth values that have been inserted into the lags

        # Change model according to the used window strategy
        X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, model = windowStrategy(X_train_tensor,
                                                                                           y_train_tensor,
                                                                                           X_val_tensor,
                                                                                           y_val_tensor,
                                                                                           model=model,
                                                                                           batchSize=batchSize,
                                                                                           epochs=epochs,
                                                                                           lr=lr,
                                                                                           device=device,
                                                                                           horizon=horizon)

    return y_val_pred_scaled


def forecastPytorch(windowStrategy, X_train, y_train, X_val, y_val, model, batchSize, epochs, lr, device, lagColName, horizon):

    _, scaler_y, X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, _, X_val_df = scaleData(X_train,
                                                                                                     y_train,
                                                                                                     X_val,
                                                                                                     y_val)

    # Turn the scaled lists into tensors
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = createTensors(X_train_scaled,
                                                                               y_train_scaled,
                                                                               X_val_scaled,
                                                                               y_val_scaled)

    # Turn the tensors into dataloaders
    initial_trainLoader, initial_valLoader = createDataLoaders(X_train_tensor,
                                                               y_train_tensor,
                                                               X_val_tensor,
                                                               y_val_tensor,
                                                               batchSize=batchSize)

    # Estimate NN model from training data
    estimated_model, epoch_range, train_loss_list, val_loss_list = train_NN(model,
                                                                            initial_trainLoader,
                                                                            initial_valLoader,
                                                                            totalEpochs=epochs,
                                                                            lr=lr,
                                                                            device=device,
                                                                            save_best=False)
    
    # Get fitted values from train
    y_train_pred_scaled = estimated_model(X_train_tensor.unsqueeze(-1)).detach().squeeze().numpy().tolist()

    X_val_df.apply(removeGTLags, lagColName=lagColName)  # Calls the removeGTLags function on all columns, which removes the lags that would not yet be known from the starting point of the forecasting
    y_gt = y_val_scaled.squeeze().tolist()  # Initiate y_gt as the ground truth observations

    y_val_pred_scaled = forecast(windowStrategy,
                                 X_train_tensor,
                                 y_train_tensor,
                                 X_val_tensor,
                                 y_val_tensor,
                                 X_val_df,
                                 y_gt,
                                 model=copy.deepcopy(estimated_model),
                                 batchSize=batchSize,
                                 epochs=epochs,
                                 lr=lr,
                                 device=device,
                                 lagColName=lagColName,
                                 horizon=horizon)
    
    # Inverse transform the scaled predictions and output as type pd.Series
    y_train_pred = pd.Series(scaler_y.inverse_transform(np.array(y_train_pred_scaled).reshape(-1, 1)).squeeze())
    y_val_pred = pd.Series(scaler_y.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1)).squeeze())
    return estimated_model, y_train_pred, y_val_pred, epoch_range, train_loss_list, val_loss_list


def fixedWindowPytorch(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, batchSize, epochs, lr, device, horizon):
    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, model


def expandingWindowPytorch(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, batchSize, epochs, lr, device, horizon):
    # Add the next <horizon> predicted steps of validation data to the training data and remove it from the validation data
    X_train_tensor = torch.cat((X_train_tensor, X_val_tensor[:horizon]), dim=0)
    X_val_tensor = X_val_tensor[horizon:]
    y_train_tensor = torch.cat((y_train_tensor, y_val_tensor[:horizon]), dim=0)
    y_val_tensor = y_val_tensor[horizon:]

    # Turn the updated tensors into dataloaders
    trainLoader, _ = createDataLoaders(X_train_tensor,
                                       y_train_tensor,
                                       X_val_tensor,
                                       y_val_tensor,
                                       batchSize=batchSize)
    
    # Reestimate NN model from training data + next <horizon> predicted steps of validation data
    reestimated_model, _, _, _ = train_NN(model,
                                          trainLoader,
                                          totalEpochs=epochs,
                                          lr=lr,
                                          device=device,
                                          save_best=False,
                                          reestimate=True)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, reestimated_model


def rollingWindowPytorch(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, model, batchSize, epochs, lr, device, horizon):
    # Add the next <horizon> predicted steps of validation data to the training data, remove it from the validation data and remove the first <horizon> steps of training data
    X_train_tensor = torch.cat((X_train_tensor, X_val_tensor[:horizon]), dim=0)
    X_val_tensor = X_val_tensor[horizon:]
    X_train_tensor = X_train_tensor[horizon:]
    y_train_tensor = torch.cat((y_train_tensor, y_val_tensor[:horizon]), dim=0)
    y_val_tensor = y_val_tensor[horizon:]
    y_train_tensor = y_train_tensor[horizon:]

    # Turn the updated tensors into dataloaders
    trainLoader, _ = createDataLoaders(X_train_tensor,
                                       y_train_tensor,
                                       X_val_tensor,
                                       y_val_tensor,
                                       batchSize=batchSize)
    
    # Reestimate NN model from training data + next <horizon> predicted steps of validation data
    reestimated_model, _, _, _ = train_NN(model,
                                          trainLoader,
                                          totalEpochs=epochs,
                                          lr=lr,
                                          device=device,
                                          save_best=False,
                                          reestimate=True)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, reestimated_model