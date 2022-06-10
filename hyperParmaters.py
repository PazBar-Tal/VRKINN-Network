import torch
import torch as tr
import configuration as cfg
from torch.utils.data import DataLoader
import optuna
import torch.nn as nn
from model import VRKINNN
from DataPreparation import dataPreparation
from pytorchtools_stoping import EarlyStopping

train_dataset, test_dataset, validation_dataset = dataPreparation()
device = 'cuda' if tr.cuda.is_available() else 'cpu'
print(f"we are working on: {device}")


def eval(outputs_to_eval, label):
    label_hat = tr.round(outputs_to_eval)
    correct = tr.eq(label, label_hat)
    num_of_correct = tr.sum(correct)
    return num_of_correct.item()



# training and val loss were calculated after every epoch
def train(model, train_dataset,val_dataset, optim, num_epochs, criterion, lr,BATCH_SIZE, patience=cfg.patience):
    train_losses = []
    val_losses = []

    train_acc_total = []

    val_acc_total = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=True, pin_memory=True)


    parameters = model.parameters()
    # define the optimizer
    if optim == "SGD":
        optimizer = tr.optim.SGD(parameters, lr=lr)
    elif optim == "ADAM":
        optimizer = tr.optim.Adam(parameters, lr=lr)


    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_trails = 0
        i = 0
        for i, (room1, room2, labels) in enumerate(train_loader, 0):
            model.train()

            room1 = room1.to(device)
            room2 = room2.to(device)
            label = labels[0]
            label = label.to(device)
            label = label[:, None]
            label = label.to(torch.float64)

            outputs = model(room1, room2)
            outputs_train_to_eval = outputs

            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct = eval(outputs_train_to_eval, label)
            correct_trails += correct




        avg_train_acc = (correct_trails) / len(train_dataset)
        train_acc_total.append(avg_train_acc)
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        # check val loss after every epoch
        with torch.no_grad():
            model.eval()

            val_running_loss = 0.0
            for i, (room1, room2, labels) in enumerate(val_loader, 0):

                room1 = room1.to(device)
                room2 = room2.to(device)
                label = labels[0]
                label = label.to(device)
                label = label[:, None]
                label = label.to(torch.float64)

                outputs = model(room1, room2)
                outputs_val_to_eval = outputs

                loss = criterion(outputs, label)
                val_running_loss += loss.item()
                correct_val = eval(outputs_val_to_eval, label)
                acc_now_val = correct_val / len(val_dataset)

        early_stopping(val_running_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        val_acc_total.append(acc_now_val)
        val_losses.append(val_running_loss)



    return val_losses

def objective(trial):

    BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 5,30)
    num_of_filters1 = trial.suggest_int("num_of_filters1", 5, 50 )
    num_of_filters2 = trial.suggest_int("num_of_filters2",5,50, step = 5)
    drop_out1 = trial.suggest_float("drop_out1",0.1, 0.7,step=0.1)
    drop_out2 = trial.suggest_float("drop_out2",0.1, 0.7,step=0.1)
    output_fc1 = trial.suggest_int("output_fc1",5,20)
    output_fc2 = trial.suggest_int("output_fc2",5,20)
    hidden_size_lstm = trial.suggest_int("hidden_size_lstm", 1, 3)
    lr = trial.suggest_float("lr",1e-5, 1e-1, log=True)
    criterion = nn.BCELoss(reduction='mean')
    train_dataset, test_dataset, validation_dataset = dataPreparation()
    model = VRKINNN(BATCH_SIZE = BATCH_SIZE,num_of_filters1 = num_of_filters1,num_of_filters2 =num_of_filters2,karnel1 =cfg.karnel1,stride1 =cfg.stride1,karnel2=cfg.karnel2,stride2= cfg.stride2,drop_out1 = drop_out1,drop_out2 = drop_out2,output_fc1 = output_fc1,output_fc2 = output_fc2,hidden_size_lstm = hidden_size_lstm)

    return loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100  )
best_parmas = study.best_params
print(f"Optimized parameters: {best_parmas}\n") # without LSTM Optimized parameters: {'BATCH_SIZE': 13, 'num_of_filters1': 38, 'num_of_filters2': 45, 'drop_out1': 0.5, 'drop_out2': 0.6, 'output_fc1': 5, 'output_fc2': 16, 'hidden_size_lstm': 2, 'lr': 0.00981755150307507}
#WITH LSTM :Optimized parameters: {'BATCH_SIZE': 28, 'num_of_filters1': 34, 'num_of_filters2': 25, 'drop_out1': 0.2, 'drop_out2': 0.4, 'output_fc1': 20, 'output_fc2': 5, 'hidden_size_lstm': 2, 'lr': 7.983877248912919e-05}
#WITH LSTM ONLY POSITION: Optimized parameters: {'BATCH_SIZE': 11, 'num_of_filters1': 23, 'num_of_filters2': 20, 'drop_out1': 0.5, 'drop_out2': 0.30000000000000004, 'output_fc1': 9, 'output_fc2': 14, 'hidden_size_lstm': 3, 'lr': 2.192740451019604e-05}
# 100 trails : Optimized parameters: {'BATCH_SIZE': 11, 'num_of_filters1': 34, 'num_of_filters2': 15, 'drop_out1': 0.1, 'drop_out2': 0.6, 'output_fc1': 20, 'output_fc2': 10, 'hidden_size_lstm': 3, 'lr': 1.0322290446428317e-05}