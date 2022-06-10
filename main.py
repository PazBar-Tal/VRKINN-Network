from DataPreparation import dataPreparation
from model import VRKINNN
from training import train
import configuration as cfg
import torch.nn as nn
import matplotlib.pyplot as plt


count = 0
# loading the data and split it to train, validation and test sets
train_dataset, test_dataset, validation_dataset = dataPreparation()
for i in range(len(validation_dataset.label)):
    if train_dataset.label[i] == [0]:
        count = count + 1


# creating our best performence model
model = VRKINNN(BATCH_SIZE = cfg.BATCH_SIZE,num_of_filters1 = cfg.num_of_filters1,num_of_filters2 =cfg.num_of_filters2,karnel1 =cfg.karnel1,stride1 =cfg.stride1,karnel2=cfg.karnel2,stride2= cfg.stride2,drop_out1 = cfg.drop_out1,drop_out2 = cfg.drop_out2,output_fc1 = cfg.output_fc1,output_fc2 = cfg.output_fc2,hidden_size_lstm = cfg.hidden_size_lstm) # add the variables

# define loss function
criterion = nn.BCELoss(reduction='mean')

# train the model with his optimal hyperparameters
train_losses, test_losses, acc_train, acc_test,epoch, error_ratio = train(model, train_dataset, test_dataset ,optim="ADAM", num_epochs=cfg.EPOCHS, lr=1.0322290446428317e-05, criterion=criterion)
plt.subplot(2, 2, 1)
plt.plot(list(range(0, len(train_losses) )), train_losses)
plt.title('Train Error')
plt.ylim(0,0.8)


plt.subplot(2, 2, 2)
plt.plot(list(range(0, len(test_losses))), test_losses)
plt.title('Test Error')
plt.ylim(0,0.8)


plt.subplot(2, 2, 3)
plt.plot(list(range(0, len(acc_train))), acc_train)
plt.title('Acc Train')
plt.ylim(0.5,1)

plt.subplot(2, 2, 4)
plt.plot(list(range(0, len(acc_test))), acc_test)
plt.title('Acc Test')
plt.ylim(0.5,1)
plt.show()


plt.plot(list(range(0, len(error_ratio-1) )), error_ratio[1:])
plt.title('Error Ratio in Percentage')
plt.ylabel('Percentage of change [%]')
plt.show()