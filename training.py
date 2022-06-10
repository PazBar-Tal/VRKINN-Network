import torch
import torch as tr
import configuration as cfg
from torch.utils.data import DataLoader




def eval(outputs_to_eval, label):
    label_hat = tr.round(outputs_to_eval)
    correct = tr.eq(label, label_hat)
    num_of_correct = tr.sum(correct)
    return num_of_correct.item()



# training and test loss were calculated after every epoch
def train(model, train_dataset,test_dataset, optim, num_epochs, criterion, lr, patience=cfg.patience):
    train_losses = []
    test_losses = []

    train_acc_total = []

    test_acc_total = []
    error_ratio = []
    trigger_times =0

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True, pin_memory=True)

    parameters = model.parameters()
    # define the optimizer
    if optim == "SGD":
        optimizer = tr.optim.SGD(parameters, lr=lr)
    elif optim == "ADAM":
        optimizer = tr.optim.Adam(parameters, lr=lr)
    last_lost = 100
    for epoch in range(num_epochs):

        running_loss = 0.0
        correct_trails = 0
        print("Starting epoch " + str(epoch + 1))
        i = 0
        for i, (room1, room2, labels) in enumerate(train_loader, 0):
            model.train()

            room1 = room1.to(device)
            room2 = room2.to(device)
            label = labels[0]
            label = label.to(device)
            label = label[:, None]
            label = label.to(torch.float64)

            outputs_train = model(room1, room2)
            outputs_train_to_eval = outputs_train

            loss = criterion(outputs_train, label)

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

        error_ratio_now = (100 - (avg_train_loss / last_lost) * 100)
        if error_ratio_now < 0.1:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= cfg.patience:
                print('Early Stopping!\nStart to test process.')
                break

        else:
            print(''.join(['Trigger Times: ', str(trigger_times), '\nloss decreased (',str(last_lost),' --> ', str(avg_train_loss), '.)']))
            trigger_times = 0

        error_ratio.append((error_ratio_now))
        last_lost = avg_train_loss

        # check test loss after every epoch
        with torch.no_grad():
            model.eval()

            test_running_loss = 0.0
            for i, (room1, room2, labels) in enumerate(test_loader, 0):
                # Forward


                room1 = room1.to(device)
                room2 = room2.to(device)
                label = labels[0]
                label = label.to(device)
                label = label[:, None]
                label = label.to(torch.float64)

                outputs_test = model(room1, room2)
                outputs_test_to_eval = outputs_test

                loss = criterion(outputs_test, label)
                test_running_loss += loss.item()
                correct_test = eval(outputs_test_to_eval, label)
                acc_now_test = correct_test / len(test_dataset)




        test_acc_total.append(acc_now_test)
        test_losses.append(test_running_loss)
        print('Epoch [{}/{}],Train Loss: {:.4f}, test Loss: {:.8f}'
              .format(epoch + 1, num_epochs, avg_train_loss, test_running_loss))
        print( "train acc : ",train_acc_total[epoch],"test acc:",test_acc_total[epoch])
    print("Finished Training")
    return train_losses, test_losses, train_acc_total, test_acc_total, epoch, error_ratio



device = 'cuda' if tr.cuda.is_available() else 'cpu'
print(f"we are working on: {device}")
