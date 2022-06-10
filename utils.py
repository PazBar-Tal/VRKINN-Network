import torch

alteration_train_list = torch.load('alteration_train_list.pt')
alteration_test_list = torch.load('alteration_test_list.pt')
alteration_val_list = torch.load('alteration_val_list.pt')


## need to change the num of data points according the changes after the down samplpling 
def paired_rooms(room, positive,num_data_points=450):  # positive = 0 make pairs of diff rooms positive =1 make pairs of samiler rooms
    paired_rooms = []
    paired_labels = []
    paired_roomsID = []
    check_size =torch.zeros(1 ,6, num_data_points)
    if positive == 1:
        for roomID in range(room.size(dim=0)):
            for room2ID in range(roomID + 1, room.size(dim=0)):
                # Room position - dim 1, room rotation dim 2
                room1 = room[roomID]
                room2 = room[room2ID]


                if room1.size() != check_size.size() or room2.size() != check_size.size():
                    Exception
                paired_rooms.append([room1, room2])
                paired_roomsID.append([roomID, room2ID])
                paired_labels.append([1])

    else:
        the_middele = int(room.size(dim=0) / 2)
        for roomID in range(the_middele):
            for room2ID in range(the_middele, room.size(dim=0)):
                room1 = room[roomID]
                room2 = room[room2ID]


                if room1.size() != check_size.size() or room2.size() != check_size.size():
                    Exception
                paired_rooms.append([room1, room2])
                paired_roomsID.append([roomID, room2ID])
                paired_labels.append([0])

    return paired_rooms, paired_labels, paired_roomsID

# the function take one alteration type tensor and make positive and negative pairs
def make_pairs(room, middle, numOftrails):
    pairs = []
    labels = []
    baseline = room[0:middle]
    alterted = room[middle:numOftrails]
    # create positive pairs
    try:
        baseline_pairs, labelsB, IDB = paired_rooms(baseline, 1)
    except:
        exit("not in the right len! -1")
    try:
        alterted_pairs, labelsA, IDA = paired_rooms(alterted, 1)
    except:
        exit("not in the right len! -2")
    # create negative pairs
    try:
        different_pairs, labelsD, IDD = paired_rooms(room, 0)
    except:
        exit("not in the right len! -3")
    pairs = baseline_pairs + alterted_pairs + different_pairs
    labels = labelsB + labelsA + labelsD
    return pairs, labels


def all_tensor_paired(listOfTensors):
    all_paired_rooms = []
    all_labels = []

    for i in range(len(listOfTensors)):
        rooms_data, rooms_label = make_pairs(listOfTensors[i], int(listOfTensors[i].size(0) / 2),
                                             int(listOfTensors[i].size(0)))
        all_paired_rooms = all_paired_rooms + rooms_data
        all_labels = all_labels + rooms_label
    return all_paired_rooms, all_labels



[train_list, labels_train_list] = all_tensor_paired(alteration_train_list)
[test_list, labels_test_list] = all_tensor_paired(alteration_test_list)
[val_list, labels_val_list] = all_tensor_paired(alteration_val_list)

torch.save(train_list, 'train_list.pt')
torch.save(labels_train_list, 'labels_train_list.pt')
torch.save(test_list, 'test_list.pt')
torch.save(labels_test_list, 'labels_test_list.pt')
torch.save(val_list, 'val_list.pt')
torch.save(labels_val_list, 'labels_val_list.pt')