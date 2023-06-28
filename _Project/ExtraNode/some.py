import time
import datetime
import torch
import sys
import os
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from customdataset_2 import CustomImageDataset
from VGG_kcc_for_train_2 import VGG
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_210820_2 import visualize_features

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


now = time.localtime()
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


print("[%04d/%02d/%02d %02d:%02d:%02d] Training_start" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))


hyper_param_epoch = 500
hyper_param_batch = 50
hyper_param_learning_rate = 0.01


class_number = 10 # For making a toy example only in Nth class. If the class_number is 10, the toy example is made in full color.

model_number = 2
model_name = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
print("model_name : ", model_name[model_number])
# path = './weights/init_weight_original_210429_(VGG16)_cifar10.pth'
# path = './weights/init_weight_2node_210821_~.pth' # load the initial weight

image_transforms = 1 # 0 : resize, 1 : randomcrop

optimizer_select = 1 # Adam == 0, SGD == 1
scheduler_select = 1 # 0 : not use, 1 : Use the CosineAnnealingLR

t_max = 50 # scheduler

toy_example_mode = 0 # 0 : Basic, 1 : fixed one class, 2 : fixed two class
print("toy_example_mode :",toy_example_mode)

angle_number = 1 #This part is used only when toy_example_model is 0.
# added_angle = [0, 30, 60, 90, 120, 180]
# added_angle = [0, 90, 180, 270]
added_angle = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]
print("added_angle :",added_angle[angle_number])

exchange_number = 0 # Number(1~9) for exchange between each coordinate(index 0 means the initial shape)
# ['bird', 'airplane', 'dog', 'cat', 'ship', 'frog', 'automobile', 'truck', 'deer', 'horse']
print("exchange_number :",exchange_number)

save_folder_name = f'202.cifar10_freezing_{model_name[model_number]}_2node_{added_angle[angle_number]}_210917'
weights_path = fr'./weights/{save_folder_name}'
os.makedirs(weights_path, exist_ok=True)



epoch_startpoint = 0

#Load the initial model
model_path = "./weights/init_model/init_weight_original_210419_(VGG16)_cifar10.pth"
# model_path = "./weights/init_model/init_weight_original_210419_(VGG19)_cifar10.pth"



# Load the pretrained model
# model_path = f'./weights/{save_folder_name}/train_weight_199_(VGG16).pth'

# shape_number = 0
# shape_select = ["basic", "fixed_one", "fixed_two"]


# !!! include cd and then start training!!!!

def main():

    if image_transforms == 0 :
        transforms_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                    std=[0.2023, 0.1994, 0.2010]), ])
        transforms_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                   std=[0.2023, 0.1994, 0.2010]), ])


    else :
        transforms_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                    std=[0.2023, 0.1994, 0.2010]), ])

        transforms_test = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                   std=[0.2023, 0.1994, 0.2010]), ])

    train_data_set = CustomImageDataset(data_set_path="/home/prml/PSH/SH_dataset/cifar10/train",
                                        transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path="/home/prml/PSH/SH_dataset/cifar10/test",
                                       transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = train_data_set.num_classes
    class_names = train_data_set.class_names

    print("num_classes :", num_classes)
    print("classe_names :", class_names)

    hidden_model = VGG(m_name=model_name[model_number], num_classes=num_classes, device=device, batch_size=hyper_param_batch)
    hidden_model.load_state_dict(torch.load(model_path)) #model file load

    # print(hidden_model)

    # print("before changed_weight :", hidden_model.classifier2.weight.data.cpu().numpy())
    # print("bias 1 :", hidden_model.classifier2.bias.data.cpu().numpy())
    # print("before changed_weight :", hidden_model.classifier2.weight.data.cpu().numpy())
    # print("before changed_bias :", hidden_model.classifier2.bias.data.cpu().numpy())
    # print("True in weight :", hidden_model.classifier2.weight.requires_grad)
    # print("True in bias :", hidden_model.classifier2.bias.requires_grad)



    if toy_example_mode == 0 :

        changed_weight = []
        for a in range(0, 360, 360 // num_classes):
            phi = np.deg2rad(a + added_angle[angle_number])
            x = np.cos(phi)
            y = np.sin(phi)
            changed_weight.append([x, y])
        # print("changed_weight :",changed_weight)
        # print("changed_weight :", type(changed_weight))

    elif toy_example_mode == 1:
        angle = np.linspace(-45, 45, 9)
   
        angle_list = angle.tolist()

        print(type(angle_list[0]))
        # for i in range(4): # change negative angle with postive angle
        #     angle_list[i] += 360

        changed_weight = []
        for a in range(num_classes):
            if a == 0:

                phi = np.deg2rad(180)
                x = np.cos(phi)
                y = np.sin(phi)
                # fixed_location = [x, y]
                changed_weight.append([x, y])
                # print("180:",changed_weight)
            else:
                phi = np.deg2rad(angle_list[a - 1])
                x = np.cos(phi)
                y = np.sin(phi)
                changed_weight.append([x, y])

        # print("before_changed_weight :", changed_weight)
        if exchange_number == 0:
            pass
            print("<<<<<the initial shape>>>>>")
        else :

            print("<<<<<the exchanged shape>>>>>")
            exchange_a = copy.deepcopy(changed_weight[0]) # 180 degree coordinates
            exchange_b = copy.deepcopy(changed_weight[exchange_number]) # the remaining coordinates

            changed_weight[0] = exchange_b
            changed_weight[exchange_number] = exchange_a


            # changed_weight.insert(insert_number, fixed_location)
            # print("after_changed_weight :", changed_weight)
            # print("changed_weight :", len(changed_weight))




    hidden_model.classifier2.weight = torch.nn.Parameter(torch.FloatTensor(changed_weight))
    hidden_model.classifier2.bias = torch.nn.Parameter(torch.zeros(num_classes))

    # print("after changed_weight :", hidden_model.classifier2.weight.data)
    # print("after changed_bias :", hidden_model.classifier2.bias.data)

    for param in hidden_model.classifier2.parameters():  # freezing weight
        param.requires_grad = False

    print("false in weight :", hidden_model.classifier2.weight.requires_grad)
    print("false in bias :", hidden_model.classifier2.bias.requires_grad)



    hidden_model.to(device)


    #<<<<< Loss and optimizer >>>>>
    writer = SummaryWriter(f'runs/{save_folder_name}')
    criterion = nn.CrossEntropyLoss()

    if optimizer_select == 0 :

        optimizer = torch.optim.Adam(hidden_model.parameters(), lr=hyper_param_learning_rate, weight_decay=5e-4)
        print("optimizer : Adam")

    else :
        optimizer = torch.optim.SGD(hidden_model.parameters(), lr=hyper_param_learning_rate, momentum=0.9, weight_decay=5e-4)
        print("optimizer : SGD")

    if scheduler_select == 0:
        print("scheduler is disabled")

    else:

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        print(f"scheduler : CosineAnnealingLR(t_max {t_max})")


    num_iter_per_epoch = len(train_loader)
    best_acc = 0
    try:
        for e in range(epoch_startpoint, hyper_param_epoch):
            per_epoch_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader)
            hidden_model.train()

            features_loader_train = []
            labels_loader_train = []



            # with torch.no_grad():
            #     hidden_model.classifier2.weight = torch.nn.Parameter(torch.randn(10, 2))
            #
            #     print("weight2 :", hidden_model.classifier2.weight.data.cpu().numpy())
            for i_batch, item in enumerate(progress_bar):

                images = item['image'].to(device)
                labels = item['label'].to(device)



                # <<<<< Forward pass >>>>

                features, outputs = hidden_model(images)

                # print("outputs :",outputs.shape)

                # hidden_output = hidden_model(pretrain_out=features)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                total += len(labels)
                correct += (predicted == labels).sum().item()

                # <<<<< Backward and optimize >>>>>
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                features_loader_train.append(features)
                labels_loader_train.append((labels))

                now = time.localtime()
                current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

                writer.add_scalar(f"Loss of train({model_name[model_number]})", loss, e)

                progress_bar.set_description(
                    '[{}] Epoch: {}/{}. Iteration: {}/{}. current batch loss: {:.4f}. '.format(
                        current_time, e + 1, hyper_param_epoch, i_batch + 1, num_iter_per_epoch, loss.item()))

                per_epoch_loss += loss.item()

            train_feat = torch.cat(features_loader_train, 0)
            train_labels = torch.cat(labels_loader_train, 0)

            train_epoch_acc = round((100 * correct / total), 2)
            visualize_features(hidden_model, train_feat.data.cpu().numpy(), train_labels.data.cpu().numpy(), e,
                               class_names, added_angle[angle_number], model_name[model_number],save_folder_name ,
                               toy_example_mode, exchange_number ,mode='train', class_number=class_number)


            if (e) % 3 == 0:

                hidden_model.eval()
                features_loade_test = []
                labels_loader_test = []

                with torch.no_grad():
                    correct_eval = 0
                    total_eval = 0

                    for item in test_loader:
                        images_eval = item['image'].to(device)
                        labels_eval = item['label'].to(device)

                        hidden_features, hidden_outputs = hidden_model(images_eval)

                        _, predicted_eval = torch.max(hidden_outputs.data, 1)

                        total_eval += len(labels_eval)
                        correct_eval += (predicted_eval == labels_eval).sum().item()

                        features_loade_test.append(hidden_features)
                        labels_loader_test.append((labels_eval))

                    test_feat = torch.cat(features_loade_test, 0)
                    test_labels = torch.cat(labels_loader_test, 0)
                    visualize_features(hidden_model, test_feat.data.cpu().numpy(), test_labels.data.cpu().numpy(), e,
                                       class_names,added_angle[angle_number], model_name[model_number], save_folder_name,toy_example_mode, exchange_number, mode='validation', class_number=class_number)


                    test_acc = round((100 * correct_eval / total_eval), 2)
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(hidden_model.state_dict(),
                               f'./weights/{save_folder_name}/train_weight_{e + 1}_({model_name[model_number]})_best.pth')
                else:
                    torch.save(hidden_model.state_dict(),
                               f'./weights/{save_folder_name}/train_weight_{e + 1}_({model_name[model_number]}).pth')

                now = time.localtime()
                current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

                writer.add_scalar(f"Accuracy of test({model_name[model_number]})", test_acc, e)
                print(
                    '[{}] Epoch : {}, Test Accuracy : {}% (completed)'.format(current_time, e + 1, test_acc))

            writer.add_scalar(f"per_epoch_loss_mean({model_name[model_number]})",
                              round(per_epoch_loss / num_iter_per_epoch, 4),
                              e)

            now = time.localtime()
            current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

            writer.add_scalar(f"Accuracy of train({model_name[model_number]})", train_epoch_acc, e)
            writer.flush()
            print('[{}] Epoch : {}, Train Accuracy : {}%, loss_mean : {} (completed)'.format(current_time, e + 1,
                                                                                             train_epoch_acc
                                                                                             , round(
                    (per_epoch_loss / num_iter_per_epoch), 4)))

            if scheduler_select == 0:
                pass
                # print("deactivate scheduler.step")
            else :
                scheduler.step()

    except KeyboardInterrupt:
        writer.close()

    writer.close()

    end = time.localtime()
    print("[%04d/%02d/%02d %02d:%02d:%02d] Training_finished" % (
    end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))


if __name__ == '__main__':
    main()