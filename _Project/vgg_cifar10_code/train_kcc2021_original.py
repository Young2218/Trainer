import time
import datetime
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from customdataset import CustomImageDataset
from VGG_model_original import VGG
import matplotlib.pyplot as plt
from tqdm import tqdm

now = time.localtime()
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


print("[%04d/%02d/%02d %02d:%02d:%02d] Training_start" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))


hyper_param_epoch = 200
hyper_param_batch = 100
hyper_param_learning_rate = 0.01
model_number = 2
m_name = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
# path = './weights/init_weight_original_210429_(VGG16)_cifar10.pth'
# path = './weights/init_weight_original_210419_(VGG16)_cifar10.pth'

path = './weights/init_weight_original_210525_(VGG16)_cifar10.pth'

# path = './weights/init_weight_original_210525_(VGG19)_cifar10.pth'
# angle_number = 9
# # added_angle = [0, 30, 60, 90, 120, 180]
# # added_angle = [0, 90, 180, 270]
# added_angle = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]
#
# # !!! include cd and then start training!!!!
# def visualize_features(model, feat, labels, epoch, class_names, added_angle, model_name, mode = 'train'):
#     # plt.ion()
#     plt.ioff()
#     c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
#          '#ff00ff', '#990000', '#999900', '#009900', '#009999']
#     # plt.clf()
#
#     for i in range(len(class_names)):
#         plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
#
#     leg = plt.legend(class_names, loc='center left', bbox_to_anchor=(1, 0.5))
#     # print("x, y", plt.xlim(), plt.ylim())
#     # x1_axis = np.linspace(plt.xlim()[0], plt.xlim()[1], 10)
#     # print("x1_axis :",x1_axis)
#     # print("plt.xlim :",plt.xlim())
#
#     # for j in range(len(class_names)):
#     #     w1 = model.classifier2.weight.data.cpu().numpy()[j][0]
#     #
#     #     w2 = model.classifier2.weight.data.cpu().numpy()[j][1]
#     #
#     #     b = model.classifier2.bias.data.cpu().numpy()[j]
#     #
#     #     x2_axis = -(w1 * x1_axis + b) / w2
#     #
#     #     filtered_x1 = x1_axis[np.where((x2_axis > plt.ylim()[0]) & (x2_axis < plt.ylim()[1]))]
#     #     # temp = filtered_x1[]
#     #     filtered_x2 = -(w1 * filtered_x1 + b) / w2
#     #     plt.plot(filtered_x1, filtered_x2,'-', color=c[j], linewidth=1)
#
#     center = [0, 0]
#     # L = np.array([0, 10])
#     w1_max = np.array([0, plt.xlim()[1]])
#     w2_max = np.array([0, plt.ylim()[1]])
#
#
#     for j in range(len(class_names)):
#         w1 = center[0] + model.classifier2.weight.data.cpu().numpy()[j][0] * (w1_max)
#         w2 = center[1] + model.classifier2.weight.data.cpu().numpy()[j][1] * (w2_max)
#
#         plt.plot(w1, w2, '-', color=c[j], linewidth=1)
#
#     plt.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
#     plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
#
#     plt.title("CIFAR10, {}_bn, added_angle = {}, epoch = {}".format(model_name, added_angle, epoch+1))
#     #   plt.xlim(xmin=-5,xmax=5)
#     #   plt.ylim(ymin=-5,ymax=5)
#     # plt.show(block=False)
#     # plt.pause(1)
#     plt.savefig('./visualize_{}/CIFAR10_+{}angle_{}_bn_epoch_{}.jpg'.format(mode,added_angle,model_name,epoch+1), bbox_extra_artists=(leg,), bbox_inches='tight')
#     plt.close()


def main():
    transforms_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                std=[0.2023, 0.1994, 0.2010]), ])

    transforms_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010]), ])
    # transforms_train = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    #
    # transforms_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    train_data_set = CustomImageDataset(data_set_path="/home/prml/PSH/SH_dataset/cifar10/train",
                                        transforms=transforms_train)
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

    test_data_set = CustomImageDataset(data_set_path="/home/prml/PSH/SH_dataset/cifar10/test",
                                       transforms=transforms_test)
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=False)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = train_data_set.num_classes
    class_names = train_data_set.class_names

    print("num_classes :", num_classes)
    print("classe_names :", class_names)

    # hidden_model = VGG(m_name=m_name[model_number], num_classes=num_classes, device=device, batch_size=hyper_param_batch)

    hidden_model = VGG(m_name=m_name[model_number], num_classes=num_classes)
    hidden_model.load_state_dict(torch.load(path)) #original file load
    hidden_model = torch.nn.DataParallel(hidden_model)
    hidden_model.to(device)


    # print("before changed_weight :", hidden_model.classifier2.weight.data.cpu().numpy())
    # print("bias 1 :", hidden_model.classifier2.bias.data.cpu().numpy())


    # changed_weight = []
    # for a in range(0, 360, 360 // num_classes):
    #     phi = np.deg2rad(a + added_angle[angle_number])
    #     x = np.cos(phi)
    #     y = np.sin(phi)
    #     changed_weight.append([x, y])
    #
    #
    # hidden_model.classifier2.weight = torch.nn.Parameter(torch.FloatTensor(changed_weight))
    # hidden_model.to(device)




    # print("after changed_weight :", hidden_model.classifier2.weight.data.cpu().numpy())
    # print("bias 1 :", hidden_model.classifier2.bias.data.cpu().numpy())
    # print("hidden_model :",hidden_model)
    # print(hidden_model.fc_layer.weight[0])
    # print(hidden_model.fc_layer.weight[1])
    # print(hidden_model.fc_layer)
    # print(aaa)

    # hidden_model.classifier2.weight.detach()
    # hidden_model.classifier2.weight.require_grad = False

    # hidden_model.classifier2.trainable = False

    # for param in hidden_model.classifier2.parameters(): # freezing weight
    #     param.requires_grad = False

    #<<<<< Loss and optimizer >>>>>
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hidden_model.parameters(), lr=hyper_param_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # print("hidden_model.parameters() :",hidden_model.parameters())

    # path = './weights/train_weight_{}.pth'.format(hyper_param_epoch)




    num_iter_per_epoch = len(train_loader)
    best_acc = 0
    try:
        for e in range(hyper_param_epoch):
            per_epoch_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader)
            hidden_model.train()

            # features_loader_train = []
            # labels_loader_train = []



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
                # features_loader_train.append(features)
                # labels_loader_train.append((labels))

                now = time.localtime()
                current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

                writer.add_scalar(f"Loss of train({m_name[model_number]})", loss, e)

                progress_bar.set_description(
                    '[{}] Epoch: {}/{}. Iteration: {}/{}. current batch loss: {:.4f}. '.format(
                        current_time, e + 1, hyper_param_epoch, i_batch + 1, num_iter_per_epoch, loss.item()))

                per_epoch_loss += loss.item()
                # print("weight 2 :", hidden_model.classifier2.weight.data.cpu().numpy())
                # print("bias 2 :", hidden_model.classifier2.bias.data.cpu().numpy())

            # train_feat = torch.cat(features_loader_train, 0)
            # train_labels = torch.cat(labels_loader_train, 0)

            train_epoch_acc = round((100 * correct / total), 2)
            # print("weight_end of epoch :", hidden_model.classifier2.weight.data.cpu().numpy())
            # print("bias_end of epoch :", hidden_model.classifier2.bias.data.cpu().numpy())
            # hidden_model.classifier2.weight.require_grad = False
            # visualize_features(hidden_model, train_feat.data.cpu().numpy(), train_labels.data.cpu().numpy(), e,
            #                    class_names, added_angle[angle_number], m_name[model_number], mode='train')


            if (e) % 3 == 0:

                hidden_model.eval()
                # features_loade_test = []
                # labels_loader_test = []

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

                        # features_loade_test.append(hidden_features)
                        # labels_loader_test.append((labels_eval))

                    # test_feat = torch.cat(features_loade_test, 0)
                    # test_labels = torch.cat(labels_loader_test, 0)
                    # visualize_features(hidden_model, test_feat.data.cpu().numpy(), test_labels.data.cpu().numpy(), e,
                    #                    class_names,added_angle[angle_number], m_name[model_number], mode='validation')
                    #

                    test_acc = round((100 * correct_eval / total_eval), 2)
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(hidden_model.state_dict(),
                               f'./weights/train_weight_{e + 1}_({m_name[model_number]}_original)_best.pth')
                else:
                    torch.save(hidden_model.state_dict(),
                               f'./weights/train_weight_{e + 1}_({m_name[model_number]}_original).pth')

                now = time.localtime()
                current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

                writer.add_scalar(f"Accuracy of test({m_name[model_number]})", test_acc, e)
                print(
                    '[{}] Epoch : {}, Test Accuracy : {}% (completed)'.format(current_time, e + 1, test_acc))

                # scheduler.step()

            writer.add_scalar(f"per_epoch_loss_mean({m_name[model_number]})",
                              round(per_epoch_loss / num_iter_per_epoch, 4),
                              e)

            now = time.localtime()
            current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

            writer.add_scalar(f"Accuracy of train({m_name[model_number]})", train_epoch_acc, e)
            writer.flush()
            print('[{}] Epoch : {}, Train Accuracy : {}%, loss_mean : {} (completed)'.format(current_time, e + 1,
                                                                                             train_epoch_acc
                                                                                             , round(
                    (per_epoch_loss / num_iter_per_epoch), 4)))

            scheduler.step()

    except KeyboardInterrupt:
        writer.close()

    writer.close()

    end = time.localtime()
    print("[%04d/%02d/%02d %02d:%02d:%02d] Training_finished" % (
    end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))


if __name__ == '__main__':
    main()