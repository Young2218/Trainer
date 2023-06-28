import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np

class Trainer_ang:
    def __init__(self, model, max_epoch, early_stop, 
                 train_loader, val_loader, test_loader,
                 save_path, log_save_path,
                 optimizer, criterion, evaluate_dic, scheduler,
                 model_name, classes_name) -> None:
        
        self.model = model
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.log_save_path = log_save_path
        self.save_path = save_path
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluate_dic = evaluate_dic
        self.scheduler = scheduler
        
        self.model_name = model_name
        self.classes_names = classes_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_save_path)
        print(self.device)
     

    def train(self):
        self.model.to(self.device)
        min_val_loss = float('inf')
        check_early_stop = 0
        
        for epoch in range(1, self.max_epoch):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)
            
            if self.scheduler is None:
                pass
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            self.writer.add_scalar("LOSS/train_loss", train_loss, epoch)
            self.writer.add_scalar("LOSS/val_loss", val_loss, epoch)
            for name, met in self.evaluate_dic.items():
                self.writer.add_scalar(name, met.compute(), epoch)
            
            self.writer.flush()
            
            print(f'EPOCH[{epoch}] TRAIN_LOSS[{train_loss:.5f}] VAL_LOSS[{val_loss:.5f}]', end=' ')
            for name, met in self.evaluate_dic.items():
                print(f"{name}[{met.compute():.5f}]", end=' ')
            print()
            
            
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                check_early_stop = 1
                torch.save(self.model.state_dict(), self.save_path)
            else:
                check_early_stop += 1
                if check_early_stop > self.early_stop:
                    print("EARLY STOP")
                    self.writer.close()
                    break
        
        print(f"End Train in {epoch} epochs, Min Loss[{min_val_loss}]")
        return str(min_val_loss)
        
    def train_one_epoch(self, epoch) -> float:
        self.model.train()        
        loss_list = []
        
        feautres_list = []
        labels_list = []
        
        for inputs, labels in tqdm(self.train_loader):
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            feature, outputs = self.model(inputs, labels)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss_list.append(loss)
            
            loss.backward()
            
            self.optimizer.step()
            
            feautres_list.append(feature)
            labels_list.append(labels)
        
        fs = torch.cat(feautres_list, 0)
        ls = torch.cat(labels_list, 0)
        self.visualize(fs.data.cpu().numpy(), ls.data.cpu().numpy(), epoch, "train")
        
        train_loss = sum(loss_list)/len(loss_list)
        return train_loss
    
    def validate_one_epoch(self, epoch) -> float:
        self.model.eval()        
        loss_list = []
        
        for eval_met in self.evaluate_dic.values():
            eval_met.reset()
        
        feautres_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                feature, outputs = self.model(inputs, labels)
                loss = self.criterion(outputs, labels)
                loss_list.append(loss)
                
                # _, predicted = torch.max(outputs.data, 1)
                feautres_list.append(feature)
                labels_list.append(labels)
                
                for eval_met in self.evaluate_dic.values():
                    eval_met.update(outputs, labels)
        
        fs = torch.cat(feautres_list, 0)
        ls = torch.cat(labels_list, 0)
        self.visualize(fs.data.cpu().numpy(), ls.data.cpu().numpy(), epoch, "val")
                     
        val_loss = sum(loss_list)/len(loss_list)
        
        return val_loss

    def inference_with_test_loader(self):
        self.model.to(self.device)
        self.model.eval()        
        loss_list = []
        
        for eval_met in self.evaluate_dic.values():
            eval_met.reset()
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader):
                # get the inputs
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                feature, outputs = self.model(inputs, labels)
                loss = self.criterion(outputs, labels)
                loss_list.append(loss)
                
                _, predicted = torch.max(outputs.data, 1)

                for eval_met in self.evaluate_dic.values():
                    eval_met.update(predicted, labels)
        
        print("LOSS", sum(loss_list)/len(loss_list))       
        for name, met in self.evaluate_dic.items():
            print(f"{name}[{met.compute():.5f}]", end=' ')
        print()
        

    def visualize(self, features, labels, epoch, mode):
        plt.ioff()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
            '#ff00ff', '#990000', '#999900', '#009900', '#009999', '#ff9900']
        plt.clf()
        

        for i in range(len(self.classes_names)):
            plt.plot(features[labels == i, 0], features[labels == i, 1], '.', c=c[i])
        leg = plt.legend(self.classes_names, loc='center left', bbox_to_anchor=(1, 0.5))

        if not os.path.isdir(f'{self.log_save_path}/visualize_{mode}/'):
            os.makedirs(f'{self.log_save_path}/visualize_{mode}/')
        plt.savefig(f'{self.log_save_path}/visualize_{mode}/epoch_{epoch}.jpg', bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close()