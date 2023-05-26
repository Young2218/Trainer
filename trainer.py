import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, max_epoch, early_stop, 
                 train_loader, val_loader, test_loader,
                 save_path, log_save_path,
                 optimizer, criterion, evaluate_dic, scheduler) -> None:
        
        self.model = model
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.save_path = save_path
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.evaluate_dic = evaluate_dic
        self.scheduler = scheduler
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_save_path)

     

    def train(self):
        self.model.to(self.device)
        min_val_loss = float('inf')
        check_early_stop = 0
        
        for epoch in range(1, self.max_epoch):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            self.writer.add_scalar("LOSS/train_loss", train_loss, epoch)
            self.writer.add_scalar("LOSS/val_loss", val_loss, epoch)
            for name, met in self.evaluate_dic.items():
                self.writer.add_scalar(name, met.get_value(), epoch)
            
            self.writer.flush()
            
            print(f'EPOCH[{epoch}] TRAIN_LOSS[{train_loss:.5f}] VAL_LOSS[{val_loss:.5f}]', end=' ')
            for name, met in self.evaluate_dic.items():
                print(f"{name}[{met.get_value():.5f}]", end=' ')
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
        
    def train_one_epoch(self) -> float:
        self.model.train()        
        loss_list = []
        for images, labels in tqdm(self.train_loader):
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss_list.append(loss)
            
            loss.backward()
            
            self.optimizer.step()
            
        train_loss = sum(loss_list)/len(loss_list)
        return train_loss
    
    def validate_one_epoch(self) -> float:
        self.model.eval()        
        loss_list = []
        
        for eval_met in self.evaluate_dic.values():
            eval_met.reset()
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                # get the inputs
                images = images.to(self.device)
                labels = labels.to(self.device)

                # predict classes using images from the training set
                outputs = self.model(images)
                # compute the loss based on model output and real labels
                loss = self.criterion(outputs, labels)
                loss_list.append(loss)
                
                _, predicted = torch.max(outputs.data, 1)

                for eval_met in self.evaluate_dic.values():
                    eval_met.update(predicted, labels)
                     
        val_loss = sum(loss_list)/len(loss_list)
        
        return val_loss

    def inference(self):
        self.model.eval()        
        
        for eval_met in self.evaluate_dic.values():
            eval_met.reset()
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                # get the inputs
                images = images.to(self.device)
                labels = labels.to(self.device)

                # predict classes using images from the training set
                outputs = self.model(images)
                # compute the loss based on model output and real labels
                
                _, predicted = torch.max(outputs.data, 1)

                for eval_met in self.evaluate_dic.values():
                    eval_met.update(predicted, labels)
        for name, met in self.evaluate_dic.items():
            print(f"{name}[{met.get_value():.5f}]", end=' ')
        print()
