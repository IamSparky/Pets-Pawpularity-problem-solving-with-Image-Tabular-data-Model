from importsfolder import *
from config_file import AverageMeter


def loss_fn(x, y):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(x, y))
    return loss

def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    running_loss = 0
    model.train()
    
    losses = AverageMeter()
    tqdm_ob = tqdm(data_loader, total = len(data_loader))
    
    for batch_index,dataset in enumerate(data_loader):
        image = dataset["image"]
        tabular_data = dataset["tabular_data"]
        target = dataset["target"]
        
        image = image.to(device, dtype=torch.float)
        tabular_data = tabular_data.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        
        optimizer.zero_grad()

        outputs = model(image, tabular_data)
        loss = loss_fn(outputs , target)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), image.size(0))
        tqdm_ob.set_postfix(loss = losses.avg)
            
        del image, tabular_data, target
        gc.collect()
        torch.cuda.empty_cache()
        
        running_loss += loss.item() 
    train_loss = running_loss/ (batch_index + 1)
    return train_loss

def eval_loop_fn(data_loader, model, device):
    running_loss = 0
    model.eval()
    
    losses = AverageMeter()
    tqdm_ob = tqdm(data_loader, total = len(data_loader))
    
    for batch_index,dataset in enumerate(data_loader):
        image = dataset["image"]
        tabular_data = dataset["tabular_data"]
        target = dataset["target"]
        
        image = image.to(device, dtype=torch.float)
        tabular_data = tabular_data.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        outputs = model(image, tabular_data)
        loss = loss_fn(outputs , target)
        losses.update(loss.item(), image.size(0))
        tqdm_ob.set_postfix(loss = losses.avg)
            
        del image, tabular_data, target
        gc.collect()
        torch.cuda.empty_cache()
        
        running_loss += loss.item() 
    val_loss = running_loss/ (batch_index + 1)
    return val_loss