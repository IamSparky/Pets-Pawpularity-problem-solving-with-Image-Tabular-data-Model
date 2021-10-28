from importsfolder import *
from engine import loss_fn, train_loop_fn, eval_loop_fn
from config_file import config
from create_models import TimmEfficientNet_b0
from create_folds import create_the_folds
from create_dataset import petFinderDataset

def run():
    new_train = pd.read_csv(config.TRAINING_FILE)
    new_train = create_the_folds(no_of_folds = 5, new_train = new_train)

    model = TimmEfficientNet_b0()
    model = model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3 * 0.95)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
    
    a_string = "*" * 20
    for i in range(5):
        print(a_string, " FOLD NUMBER ", i, a_string)
        df_train = new_train[new_train.kfold != i].reset_index(drop=True)
        df_valid = new_train[new_train.kfold == i].reset_index(drop=True)

        train_data = petFinderDataset(config, df_train)
        val_data = petFinderDataset(config, df_valid, is_valid = 1)

        train_data_loader = DataLoader(train_data,
                                num_workers=4,
                                batch_size=config.TRAIN_BATCH_SIZE,
                                shuffle=True,
                                drop_last=True)

        valid_data_loader = DataLoader(val_data,
                                num_workers=4,
                                batch_size=config.VALID_BATCH_SIZE,
                                shuffle=False,
                                drop_last=False)
        
        all_rmse = []
        for epoch in range(config.EPOCHS):
            print(f"Epoch --> {epoch+1} / {config.EPOCHS}")
            print(f"-------------------------------")
            train_rmse = train_loop_fn(train_data_loader, model, optimizer, config.DEVICE, scheduler)
            print(f"Training Root Mean Square Error = {train_rmse}")
            val_rmse = eval_loop_fn(valid_data_loader, model, config.DEVICE)
            print(f"Validation Root Mean Square Error = {val_rmse}")
            
            all_rmse.append(val_rmse)
        print('\n')
        
        if i < 1:
            best_RMSE = min(all_rmse)
            best_model = copy.deepcopy(model)
            all_rmse = []
        else:
            if min(all_rmse) < best_RMSE:
                best_RMSE = min(all_rmse)
                best_model = copy.deepcopy(model)
                all_rmse = []
                
    torch.save(best_model,'swin_large_patch4_window12_384.bin')
    print()
    print(f"The lowest RMSE score that we got across all the folds is : {best_RMSE}")
    
    return best_model
                
if __name__ == "__main__":
    run()