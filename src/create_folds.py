from importsfolder import *
from config_file import config
 
# create folds
def create_the_folds(no_of_folds, new_train):
    new_train["kfold"] = -1    
    new_train = new_train.sample(frac=1).reset_index(drop=True)
    y = new_train.Pawpularity.values
    kf = model_selection.StratifiedKFold(n_splits=no_of_folds)

    for f, (t_, v_) in enumerate(kf.split(X = new_train, y = y)):
        new_train.loc[v_, 'kfold'] = f
    
    return new_train

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df = create_the_folds(no_of_folds = 5, new_train = df)
    print(df.head(15))