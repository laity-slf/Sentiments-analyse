# this is for filtering the warnings
import warnings

import pandas as pd
import torch.optim as optim
import os
import torch
from torch import nn
from data_preprocess import Preprocess
from dataset import TwitterDataset
from model import LSTM_Net
from utils import evaluation, load_training_data, load_testing_data

warnings.filterwarnings('ignore')
path_prefix = ''

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()  # 將model的模式設為train，這樣optimizer就可以更新model的參數
    criterion = nn.BCELoss()  # 定義損失函數，這裡我們使用binary cross entropy loss
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 將模型的參數給optimizer，並給予適當的learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)  # device為"cuda"，將inputs轉成torch.cuda.LongTensor
            labels = labels.to(device,
                               dtype=torch.float)  # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
            optimizer.zero_grad()  # 由於loss.backward()的gradient會累加，所以每次餵完一個batch後需要歸零
            outputs = model(inputs)  # 將input餵給模型
            outputs = outputs.squeeze()  # 去掉最外面的dimension，好讓outputs可以餵進criterion()
            loss = criterion(outputs, labels)  # 計算此時模型的training loss
            loss.backward()  # 算loss的gradient
            optimizer.step()  # 更新訓練模型的參數
            correct = evaluation(outputs, labels)  # 計算此時模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        # 這段做validation
        model.eval()  # 將model的模式設為eval，這樣model的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)  # device為"cuda"，將inputs轉成torch.cuda.LongTensor
                labels = labels.to(device,
                                   dtype=torch.float)  # device為"cuda"，將labels轉成torch.cuda.FloatTensor，因為等等要餵進criterion，所以型態要是float
                outputs = model(inputs)  # 將input餵給模型
                outputs = outputs.squeeze()  # 去掉最外面的dimension，好讓outputs可以餵進criterion()
                loss = criterion(outputs, labels)  # 計算此時模型的validation loss
                correct = evaluation(outputs, labels)  # 計算此時模型的validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                # 如果validation的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, os.path.join(model_dir,'ckpt.model'))
                print('saving model with acc {:.3f}'.format(total_acc / v_batch * 100))
        print('-----------------------------------------------')
        model.train()  # 將model的模式設為train，這樣optimizer就可以更新model的參數（因為剛剛轉成eval模式）

def testing(test_loader, model, device):
        model.eval()
        ret_output = []
        with torch.no_grad():
            for i, inputs in enumerate(test_loader):
                inputs = inputs.to(device, dtype=torch.long)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                outputs[outputs >= 0.5] = 1  # 大於等於0.5為負面
                outputs[outputs < 0.5] = 0  # 小於0.5為正面
                ret_output += outputs.int().tolist()
        return ret_output

if __name__ == '__main__':
    # 通過torch.cuda.is_available()的回傳值進行判斷是否有使用GPU的環境，如果有的話device就設為"cuda"，沒有的話就設為"cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個data的路徑
    train_with_label = os.path.join(path_prefix, 'training_label.txt')
    train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
    testing_data = os.path.join(path_prefix, 'testing_data.txt')

    w2v_path = os.path.join(path_prefix, 'w2v_all.model')  # 處理word to vec model的路徑

    # 定義句子長度、要不要固定embedding、batch大小、要訓練幾個epoch、learning rate的值、model的資料夾路徑
    sen_len = 30
    fix_embedding = True  # fix embedding during training
    batch_size = 128
    epoch = 5
    lr = 0.001
    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = path_prefix  # model directory for checkpoint model

    print("loading data ...")  # 把'training_label.txt'跟'training_nolabel.txt'讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    # 對input跟labels做預處理
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個model的對象
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=250, num_layers=1, dropout=0.5,
                     fix_embedding=fix_embedding)
    model = model.to(device)  # device為"cuda"，model使用GPU來訓練(餵進去的inputs也需要是cuda tensor)

    # 把data分為training data跟validation data(將一部份training data拿去當作validation data)
    X_train, X_val, y_train, y_val = train_x[:190000], train_x[190000:], y[:190000], y[190000:]

    # 把data做成dataset供dataloader取用
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    # 把data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)

    # 開始訓練
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
    # 開始測試模型並做預測
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)
    print('\nload model ...')
    model = torch.load(os.path.join(model_dir, 'ckpt.model'))
    outputs = testing(test_loader, model, device)

    # 寫到csv檔案供上傳kaggle
    tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
    print("Finish Predicting")
