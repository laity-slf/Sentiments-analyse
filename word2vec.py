# w2v.py
# 這個block是用來訓練word to vector 的 word embedding
# 注意！這個block在訓練word to vector時是用cpu，可能要花到10分鐘以上
from gensim.models import word2vec
from utils import  load_training_data,load_testing_data

def train_word2vec(x):
    # 訓練word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('training_label.txt')
    train_x_no_label = load_training_data('training_nolabel.txt')

    print("loading testing data ...")
    test_x = load_testing_data('testing_data.txt')

    model = train_word2vec(train_x + train_x_no_label + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save('w2v_all.model')