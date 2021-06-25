import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding
from sself_data import readid
import keras
import keras
from keras_multi_head import MultiHeadAttention
from keras_self_attention import SeqSelfAttention
from keras.layers import Bidirectional,LSTM
from keras_contrib.layers import CRF
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import self_data

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # need ~700MB GPU memory


## hyperparameters #超参数
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--id_path', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=10, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=100, help='random init char embedding_dim')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')

args = parser.parse_args()


## get char embeddings
# word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
word2id = readid(os.path.join('.', args.id_path, 'vocab.txt'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

print('embedding shape',embeddings.shape)

train_path = os.path.join('.', args.train_data, 'train_data.txt')
test_path = os.path.join('.', args.test_data, 'test_data.txt')
train_data = read_corpus(train_path)
test_data = read_corpus(test_path); test_size = len(test_data)

(train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = sself_data.load_data(word2id,train_path,test_path)
## training model




if args.mode == 'train':
    model = keras.models.Sequential()
    model.add(Embedding(input_dim=len(vocab), output_dim=args.embedding_dim, mask_zero=True))
    model.add(MultiHeadAttention(head_num=100,history_only=True))
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dropout(0.5))
    # model.add(MultiHeadAttention(head_num=100,history_only=True))
    #
    #
    # model.add(keras.layers.Dense(100))
    # model.add(keras.layers.Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))


    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(units=))
    crf = CRF(7, sparse_target=True)
    model.add(crf)
    model.compile(
        optimizer='adam',
        loss=crf.loss_function,
        metrics=[crf.accuracy],
    )
    model.summary()






    # train_data =np.array(train_data)
    # test_data = np.array(test_data)
    model.fit(train_x, train_y,batch_size=64,epochs=15, validation_data=[test_x, test_y])
    model.save('model/models.h5')

    score = model.evaluate(test_x, test_y, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


