
class Config:
    def __init__(self, history_size=4,
                 predict_size=1,
                 input_size=1,
                 hidden_size=64,
                 num_layers=2,
                 batch_size=32,
                 epoch=20,
                 dropout=0,

                 lstm_hidden_size=64,
                 lstm_num_layers=2,

                 gru_hidden_size=64,
                 gru_num_layers=2,

                 dim_k=10,
                 dim_v=10

                 ):
        self.history_size = history_size
        self.predict_size = predict_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epoch = epoch
        self.dropout = dropout
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        self.dim_k = dim_k
        self.dim_v = dim_v


if __name__ == '__main__':
    config = Config()
    print(config.batch_size)
    pass
