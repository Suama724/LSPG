class Config:
    def __init__(self):
        self.dim = 100
        self.surrogate_method = 'mlp' # mlp or set_transformer
        self.surrogate_path = 'surrogate_mlp/mlp_surrogate_best.pth'
        self.epoch = 15000
        self.out_dir = "output"
        