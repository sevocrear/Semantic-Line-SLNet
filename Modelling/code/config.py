import os

class Config(object):
    def __init__(self):

        # proj & output dir
        self.proj_dir = os.path.dirname(os.getcwd()) + '/'
        self.output_dir = self.proj_dir + 'output/'

        # dataset dir
        self.dataset = 'SEL'  # ['SEL', 'SEL_Hard']
        if self.dataset == 'SEL':
            self.dataset_dir = '--SEL dataset root/'  # need to modify
            self.img_dir = self.dataset_dir + 'ICCV2017_JTLEE_images/'
        elif self.dataset == 'SEL_Hard':
            self.dataset_dir = '--SEL_Hard dataset root/'  # need to modify
            self.img_dir = self.dataset_dir + 'images/'

        # other dir
        self.pickle_dir = self.dataset_dir + 'data/detector/' # -- preprocessed data root ## need to modify
        self.weight_dir = self.output_dir + '/train/weight/'
        self.paper_weight_dir = self.proj_dir + '/paper_weight/'  # need to modify

        # setting for train & test
        self.run_mode = 'test_paper'  # ['train', 'test', 'test_paper']
        self.resume = True

        self.gpu_id = "0"
        self.seed = 123
        self.num_workers = 4
        self.epochs = 500
        self.ratio_pos = 0.4
        self.batch_size = {'img': 1,
                           'train_line': 50,
                           'test_line': 200}

        # optimizer
        self.lr = 1e-5
        self.milestones = [40, 80, 120, 160, 200, 240, 280]
        self.weight_decay = 5e-4
        self.gamma = 0.5

        # other setting
        self.height = 400
        self.width = 400
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


        # option for visualization
        self.draw_auc_graph = True
