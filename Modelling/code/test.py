import numpy as np

from evaluation.eval_process import *
from libs.utils import *
from PIL import Image
import torchvision.transforms as transforms

class Test_Process_NMS(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.dataloader = dict_DB['testloader']

        self.SLNet = dict_DB['SLNet']
        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['NMS_process']

        self.eval_func = dict_DB['eval_func']

        self.batch_size = self.cfg.batch_size['test_line']
        self.size = to_tensor(np.float32(cfg.size), cfg)
        self.candidates = load_pickle(self.cfg.pickle_dir + 'detector_test_candidates')
        self.candidates = to_tensor(self.candidates, cfg).unsqueeze(0)
        self.cand_num = self.candidates.shape[1]
        self.step = create_forward_step(self.candidates.shape[1],
                                        cfg.batch_size['test_line'])

        self.visualize = dict_DB['visualize']

        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
    @calculate_time
    def inference(self, img, batch,  SLNet, cfg):
        # semantic line detection
        if type(img) != torch.Tensor:
            image_pil = Image.fromarray(img.astype('uint8'), 'RGB')
            image_tensor = self.transform(image_pil).to(cfg.device).unsqueeze(0)
        else: 
            image_tensor = img
        out = self.forward_model.run_detector(img=image_tensor,
                                                line_pts=self.candidates,
                                                step=self.step,
                                                model=SLNet,
                                                cfg=cfg)
        # reg result
        out['pts'] = self.candidates[0] + out['reg'] * self.size
        # cls result
        pos_check = torch.argmax(out['cls'], dim=1)
        out['pos'] = out['pts'][pos_check == 1]
        # primary line
        sorteed = torch.argsort(out['cls'][:, 1], descending=True)
        out['pri'] = out['pts'][sorteed[0], :].unsqueeze(0)
        if torch.sum(pos_check == 1) == 0:
            out['pos'] = out['pri']

        # post process
        self.post_process.update_data(batch, out['pos'])
        out['mul'] = self.post_process.run(cfg=cfg, img = img)

        # visualize
        self.visualize.display_for_test(batch=batch, out=out, img = img)
        return out
    
    def run(self, SLNet, cfg, mode='test'):
        result = {'out': {'pri': [], 'mul': []},
                  'gt': {'pri': [], 'mul': []}}

        with torch.no_grad():
            SLNet.eval()

            for i, self.batch in enumerate(self.dataloader):  # load batch data

                self.img_name = self.batch['img_name'][0]
                self.img = self.batch['img'].to(cfg.device)
                pri_gt = self.batch['pri_gt'][0][:, :4]
                mul_gt = self.batch['mul_gt'][0][:, :4]

                # inference
                out = self.inference(self.img, self.batch, SLNet, cfg)

                # record output data
                result['out']['pri'].append(out['pri'])
                result['out']['mul'].append(out['mul'])
                result['gt']['pri'].append(pri_gt)
                result['gt']['mul'].append(mul_gt)

                print('image %d ---> %s done!' % (i, self.img_name))


        # save pickle
        save_pickle(dir_name=self.cfg.output_dir + 'test/pickle/',
                    file_name='result',
                    data=result)

        if mode == 'test':
            self.evaluation(cfg=cfg)

    def evaluation(self, cfg):
        # evaluation
        data = load_pickle(self.cfg.output_dir + 'test/pickle/result')
        auc_a = eval_AUC_A(out=data['out']['pri'],
                           gt=data['gt']['pri'],
                           eval_func=self.eval_func, cfg = cfg)
        auc_p, auc_r = eval_AUC_PR(out=data['out']['mul'],
                                   gt=data['gt']['mul'],
                                   eval_func=self.eval_func, cfg = cfg)
        return auc_a, auc_p, auc_r