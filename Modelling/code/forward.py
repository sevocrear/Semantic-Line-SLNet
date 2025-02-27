from libs.utils import *

class Forward_Model(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def run_detector(self, img, line_pts, step, model, cfg):
        out = create_test_dict(cfg)

        # semantic line detection (SLNet)
        feat1, feat2 = model.feature_extraction(img)

        for i in range(len(step) - 1):
            start, end = step[i], step[i + 1]
            batch_line_pts = line_pts[:, start:end, :]

            batch_out = model(img=img,
                              line_pts=batch_line_pts,
                              feat1=feat1, feat2=feat2)

            out['cls'] = torch.cat((out['cls'], batch_out['cls']), dim=0)
            out['reg'] = torch.cat((out['reg'], batch_out['reg']), dim=0)

        return out
