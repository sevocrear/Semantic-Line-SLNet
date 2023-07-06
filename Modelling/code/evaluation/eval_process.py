import torch


def eval_AUC_PR(out, gt, eval_func, cfg):
    miou, match = create_eval_dict()

    num = len(out)

    # eval process
    for i in range(num):
        out[i] = out[i].to(cfg.device)
        gt[i] = gt[i].to(cfg.device)
        out_num = out[i].shape[0]
        gt_num = gt[i].shape[0]

        if gt_num == 0:
            match['p'][i] = torch.zeros(out_num, dtype=torch.float32).to(cfg.device)
        elif out_num == 0:
            match['r'][i] = torch.zeros(gt_num, dtype=torch.float32).to(cfg.device)
        else:
            miou['p'][i], miou['r'][i] = eval_func.measure_miou(out=out[i],
                                                                gt=gt[i], cfg = cfg)
            match['p'][i], match['r'][i] = eval_func.matching(miou=miou,
                                                              idx=i, cfg = cfg)

    # performance
    auc_p = eval_func.calculate_AUC(miou=match,
                                    metric='precision')

    auc_r = eval_func.calculate_AUC(miou=match,
                                    metric='recall')

    print('---------Performance---------\n'
          'AUC_P %5f / AUC_R %5f' % (auc_p, auc_r))

    return auc_p, auc_r


def eval_AUC_A(out, gt, eval_func, cfg):
    miou, match = create_eval_dict()

    num = len(out)

    # eval process
    for i in range(num):
        out[i] = out[i].to(cfg.device)
        gt[i] = gt[i].to(cfg.device)

        miou['a'][i], _ = eval_func.measure_miou(out=out[i],
                                                 gt=gt[i], cfg = cfg)

    # performance
    auc_a = eval_func.calculate_AUC(miou=miou,
                                    metric='accuracy')

    print('---------Performance---------\n'
          'AUC_A %5f' % (auc_a))

    return auc_a

def create_eval_dict():
    # a : accuracy / p : precision / r : recall

    miou = {'a': {},
            'p': {},
            'r': {}}

    match = {'p': {},
             'r': {}}

    return miou, match
