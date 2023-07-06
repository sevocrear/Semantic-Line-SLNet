from config import Config
from test import *
from train import *
from libs.prepare import *
import click

def main_test(cfg, dict_DB):

    # test optoin
    test_process = Test_Process_NMS(cfg, dict_DB)
    test_process.run(dict_DB['SLNet'], cfg)
    
def main_test_imgs(cfg, dict_DB, source_dir):
    # test optoin
    test_process = Test_Process_NMS(cfg, dict_DB)
    
    cap = cv2.VideoCapture(source_dir)
    idx = 0
    with torch.no_grad():
        dict_DB['SLNet'].eval()
            
        while (cap.isOpened() and idx < 10):
            ret, img = cap.read()
            if not ret:
                break
            out = test_process.inference(img, None,  dict_DB['SLNet'], cfg)
            
def main_train(cfg, dict_DB):

    # train option
    dict_DB['test_process'] = Test_Process_NMS(cfg, dict_DB)
    train_process = Train_Process_SLNet(cfg, dict_DB)
    train_process.run(cfg)

@click.command()
@click.option('--source_path', default='video.mp4', help='Path to source')
def main(source_path):

    # Config
    cfg = Config()

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "all"
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)
    dict_DB = prepare_model(cfg, dict_DB)
    dict_DB = prepare_postprocessing(cfg, dict_DB)
    dict_DB = prepare_evaluation(cfg, dict_DB)
    dict_DB = prepare_training(cfg, dict_DB)

    if 'test_imgs' in cfg.run_mode:
        main_test_imgs(cfg, dict_DB, source_path)
    elif 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
if __name__ == '__main__':
    main()
