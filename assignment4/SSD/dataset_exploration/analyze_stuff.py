from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    box_width = [[-5]] *9
    box_height = [[-5]] *9
    box_ratio = [[-5]] *9
                      
    for batch in tqdm(dataloader):
        labels = (batch['labels'][0])
        for ind, box in enumerate(batch['boxes'][0]):
            
            width = (box[2] - box[0]).numpy()*batch['width'].numpy()[0] #account for 0-1
            height = (box[3] - box[1]).numpy()*batch['height'].numpy()[0]
            ratio = width/height
            
            box_width[labels[ind].item()] = np.append(box_width[labels[ind].item()], width)
            box_height[labels[ind].item()] = np.append(box_height[labels[ind].item()], height)
            box_ratio[labels[ind].item()] = np.append(box_ratio[labels[ind].item()], ratio)

    #Make graphs for each class
    for i in range(1, len(cfg.label_map)):
        plt.plot(box_width[labels[i].item()])
        plt.plot(box_height[labels[i].item()])
        plt.plot(box_ratio[labels[i].item()])
        
        plt.xlabel("number")
        plt.ylabel("size")
        
        plt.savefig('./dataset_exploration/anal_figures/{0}'.format(cfg.label_map[i]))
        plt.close()
            
        


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
