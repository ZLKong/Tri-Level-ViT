# Peeling the Onion: Hierarchical Reduction of Data Redundancy for Efficient Vision Transformer Training (AAAI 2023)

arXiv https://arxiv.org/abs/2211.10801

<p align="center">
  <img src="images/eformerv2.png" width=70%> <br>
  Comparison of different models with various accuracy-training time trade-off..
</p>

## Usage

### Requirements

- torch>=1.8.0
- torchvision>=0.9.0
- timm==0.4.5

**Data preparation**: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Training

To train the model on ImageNet from scratch, run:

**DeiT**  

```
CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py 
                               --model deit_tiny_patch16_224_attn_dst 
                               --batch-size 512 
                               --data-path /datasets/imagenet 
                               --keep_ratio 0.9 
                               --attn_ratio 0.1 
                               --output_dir output_dir 
                               --remove-n 64058
```

You can train models with different ratio by adjusting token ratio ```keep_ratio``` and attention ratio ```attn_ratio``` .
 For the ratio of example level, modify the amount of examples to remove and restore in ```remove-n``` and also the random remove before training in ```train_example_idx```, ```removed_example_idx``` of the code ```main.py```  
 
 For DeiT-S and DeiT-B, replace the ```--model``` as in [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md)
 
 **Swin Transformer**  

```
cd Swin
```
```
CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py 
                               --cfg configs/swin/swin_tiny_patch4_window7_224_token.yaml 
                               --batch-size 128 
                               --data-path /datasets/imagenet 
                               --output output_dir 

```

## License

MIT License

## Acknowledgements

Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

## Citation
If you find our work useful in your research, please consider citing:
```
@article{kong2022peeling,
  title={Peeling the Onion: Hierarchical Reduction of Data Redundancy for Efficient Vision Transformer Training},
  author={Kong, Zhenglun and Ma, Haoyu and Yuan, Geng and Sun, Mengshu and Xie, Yanyue and Dong, Peiyan and Meng, Xin and Shen, Xuan and Tang, Hao and Qin, Minghai and others},
  journal={arXiv preprint arXiv:2211.10801},
  year={2022}
}
```
