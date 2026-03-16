# Single Image Dehazing Using DehazeFormer-B

## Project Overview
This project implements single image dehazing using the DehazeFormer-B model on a prepared subset of the RESIDE-IN dataset.

## Model Used
- DehazeFormer-B
- Transformer-based image restoration model
- Supervised learning with paired hazy and ground-truth images

## Dataset
- Training: 6000 image pairs
- Testing: 1000 image pairs
- Approximate split: 86% training, 14% testing

## Training Details
- Framework: PyTorch
- Environment: Google Colab
- Epochs: 10
- Training Type: Fine-tuning with pretrained weights

## Final Results
- Average PSNR: 32.62 dB
- Average SSIM: 0.9808

## How to Run
Training:
python train.py --model dehazeformer-b --dataset RESIDE-IN --exp reside6k

Testing / Inference:
python test.py --model dehazeformer-b --dataset RESIDE-IN --exp reside6k

## Notes
- Full dataset and model checkpoints are not included because of size limitations.
- Large files such as .pth models and dataset folders are stored separately in Google Drive.

## Future Improvements
- Train for more epochs and analyze PSNR/SSIM trends
- Test on images with different haze densities
- Compare with traditional dehazing methods such as Dark Channel Prior (DCP)
- Improve inference efficiency in limited-resource environments

## Author
- Mohammad SAMEER
- Prathmesh Bogar
- Soham Naukudkar
