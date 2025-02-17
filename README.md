# Generative Modeling Algorithms for Unconditional MNIST Generation

This repository serves as an educational resource, providing implementations of various generative modeling algorithms for unconditional image generation on the MNIST dataset. Each algorithm is implemented in a separate trainer class within the `trainers` module, allowing for easy comparison and study of different approaches.

## Repository Outline

- **networks/**: Contains model architectures used by the trainers.
- **trainers/**: Implements various generative modeling algorithms for MNIST generation.
- **main.py**: Script to run training and sampling for selected models.
- **README.md**: This file, providing an overview and citations.

## Implemented Algorithms

### 1. Denoising Diffusion Probabilistic Models (DDPM)
- **Trainer**: `trainers/ddpm.py`
- **Citation**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2006.11239. [arXiv](https://arxiv.org/abs/2006.11239)

### 2. Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
- **Trainer**: `trainers/edm.py`
- **Citation**: Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. arXiv preprint arXiv:2206.00364. [arXiv](https://arxiv.org/abs/2206.00364)

### 3. Rectified Flow (RF)
- **Trainer**: `trainers/rf.py`
- **Citation**: Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. arXiv preprint arXiv:2209.03003. [arXiv](https://arxiv.org/abs/2209.03003)

### 4. Variational Rectified Flow (VRF)
- **Trainer**: `trainers/vrf.py`
- **Citation**: Guo, P., & Schwing, A.G. (2025). Variational Rectified Flow. arXiv preprint arXiv:2502.09616. [arXiv](https://arxiv.org/pdf/2502.09616)

### 5. PFGM++ (PFGMpp)
- **Trainer**: `trainers/pfgmpp.py`
- **Citation**: Xu, Y., Liu, Z., Tian, Y., Tong, S., Tegmark, M., & Jaakola, T. (2023). PFGM++: Unlocking the Potential of Physics-Inspired Generative Models. arXiv preprint arXiv:2206.06910. [arXiv](https://arxiv.org/abs/2302.04265)

### 6. Consistency Models (CM)
- **Trainer**: `trainers/cm.py`
- **Citation**: Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency Models. arXiv preprint arXiv:2303.01469. [arXiv](https://arxiv.org/abs/2303.01469)

### 7. Cold Diffusion (CD)
- **Trainer**: `trainers/cd.py`
- **Citation**: Bansal, A., Borgnia, E., Chu, H.M., Li, J.S., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., & Goldstein, T. (2022). Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise. arXiv preprint arXiv:2208.09392. [arXiv](https://arxiv.org/abs/2208.09392)

### 8. Variational Diffusion Models (VDM)
- **Trainer**: `trainers/vdm.py`
- **Citation**: Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). Variational Diffusion Models. arXiv preprint arXiv:2107.00630. [arXiv](https://arxiv.org/abs/2107.00630)

## Usage

To run the training and sampling for a specific model:

```bash
python main.py --model <model_name> [additional arguments]
```

Replace `<model_name>` with one of the implemented algorithms (e.g., `ddpm`, `edm`, `rf`, etc.). Use `--help` to see all available options.

## Code References

- Variational Encoder is based on [black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py)
- DiT is from [facebookresearch/DiT](https://github.com/facebookresearch/DiT/blob/main/models.py)
- EDM is based on [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion.py)


## Notes

This repository was built for personal use and for teaching a section on generative modeling. A few bugs crept in while refactoring so if you happen to find any, or see any errors, please let me know!

---
*This README was generated with assistance from Claude-3.5-Sonnet*
