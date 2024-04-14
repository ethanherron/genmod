from functools import partial
from pathlib import Path
from copy import deepcopy
import time
import argparse

import dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from unet import UNetModel
from mlx.utils import tree_flatten
from PIL import Image

def pad(array):
    '''
    Reshape an array of timesteps into 4D array to match image shapes.
    
    array: (B) array of timesteps
    
    returns: (B, 1, 1, 1) array of timesteps
    '''
    return mx.reshape(array, (array.shape[0], 1, 1, 1))


def grid_image_from_batch(image_batch, num_rows):
    """
    Generate a grid image from a batch of images.
    Assumes input has shape (B, H, W, C).
    """

    B, H, W, _ = image_batch.shape

    num_cols = B // num_rows

    # Calculate the size of the output grid image
    grid_height = num_rows * H
    grid_width = num_cols * W

    # Normalize and convert to the desired data type
    image_batch = np.array(image_batch * 255).astype(np.uint8)

    # Reshape the batch of images into a 2D grid
    grid_image = image_batch.reshape(num_rows, num_cols, H, W, -1)
    grid_image = grid_image.swapaxes(1, 2)
    grid_image = grid_image.reshape(grid_height, grid_width, -1)

    # Convert the grid to a PIL Image
    return Image.fromarray(grid_image.squeeze())

def loss_fn(model, xt, t, velocity):
    '''
    Rectified flow loss function.
    
    model: UNetModel
    xt: tensor of shape (B, H, W, C)
    t: tensor of shape (B,)
    target: tensor of shape (B, H, W, C)
    
    velocity_hat: tensor of shape (B, H, W, C)
    '''
    velocity_hat = model(xt, t)
    
    return nn.losses.mse_loss(velocity_hat, velocity, reduction="mean")

def get_train_tuple(x0, x1):
    '''
    Rectified flow training tuple.
    
    x0: tensor of shape (B, H, W, C)
    x1: tensor of shape (B, H, W, C)
    
    t: tensor of shape (B,)
    xt: tensor of shape (B, H, W, C)
    target: tensor of shape (B, H, W, C)
    '''
    t = mx.random.uniform(low=0.0, high=1.0, shape=(x0.shape[0],))
    xt = pad(t) * x0 + (1. - pad(t)) * x1
    velocity = x1 - x0
    return xt, t, velocity

def euler_ode(model, x0, nfe):
    '''
    Euler method for solving ODE. Trained model is used to predict velocity term.
    
    model: UNetModel
    x0: tensor of shape (B, H, W, C)
    nfe: number of forward euler steps
    '''
    dt = 1.0 / nfe
    x_t = deepcopy(x0)
    
    for i in range(nfe):
        t = mx.ones((x_t.shape[0],)) * (i * dt)
        velocity = model(x_t, t)
        x_t = x_t + velocity * dt
        
    return x_t

def generate(model, out_file, num_samples=9, nfe=100):
    '''
    Generate a grid of images with recfified flow model.
    
    model: UNetModel
    out_file: str
    nfe: number of forward euler steps
    '''
    x0 = mx.random.normal(shape=(num_samples, 32, 32, 1))
    
    images = euler_ode(model, x0, nfe)
    
    grid_image = grid_image_from_batch(images, num_rows=3)
    grid_image.save(out_file)
    
def main(args):
    # Load the data
    img_size = (32, 32, 1)
    train_iter, _ = dataset.mnist(
        batch_size=args.batch_size, img_size=img_size[:2]
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    model = UNetModel()
    mx.eval(model.parameters())

    num_params = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    print("Number of trainable params: {:0.04f} M".format(num_params / 1e6))

    optimizer = optim.AdamW(learning_rate=args.lr)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(xt, t, velocity):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, xt, t, velocity)
        optimizer.update(model, grads)
        return loss

    for e in range(1, args.epochs + 1):
        # Reset iterators and stats at the beginning of each epoch
        train_iter.reset()
        model.train()

        # Train one epoch
        tic = time.perf_counter()
        loss_acc = 0.0
        throughput_acc = 0.0

        # Iterate over training batches
        for batch_count, batch in enumerate(train_iter):
            X1 = mx.array(batch["image"])
            X0 = mx.random.normal(shape=(X1.shape[0], 32, 32, 1))
            Xt, t, velocity = get_train_tuple(X0, X1)
            throughput_tic = time.perf_counter()

            # Forward pass + backward pass + update
            loss = step(Xt, t, velocity)

            # Evaluate updated model parameters
            mx.eval(state)

            throughput_toc = time.perf_counter()
            throughput_acc += X1.shape[0] / (throughput_toc - throughput_tic)
            loss_acc += loss.item()

            if batch_count > 0 and (batch_count % 10 == 0):
                print(
                    " | ".join(
                        [
                            f"Epoch {e:4d}",
                            f"Loss {(loss_acc / batch_count):10.2f}",
                            f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                            f"Batch {batch_count:5d}",
                        ]
                    ),
                    end="\r",
                )
        toc = time.perf_counter()

        print(
            " | ".join(
                [
                    f"Epoch {e:4d}",
                    f"Loss {(loss_acc / batch_count):10.2f}",
                    f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                    f"Time {toc - tic:8.1f} (s)",
                ]
            )
        )

        model.eval()

        # Generate images
        generate(model, save_dir / f"generated_{e:03d}.png")

        model.save_weights(str(save_dir / "weights.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU acceleration",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed"
        )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=15, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/rectified_flow/",
        help="Path to save the model and reconstructed images.",
    )

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")

    main(args)