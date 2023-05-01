# DDPM Model on CIFAR-10

**Model Name**

* Denoise Diffusion Probabilistic Model (DDPM) for generating CIFAR-10 images.

This model is a generative model that can be used to generate CIFAR-10 images. It is based on the Denoise Diffusion Probabilistic Model (DDPM) proposed by Ho et al. in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model is trained to generate images that are similar to the images in the CIFAR-10 dataset.

**Model Description**

* DDPM Model

## `DDPM` class

DDPM stands for Denoising Diffusion Probabilistic Model. It is a type of generative model that is used to generate images. DDPM works by first adding noise to an image, then training a model to denoise the image. The model is then used to generate new images by adding noise to a random image and then denoising the image.

DDPM is a relatively new generative model, but it has quickly become one of the most popular generative models for image generation. This is because DDPM is able to generate high-quality images with a relatively small amount of training data.

DDPM was first proposed by Jonathan Ho, Ajay Jain, and Pieter Abbeel in their paper "Denoising Diffusion Probabilistic Models". The paper was published in 2020 at the International Conference on Machine Learning (ICML).

DDPM has been used to generate a variety of different types of images, including faces, animals, and objects. DDPM has also been used to generate images of text, code, and music.

DDPM is a powerful generative model that is still under development. However, it has the potential to be used for a variety of different applications, such as image generation, image editing, and text generation.

Here are some of the advantages of DDPM:

* It can generate high-quality images with a relatively small amount of training data.
* It is relatively easy to train.
* It is versatile and can be used to generate a variety of different types of images.

Here are some of the disadvantages of DDPM:

* It can be computationally expensive to train.
* It can be difficult to control the output of the model.
* It can sometimes generate blurry or distorted images.

Overall, DDPM is a powerful generative model that has the potential to be used for a variety of different applications.

The `DDPM` class is a PyTorch module that implements the diffusion probabilistic model. The class takes the following arguments:

* `network`: A neural network that is used to predict the noise at each diffusion step.
* `n_steps`: The number of diffusion steps.
* `min_beta`: The minimum value of $\beta_t$.
* `max_beta`: The maximum value of $\beta_t$.
* `device`: The device to use for training and inference.
* `image_chw`: The shape of the input images.

The `DDPM` class has two methods:

* `forward()`: This method takes an input image and a time step and returns the noisy image at the specified time step.
* `backward()`: This method takes an input image and a time step and returns the noise that was added to the image at the specified time step.

## `forward()` method

The `forward()` method takes the following arguments:

* `x0`: The input image.
* `t`: The time step.
* `eta`: The noise.

The `forward()` method first computes the alpha bar value for the specified time step. The alpha bar value is a scaling factor that is used to add noise to the input image. The `forward()` method then computes the noisy image by adding the noise to the input image, scaled by the alpha bar value.

## `backward()` method

The `backward()` method takes the following arguments:

* `x`: The input image.
* `t`: The time step.

The `backward()` method first runs the input image through the neural network. The neural network predicts the noise that was added to the input image at the specified time step. The `backward()` method then returns the predicted noise.

* Unet Model with Sinuoidal Positional Embedding

UNet with sinusoidal embedding is a type of convolutional neural network (CNN) that is used for image denoising. It is a variation of the U-Net architecture, which is a popular CNN architecture for image segmentation.

The main difference between UNet with sinusoidal embedding and the standard U-Net architecture is the use of sinusoidal embeddings in the skip connections. Skip connections are connections that skip over some of the layers in a CNN. In the standard U-Net architecture, the skip connections use linear embeddings. In UNet with sinusoidal embedding, the skip connections use sinusoidal embeddings.

Sinusoidal embeddings are a type of embedding that uses sinusoidal functions to represent the data. This has several advantages over using linear embeddings. First, sinusoidal embeddings are more efficient to compute. Second, sinusoidal embeddings are more expressive and can represent more complex relationships in the data.

UNet with sinusoidal embedding has been shown to be more effective than the standard U-Net architecture for image denoising. This is because the sinusoidal embeddings are able to better represent the spatial relationships in the data.

Here are some of the advantages of UNet with sinusoidal embedding:

* It is more efficient to compute than the standard U-Net architecture.
* It is more expressive and can represent more complex relationships in the data.
* It is more effective for image denoising than the standard U-Net architecture.

Here are some of the disadvantages of UNet with sinusoidal embedding:

* It is more complex to train than the standard U-Net architecture.
* It requires more training data than the standard U-Net architecture.
* It is not as well-suited for other tasks, such as image classification, as the standard U-Net architecture.

Overall, UNet with sinusoidal embedding is a powerful architecture for image denoising. It is more efficient to compute and more expressive than the standard U-Net architecture. However, it is more complex to train and requires more training data.

## `MyUNet` class

The `MyUNet` class is a PyTorch module that implements a U-Net model. The class takes the following arguments:

* `n_steps`: The number of diffusion steps.
* `time_emb_dim`: The dimension of the positional embedding.

The `MyUNet` class has two methods:

* `forward()`: This method takes an input image and a time step and returns the denoised image at the specified time step.
* `_make_te()`: This method is a helper method that creates a temporal embedding layer.

## `forward()` method

The `forward()` method takes the following arguments:

* `x`: The input image.
* `t`: The time step.

The `forward()` method first computes the positional embedding for the input image. The positional embedding is a learned embedding that encodes the temporal information of the input image. The `forward()` method then denoises the input image by passing it through the U-Net model. The U-Net model is a convolutional neural network that has been specifically designed for image denoising. The `forward()` method finally returns the denoised image.

## `_make_te()` method

The `_make_te()` method takes the following arguments:

* `dim_in`: The input dimension of the temporal embedding.
* `dim_out`: The output dimension of the temporal embedding.

The `_make_te()` method creates a temporal embedding layer. The temporal embedding layer is a linear layer that maps the input dimension to the output dimension. The `_make_te()` method then applies a non-linear activation function to the output of the linear layer. The `_make_te()` method finally returns the temporal embedding layer.

* Final Model 

```python
# Defining model

n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors

ddpm = DDPM(
    MyUNet(n_steps),
    n_steps=n_steps,
    min_beta=min_beta,
    max_beta=max_beta,
    device=device,
)
```

The model definition defines a diffusion probabilistic model (DDPM) model. The model consists of two parts:

* A U-Net model that is used to denoise images.
* A diffusion model that is used to add noise to images.

The U-Net model is a convolutional neural network that has been specifically designed for image denoising. The diffusion model is a probabilistic model that is used to add noise to images.

The model definition takes the following arguments:

* `MyUNet`: The U-Net model.
* `n_steps`: The number of diffusion steps.
* `min_beta`: The minimum value of $\beta_t$.
* `max_beta`: The maximum value of $\beta_t$.
* `device`: The device to use for training and inference.

The model definition returns a DDPM model.

* Hyperparameters and Execution Options.

## Setting Reproducibility

The following code is used to set reproducibility:

```
SEED = 0 # Set random seed
random.seed(SEED) # Set Python random seed
np.random.seed(SEED) # Set NumPy random seed
torch.manual_seed(SEED) # Set PyTorch random seed
```

This code sets the random seed to a constant value, which ensures that the results of the code are reproducible. This is useful for debugging and testing code.

## Setting Execution Options

The following code is used to set execution options:

```
no_train = False # If True, no training is performed
batch_size = 128 # Batch size
n_epochs = 50 # Number of epochs
lr = 0.001 # Learning rate
store_path = "ddpm_model.pt" # Path to store the model
```

This code sets the following execution options:

* `no_train`: If `True`, no training is performed.
* `batch_size`: The batch size for training.
* `n_epochs`: The number of epochs to train for.
* `lr`: The learning rate for training.
* `store_path`: The path to store the trained model.

## Getting Device

The following code is used to get the device to use for training:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This code checks if a GPU is available, and if so, uses the GPU for training. Otherwise, it uses the CPU for training.

## Printing Device

The following code is used to print the device that will be used for training:


print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


This code prints the device that will be used for training, along with the name of the device if a GPU is being used.

**Model Training Data**

The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

The CIFAR-10 dataset is commonly used to train machine learning models for image classification. The dataset is relatively small, but it is still challenging to train a model to achieve good performance on the dataset.

The CIFAR-10 dataset was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The dataset was first released in 2009.

The CIFAR-10 dataset is freely available for download from the website of the Canadian Institute for Advanced Research (CIFAR).

* Loading Dataset

## `transform`

The `transform` is a composition of image transformations. The transformations are applied in the order they are specified in the list. The following transformations are applied:

* `Grayscale(1)`: Converts the image to grayscale.
* `Resize((28, 28))`: Resizes the image to 28x28 pixels.
* `ToTensor()`: Converts the image to a tensor.
* `Lambda(lambda x: (x - 0.5) * 2)`: Normalizes the image to the range [-1, 1].

## `dataset`

The `dataset` is a CIFAR10 dataset. The CIFAR10 dataset is a collection of 60,000 32x32 color images of 10 classes, with 6,000 images per class. The dataset is split into a training set of 50,000 images and a test set of 10,000 images.

## `loader`

The `loader` is a data loader. The data loader loads the data from the dataset in batches. The batch size is specified by the `batch_size` variable. The data loader also shuffles the data, which helps to prevent overfitting.


**Model Training Process**

The training loop is a function that trains a DDPM model. The function takes the following arguments:

* `ddpm`: The DDPM model.
* `loader`: The data loader.
* `n_epochs`: The number of epochs to train for.
* `optim`: The optimizer.
* `device`: The device to use for training and inference.
* `display`: A boolean value that indicates whether to display images generated at each epoch.
* `store_path`: The path to the file where the model should be stored.

The training loop works as follows:

1. For each epoch:
    * Initialize the epoch loss to 0.
    * For each batch in the data loader:
        * Load the batch of images to the device.
        * Generate noisy images from the batch of images.
        * Get the model's estimation of the noise from the noisy images.
        * Compute the mean squared error between the model's estimation of the noise and the actual noise.
        * Update the epoch loss.
        * Backpropagate the loss.
        * Update the model parameters.
    * Display images generated at this epoch, if `display` is True.
    * Log the epoch loss.
    * If the epoch loss is better than the best loss so far, store the model to the file specified by `store_path`.

The training loop returns a list of epoch losses.

**Model Evaluation**

* Utility Functions

## `show_images`

The `show_images` function takes a list of images and displays them as sub-pictures in a square. The function takes the following arguments:

* `images`: A list of images. Each image should be a NumPy array with shape `(h, w, c)`, where `h` is the height of the image, `w` is the width of the image, and `c` is the number of channels.
* `title`: The title of the figure.

The function first converts the images to CPU NumPy arrays. It then defines the number of rows and columns in the figure. It then populates the figure with sub-plots. For each sub-plot, the function displays the image at the corresponding index in the list of images. The function finally displays the figure.

## `generate_new_images`

The `generate_new_images` function takes a DDPM model, a number of samples to be generated, and a device, and returns some newly generated samples. The function takes the following arguments:

* `ddpm`: A DDPM model.
* `n_samples`: The number of samples to be generated.
* `device`: The device to use for generating the samples.
* `frames_per_gif`: The number of frames to generate in the GIF.
* `gif_name`: The name of the GIF file to save the generated images to.
* `c`: The number of channels in the generated images.
* `h`: The height of the generated images.
* `w`: The width of the generated images.

The function first creates a list of frames. It then starts from random noise and denoises the image iteratively. For each iteration, the function adds some more noise to the image. The function adds frames to the list of frames at the specified intervals. The function finally saves the list of frames to a GIF file.

The model testing and evaluation code is used to test and evaluate the performance of a trained DDPM model. The code takes the following arguments:

* `best_model`: The trained DDPM model.
* `store_path`: The path to the file where the model was stored.
* `no_train`: A boolean value that indicates whether to train the model.

The model testing and evaluation code works as follows:

1. If `no_train` is False, the code loads the trained model from the file specified by `store_path`.
2. The code plots the loss over epochs.
3. The code generates a new image from the model.
4. The code calculates the FID score between the generated image and a real image.
5. The code generates 100 new images from the model and saves them to a GIF file.
6. The code displays the generated images.

The model testing and evaluation code returns the FID score between the generated image and a real image.

* Observations and metrics

FID stands for Fréchet Inception Distance. It is a metric that measures the similarity between two distributions of images. The FID score is calculated by first calculating the mean and covariance matrices of the two distributions. The mean and covariance matrices are then used to calculate the Fréchet distance, which is a measure of the distance between two multivariate Gaussian distributions.

A lower FID score indicates that the two distributions are more similar. A score of 177 is a relatively high FID score, which suggests that the two distributions are not very similar. This could be due to a number of factors, such as the quality of the generated images, the quality of the real images, or the choice of the FID metric.

Here are some possible reasons why you got a score of 177:

* The quality of the generated images is not very good. This could be due to a number of factors, such as the training dataset, the architecture of the model, or the hyperparameters used to train the model.
* The quality of the real images is not very good. This could be due to a number of factors, such as the dataset used to collect the images, the way the images were captured, or the way the images were processed.
* The choice of the FID metric is not appropriate for the task at hand. The FID metric is a good measure of the similarity between two distributions of images, but it may not be the best measure for all tasks. For example, the FID metric may not be the best measure for tasks that require a high degree of realism in the generated images.

If you are concerned about the quality of your generated images, you can try to improve the quality of the images by improving the quality of the training dataset, the architecture of the model, or the hyperparameters used to train the model. You can also try to improve the quality of the real images by using a better dataset, capturing the images in a better way, or processing the images in a better way. Finally, you can try to use a different metric to evaluate the quality of your generated images.

**Model Deployment**

* Hardware and Software Requirements

* Hardware:
    * A Google Colaboratory notebook.
    * A GPU with at least 4GB of memory.
* Software:
    * Python 3.6 or higher.
    * The following Python libraries:
        * NumPy
        * TensorFlow
        * Matplotlib

* Deployment Process

1. Clone the DDPM repository to your Google Colaboratory notebook.
2. Install the required Python libraries.

**Model Usage**

* Input Data Format

The input data format for the DDPM model is a NumPy array of shape (height, width, channels). The height and width of the image must be divisible by 8. The channels of the image must be either 1 (grayscale) or 3 (RGB).

* Output Data Format

The output data format for the DDPM model is a NumPy array of shape (height, width, channels). The height and width of the image will be the same as the input image. The channels of the image will be the same as the input image.

* Usage

To use the DDPM model, you can use the following steps:

1. Load the DDPM model.
2. Generate an image.
3. Save the image.


**Model References**

* Papers
- *Denoising Diffusion Implicit Models* by Song et. al. (https://arxiv.org/abs/2010.02502);
- *Improved Denoising Diffusion Probabilistic Models* by Nichol et. al. (https://arxiv.org/abs/2102.09672);
- *Hierarchical Text-Conditional Image Generation with CLIP Latents* by Ramesh et. al. (https://arxiv.org/abs/2204.06125);
- *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* by Saharia et. al. (https://arxiv.org/abs/2205.11487);

* Acknowledgements

 - <b>Lilian Weng</b>'s [blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): <i>What are Diffusion Models?</i>
 - <b>abarankab</b>'s [Github repository](https://github.com/abarankab/DDPM)
 - <b>Jascha Sohl-Dickstein</b>'s [MIT class](https://www.youtube.com/watch?v=XCUlnHP1TNM&ab_channel=AliJahanian)
 - <b>Niels Rogge</b> and <b>Kashif Rasul</b> [Huggingface's blog](https://huggingface.co/blog/annotated-diffusion): <i>The Annotated Diffusion Model</i>
 - <b>Outlier</b>'s [Youtube video](https://www.youtube.com/watch?v=HoKDTa5jHvg&ab_channel=Outlier)
 - <b>AI Epiphany</b>'s [Youtube video](https://www.youtube.com/watch?v=y7J6sSO1k50&t=450s&ab_channel=TheAIEpiphany)
