# 🤖 Makemore Remake: A Deep Dive into Neural Networks

## Project Overview 📚

This project is a **remake** of the `makemore` project originally created by **Andrej Karpathy** (available [here](https://github.com/karpathy/makemore)).

My primary goal is not to create a novel piece of software, but to **deeply understand** the foundational concepts of modern neural network training. By meticulously following and recreating the code from Karpathy's video series, I am aiming to solidify my understanding of the mechanisms that power large language models and other deep learning applications.

---

## 🎯 Learning Objectives

This project is a hands-on exercise focused on demystifying core concepts, including:

* **Understanding Backpropagation from First Principles:** Implementing and tracing the chain rule to see exactly how gradients flow backward through a computational graph.
* **Gradient Descent and Optimization:** Observing how gradients are used to iteratively adjust weights and minimize the loss function.
* **Batch Normalization:** Learning why and how Batch Normalization works, its implementation details, and its effect on training stability and speed.
* **Building a Multi-Layer Perceptron (MLP):** Constructing a simple neural network from scratch, including layers, non-linearities, and the loss function.
* **PyTorch Mechanics:** Gaining practical experience with PyTorch tensors, automatic differentiation (`torch.autograd`), and module creation.

---

## 💻 Project Structure (Based on the Series)

The project mirrors the progression of the `makemore` video series, with different files or notebooks corresponding to key evolutionary stages of the code:

* **`01_Bigram.ipynb`**: The initial step: a simple **bigram language model** based on basic counts.
* **`02_MLP.ipynb`**: Transitioning from counts to weights: creating a simple **Multi-Layer Perceptron (MLP)** character-level model.
* **`03_Activation_Gradiants_BatchNorm.ipynb`**: Implementing and exploring the mechanics of **Activation**, **Gradiants** & **Batch Normalization**.
* **`04_Backprop.ipynb`**: Exploring the mechanics of **Back Propagation**.
* **... and so on (as I continue through the series)**

---

## 🙏 Attribution and Disclaimer

**I am not the original creator of this concept or the foundational code structure.**

This project is a **learning exercise**. Substantial portions of the code are direct implementations of the techniques and logic taught by Andrej Karpathy.

* **Original Creator:** **Andrej Karpathy**
* **Original Project:** **makemore**
* **Video Tutorial Series:** [YouTube Playlist]([https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cEbeCmnTRg7hQRto](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ))

My contribution is the **active effort to implement, debug, annotate, and understand** every line of code, ensuring that the concepts (like backpropagation and gradient calculation) are clear and well-documented for my own learning.
