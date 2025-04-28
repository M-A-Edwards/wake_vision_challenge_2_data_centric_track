# 🚀 **Data-Centric Track**

Welcome to the **Data-Centric Track** of the **Wake Vision Challenge 2**! 🎉

The goal of this track is to **push the boundaries of tiny computer vision** by enhancing the data quality of the [Wake Vision Dataset](https://wakevision.ai/).

🔗 **Learn More**: Wake Vision Challenge Details (TBD)

---

## 🌟 **Challenge Overview**

Participants are invited to:

1. **Enhance the provided dataset** to improve person detection accuracy.
2. Train the [MCUNet-VWW2 model](https://github.com/mit-han-lab/mcunet), a state-of-the-art person detection model, on the enhanced dataset.
3. Assess quality improvements on the public test set.

You can modify the **dataset** however you like, but the **model architecture must remain unchanged**. 🛠️

---

## 🛠️ **Getting Started**

### Step 1: Install Docker Engine 🐋

First, install Docker on your machine:
- [Install Docker Engine](https://docs.docker.com/engine/install/).

---

### Step 2: Download the Wake Vision dataset

1. [Sign up](https://dataverse.harvard.edu/dataverseuser.xhtml;jsessionid=b78ff6ae13347e089bc776b916e9?editMode=CREATE&redirectPage=%2Fdataverse_homepage.xhtml) on Harvard Dataverse

2. On your account information page go to the API Token tab and create a new API Token for Harvard Dataverse

3. Substitute "your-api-token-goes-here" with your API token in the following command and run it inside the directory where you cloned this repository to download and build the Wake Vision Dataset:

```bash
sudo docker run -it --rm -v "$(pwd):/tmp" -w /tmp tensorflow/tensorflow:2.19.0 python download_and_build_wake_vision_dataset.py your-api-token-goes-here
```

💡 **Note**: Make sure to have at least 600 GB of free disk space.

---

### 💻 **Training Without a GPU**

Run the following command inside the directory where you cloned this repository:

```bash
sudo docker run -it --rm -v "$(pwd):/tmp" -w /tmp tensorflow/tensorflow:2.19.0 python data_centric.py
```

- This trains the [MCUNet-VWW2 model](https://github.com/mit-han-lab/mcunet) on the original dataset.
- Modify the dataset to improve the model's test accuracy by correcting labels or augmenting data.

### ⚡ **Training With a GPU**

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Verify your [GPU drivers](https://ubuntu.com/server/docs/nvidia-drivers-installation).

Run the following command inside the directory where you cloned this repository:

```bash
sudo docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:2.19.0-gpu python data_centric.py
```

- This trains the [MCUNet-VWW2 model](https://github.com/mit-han-lab/mcunet) on the original dataset.
- Modify the dataset to enhance test accuracy while keeping the model architecture unchanged.

---

## 🎯 **Tips for Success**

- **Focus on Data Quality**: Explore label correction, data augmentation, and other preprocessing techniques.
- **Stay Efficient**: The dataset is large, so plan your modifications carefully.
- **Collaborate**: Join the community discussion on [Discord](https://discord.com/channels/803180012572114964/1323721087170773002) to share ideas and tips!

---

## 📚 **Resources**

- [MCUNet-VWW2 Model Documentation](https://github.com/mit-han-lab/mcunet)
- [Docker Documentation](https://docs.docker.com/)
- [Wake Vision Dataset](https://wakevision.ai/)

---

## 📞 **Contact Us**

Have questions or need help? Reach out on Discord (TBD).

---

🌟 **Happy Innovating and Good Luck!** 🌟
