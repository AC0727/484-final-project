{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e0f02ad-6163-4593-8d98-44db63300d1c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "from torchvision.models import resnet18, ResNet18_Weights\n",
    "\n",
    "# we use resnet18 over 34 since we prioritize speed and efficiency here, as our classifier\n",
    "# task is quite simple: output 1 if the image contains a cat, 0 otherwise\n",
    "\n",
    "class CatClassifier(nn.Module):\n",
    "    def __init__(self, backbone=\"resnet18\", pretrained=True, freeze_backbone=False):\n",
    "        super().__init__()\n",
    "        weights = ResNet18_Weights.DEFAULT if pretrained else None\n",
    "        self.backbone = resnet18(weights=weights)\n",
    "\n",
    "         if freeze_backbone:\n",
    "            for param in self.backbone.parameters():\n",
    "                param.requires_grad = False\n",
    "\n",
    "        in_features = self.backbone.fc.in_features\n",
    "        self.backbone.fc = nn.Linear(in_features, 1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # x: image tensor of shape [B, 3, H, W]\n",
    "        # returns: logits of shape [B, 1]\n",
    " \n",
    "        return self.backbone(x)\n",
    "\n",
    "    def predict_probs(self, x):\n",
    "        # returns probabilities in [0, 1], shape [B, 1]\n",
    "        \n",
    "        logits = self.forward(x)\n",
    "        return torch.sigmoid(logits)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (myenv)",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
