from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class Pipeline:
    def __init__(self):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])  # Normalize for grayscale images

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.lossFunc = nn.CrossEntropyLoss()

    def train_step(self, model, optimizer):
        model.train()
        epochloss = 0
        for batchcount, (images, labels) in enumerate(self.trainloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()

            y = model(images)

            loss = self.lossFunc(y, labels)     
            loss.backward()

            optimizer.step()
            
            epochloss += loss.item()

        return epochloss

    def val_step(self, model):
        correct = 0
        total = 0
        model.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batchcount, (images, labels) in enumerate(self.testloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                y = model(images)
                _, predicted = torch.max(y, 1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()       
        ConfusionMatrix(LabelsTrue=all_labels, LabelsPred=all_preds, class_names=self.classes)
        return correct*100/total
        
def ConfusionMatrix(LabelsTrue, LabelsPred, class_names):
    LabelsTrue = np.array(LabelsTrue)
    LabelsPred = np.array(LabelsPred)

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(LabelsTrue, LabelsPred):
        cm[t, p] += 1

    class_numbers = [f" ({i})" for i in range(n_classes)]

    accuracy = 100.0 * np.sum(LabelsTrue == LabelsPred) / len(LabelsTrue)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
            )

    fig.tight_layout()
    fig.savefig("Confusion.pdf", bbox_inches="tight")
    plt.close(fig)
    return None


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

LOSS_FN = nn.CrossEntropyLoss()
