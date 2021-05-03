import Config
import torch
from models import Preprocessing, TransformerEncoder, TransformerEncoderLayer, VisionTransformer
from dataset import my_Cifar10
from tqdm.notebook import tqdm
import numpy as np

"""
    Model Setup
"""

CONFIGS = {
    'ViT-B/16' : Config.get_ViT_B_16(),
    'ViT-B/32' : Config.get_ViT_B_32(),
    'ViT-L/16' : Config.get_ViT_L_16(),
    'ViT-L/32' : Config.get_ViT_L_32(),
    'ViT-H/16' : Config.get_ViT_H_16(),
    'ViT-H/32' : Config.get_ViT_H_32(),
    'for_test' : Config.for_test_ViT_B_14()
}

config = CONFIGS['for_test'] # architecture

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

preprocessing = Preprocessing(image_size=(224,224), hid_dim=config.hid_dim, patch_size=config.patch_size, is_hybrid = True).to(DEVICE)
encoder_layer = TransformerEncoderLayer(hid_dim=config.hid_dim, ff_dim=config.ff_dim, n_heads=config.n_heads).to(DEVICE)
transformer = TransformerEncoder(encoder_layer,config.num_layers).to(DEVICE)
vit = VisionTransformer(preprocessing,transformer,num_classes=100, hid_dim=config.hid_dim).to(DEVICE)

"""
    Data Load
"""

train_dataset, test_dataset, train_dataloader, test_dataloader = my_Cifar10()


"""
    Train
"""

optim = torch.optim.Adam(params=vit.parameters(),lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
vit.train()

epochs = 30

for epoch in range(epochs):
    losses = []
    print("\n Epoch {}/{}".format(epoch+1, epochs))
    for x, y in tqdm(train_dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = vit(x) # [BS, N+1]가 나옴.
        loss = criterion(pred,y)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("This Loss is : {}".format(np.mean(losses)))


"""
    Evaluate
"""

vit.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in tqdm(test_dataloader):
        x, test_y = x.to(DEVICE), y.to(DEVICE)
        pred = vit(x)  # [BS, N+1]가 나옴.
        # pred = pred.index_select(index=torch.tensor([0], device=DEVICE),dim=1).squeeze(1) # N+1개 중 class token인 가장 앞에거 가져옴.
        _, pred_idx = torch.max(pred, 1)

        now_correct = (pred_idx == test_y).sum().item()
        print("이번에 맞은 횟수는 {}, {:.2f}% 입니다.".format(now_correct, 100 * now_correct / 16))

        correct += now_correct
        total += 16
print("===========================\n결과 : {:.2f}%\===========================".format(100 * correct / total))