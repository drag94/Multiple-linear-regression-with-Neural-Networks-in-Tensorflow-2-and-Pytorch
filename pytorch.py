import pandas as pd
from datetime import datetime
import torch
from torchvision import models
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

# per la gpu
# torch.cuda.is_available()
    # model.cuda()
model_torch = torch.nn.Sequential(
          torch.nn.Linear(in_features=X_true.shape[1], out_features=1,bias=True),
          )
loss_torch = torch.nn.MSELoss()
opt_torch = torch.optim.SGD(model_torch.parameters(), lr=1e-2)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_torch, gamma=0.9)

#converto il dataframe di input e target in tensore con i float
X_true_torch = torch.from_numpy(X_true.values)
X_true_torch= X_true_torch.float()
y_torch = torch.from_numpy(y.values).float()

# keras-like summary()
vgg = models.vgg16()
summary(model_torch)

#parameters logs
writer = SummaryWriter("/home/davide/PycharmProjects/dnn_linearReg/tensorboard-logs_pytorch"+ \
          datetime.now().strftime("%Y%m%d-%H%M%S"))

for epoch in range(200):
    # zero the parameter gradients
    opt_torch.zero_grad()
    # forward + backward + optimize
    output = model_torch(X_true_torch)
    loss = loss_torch(output, y_torch)
    #log the loss
    writer.add_scalar('Loss/train', loss.item(),epoch)
    loss.backward()
    opt_torch.step()
    # scheduler.step()
    print(f'epoch {epoch}, loss {loss.item()}')
model_torch.state_dict()

# vedere i pesi e bias del primo layer
pesi_torch =pd.DataFrame([model_torch.state_dict()['0.weight'][0].tolist()],columns=X_true.columns)  #c'è solo un layer che è quello di input
bias_torch =pd.DataFrame([model_torch.state_dict()['0.bias'][0].tolist()],columns=['bias'])
df_torch = pd.concat([pesi_torch,bias_torch],axis=1)





