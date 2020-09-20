from tqdm import tqdm 
import config 
import torch

def train_fn(model, dataloader, optimizer, loss_fn):
    model.train()
    fin_loss = 0 
    tk0 = tqdm(dataloader, total = len(dataloader))
    for data in tk0:
        inp1, inp2, label = data["images0"], data["images1"], data["label"]
        inp1, inp2, label = inp1.to(config.DEVICE), inp2.to(config.DEVICE), label.to(config.DEVICE)

        optimizer.zero_grad()
        op1, op2 = model(inp1, inp2)
        loss = loss_fn(op1, op2, label)
        loss.backward()
        optimizer.step()

        del inp1, inp2, label, op1, op2
        torch.cuda.empty_cache()
        fin_loss += loss.item()
    return fin_loss / len(dataloader)





