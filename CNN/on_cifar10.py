import torch

from Datasets.cifar10 import init_dataloaders
from utils import get_model, DefinedModels, train_eval, train_eval_lightning, load_lightning_model_and_test

if __name__ == '__main__':
    cnn = get_model(model=DefinedModels.CNN, number_of_inputs=3, number_of_outputs=10)
    resnet18 = get_model(model=DefinedModels.RESNET18, number_of_inputs=3, number_of_outputs=10)
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(batch_size=128)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "cnn": cnn,
        "resnet18": resnet18
    }
    for model_name, model in models.items():
        train_dataloader, val_dataloader, test_dataloader = init_dataloaders(batch_size=128)
        train_eval(model=model,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   test_dataloader=test_dataloader,
                   num_epochs=num_epochs,
                   device=device)

        train_eval_lightning(model=model,
                             train_dataloader=train_dataloader,
                             val_dataloader=val_dataloader,
                             test_dataloader=test_dataloader,
                             num_epochs=num_epochs,
                             name=f"cifar10_{model_name}")

        load_lightning_model_and_test(model=model,
                                      test_dataloader=test_dataloader,
                                      device=device,
                                      name=f"cifar10_{model_name}",
                                      version_no=0)
