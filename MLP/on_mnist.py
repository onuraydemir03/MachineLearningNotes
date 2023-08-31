import torch

from Datasets.mnist import init_dataloaders
from utils import get_model, DefinedModels, train_eval, train_eval_lightning, load_lightning_model_and_test

if __name__ == '__main__':
    mlp = get_model(model=DefinedModels.MLP, number_of_inputs=784, number_of_outputs=10)
    cnn = get_model(model=DefinedModels.CNN, number_of_inputs=1, number_of_outputs=10)
    resnet18 = get_model(model=DefinedModels.RESNET18, number_of_inputs=1, number_of_outputs=10)
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {
        "mlp": mlp,
        "cnn": cnn,
        "resnet18": resnet18
    }
    for model_name, model in models.items():
        if model_name == "mlp":
            train_dataloader, val_dataloader, test_dataloader = init_dataloaders(flat=True)
        else:
            train_dataloader, val_dataloader, test_dataloader = init_dataloaders(flat=False)

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
                             name=f"mnist_{model_name}")

        load_lightning_model_and_test(model=model,
                                      test_dataloader=test_dataloader,
                                      device=device,
                                      name=f"mnist_{model_name}",
                                      version_no=0)
