from fastapi import FastAPI
from fastapi.responses import FileResponse
from utils import run_train, load_dataset, run_predict


app = FastAPI()

@app.get("/")
def root():
 return {"message": "Hello World"}

@app.get("/training")
def training(n_train:int, batch_size:int, n_epochs:int, n_early_stop:int, save_losses:bool):
    train_loader, val_loader = load_dataset(n_train, batch_size)
    train_losses, val_losses = run_train(n_epochs=n_epochs, train_loader=train_loader, val_loader=val_loader, 
                                        batch_size=batch_size, n_train=n_train, n_early_stop=n_early_stop, save_losses=save_losses)
    return FileResponse(path="./results/losses.png", filename="losses.png", media_type="")

@app.get("/prediction")
def prediction(run: bool):
    if run:
        run_predict()
        return FileResponse(path="./results/predictions.txt", filename="predictions.txt", media_type='text/plain')