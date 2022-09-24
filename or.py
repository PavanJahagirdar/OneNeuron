from utils.model import perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np


def main(data, model_name, plot_name, eta, epochs):

    df = pd.DataFrame(data)
    df

    X,y = prepare_data(df)

    model = perceptron(eta= eta, epochs= epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=model_name)
    save_plot(df, plot_name, model)

if __name__=='__main__':
    OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1]
    }
    ETA = 0.3 # should be between 0 and 1
    EPOCHS = 10
    
    main( OR,  "or.py", "or.png", ETA, EPOCHS)
