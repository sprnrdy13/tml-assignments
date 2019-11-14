from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.config import Config
import uvicorn
import os
from fastai import *
from fastai.vision import *
import urllib

app = Starlette(debug=True)

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])

### EDIT CODE BELOW ###

answer_question_1 = """ 
Underfitting: performs poorly on training data, and poor generalization on other data

Overfitting: performs well on training data, but poor generalization to other data
"""

answer_question_2 = """ 
Given a function (or dataset) and a starting point (line of best fit/prediction), the differences between the two can be minimized by using Mean squared error (MSE). However, this process is tedious and calculus can generalize and minimize via gradient descent. 
The prediction will have a calculated gradient which will be applied to the prediction to adjust. This process will iterate until the gradient reaches a minimum. Once this occurs the prediction is optimized for the function. 


"""

answer_question_3 = """ 
The goal of regression is to create a prediction of a dependent variable based upon an independent variable. For this problem its classification of classes (yeezy).  
""

## Replace none with your model
pred_model = None 

@app.route("/api/answers_to_hw", methods=["GET"])
async def answers_to_hw(request):
    return JSONResponse([answer_question_1, answer_question_2, answer_question_3])

@app.route("/api/class_list", methods=["GET"])
async def class_list(request):
    return JSONResponse(learn.data.classes)

@app.route("/api/classify", methods=["POST"])
async def classify_url(request):
    body = await request.json()
    url_to_predict = body["url"]

    ## Make your prediction and store it in the preds variable
    bytes = await get_bytes(url_to_predict)
    img = open_image(BytesIO(bytes))
    pred_class, pred_idx, losses = pred_model.predict(img)
    
    return JSONResponse({
        "predictions": sorted(
        zip(learn.data.classes, map(float, losses)),
        key=lambda p: p[1], 
        reverse=True
    })

### EDIT CODE ABOVE ###

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ['PORT']))
