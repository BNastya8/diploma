from flask import Flask, request, jsonify
import pickle
import numpy as np

# Загружаем модель из файла и десериализуем ее
with open("E:/course/diploma/web/model/model_forest.pkl", "rb") as pkl_file:
    model = pickle.load(pkl_file)

app = Flask(__name__)


@app.route('/')
def index():
    message = "Сервер запущен!"
    return message

@app.route("/add", methods=["POST"])


def predict():
    # Получаем запрос от клиента и выполняем предсказание
    test_data = np.array(request.json).reshape(1, -1) # преобразуем в датафрейм
    prediction = model.predict(test_data)
    target = np.exp(prediction)
    rounded_target = np.round(target) 
    # Возвращаем предсказание
    return jsonify({"result'": rounded_target})


if __name__ == "__main__":
    app.run("localhost", 5000)
    
    
