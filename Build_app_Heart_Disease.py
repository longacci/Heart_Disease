from flask import Flask,render_template, request
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = pickle.load(open('./Heart_Disease_model.pkl','rb'))
scale = MinMaxScaler()
@app.route("/")
def hello():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    age = int(request.form["Age"])
    sex = int(request.form["Sex"])
    chest_pain_type = int(request.form["chest_pain_type"])
    bp = int(request.form["bp"])
    cholesterol = int(request.form["cholesterol"])
    fbs = int(request.form["fbs"])
    ekg = int(request.form["ekg"])
    max_hr = int(request.form["max_hr"])
    exercise_angina = int(request.form["exercise_angina"])
    st_depression = request.form['st_depression']
    slope_of_st = int(request.form["slope_of_st"])
    number_of_vessels_fluro = int(request.form["number_of_vessels_fluro"])
    thallium = int(request.form['thallium'])
    data_sc = scale.fit_transform([[age,sex,chest_pain_type,bp,cholesterol,fbs,ekg,max_hr,exercise_angina,st_depression,slope_of_st,number_of_vessels_fluro,thallium]])
    result = model.predict(data_sc)[0]
    if result == 1:
        return render_template('index.html', predict_text=f"Kết quả dự đoán có khả năng mắc bệnh")
    else:
        return render_template('index.html', predict_text=f"Kết quả dự đoán không có khả năng mắc bệnh")
if __name__ == '__main__':
    app.run()