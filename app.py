from flask import Flask,request,jsonify
import json
import joblib
import pandas as pd
import pickle


app= Flask(__name__)
# model=joblib.load('model.apk')
# import pickle
# pickle.dump(model, open("model2.pkl","wb"))
# lr=pickle.load(open('model2.pkl','rb'))
@app.route('/predict',methods=['POST'])
def prediction():

	data=request.json
	data=pd.DataFrame(data)
	#data['AGE'] = pd.cut(data['AGE'], 6, labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
	dumData=pd.get_dummies(data)
	finalData=pd.concat(dumData)

	prediction= lr.predict(finalData)
	return jsonify({'prediction':str(prediction)})





if __name__ == '__main__':
	model=joblib.load('model.apk')
	app.run_server(debug=True)