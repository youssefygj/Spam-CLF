from flask import Flask, render_template,request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


loaded_model = pickle.load(open("finalized_model.sav",'rb'))

app = Flask(__name__)
@app.route("/")
def form():
	return render_template('home11.html')

@app.route('/',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['msg']
		data = [message]
		prediction=loaded_model.predict(data)[0]
			
	return render_template('result11.html', res = prediction)

if __name__ == "__main__":
	app.run(debug = True)




