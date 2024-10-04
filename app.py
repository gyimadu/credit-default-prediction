from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def predict_default():
    if request.method == 'POST':
        user_input = [
            request.form['limit_bal'],
            request.form['sex'],
            request.form['education'],
            request.form['marriage'],
            request.form['age'],
            request.form['pay_0'],
            request.form['pay_2'],
            request.form['pay_3'],
            request.form['pay_4'],
            request.form['pay_5'],
            request.form['pay_6'],
            request.form['bill_amt1'],
            request.form['bill_amt2'],
            request.form['bill_amt3'],
            request.form['bill_amt4'],
            request.form['bill_amt5'],
            request.form['bill_amt6'],
            request.form['pay_amt1'],
            request.form['pay_amt2'],
            request.form['pay_amt3'],
            request.form['pay_amt4'],
            request.form['pay_amt5'],
            request.form['pay_amt6']
        ]
        
        user_input = np.array(user_input, dtype=float).reshape(1, -1)

        prediction = model.predict(user_input)[0]

        return render_template('result.html', prediction=prediction)
    
    return render_template('form.html')

if __name__=='__main__':
    app.run(debug=True)
    