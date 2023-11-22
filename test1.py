# https://wings2pc.tistory.com/entry/%EC%9B%B9-%EC%95%B1%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%94%8C%EB%9D%BC%EC%8A%A4%ED%81%ACPython-Flask-URL-variablename

from flask import Flask, request,render_template
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/method', methods=['GET', 'POST'])
def method():
    if request.method == 'GET':
        return "GET으로 전달"
    else:
        return "POST로 전달"
    
@app.route('/image')
def kona():
    return render_template('car.html',image_file='static/img/kona.jpg')    

@app.route('/angle')
def angle():
    return render_template('randomangle.html')    

@app.route('/rear')
def camera1():
    return render_template('cam1.html')


if __name__ == '__main__':
    app.run(debug=True)