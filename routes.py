from flask import Flask, render_template, request, jsonify
from modelo import SistemaChatEducativo

app = Flask(__name__)
sistema_chat = SistemaChatEducativo()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.get_json()
    pergunta = data.get('pergunta', '')
    
    if not pergunta:
        return jsonify({'erro': 'Pergunta não fornecida'}), 400
    
    resposta = sistema_chat.responder_pergunta(pergunta)
    
    return jsonify({
        'pergunta': pergunta,
        'resposta': resposta
    })

if __name__ == '__main__':
    app.run(debug=True)