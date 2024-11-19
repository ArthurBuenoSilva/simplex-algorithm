from flask import render_template, request

from app.simplex import bp
from app.simplex.controller import controller
from app.extensions import socketio


@bp.route('/')
def index():
    return render_template("config.html")


@bp.route('/generate_result', methods=['POST'])
def generate_result():
    data = request.get_json()
    var_qtd = int(data.get('varQtd'))
    objective_function = data.get('objectiveFunction')
    restrictions = data.get('restrictions')

    controller.var_qtd = var_qtd
    controller.objective_func = objective_function
    controller.restrictions = restrictions

    return {}, 200


@bp.route('/result', methods=['GET'])
def result():
    context = {
        "result": controller.generate_result(),
        "var_qtd": controller.var_qtd,
    }
    return render_template("result.html", context=context)


@socketio.on('get-result')
def get_result():
    socketio.emit('result', controller.generate_result())


@socketio.on('post-optimization')
def post_optimization(data):
    op_result = controller.post_optimization(data.get('deltas', []))

    socketio.emit('post-optimization-result', op_result)