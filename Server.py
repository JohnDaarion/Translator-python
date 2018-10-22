from flask import Flask, request
import numpy as np
import tensorflow as tf
import operator

app = Flask(__name__)
session = tf.Session()
saver = tf.train.import_meta_graph('network-model.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))


@app.route('/hello', methods = ['GET'])
def hello():
    return 'Hi!'


@app.route('/translate', methods = ['POST'])
def translate():
    content = request.json
    array = np.reshape(content['image'], (1, 256, 256, 3))
    array = array.astype(np.float32)
    array = np.multiply(array, 1.0 / 255.0)
    z_pred = session.run("y_pred:0", feed_dict={"x:0": array})
    prediction = {'A': z_pred[0][0], 'B': z_pred[0][1], 'C': z_pred[0][2], '0': z_pred[0][3]}
    text = max(prediction.items(), key=operator.itemgetter(1))[0]
    return text


if __name__ == '__main__':
    app.run(debug=True)


