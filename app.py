from flask import Flask, request, jsonify
import pandas as pd
from indicators import indicators, FEATURE_COLUMNS

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/indicator', methods=['POST'])
def indicator():
    req = request.get_json()
    symbol   = req['symbol']
    interval = req['interval']
    data     = req['data']

    df = pd.DataFrame({
        'open':   data['open'],
        'high':   data['high'],
        'low':    data['low'],
        'close':  data['close'],
        'volume': data['volume'],
    })

    timestamps = req.get('timestamps')  # optional ISO strings, enables hour_sin/hour_cos
    features = indicators(df, timestamps=timestamps)

    return jsonify({
        'symbol':   symbol,
        'interval': interval,
        'columns':  FEATURE_COLUMNS,
        'data':     features.to_dict(orient='records'),
    })
