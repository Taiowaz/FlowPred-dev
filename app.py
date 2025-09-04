from flask import Flask, request, jsonify
import pandas as pd
import sys
import os
import torch
import logging
from gevent import monkey
monkey.patch_all()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pred.pred import predict

app = Flask(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

@app.route('/flow_predict', methods=['POST'])
def flow_predict():
    try:
        # Get JSON data from request
        input_json = request.get_json(force=True)

        kpi_time=input_json.get('kpi_time', [])
        kpi_value = input_json.get('kpi_value', [])
    
        # Convert JSON to DataFrame
        input_df = pd.DataFrame({
            "kpi_time": input_json.get('kpi_time', []),
            "kpi_value": pd.to_numeric(input_json.get('kpi_value', []), errors='coerce'),
        }
        )
        
        pred_mode = input_json.get('pred_mode', 'normal_predict')
        print (f"\n>>>>>>>>>>>Prediction mode: {pred_mode}")
        # Validate required columns
        required_columns = ['kpi_time', 'kpi_value']
        if not all(col in input_df.columns for col in required_columns):
            return jsonify({
                'error': 'Missing required fields',
                'required_fields': required_columns,
                'received_fields': list(input_df.columns)
            }), 400

        # Perform prediction
        try:
            his_hour = input_json.get('his_hour', 24)  # Default to 24 if not provided
            pred_hour = input_json.get('pred_hour', 6)  # Default to 24 if not provided
            spot_id = input_json.get('spot_id', -1)  # Get spot_id from the first row
            output_df = predict(spot_id,his_hour, pred_hour, input_df)
            output_json = output_df.to_dict(orient='records')
            return jsonify(output_json)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({
                'error': 'Prediction failed',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({
            'error': 'Request processing failed',
            'details': str(e)
        }), 500

@app.route('/test', methods=['POST'])
def test_post():
    data = request.json
    return jsonify(data), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'gpu_device': str(device)
    })

