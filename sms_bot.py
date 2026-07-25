from flask import Flask, request
import africastalking
from model_new import model   # Using your existing model file

# ========== Africa's Talking Credentials ==========
# You will replace these later
username = "sandbox"          # Temporary for testing
api_key = "atsk_9e153b3eee2397c6dfa5c0ceda51c2393b1d820c1c2146452fccf223b52db168a9e7174dd" # You will get this later

africastalking.initialize(username, api_key)
sms = africastalking.SMS

app = Flask(__name__)

@app.route('/sms', methods=['POST'])
def sms_callback():
    from_number = request.values.get('from')
    message = request.values.get('text', '').strip().lower()

    try:
        # Expected format from farmer:
        # moisture 35 temp 32 rain 5 crop maize
        parts = message.split()

        data = {
            'Soil_Moisture': float(parts[1]),
            'Temperature_C': float(parts[3]),
            'Rainfall_mm': float(parts[5]),
            'Humidity': 65,
            'Wind_Speed_kmh': 8,
            'Soil_Type': 'Loamy',
            'Crop Type': parts[7].capitalize(),
            'Crop_Growth_Stage': 'Vegetative'
        }

        result = model.predict(data)

        reply = f"Recommendation: {result['recommendation']}\nConfidence: {result['confidence']:.0%}"

    except Exception as e:
        reply = (
            "Please send in this format:\n"
            "moisture 35 temp 32 rain 5 crop maize\n\n"
            "Example: moisture 28 temp 31 rain 2 crop rice"
        )

    # Send reply
    try:
        sms.send(reply, [from_number])
    except:
        print("Reply:", reply)

    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)