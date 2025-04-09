from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Dummy encoding map — replace keys/values based on your training dataset
label_encoding_map = {
    'Male': 0,
    'Female': 1,
    'Science': 0,
    'Commerce': 1,
    'Arts': 2,
    'Yes': 1,
    'No': 0,
    # Add more mappings as needed
}

@app.route('/')
def career():
    return render_template("hometest.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print("Raw form data:", result)
        res = result.to_dict(flat=True)
        print("Form dict:", res)

        # Convert values: map strings to numbers or cast to float
        arr = []
        for value in res.values():
            if value in label_encoding_map:
                arr.append(label_encoding_map[value])
            else:
                try:
                    arr.append(float(value))
                except ValueError:
                    arr.append(0)  # default fallback value if unexpected string

        data = np.array(arr).reshape(1, -1)
        print("Formatted input data:", data)

        # Load model
        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))

        # Make predictions
        predictions = loaded_model.predict(data)
        print("Top prediction:", predictions)
        
        pred = loaded_model.predict_proba(data)
        print("Probabilities:", pred)

        pred = pred > 0.05  # threshold
        res_indices = {i: j for i, j in enumerate(range(pred.shape[1])) if pred[0, j]}

        final_res = {}
        index = 0
        for key, value in res_indices.items():
            if value != predictions[0]:
                final_res[index] = value
                print('Suggested career:', value)
                index += 1

        jobs_dict = {
            0: 'AI ML Specialist',
            1: 'API Integration Specialist',
            2: 'Application Support Engineer',
            3: 'Business Analyst',
            4: 'Customer Service Executive',
            5: 'Cyber Security Specialist',
            6: 'Data Scientist',
            7: 'Database Administrator',
            8: 'Graphics Designer',
            9: 'Hardware Engineer',
            10: 'Helpdesk Engineer',
            11: 'Information Security Specialist',
            12: 'Networking Engineer',
            13: 'Project Manager',
            14: 'Software Developer',
            15: 'Software Tester',
            16: 'Technical Writer'
        }

        predicted_job = predictions[0]
        return render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=predicted_job)

if __name__ == '__main__':
    app.run(debug=True)
